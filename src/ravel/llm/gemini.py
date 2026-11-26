"""Google Gemini: ответ строго по pydantic-схеме; любой сбой SDK превращается в подкласс LLMError."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ravel.llm.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    SEED,
    LLMClient,
    LLMError,
    LLMUnavailable,
    QuotaExhausted,
    TransientError,
    TruncatedResponse,
)

try:  # google-genai 3.x по умолчанию ходит через httpx2 — форк httpx со своими классами исключений
    import httpx2
except ImportError:  # pragma: no cover - httpx2 ставится вместе с google-genai
    httpx2 = None

DEFAULT_TIMEOUT = 180.0  # секунд на один запрос, RAVEL_LLM_TIMEOUT


def _exceptions(name: str) -> tuple[type[Exception], ...]:
    return tuple(getattr(module, name) for module in (httpx, httpx2) if module is not None)


_TIMEOUT = _exceptions("TimeoutException")  # ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
_TRANSPORT = _exceptions("TransportError")  # ConnectError, ReadError, RemoteProtocolError, ProxyError…
_REQUEST = _exceptions("RequestError")  # плюс DecodingError, TooManyRedirects

_RETRYABLE = {408, 500, 502, 503, 504}
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")  # google.protobuf.Duration: "38s", "1.5s"
_RETRY_IN_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

_FINISH_TEXT = {
    "MAX_TOKENS": "ответ обрезан на лимите выходных токенов ({limit}, RAVEL_MAX_OUTPUT_TOKENS)",
    "SAFETY": "ответ заблокирован фильтром безопасности",
    "RECITATION": "ответ остановлен: модель начала дословно цитировать чужой текст",
    "LANGUAGE": "ответ остановлен: язык не поддерживается моделью",
    "BLOCKLIST": "ответ заблокирован: запрещённые термины",
    "PROHIBITED_CONTENT": "ответ заблокирован: запрещённое содержимое",
    "SPII": "ответ заблокирован: в нём персональные данные",
    "MALFORMED_FUNCTION_CALL": "ответ остановлен: некорректный вызов функции",
}
_FINISH_OK = {"STOP", "FINISH_REASON_UNSPECIFIED"}


def _name(value: Any) -> str:
    """Имя значения enum SDK ('STOP', 'MAX_TOKENS'); неизвестные значения SDK тоже строки."""
    return str(getattr(value, "value", value) or "")


def _error_details(payload: Any) -> list[dict[str, Any]]:
    """Список google.rpc-деталей из тела ошибки: ErrorInfo, QuotaFailure, RetryInfo…"""
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        return []
    err = payload.get("error", payload)
    items = err.get("details") if isinstance(err, dict) else None
    return [d for d in items if isinstance(d, dict)] if isinstance(items, list) else []


def _retry_after(e: errors.APIError, details: list[dict[str, Any]]) -> float | None:
    """Пауза, которую просит провайдер: RetryInfo.retryDelay, заголовок Retry-After, «retry in Ns»."""
    for d in details:
        m = _DURATION_RE.match(str(d.get("retryDelay", "")))
        if m:
            return float(m.group(1))
    headers = getattr(getattr(e, "response", None), "headers", None)
    value = headers.get("retry-after") if headers is not None and hasattr(headers, "get") else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass  # HTTP-дата: проще подождать по экспоненте
    m = _RETRY_IN_RE.search(e.message or "")
    return float(m.group(1)) if m else None


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        cache_dir: Path | None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_calls: int | None = None,
    ):
        super().__init__(model, cache_dir, max_calls)
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._secrets = (api_key,)
        # HttpOptions.timeout — в миллисекундах. retry_options не задаём: у SDK тогда ровно одна попытка,
        # повторы делает LLMClient._call с общим бюджетом.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    def _raw(self, system: str, user: str, schema: type[BaseModel]) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            seed=SEED,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )
        try:
            resp = self._client.models.generate_content(model=self.model, contents=user, config=config)
        except errors.APIError as e:
            raise self._api_error(e) from e
        except _TIMEOUT as e:
            raise TransientError(
                f"Gemini не ответил за {self.timeout:g} с ({type(e).__name__}); "
                "лимит задаётся RAVEL_LLM_TIMEOUT",
                attempts=2,
                final=LLMUnavailable,
            ) from e
        except _TRANSPORT as e:
            raise TransientError(
                f"Нет связи с Gemini ({type(e).__name__}): {self._text(e)}", final=LLMUnavailable
            ) from e
        except _REQUEST as e:
            raise LLMError(f"Gemini: ошибка HTTP-запроса ({type(e).__name__}): {self._text(e)}") from e
        except Exception as e:  # прочие сбои SDK (разбор ответа и т.п.) не должны ронять прогон
            raise LLMError(f"Gemini: сбой SDK ({type(e).__name__}): {self._text(e)}") from e
        return self._response_text(resp)

    def _response_text(self, resp: types.GenerateContentResponse) -> str:
        candidates = resp.candidates or []
        if not candidates:
            feedback = resp.prompt_feedback
            blocked = _name(feedback.block_reason) if feedback else ""
            if blocked:
                note = self._text(feedback.block_reason_message or "") if feedback else ""
                raise TruncatedResponse(f"Gemini отказался обрабатывать запрос: {blocked}. {note}".strip())
        else:
            reason = _name(candidates[0].finish_reason)
            if reason and reason not in _FINISH_OK:
                why = _FINISH_TEXT.get(reason, "ответ остановлен досрочно").format(
                    limit=self.max_output_tokens
                )
                raise TruncatedResponse(f"Gemini: {why} [finish_reason={reason}]; результат не принят")
        text = resp.text
        if not text or not text.strip():
            raise TransientError("Gemini вернул пустой ответ", attempts=2, final=LLMError)
        return text

    def _api_error(self, e: errors.APIError) -> LLMError:
        code, status = e.code or 0, e.status or ""
        where = f"Gemini {code} {status}".strip()
        msg = self._text(e.message or e)
        details = _error_details(e.details)
        reasons = {str(d.get("reason", "")) for d in details}
        if code in (401, 403) or any(r.startswith("API_KEY") for r in reasons):
            return LLMUnavailable(f"{where}: ключ или доступ к API отклонён, проверьте GEMINI_API_KEY. {msg}")
        if code == 404:
            return LLMUnavailable(
                f"{where}: модель {self.model} не найдена или недоступна этому ключу, проверьте GEMINI_MODEL. {msg}"
            )
        if code == 429:
            return self._quota_error(e, where, msg, details)
        if code == 400 and status == "FAILED_PRECONDITION":
            return LLMUnavailable(f"{where}: Gemini API недоступен для этого ключа или региона. {msg}")
        if code in _RETRYABLE or code >= 500:
            return TransientError(
                f"{where}: {msg}", retry_after=_retry_after(e, details), final=LLMUnavailable
            )
        return LLMError(f"{where}: {msg}")

    def _quota_error(
        self, e: errors.APIError, where: str, msg: str, details: list[dict[str, Any]]
    ) -> LLMError:
        violations = [v for d in details for v in d.get("violations", []) if isinstance(v, dict)]
        quota_ids = [str(v.get("quotaId", "")) for v in violations if v.get("quotaId")]
        per_day = any("PerDay" in q for q in quota_ids)
        no_quota = any(str(v.get("quotaValue", "")) == "0" for v in violations)  # модели нет на этом тарифе
        ids = ", ".join(quota_ids) or "квота"
        if per_day:
            return QuotaExhausted(f"{where}: дневная квота Gemini исчерпана ({ids}). {msg}")
        if no_quota:
            return QuotaExhausted(
                f"{where}: на тарифе этого ключа у {self.model} нулевая квота ({ids}). {msg}"
            )
        return TransientError(
            f"{where}: превышен лимит запросов Gemini. {msg}",
            retry_after=_retry_after(e, details),
            final=QuotaExhausted,
        )
