"""Общая логика клиентов: кэш, ограниченные повторы, бюджет запросов, починка ответа не по схеме."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import RetryCallState, Retrying, retry_if_exception_type, wait_exponential_jitter

from ravel.fsutil import atomic_write
from ravel.prompts import PROMPT_VERSION

T = TypeVar("T", bound=BaseModel)
N = TypeVar("N", int, float)
SEED = 17

DEFAULT_MAX_CALLS = 50  # запросов к провайдеру за прогон, RAVEL_MAX_CALLS; 0 — без ограничения
DEFAULT_MAX_OUTPUT_TOKENS = 16384  # RAVEL_MAX_OUTPUT_TOKENS; у Gemini сюда входят и токены «размышлений»
MAX_ATTEMPTS = 3  # попыток на один запрос, считая первую: сбойный этап тратит не больше трёх вызовов
MAX_RETRY_WAIT = 60.0  # секунд; если провайдер просит ждать дольше, повтор не имеет смысла
ERROR_TEXT_LIMIT = 300  # символов текста ошибки провайдера в отчёте

_HIDDEN = "[ключ скрыт]"
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")
_URL_KEY_RE = re.compile(r"([?&](?:key|api_key)=)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Вырезать ключи: значение GEMINI_API_KEY, переданные секреты, всё похожее на AIza…, key=… в URL."""
    text = _URL_KEY_RE.sub(r"\1" + _HIDDEN, text)
    for secret in (*secrets, os.getenv("GEMINI_API_KEY", "")):
        if secret and len(secret) >= 8:
            text = text.replace(secret, _HIDDEN)
    return _GOOGLE_KEY_RE.sub(_HIDDEN, text)


def provider_text(text: object, secrets: Iterable[str] = (), limit: int = ERROR_TEXT_LIMIT) -> str:
    """Текст ошибки провайдера для исключения и отчёта: без ключей, в одну строку, не длиннее limit."""
    clean = " ".join(redact(str(text), secrets).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


class LLMError(Exception):
    """Ответ не получен или не разобран после всех попыток.

    model_issue — что случилось именно с этой моделью («перегружена», «дневная квота исчерпана»).
    Непустое значение значит, что запасная модель может ответить; пустое — сбой общий (ключ, сеть,
    бюджет прогона, плохой запрос), и смена модели не поможет.
    """

    def __init__(self, message: str = "", *, model_issue: str = ""):
        # страховка на любом пути: текст исключения попадает в отчёт, ключа в нём быть не должно
        super().__init__(redact(str(message)))
        self.model_issue = model_issue


class LLMUnavailable(LLMError):
    """Провайдер недоступен вообще (нет ключа или сети, модель не найдена): дальше прогон идёт без LLM."""


class QuotaExhausted(LLMUnavailable):
    """Квота провайдера или бюджет запросов на прогон исчерпаны: новых запросов клиент не делает."""


class TruncatedResponse(LLMError):
    """Модель оборвала ответ (лимит токенов, фильтр безопасности, цитирование): результат неполный."""


class TransientError(LLMError):
    """Временный сбой (429, 5xx, таймаут, обрыв связи): стоит повторить, но не больше attempts раз.

    retry_after — сколько секунд просит подождать провайдер (Retry-After, RetryInfo.retryDelay).
    final — во что превращается ошибка, когда попытки кончились: LLMUnavailable и QuotaExhausted
    выключают LLM до конца прогона, LLMError проваливает только текущий этап.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        attempts: int = MAX_ATTEMPTS,
        final: type[LLMError] = LLMError,
        model_issue: str = "",
    ):
        super().__init__(message, model_issue=model_issue)
        self.retry_after = retry_after
        self.attempts = attempts
        self.final = final


class _ModelDown(Exception):
    """Внутренний сигнал _call: текущая модель не отвечает по своей причине, а запасная модель есть."""

    def __init__(self, issue: str):
        super().__init__(issue)
        self.issue = issue


def env_number(name: str, default: N, cast: Callable[[str], N], minimum: N) -> N:
    """Число из переменной окружения; мусор или значение меньше minimum — понятная ошибка, а не трейсбек."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        value = None
    if value is None or value < minimum:
        raise LLMUnavailable(f"{name}={raw!r}: ожидается число не меньше {minimum}.")
    return value


class LLMClient(ABC):
    provider: str = "?"
    max_attempts: int = MAX_ATTEMPTS
    max_retry_wait: float = MAX_RETRY_WAIT
    # пауза между повторами; в тестах подменяется, чтобы не ждать
    retry_sleep: Callable[[float], None] = staticmethod(time.sleep)
    # строки прогресса (CLI печатает их в stderr): без них долгий запрос неотличим от зависания
    on_event: Callable[[str], None] | None = None

    def __init__(
        self,
        model: str,
        cache_dir: Path | None,
        max_calls: int | None = None,
        fallback_models: Iterable[str] = (),
    ):
        self.model = model
        # Запасные модели по порядку: если текущая перегружена или исчерпала свою квоту, клиент
        # переходит на следующую до конца прогона
        self.fallback_models = [m for m in dict.fromkeys(fallback_models) if m and m != model]
        self.models_tried = [model]  # цепочка моделей за прогон, попадает в паспорт отчёта
        self.model_issues: list[str] = []  # почему уходили с каждой модели цепочки
        self.cache_dir = cache_dir
        self.max_calls = (
            env_number("RAVEL_MAX_CALLS", DEFAULT_MAX_CALLS, int, 0) if max_calls is None else max_calls
        )
        self.calls = 0
        self.cache_hits = 0
        self.exhausted: QuotaExhausted | None = None
        self._secrets: tuple[str, ...] = ()
        self._backoff = wait_exponential_jitter(initial=1, max=MAX_RETRY_WAIT)
        self._label = ""

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def models_label(self) -> str:
        """Модель для паспорта отчёта: с цепочкой переходов на запасные и их причинами."""
        if len(self.models_tried) == 1:
            return self.model
        return " → ".join(self.models_tried) + f" ({'; '.join(self.model_issues)})"

    @abstractmethod
    def _raw(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Один запрос к модели, вернуть JSON-строку. Сбои — только подклассы LLMError."""

    def _text(self, text: object) -> str:
        return provider_text(text, self._secrets)

    def _emit(self, message: str) -> None:
        if self.on_event:
            self.on_event(message)

    def complete(self, system: str, user: str, schema: type[T], label: str = "") -> T:
        """label — что проверяется этим запросом (например, «R-001–R-006»), только для прогресса."""
        what = label or schema.__name__
        key = self._key(system, user, schema)
        cached = self._cache_get(key)
        if cached is not None:
            try:
                result = schema.model_validate_json(cached)
                self.cache_hits += 1
                self._emit(f"{what}: из кэша")
                return result
            except ValidationError:
                pass  # схема поменялась — перезапросим

        self._label = what
        raw = self._call(system, user, schema)
        try:
            result = schema.model_validate_json(raw)
        except ValidationError as e:
            # одна попытка починки: показываем модели её ошибку
            self._emit(f"{what}: ответ не по схеме, прошу исправить")
            fix = (
                f"{user}\n\nТвой предыдущий ответ не прошёл проверку схемы:\n{str(e)[:1500]}\n"
                "Верни исправленный JSON строго по схеме."
            )
            raw = self._call(system, fix, schema)
            try:
                result = schema.model_validate_json(raw)
            except ValidationError as e2:
                raise LLMError(
                    f"Ответ модели не соответствует схеме {schema.__name__}: {e2.errors()[:3]}"
                ) from e2
        # ключ по модели, которая ответила: ответ запасной модели не должен выдаваться за ответ основной
        self._cache_put(self._key(system, user, schema), raw)
        return result

    # ---------------------------------------------------------------- retries

    def _call(self, system: str, user: str, schema: type[BaseModel]) -> str:
        while True:
            try:
                return self._call_model(system, user, schema)
            except _ModelDown as e:
                # текущая модель не отвечает, а запасная есть: повторяем тот же запрос на ней
                old, self.model = self.model, self.fallback_models.pop(0)
                self.models_tried.append(self.model)
                self.model_issues.append(f"{old}: {e.issue}")
                self._emit(f"{self._label}: {old} — {e.issue}, переключаюсь на {self.model}")
                cached = self._cache_get(self._key(system, user, schema))
                if cached is not None:  # запасная уже отвечала на этот запрос в прошлом прогоне
                    self.cache_hits += 1
                    self._emit(f"{self._label}: из кэша")
                    return cached

    def _call_model(self, system: str, user: str, schema: type[BaseModel]) -> str:
        tries = 0

        def attempt() -> str:
            nonlocal tries
            tries += 1
            return self._request(system, user, schema)

        retrying = Retrying(
            retry=retry_if_exception_type(TransientError),
            stop=self._should_stop,
            wait=self._retry_wait,
            sleep=self.retry_sleep,
            before_sleep=self._before_sleep,
            reraise=True,
        )
        try:
            return retrying(attempt)
        except TransientError as e:
            self._emit(f"{self._label}: не удалось после {tries} попыток")
            if e.model_issue and self.fallback_models:
                raise _ModelDown(e.model_issue) from e
            final = e.final(f"{e} (попыток: {tries})")
            if isinstance(final, QuotaExhausted):
                self.exhausted = final
            raise final from e
        except QuotaExhausted as e:
            if e.model_issue and self.fallback_models:
                raise _ModelDown(e.model_issue) from e  # квота этой модели; у запасной она своя
            self.exhausted = self.exhausted or e
            raise

    def _request(self, system: str, user: str, schema: type[BaseModel]) -> str:
        if self.exhausted is not None:
            raise QuotaExhausted(str(self.exhausted))
        if self.max_calls and self.calls >= self.max_calls:
            raise QuotaExhausted(
                f"Исчерпан бюджет запросов к LLM на прогон: {self.max_calls} (RAVEL_MAX_CALLS). "
                "Остальные проверки выполнены только по правилам."
            )
        self.calls += 1
        budget = f"/{self.max_calls}" if self.max_calls else ""
        timeout = getattr(self, "timeout", None)
        limit = f", ждём до {timeout:g} с" if timeout else ""
        self._emit(f"{self._label}: запрос {self.calls}{budget} к {self.model}{limit}…")
        started = time.monotonic()
        raw = self._raw(system, user, schema)
        self._emit(f"{self._label}: ответ за {time.monotonic() - started:.0f} с")
        return raw

    def _before_sleep(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0.0
        total = min(self.max_attempts, exc.attempts) if isinstance(exc, TransientError) else self.max_attempts
        reason = str(exc).splitlines()[0][:100] if exc else "сбой"
        self._emit(f"{self._label}: {reason} — повтор {state.attempt_number + 1}/{total} через {wait:.0f} с")

    def _should_stop(self, state: RetryCallState) -> bool:
        exc = state.outcome.exception() if state.outcome else None
        if not isinstance(exc, TransientError):
            return True
        if exc.retry_after is not None and exc.retry_after > self.max_retry_wait:
            return True  # провайдер просит ждать дольше, чем стоит держать прогон
        return state.attempt_number >= min(self.max_attempts, exc.attempts)

    def _retry_wait(self, state: RetryCallState) -> float:
        """Экспоненциальная пауза с джиттером, но не меньше, чем просит провайдер."""
        exc = state.outcome.exception() if state.outcome else None
        hint = exc.retry_after if isinstance(exc, TransientError) and exc.retry_after else 0.0
        return min(max(self._backoff(state), hint), self.max_retry_wait)

    # ------------------------------------------------------------------ cache

    def _key(self, system: str, user: str, schema: type[BaseModel]) -> str:
        payload = json.dumps(
            [self.provider, self.model, PROMPT_VERSION, SEED, system, user, schema.model_json_schema()],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _cache_put(self, key: str, raw: str) -> None:
        if self.cache_dir:
            atomic_write(self.cache_dir / f"{key}.json", raw)
