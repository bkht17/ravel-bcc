"""Локальный режим: документ не покидает машину."""

from __future__ import annotations

from pathlib import Path

import httpx
import ollama
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

DEFAULT_TIMEOUT = 600.0  # секунд: локальная модель на CPU отвечает долго, RAVEL_LLM_TIMEOUT


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        host: str,
        cache_dir: Path | None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_calls: int | None = None,
    ):
        super().__init__(model, cache_dir, max_calls)
        self.host = host
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        # timeout уходит в httpx.Client; по умолчанию у ollama его нет совсем
        self._client = ollama.Client(host=host, timeout=timeout)

    def _raw(self, system: str, user: str, schema: type[BaseModel]) -> str:
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema.model_json_schema(),
                options={
                    "temperature": 0,
                    "seed": SEED,
                    "num_ctx": 16384,
                    "num_predict": self.max_output_tokens,
                },
            )
        except (ConnectionError, httpx.ConnectError) as e:
            # ollama превращает httpx.ConnectError во встроенный ConnectionError
            raise LLMUnavailable(f"Ollama не отвечает на {self.host}. Запустите `ollama serve`.") from e
        except httpx.TimeoutException as e:
            # повтор того же запроса к той же локальной модели займёт столько же времени
            raise LLMUnavailable(
                f"Ollama не ответила за {self.timeout:g} с: увеличьте RAVEL_LLM_TIMEOUT или возьмите модель поменьше."
            ) from e
        except httpx.TransportError as e:
            raise TransientError(
                f"Связь с Ollama оборвалась ({type(e).__name__}): {self._text(e)}", final=LLMUnavailable
            ) from e
        except ollama.ResponseError as e:
            raise self._response_error(e) from e
        except Exception as e:  # прочие сбои клиента не должны ронять прогон
            raise LLMError(f"Ollama: сбой клиента ({type(e).__name__}): {self._text(e)}") from e
        if resp.done_reason == "length":
            raise TruncatedResponse(
                f"Ollama: ответ обрезан на лимите {self.max_output_tokens} токенов "
                "(RAVEL_MAX_OUTPUT_TOKENS); результат не принят"
            )
        text = resp.message.content or ""
        if not text.strip():
            raise TransientError("Ollama вернула пустой ответ", attempts=2, final=LLMError)
        return text

    def _response_error(self, e: ollama.ResponseError) -> LLMError:
        code, msg = e.status_code, self._text(e.error)
        if code == 404:
            return LLMUnavailable(f"Модель {self.model} не скачана: `ollama pull {self.model}`. {msg}")
        if code in (401, 403):
            return LLMUnavailable(f"Ollama {code}: нет доступа к модели {self.model}. {msg}")
        if code == 429:
            return TransientError(f"Ollama 429: {msg}", final=QuotaExhausted)
        if code >= 500:
            return TransientError(f"Ollama {code}: {msg}", final=LLMUnavailable)
        return LLMError(f"Ollama {code}: {msg}")
