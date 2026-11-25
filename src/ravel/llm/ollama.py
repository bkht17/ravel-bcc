"""Локальный режим: документ не покидает машину."""

from __future__ import annotations

from pathlib import Path

import httpx
import ollama
from pydantic import BaseModel

from ravel.llm.base import SEED, LLMClient, LLMError, LLMUnavailable, TransientError


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, model: str, host: str, cache_dir: Path | None):
        super().__init__(model, cache_dir)
        self._client = ollama.Client(host=host)
        self.host = host

    def _raw(self, system: str, user: str, schema: type[BaseModel]) -> str:
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema.model_json_schema(),
                options={"temperature": 0, "seed": SEED, "num_ctx": 16384},
            )
        except httpx.ConnectError as e:
            raise LLMUnavailable(f"Ollama не отвечает на {self.host}. Запустите `ollama serve`.") from e
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMUnavailable(f"Модель {self.model} не скачана: `ollama pull {self.model}`") from e
            if e.status_code >= 500:
                raise TransientError(f"Ollama {e.status_code}: {e.error}") from e
            raise LLMError(f"Ollama {e.status_code}: {e.error}") from e
        return resp.message.content or ""
