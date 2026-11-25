"""Общая логика клиентов: кэш, повторы при сбоях, починка ответа не по схеме."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ravel.fsutil import atomic_write
from ravel.prompts import PROMPT_VERSION

T = TypeVar("T", bound=BaseModel)
SEED = 17


class LLMError(Exception):
    """Ответ не получен или не разобран после всех попыток."""


class LLMUnavailable(LLMError):
    """Провайдер недоступен вообще (нет ключа, сервер не запущен)."""


class TransientError(LLMError):
    """Временный сбой (429, 5xx, таймаут): стоит повторить."""


class LLMClient(ABC):
    provider: str = "?"

    def __init__(self, model: str, cache_dir: Path | None):
        self.model = model
        self.cache_dir = cache_dir
        self.calls = 0
        self.cache_hits = 0

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @abstractmethod
    def _raw(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Один запрос к модели, вернуть JSON-строку."""

    def complete(self, system: str, user: str, schema: type[T]) -> T:
        key = self._key(system, user, schema)
        cached = self._cache_get(key)
        if cached is not None:
            try:
                self.cache_hits += 1
                return schema.model_validate_json(cached)
            except ValidationError:
                pass  # схема поменялась — перезапросим

        raw = self._call(system, user, schema)
        try:
            result = schema.model_validate_json(raw)
        except ValidationError as e:
            # одна попытка починки: показываем модели её ошибку
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
        self._cache_put(key, raw)
        return result

    @retry(
        retry=retry_if_exception_type(TransientError),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call(self, system: str, user: str, schema: type[BaseModel]) -> str:
        self.calls += 1
        return self._raw(system, user, schema)

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
