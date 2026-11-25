from __future__ import annotations

from pathlib import Path

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ravel.llm.base import SEED, LLMClient, LLMError, LLMUnavailable, TransientError

_TRANSIENT = {408, 429, 500, 502, 503, 504}


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, api_key: str, cache_dir: Path | None):
        super().__init__(model, cache_dir)
        self._client = genai.Client(api_key=api_key)

    def _raw(self, system: str, user: str, schema: type[BaseModel]) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    seed=SEED,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except errors.APIError as e:
            if e.code in _TRANSIENT:
                raise TransientError(f"Gemini {e.code}: {e.message}") from e
            if e.code in (401, 403):
                raise LLMUnavailable(f"Gemini отклонил ключ ({e.code}): {e.message}") from e
            raise LLMError(f"Gemini {e.code}: {e.message}") from e
        if not resp.text:
            raise TransientError("Gemini вернул пустой ответ")
        return resp.text
