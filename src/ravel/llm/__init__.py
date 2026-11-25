"""Провайдеры LLM за одним интерфейсом: complete(system, user, schema) -> schema."""

from __future__ import annotations

import os
from pathlib import Path

from ravel.llm.base import LLMClient, LLMError, LLMUnavailable


def get_client(provider: str, model: str | None = None, cache_dir: Path | None = None) -> LLMClient | None:
    """None означает «без LLM»: работают только детерминированные проверки."""
    provider = provider.lower()
    cache = cache_dir or Path(os.getenv("RAVEL_CACHE_DIR", ".cache/llm"))
    if provider == "none":
        return None
    if provider == "gemini":
        from ravel.llm.gemini import GeminiClient

        key = os.getenv("GEMINI_API_KEY", "")
        if not key or key == "your_gemini_api_key_here":
            raise LLMUnavailable("Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и впишите ключ.")
        return GeminiClient(model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), key, cache)
    if provider == "ollama":
        from ravel.llm.ollama import OllamaClient

        return OllamaClient(
            model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
            os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            cache,
        )
    raise LLMUnavailable(f"Неизвестный провайдер: {provider}. Допустимо: gemini, ollama, none.")


__all__ = ["LLMClient", "LLMError", "LLMUnavailable", "get_client"]
