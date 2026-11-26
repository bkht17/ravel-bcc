"""Провайдеры LLM за одним интерфейсом: complete(system, user, schema) -> schema."""

from __future__ import annotations

import os
from pathlib import Path

from ravel.llm.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLMClient,
    LLMError,
    LLMUnavailable,
    QuotaExhausted,
    TransientError,
    TruncatedResponse,
    env_number,
)

# Стабильная Flash, бесплатная на Standard-тарифе; gemini-2.5-flash с 01.12.2026 Google перенаправляет
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Запасные модели на случай 503 «high demand» или исчерпанной квоты основной: полная Flash, затем Lite.
# Lite слабее, но отвечает, когда полные Flash перегружены. GEMINI_FALLBACK_MODELS= отключает переход.
DEFAULT_GEMINI_FALLBACKS = "gemini-3.6-flash,gemini-3.5-flash-lite"


def model_list(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip() and m.strip().lower() != "none"]


def get_client(provider: str, model: str | None = None, cache_dir: Path | None = None) -> LLMClient | None:
    """None означает «без LLM»: работают только детерминированные проверки."""
    provider = provider.lower()
    cache = cache_dir or Path(os.getenv("RAVEL_CACHE_DIR", ".cache/llm"))
    if provider == "none":
        return None
    max_tokens = env_number("RAVEL_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS, int, 256)
    if provider == "gemini":
        from ravel.llm.gemini import DEFAULT_TIMEOUT, GeminiClient

        key = os.getenv("GEMINI_API_KEY", "")
        if not key or key == "your_gemini_api_key_here":
            raise LLMUnavailable("Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и впишите ключ.")
        return GeminiClient(
            model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
            key,
            cache,
            timeout=env_number("RAVEL_LLM_TIMEOUT", DEFAULT_TIMEOUT, float, 1.0),
            max_output_tokens=max_tokens,
            fallback_models=model_list(os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS)),
        )
    if provider == "ollama":
        from ravel.llm.ollama import DEFAULT_TIMEOUT, OllamaClient

        return OllamaClient(
            model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
            os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            cache,
            timeout=env_number("RAVEL_LLM_TIMEOUT", DEFAULT_TIMEOUT, float, 1.0),
            max_output_tokens=max_tokens,
        )
    raise LLMUnavailable(f"Неизвестный провайдер: {provider}. Допустимо: gemini, ollama, none.")


__all__ = [
    "DEFAULT_GEMINI_FALLBACKS",
    "DEFAULT_GEMINI_MODEL",
    "LLMClient",
    "LLMError",
    "LLMUnavailable",
    "QuotaExhausted",
    "TransientError",
    "TruncatedResponse",
    "get_client",
]
