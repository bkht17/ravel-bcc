"""Промпты агентов. Меняете текст промпта — поднимайте PROMPT_VERSION (сбрасывает кэш)."""

from __future__ import annotations

from functools import cache
from importlib.resources import files

PROMPT_VERSION = "2026-11-27.1"


@cache
def load(name: str) -> str:
    return files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
