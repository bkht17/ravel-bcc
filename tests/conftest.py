from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "requirements.md"

MINI_MD = """# Мини-спецификация

## Функциональные требования

- Ф-1. Пользователь задаёт порог от 1 000 до 10 000 тенге включительно.
- Ф-2. При достижении порога клиент получает push-уведомление.

## Критерии приемки

| Условие | Результат |
| --- | --- |
| Порог 5 000, расходы 5 000 | Один push |
"""


@pytest.fixture
def sample() -> Path:
    return SAMPLE


def make_pdf(lines: list[str], **save_kw) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 18
    data = doc.tobytes(**save_kw)
    doc.close()
    return data


def make_docx(md_like: str) -> bytes:
    """Простой DOCX с заголовками, списком и таблицей — эквивалент MINI_MD."""
    from docx import Document

    d = Document()
    d.add_heading("Мини-спецификация", level=1)
    d.add_heading("Функциональные требования", level=2)
    d.add_paragraph(
        "Ф-1. Пользователь задаёт порог от 1 000 до 10 000 тенге включительно.", style="List Bullet"
    )
    d.add_paragraph("Ф-2. При достижении порога клиент получает push-уведомление.", style="List Bullet")
    d.add_heading("Критерии приемки", level=2)
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Условие", "Результат"
    t.cell(1, 0).text, t.cell(1, 1).text = "Порог 5 000, расходы 5 000", "Один push"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- сбои LLM (tests/test_failures.py)

FIXTURES = ROOT / "tests" / "fixtures"

# сценарий → как ведёт себя подменённый generate_content: исключение или текст ответа
LLM_FAILURE_SCENARIOS = (
    "403",
    "404",
    "429",
    "503",
    "connect_error",
    "read_timeout",
    "invalid_json",
    "empty",
)


def _gemini_response(text: str):
    from unittest.mock import MagicMock

    cand = MagicMock()
    cand.finish_reason = "STOP"
    resp = MagicMock()
    resp.candidates = [cand]
    resp.prompt_feedback = None
    resp.text = text
    return resp


def _scenario_effect(name: str):
    import httpx
    from google.genai import errors

    def api(code: int, status: str, message: str):
        return errors.APIError(code, {"error": {"code": code, "status": status, "message": message}})

    effects = {
        "403": {"side_effect": api(403, "PERMISSION_DENIED", "API key not valid")},
        "404": {"side_effect": api(404, "NOT_FOUND", "models/x is not found")},
        "429": {"side_effect": api(429, "RESOURCE_EXHAUSTED", "Quota exceeded")},
        "503": {"side_effect": api(503, "UNAVAILABLE", "The model is overloaded")},
        "connect_error": {"side_effect": httpx.ConnectError("connection refused")},
        "read_timeout": {"side_effect": httpx.ReadTimeout("read timed out")},
        "invalid_json": {"return_value": _gemini_response("{")},
        "empty": {"return_value": _gemini_response("")},
    }
    return effects[name]


@pytest.fixture
def no_retry_sleep(monkeypatch):
    """Паузы между повторами LLMClient обнуляются; monkeypatch вернёт штатное значение после теста."""
    from ravel.llm.base import LLMClient

    monkeypatch.setattr(LLMClient, "retry_sleep", staticmethod(lambda seconds: None))


@pytest.fixture
def gemini_failing(request, monkeypatch, no_retry_sleep):
    """(GeminiClient, MagicMock generate_content) для сценария из request.param (indirect-параметризация).

    Ключ фиктивный, кэша нет, бюджет вызовов задан явно."""
    from unittest.mock import MagicMock

    from ravel.llm.gemini import GeminiClient

    client = GeminiClient("gemini-test", "fake-key-123", None, max_calls=50)
    mock = MagicMock(**_scenario_effect(request.param))
    monkeypatch.setattr(client._client.models, "generate_content", mock)
    return client, mock
