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
