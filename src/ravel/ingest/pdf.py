"""PDF → markdown постранично. Битые и зашифрованные файлы не роняют прогон."""

from __future__ import annotations

import contextlib
import io

import pymupdf

from ravel.ingest.sniff import garbage_ratio
from ravel.models import IngestIssue, IssueLevel

GARBAGE_THRESHOLD = 0.3


def pdf_pages(data: bytes) -> tuple[list[tuple[int, str]], list[IngestIssue]]:
    """Вернуть [(номер страницы, markdown)] и список проблем чтения."""
    issues: list[IngestIssue] = []
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # FileDataError и прочие ошибки MuPDF
        return [], [
            IngestIssue(level=IssueLevel.fatal, code="broken_pdf", message=f"PDF не открывается: {e}")
        ]

    with doc:
        if doc.needs_pass:
            return [], [
                IngestIssue(
                    level=IssueLevel.fatal,
                    code="encrypted_pdf",
                    message="PDF защищён паролем. Снимите защиту и запустите проверку снова.",
                )
            ]
        if doc.is_repaired:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="repaired_pdf",
                    message="Структура PDF повреждена и восстановлена при чтении; часть текста может быть потеряна.",
                )
            )
        if doc.page_count == 0:
            return [], [IngestIssue(level=IssueLevel.fatal, code="empty_pdf", message="В PDF нет страниц.")]

        pages = _pages_via_pymupdf4llm(doc, issues) or _pages_plain(doc, issues)

    good: list[tuple[int, str]] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        ratio = garbage_ratio(text)
        if ratio > GARBAGE_THRESHOLD:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="low_text_quality",
                    message=f"Текст страницы извлечён с искажениями (доля мусора {ratio:.0%}); страница пропущена.",
                    location=f"стр. {page_no}",
                )
            )
            continue
        good.append((page_no, text))

    if not good:
        issues.append(
            IngestIssue(
                level=IssueLevel.fatal,
                code="no_text_layer",
                message="В PDF нет читаемого текстового слоя (скан?). OCR пока не поддерживается.",
            )
        )
    return good, issues


def _pages_via_pymupdf4llm(doc: pymupdf.Document, issues: list[IngestIssue]) -> list[tuple[int, str]]:
    try:
        import pymupdf4llm

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    except Exception as e:
        issues.append(
            IngestIssue(
                level=IssueLevel.warning,
                code="layout_fallback",
                message=f"Разметка PDF не распознана ({type(e).__name__}); заголовки и таблицы могут потеряться.",
            )
        )
        return []
    pages = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        page_no = meta.get("page_number") or meta.get("page") or idx
        pages.append((int(page_no), chunk.get("text") or ""))
    return pages


def _pages_plain(doc: pymupdf.Document, issues: list[IngestIssue]) -> list[tuple[int, str]]:
    pages = []
    for idx in range(doc.page_count):
        try:
            pages.append((idx + 1, doc[idx].get_text("text")))
        except Exception as e:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="page_unreadable",
                    message=f"Страница не читается: {e}",
                    location=f"стр. {idx + 1}",
                )
            )
    return pages
