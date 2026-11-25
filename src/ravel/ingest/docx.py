"""DOCX → markdown. При битом архиве достаём текст из document.xml напрямую."""

from __future__ import annotations

import html
import io
import re
import zipfile

from ravel.models import IngestIssue, IssueLevel

_HEADING_STYLE = re.compile(r"^(?:heading|заголовок)\s*(\d)$", re.IGNORECASE)


def docx_markdown(data: bytes) -> tuple[str, list[IngestIssue]]:
    issues: list[IngestIssue] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return _salvage(data, issues)

    with zf:
        bad = zf.testzip()
        if bad:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="docx_damaged_part",
                    message=f"Часть архива DOCX повреждена ({bad}); оформление может потеряться.",
                )
            )
        try:
            return _via_python_docx(data), issues
        except Exception as e:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="docx_fallback",
                    message=f"DOCX не разобран штатно ({type(e).__name__}); текст извлечён без структуры.",
                )
            )
            try:
                xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
            except Exception as e2:
                issues.append(
                    IngestIssue(level=IssueLevel.fatal, code="broken_docx", message=f"DOCX не читается: {e2}")
                )
                return "", issues
            return _xml_to_text(xml), issues


def _via_python_docx(data: bytes) -> str:
    from docx import Document
    from docx.table import Table

    document = Document(io.BytesIO(data))
    out: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            rows = [[c.text.replace("\n", " ").replace("|", "/").strip() for c in r.cells] for r in item.rows]
            if not rows:
                continue
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + " --- |" * len(rows[0]))
            out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
            out.append("")
            continue
        text = item.text.strip()
        if not text:
            continue
        style = (item.style.name if item.style is not None else "") or ""
        if m := _HEADING_STYLE.match(style):
            out.append("#" * int(m.group(1)) + " " + text)
        elif style.lower() == "title":
            out.append("# " + text)
        elif "list" in style.lower() or item._p.pPr is not None and item._p.pPr.numPr is not None:
            out.append("- " + text)
        else:
            out.append(text)
        out.append("")
    return "\n".join(out)


def _xml_to_text(xml: str) -> str:
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.DOTALL):
        runs = re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p, flags=re.DOTALL)
        text = html.unescape("".join(runs)).strip()
        if text:
            paras.append(text)
    return "\n\n".join(paras)


def _salvage(data: bytes, issues: list[IngestIssue]) -> tuple[str, list[IngestIssue]]:
    """Архив битый целиком: ищем фрагменты XML с текстом в сырых байтах."""
    raw = data.decode("utf-8", errors="ignore")
    text = _xml_to_text(raw)
    if text.strip():
        issues.append(
            IngestIssue(
                level=IssueLevel.warning,
                code="docx_salvaged",
                message="Архив DOCX повреждён; удалось восстановить только часть текста без структуры.",
            )
        )
        return text, issues
    issues.append(
        IngestIssue(level=IssueLevel.fatal, code="broken_docx", message="Архив DOCX повреждён и не читается.")
    )
    return "", issues
