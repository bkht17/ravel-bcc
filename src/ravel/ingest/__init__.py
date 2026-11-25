"""Чтение входного документа: любой файл → DocumentIR (+ список проблем чтения).

Функция никогда не бросает исключений из-за содержимого файла: всё, что не
получилось прочитать, попадает в DocumentIR.issues и затем в отчёт.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from charset_normalizer import from_bytes

from ravel.ingest.docx import docx_markdown
from ravel.ingest.markdown import normalize_text, parse_markdown
from ravel.ingest.pdf import pdf_pages
from ravel.ingest.sniff import UnsupportedFormat, ext_format, garbage_ratio, sniff
from ravel.models import DocumentIR, IngestIssue, IssueLevel

MAX_BYTES = 20 * 1024 * 1024


def load_document(path: Path) -> DocumentIR:
    issues: list[IngestIssue] = []
    name = path.name

    def fatal(code: str, message: str, sha: str = "", fmt: str = "text") -> DocumentIR:
        issues.append(IngestIssue(level=IssueLevel.fatal, code=code, message=message))
        return DocumentIR(
            title=path.stem,
            source_name=name,
            source_format=fmt,
            sha256=sha,
            markdown="",
            blocks=[],
            issues=issues,
        )

    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return fatal("not_found", f"Файл не найден: {path}")
    except IsADirectoryError:
        return fatal("is_directory", f"Ожидался файл, а не папка: {path}")
    except OSError as e:
        return fatal("unreadable", f"Файл не читается: {e}")

    sha = hashlib.sha256(data).hexdigest()
    if not data.strip():
        return fatal("empty_file", "Файл пустой.", sha)
    if len(data) > MAX_BYTES:
        return fatal("too_large", f"Файл больше {MAX_BYTES // 1024 // 1024} МБ.", sha)

    try:
        fmt = sniff(data)
    except UnsupportedFormat as e:
        return fatal(e.code, e.message, sha)

    expected = ext_format(path.suffix)
    if expected and expected != fmt:
        issues.append(
            IngestIssue(
                level=IssueLevel.warning,
                code="ext_mismatch",
                message=f"Расширение «{path.suffix}» не совпадает с содержимым (это {fmt}); файл прочитан как {fmt}.",
            )
        )

    page_of_line: dict[int, int] = {}
    if fmt == "pdf":
        pages, more = pdf_pages(data)
        issues.extend(more)
        parts: list[str] = []
        line = 1
        for page_no, text in pages:
            text = normalize_text(text).strip("\n") + "\n"
            n = text.count("\n") + 1
            for k in range(line, line + n):
                page_of_line[k] = page_no
            parts.append(text)
            line += n
        md = "\n".join(parts)
        source_format = "pdf"
    elif fmt == "docx":
        md, more = docx_markdown(data)
        issues.extend(more)
        md = normalize_text(md)
        source_format = "docx"
    else:
        text = _decode(data, issues)
        if text is None:
            return fatal("binary", "Файл не похож на текст: не удалось определить кодировку.", sha)
        md = normalize_text(text)
        source_format = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "text"
        ratio = garbage_ratio(md)
        if ratio > 0.3:
            return fatal(
                "garbage", f"Текст не читается (доля мусора {ratio:.0%}). Похоже на бинарный файл.", sha
            )

    if any(i.level == IssueLevel.fatal for i in issues):
        return DocumentIR(
            title=path.stem,
            source_name=name,
            source_format=source_format,
            sha256=sha,
            markdown="",
            blocks=[],
            issues=issues,
        )

    title, blocks = parse_markdown(md, page_of_line)
    if not blocks:
        issues.append(
            IngestIssue(level=IssueLevel.fatal, code="no_content", message="В документе не найден текст.")
        )
    return DocumentIR(
        title=title or path.stem,
        source_name=name,
        source_format=source_format,
        sha256=sha,
        markdown=md,
        blocks=blocks,
        issues=issues,
    )


def _decode(data: bytes, issues: list[IngestIssue]) -> str | None:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    best = from_bytes(data).best()
    if best is None:
        return None
    issues.append(
        IngestIssue(
            level=IssueLevel.warning,
            code="non_utf8",
            message=f"Файл не в UTF-8; кодировка определена как {best.encoding}.",
        )
    )
    return str(best)
