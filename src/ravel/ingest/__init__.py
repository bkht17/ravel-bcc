"""Чтение входного документа: любой файл → DocumentIR (+ список проблем чтения).

Функция никогда не бросает исключений из-за содержимого файла: всё, что не
получилось прочитать, попадает в DocumentIR.issues и затем в отчёт.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from charset_normalizer import from_bytes

from ravel.ingest.docx import docx_markdown
from ravel.ingest.fallback import parse_fallback
from ravel.ingest.markdown import normalize_text, parse_markdown
from ravel.ingest.pdf import read_pdf
from ravel.ingest.sniff import UnsupportedFormat, ext_format, garbage_ratio, sniff
from ravel.models import DocumentIR, IngestIssue, IssueLevel
from ravel.segment import Coverage, coverage

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
    alt_texts: list[tuple[dict[int, int], str]] = []  # другие варианты текста того же документа
    if fmt == "pdf":
        pdf = read_pdf(data)
        issues.extend(pdf.issues)
        page_of_line, md = _join_pages(pdf.pages)
        if pdf.plain:
            alt_texts.append(_join_pages(pdf.plain))
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

    def build(text: str, pages: dict[int, int], fallback: bool) -> DocumentIR:
        if fallback:
            t, bl = "", parse_fallback(text, pages)
        else:
            t, bl = parse_markdown(text, pages)
        return DocumentIR(
            title=t or path.stem,
            source_name=name,
            source_format=source_format,  # type: ignore[arg-type]
            sha256=sha,
            markdown=text,
            blocks=bl,
            issues=issues,
        )

    ir = build(md, page_of_line, False)
    if not ir.blocks:
        issues.append(
            IngestIssue(level=IssueLevel.fatal, code="no_content", message="В документе не найден текст.")
        )
        ir.issues = issues
        return ir

    cov = coverage(ir)
    if cov.low:
        # структура не распознана или почти весь текст не попал в требования: пробуем другие
        # варианты текста (для PDF — простой текстовый слой), затем нарезку по общим признакам
        # порядок от самого точного к самому грубому; берём первый с нормальным покрытием
        candidates = [(t, p, False) for p, t in alt_texts] + [(t, p, True) for t, p in [(md, page_of_line)]]
        candidates += [(t, p, True) for p, t in alt_texts]
        best, best_cov = ir, cov
        for text, pages, fb in candidates:
            cand = build(text, pages, fb)
            c = coverage(cand)
            if _better(c, best_cov):
                best, best_cov = cand, c
            if not best_cov.low:
                break
        if best is not ir:
            ir, cov = best, best_cov
            if not cov.low:
                _note_fallback(issues)
            else:
                issues.append(
                    IngestIssue(
                        level=IssueLevel.warning,
                        code="fallback_segmentation",
                        message="Структура документа не распознана; использована нарезка по общим признакам "
                        "(идентификаторы, нумерация, маркеры, абзацы).",
                    )
                )
    if cov.low:
        issues.append(_low_coverage_issue(cov))
    ir.issues = issues  # DocumentIR копирует список при создании, а проблемы добавлялись позже
    return ir


def _join_pages(pages: list[tuple[int, str]]) -> tuple[dict[int, int], str]:
    """Склеить страницы в один текст; вернуть (страница по номеру строки, текст)."""
    page_of_line: dict[int, int] = {}
    parts: list[str] = []
    line = 1
    for page_no, text in pages:
        text = normalize_text(text).strip("\n") + "\n"
        n = text.count("\n") + 1
        for k in range(line, line + n):
            page_of_line[k] = page_no
        parts.append(text)
        line += n
    return page_of_line, "\n".join(parts)


def _better(new: Coverage, old: Coverage) -> bool:
    if new.low != old.low:
        return not new.low
    return (new.requirements, new.share) > (old.requirements, old.share)


def _low_coverage_issue(cov: Coverage) -> IngestIssue:
    return IngestIssue(
        level=IssueLevel.warning,
        code="low_coverage",
        message=f"В требования и явно служебные блоки попало {cov.share:.0%} текста документа "
        f"({cov.covered_chars} из {cov.total_chars} символов без пробелов), требований выделено: "
        f"{cov.requirements}. Документ прочитан частично, возможна потеря требований.",
    )


def _note_fallback(issues: list[IngestIssue]) -> None:
    """Информационная пометка: резервная нарезка сработала и покрытие нормальное.

    Уровень info не входит в причины undecided (там только warning), поэтому прогон остаётся полным.
    """
    issues.append(
        IngestIssue(
            level=IssueLevel.info,
            code="fallback_segmentation",
            message="Структура документа не распознана; требования выделены по общим признакам "
            "(идентификаторы, нумерация, маркеры, абзацы).",
        )
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
