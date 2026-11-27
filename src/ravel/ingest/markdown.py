"""Нормализованный markdown → список блоков с адресами.

Все форматы (md, txt, pdf, docx) сначала приводятся к markdown, затем разбираются
этим парсером. Поэтому один документ в разных форматах даёт одинаковую нарезку.
"""

from __future__ import annotations

import re
import unicodedata

from ravel.ingest.sentences import split_spans
from ravel.models import Block, BlockKind

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)(?:[-*+•]|\d+[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_INLINE = re.compile(r"(\*\*|__|`)")
# Строка начинается с маркера статуса или кода ответа: «**400.**», «404 —», «HTTP 500».
_STATUS_MARKER = re.compile(
    r"^\s*(?:\*\*[1-5]\d{2}\.?\*\*|[1-5]\d{2}\s*[—–]|[1-5]\d{2}\s+-\s|HTTP\s+[1-5]\d{2}\b)"
)


def is_status_marker(line: str) -> bool:
    return bool(_STATUS_MARKER.match(line))


def normalize_text(text: str) -> str:
    """Единая нормализация входного текста для всех форматов."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace(" ", " ").replace("­", "")
    text = text.replace("\t", "    ")
    # PDF-переносы: "требова-\nния" → "требования"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # лигатуры
    for lig, rep in {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff"}.items():
        text = text.replace(lig, rep)
    return text


def strip_inline(text: str) -> str:
    """Убрать inline-разметку markdown, оставив сам текст."""
    return _INLINE.sub("", text).strip()


_CELL_BREAK = re.compile(r"\s*<br\s*/?>\s*", re.IGNORECASE)


def _split_row(line: str) -> list[str]:
    """Ячейки строки таблицы. «\\|» — вертикальная черта внутри ячейки, а не граница;
    «<br>» — перенос строки внутри ячейки (так его пишут GFM и pymupdf4llm), в тексте — пробел."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cur, i = [], [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] in "|\\":
            cur.append(line[i + 1])
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return [strip_inline(_CELL_BREAK.sub(" ", c)) for c in cells]


def compose(pieces: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Строки абзаца → (текст без inline-разметки, [(позиция начала строки в тексте, номер строки)])."""
    parts: list[str] = []
    starts: list[tuple[int, int]] = []
    pos = 0
    for line_no, piece in pieces:
        part = _INLINE.sub("", piece.strip())
        if parts:
            pos += 1  # пробел между строками
        starts.append((pos, line_no))
        parts.append(part)
        pos += len(part)
    joined = " ".join(parts)
    lead = len(joined) - len(joined.lstrip())
    return joined.strip(), [(max(0, p - lead), n) for p, n in starts]


def _line_at(starts: list[tuple[int, int]], offset: int) -> int:
    line = starts[0][1]
    for pos, n in starts:
        if pos > offset:
            break
        line = n
    return line


def add_statements(blocks: list[Block], add, kind: BlockKind, pieces: list[tuple[int, str]]) -> None:
    """Абзац или пункт списка: целиком, если в нём одно утверждение, иначе по фрагменту на каждое.

    Фрагмент — дословная часть текста абзаца со своей строкой; родитель (весь абзац), номер
    фрагмента и позиция в родителе сохраняются в полях блока.
    """
    text, starts = compose(pieces)
    if not text:
        return
    spans = split_spans(text)
    if len(spans) < 2:
        add(kind, text, starts[0][1])
        return
    first_id = f"B-{len(blocks) + 1:04d}"
    for k, (a, b) in enumerate(spans, start=1):
        add(
            kind,
            text[a:b],
            _line_at(starts, a),
            line_end=_line_at(starts, b - 1),
            parent_id=first_id,
            parent_text=text,
            part=k,
            parts=len(spans),
            offset=a,
        )


def parse_markdown(md: str, page_of_line: dict[int, int] | None = None) -> tuple[str, list[Block]]:
    """Разобрать markdown на блоки. Возвращает (заголовок документа, блоки)."""
    lines = md.split("\n")
    blocks: list[Block] = []
    section: list[str] = []
    stack: list[tuple[int, str]] = []
    title: str | None = None
    page_of_line = page_of_line or {}

    def add(kind: BlockKind, text: str, line_no: int, **kw) -> None:
        text = text.strip()
        if not text:
            return
        blocks.append(
            Block(
                id=f"B-{len(blocks) + 1:04d}",
                kind=kind,
                text=text,
                section=list(section) if kind != BlockKind.heading else list(section[:-1]),
                line=line_no,
                page=page_of_line.get(line_no),
                **kw,
            )
        )

    def add_text(kind: BlockKind, pieces: list[tuple[int, str]]) -> None:
        add_statements(blocks, add, kind, pieces)

    i = 0
    para: list[str] = []
    para_start = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            # несколько строк с маркером статуса — отдельные утверждения: делим по этим строкам
            marks = [k for k, p in enumerate(para) if is_status_marker(p)]
            cuts = [0]
            if len(marks) >= 2:
                cuts = ([0] if marks[0] != 0 else []) + marks
            for a, b in zip(cuts, [*cuts[1:], len(para)], strict=True):
                add_text(BlockKind.paragraph, [(para_start + k, para[k]) for k in range(a, b)])
            para = []

    while i < len(lines):
        line = lines[i]
        line_no = i + 1

        if line.strip().startswith("```"):
            flush_para()
            start = line_no
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            add(BlockKind.code, "\n".join(body), start)
            i += 1
            continue

        if not line.strip():
            flush_para()
            i += 1
            continue

        if m := _HEADING.match(line):
            flush_para()
            level = len(m.group(1))
            text = strip_inline(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            section = [t for _, t in stack]
            if title is None and level == 1:
                title = text
            add(BlockKind.heading, text, line_no, level=level)
            i += 1
            continue

        if _HR.match(line):
            flush_para()
            i += 1
            continue

        if line.lstrip().startswith("|"):
            flush_para()
            header: list[str] | None = None
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = lines[i]
                if _TABLE_SEP.match(row):
                    i += 1
                    continue
                cells = _split_row(row)
                if header is None:
                    header = cells
                else:
                    add(
                        BlockKind.table_row,
                        " | ".join(c for c in cells),
                        i + 1,
                        cells=cells,
                        header=header,
                    )
                i += 1
            continue

        if m := _LIST.match(line):
            flush_para()
            item = [(line_no, m.group(2))]
            indent = len(m.group(1))
            i += 1
            # продолжение пункта: строки с отступом, не начинающие новый пункт
            while (
                i < len(lines)
                and lines[i].strip()
                and not _LIST.match(lines[i])
                and len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                item.append((i + 1, lines[i]))
                i += 1
            add_text(BlockKind.list_item, item)
            continue

        if not para:
            para_start = line_no
        para.append(line)
        i += 1

    flush_para()
    return title or "", blocks
