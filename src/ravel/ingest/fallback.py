"""Резервная нарезка по общим признакам, когда структура markdown не распознана.

Не опирается на заголовки и вёрстку: границы пунктов определяют идентификаторы («Ф-1», «FR-12»),
нумерация «1.», «1)», маркеры списков, пустые строки между абзацами и строки с разделителями
таблицы. Текст и номера строк сохраняются как в исходнике.
"""

from __future__ import annotations

import re

from ravel.ingest.markdown import strip_inline
from ravel.models import Block, BlockKind

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+(?:\.\d+)*[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HR = re.compile(r"^\s*([-*_=])(\s*\1){2,}\s*$")
_ID_START = re.compile(r"^\s*(?:[-*+•]\s+)?(?:\*\*|__)?([A-Za-zА-ЯЁа-яё]{1,6}-\d+(?:\.\d+)*)")


def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2 or bool(re.search(r"\S\s{3,}\S.*\s{3,}\S", line))


def _cells(line: str) -> list[str]:
    s = line.strip().strip("|")
    parts = s.split("|") if "|" in s else re.split(r"\s{3,}", s)
    return [strip_inline(c) for c in parts]


def parse_fallback(md: str, page_of_line: dict[int, int] | None = None) -> list[Block]:
    page_of_line = page_of_line or {}
    blocks: list[Block] = []
    buf: list[str] = []
    kind = BlockKind.paragraph
    start = 0

    def add(k: BlockKind, text: str, line_no: int, **kw) -> None:
        text = text.strip()
        if text:
            blocks.append(
                Block(
                    id=f"B-{len(blocks) + 1:04d}",
                    kind=k,
                    text=text,
                    line=line_no,
                    page=page_of_line.get(line_no),
                    **kw,
                )
            )

    def flush() -> None:
        nonlocal buf, kind
        if buf:
            add(kind, strip_inline(" ".join(x.strip() for x in buf)), start)
        buf, kind = [], BlockKind.paragraph

    for n, line in enumerate(md.split("\n"), start=1):
        if not line.strip() or _HR.match(line):
            flush()
            continue
        if m := _HEADING.match(line):
            flush()
            add(BlockKind.heading, strip_inline(m.group(1)), n, level=1)
            continue
        if _TABLE_SEP.match(line):
            flush()
            continue
        if _is_table_row(line):
            flush()
            cells = _cells(line)
            add(BlockKind.table_row, " | ".join(cells), n, cells=cells)
            continue
        marker = _MARKER.match(line)
        if marker or _ID_START.match(line):
            flush()
            start = n
            kind = BlockKind.list_item if marker else BlockKind.paragraph
            buf = [marker.group(1) if marker else line]
            continue
        if not buf:
            start = n
        buf.append(line)
    flush()
    return blocks
