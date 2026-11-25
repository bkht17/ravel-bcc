"""Нормализованный markdown → список блоков с адресами.

Все форматы (md, txt, pdf, docx) сначала приводятся к markdown, затем разбираются
этим парсером. Поэтому один документ в разных форматах даёт одинаковую нарезку.
"""

from __future__ import annotations

import re
import unicodedata

from ravel.models import Block, BlockKind

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)(?:[-*+•]|\d+[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_INLINE = re.compile(r"(\*\*|__|`)")


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


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [strip_inline(c) for c in line.split("|")]


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

    i = 0
    para: list[str] = []
    para_start = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            add(BlockKind.paragraph, strip_inline(" ".join(p.strip() for p in para)), para_start)
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
            start = line_no
            item = [m.group(2)]
            indent = len(m.group(1))
            i += 1
            # продолжение пункта: строки с отступом, не начинающие новый пункт
            while (
                i < len(lines)
                and lines[i].strip()
                and not _LIST.match(lines[i])
                and len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                item.append(lines[i].strip())
                i += 1
            add(BlockKind.list_item, strip_inline(" ".join(item)), start)
            continue

        if not para:
            para_start = line_no
        para.append(line)
        i += 1

    flush_para()
    return title or "", blocks
