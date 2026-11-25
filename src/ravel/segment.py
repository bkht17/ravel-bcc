"""Нарезка документа на отдельные требования (детерминированно, по правилам).

TODO(B): LLM-доразбиение составных абзацев на единичные требования.
"""

from __future__ import annotations

import re

from ravel.models import BlockKind, DocumentIR, Requirement

# Явный идентификатор в начале: «Ф-1.», «FR-12», «НФТ-3:», «REQ-1.2»
_ID = re.compile(r"^([A-Za-zА-ЯЁа-яё]{1,6}-\d+(?:\.\d+)*)[.:)]?\s+")

# Разделы-служебки: не требования, а чек-листы и подписи (их проверяют линтеры)
_SKIP_SECTIONS = re.compile(
    r"ревью требований|блок согласующ|лист согласован|история изменений|тестирование требований"
    r"|общие сведения",
    re.IGNORECASE,
)

_CANDIDATES = {BlockKind.paragraph, BlockKind.list_item, BlockKind.table_row}


def segment(doc: DocumentIR) -> list[Requirement]:
    reqs: list[Requirement] = []
    structured = any(len(b.section) >= 2 for b in doc.blocks)
    for block in doc.blocks:
        if block.kind not in _CANDIDATES:
            continue
        # Текст до первого раздела второго уровня — преамбула документа
        # (если заголовков нет вовсе, как в PDF без разметки, берём всё)
        if structured and len(block.section) <= 1:
            continue
        if any(_SKIP_SECTIONS.search(s) for s in block.section):
            continue
        if block.kind == BlockKind.table_row and not any(c.strip() for c in block.cells or []):
            continue
        source_id = None
        if m := _ID.match(block.text):
            source_id = m.group(1)
        # обрывки вроде «или», «Не используются.» — не требования
        if source_id is None and block.kind != BlockKind.table_row and len(block.text.split()) < 3:
            continue
        text = block.text
        if block.kind == BlockKind.table_row and block.header:
            text = "; ".join(f"{h}: {c}" for h, c in zip(block.header, block.cells or [], strict=False) if c)
        reqs.append(
            Requirement(
                id=f"R-{len(reqs) + 1:03d}",
                source_id=source_id,
                block_id=block.id,
                text=text,
                section=block.section,
                location=block.location,
            )
        )
    return reqs
