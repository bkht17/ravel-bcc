"""Нарезка документа на отдельные требования (детерминированно, по правилам).

TODO(B): LLM-доразбиение составных абзацев на единичные требования.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ravel.models import BlockKind, DocumentIR, Requirement

# Явный идентификатор в начале: «Ф-1.», «FR-12», «НФТ-3:», «REQ-1.2»
ID_RE = re.compile(r"^([A-Za-zА-ЯЁа-яё]{1,6}-\d+(?:\.\d+)*)[.:)]?\s+")

# Разделы-служебки: не требования, а чек-листы и подписи (их проверяют линтеры)
_SKIP_SECTIONS = re.compile(
    r"ревью требований|блок согласующ|лист согласован|история изменений|тестирование требований"
    r"|общие сведения",
    re.IGNORECASE,
)

_CANDIDATES = {BlockKind.paragraph, BlockKind.list_item, BlockKind.table_row}

# Порог контроля покрытия: доля непустого текста документа (символы без пробелов), попавшая в
# требования или в явно не-требования (заголовки, служебные разделы, код). Ориентир из задачи — 30 %.
# На samples/requirements.md и табличном PDF покрытие около 0.9, нарезка MD «потерявшая структуру»
# (печатный PDF до исправлений) — 0; 0.3 лежит далеко от обоих.
MIN_COVERAGE = 0.3


@dataclass(frozen=True)
class Coverage:
    requirements: int
    covered_chars: int  # символы без пробелов в требованиях и явно классифицированных не-требованиях
    total_chars: int

    @property
    def share(self) -> float:
        return self.covered_chars / self.total_chars if self.total_chars else 1.0

    @property
    def low(self) -> bool:
        """Текст есть, но требований нет или покрытие ниже порога."""
        return self.total_chars > 0 and (self.requirements == 0 or self.share < MIN_COVERAGE)


def _chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def segment(doc: DocumentIR) -> list[Requirement]:
    return _walk(doc)[0]


def coverage(doc: DocumentIR) -> Coverage:
    return _walk(doc)[1]


def _walk(doc: DocumentIR) -> tuple[list[Requirement], Coverage]:
    reqs: list[Requirement] = []
    covered = total = 0
    structured = any(len(b.section) >= 2 for b in doc.blocks)
    for block in doc.blocks:
        size = _chars(block.text)
        total += size
        # заголовки и код — явные не-требования
        if block.kind not in _CANDIDATES:
            covered += size
            continue
        # Текст до первого раздела второго уровня — преамбула документа
        # (если заголовков нет вовсе, как в PDF без разметки, берём всё).
        # Преамбулу в покрытие не засчитываем: именно так молча терялся весь текст.
        if structured and len(block.section) <= 1:
            continue
        if any(_SKIP_SECTIONS.search(s) for s in block.section):
            covered += size
            continue
        if block.kind == BlockKind.table_row and not any(c.strip() for c in block.cells or []):
            continue
        source_id = None
        if m := ID_RE.match(block.text):
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
        covered += size
    return reqs, Coverage(len(reqs), covered, total)
