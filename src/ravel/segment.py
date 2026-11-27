"""Нарезка документа на отдельные требования (детерминированно, по правилам).

TODO(B): LLM-доразбиение составных абзацев на единичные требования.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ravel.models import Block, BlockKind, DocumentIR, Requirement

# Явный идентификатор в начале: «Ф-1.», «FR-12», «НФТ-3:», «REQ-1.2»
ID_RE = re.compile(r"^([A-Za-zА-ЯЁа-яё]{1,6}-\d+(?:\.\d+)*)[.:)]?\s+")

# Разделы-служебки: не требования, а чек-листы и подписи (их проверяют линтеры)
_SKIP_SECTIONS = re.compile(
    r"ревью требований|блок согласующ|лист согласован|история изменений|тестирование требований"
    r"|общие сведения",
    re.IGNORECASE,
)

_CANDIDATES = {BlockKind.paragraph, BlockKind.list_item, BlockKind.table_row, BlockKind.code}

# Служебные поля: атрибуты документа (кто написал, где лежит, какая версия), а не свойства системы.
# Строка таблицы «поле — значение» с таким полем в разделе паспорта — метаданные, не требование.
DOCUMENT_ATTRIBUTE_FIELDS = frozenset(
    {
        # русский
        "автор", "авторы", "разработчик", "разработчики", "исполнитель", "исполнители",
        "проект", "репозиторий", "фреймворк", "свэггер", "джира", "трекер",
        "версия", "дата", "владелец", "владельцы",
        # английский
        "author", "authors", "developer", "developers", "assignee", "project", "repository", "repo",
        "framework", "swagger", "jira", "tracker", "ticket", "version", "date", "owner", "owners",
        "maintainer",
    }
)  # fmt: skip

# Разделы, где такие таблицы описывают сам документ: паспорт, общие сведения/параметры
_PASSPORT_SECTIONS = re.compile(r"паспорт|общие|сведени|passport|general|overview", re.IGNORECASE)

# Содержательное слово: отсекает одинокие «или», «Да.», «404» между блоками
_MIN_WORD = re.compile(r"[A-Za-zА-ЯЁа-яё]{4,}")
_ARROWS = ("→", "->", "⇒")
_LIST_NUM = re.compile(r"^\s*(\d+)[.)]\s+")
_LIST_ANY = re.compile(r"^(\s*)(?:[-*+•]|\d+[.)])\s+")
_FENCE_LANG = re.compile(r"^\s*`{3,}\s*([\w+-]*)")

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


def is_metadata_row(block: Block) -> bool:
    """Строка таблицы «служебное поле — значение» в разделе паспорта."""
    if block.kind != BlockKind.table_row or not block.cells or len(block.cells) < 2:
        return False
    if not any(_PASSPORT_SECTIONS.search(s) for s in block.section):
        return False
    field = block.cells[0]
    tokens = re.findall(r"\w+", field.lower())
    return 0 < len(tokens) <= 5 and any(t in DOCUMENT_ATTRIBUTE_FIELDS for t in tokens)


def _span(lines: list[str], block: Block) -> tuple[int, int]:
    """Строки блока в нормализованном markdown (1-based, включительно)."""
    from ravel.ingest.markdown import is_status_marker

    start = block.line or 1
    n = len(lines)
    if start > n:
        return start, start
    if block.kind in (BlockKind.heading, BlockKind.table_row):
        return start, start
    if block.kind == BlockKind.code:
        end = start
        while end < n and not lines[end].strip().startswith("```"):
            end += 1
        return start, min(end + 1, n)
    first = _LIST_ANY.match(lines[start - 1])
    indent = len(first.group(1)) if first else 0
    end = start
    while end < n:
        nxt = lines[end]
        if not nxt.strip() or nxt.lstrip().startswith(("#", "|", "```")) or _LIST_ANY.match(nxt):
            break
        if ID_RE.match(nxt.lstrip()):
            break
        if block.kind == BlockKind.list_item and len(nxt) - len(nxt.lstrip()) <= indent:
            break
        end += 1
    if block.kind == BlockKind.paragraph:
        marks = [k for k in range(start, end + 1) if is_status_marker(lines[k - 1])]
        if len(marks) >= 2:
            end = next((k - 1 for k in marks if k > start), end)
    return start, end


def _short(block: Block) -> bool:
    return len(block.text.split()) <= 3


def _walk(doc: DocumentIR) -> tuple[list[Requirement], Coverage]:
    reqs: list[Requirement] = []
    covered = total = 0
    structured = any(len(b.section) >= 2 for b in doc.blocks)
    lines = doc.markdown.split("\n")
    seen_ids: set[str] = set()
    for block in doc.blocks:
        size = _chars(block.text)
        total += size
        # заголовки — явные не-требования
        if block.kind not in _CANDIDATES:
            covered += size
            continue
        # Текст до первого раздела второго уровня — преамбула документа
        # (если заголовков нет вовсе, как в PDF без разметки, берём всё).
        # Преамбулу в покрытие не засчитываем: именно так молча терялся весь текст.
        if structured and len(block.section) <= 1:
            if block.kind == BlockKind.code:
                covered += size
            continue
        if any(_SKIP_SECTIONS.search(s) for s in block.section):
            covered += size
            continue
        if block.kind == BlockKind.table_row and not any(c.strip() for c in block.cells or []):
            continue
        start, end = _span(lines, block)
        raw = "\n".join(lines[start - 1 : end]).rstrip() if block.line else ""
        parent = block.section[-1] if block.section else ""
        kind = "requirement"
        source_id = None
        text = block.text
        if block.kind == BlockKind.code:
            lang = m.group(1).lower() if (m := _FENCE_LANG.match(lines[start - 1])) else ""
            schema = lang in ("", "text", "txt", "plain") and any(a in block.text for a in _ARROWS)
            kind = "requirement" if schema else "example"
            body = " ".join(x.strip() for x in block.text.split("\n") if x.strip())
            text = f"{parent}: {body}" if parent else body
        else:
            if m := ID_RE.match(block.text):
                source_id = m.group(1)
            elif block.kind == BlockKind.list_item and (
                n := _LIST_NUM.match(lines[start - 1] if raw else "")
            ):
                source_id = f"{parent} п. {n.group(1)}" if parent else f"п. {n.group(1)}"
                if source_id in seen_ids:  # два списка в одном разделе: нумерация началась заново
                    k = 2
                    while f"{source_id} ({k})" in seen_ids:
                        k += 1
                    source_id = f"{source_id} ({k})"
                seen_ids.add(source_id)
            elif block.kind != BlockKind.table_row and _short(block):
                # «Не используются.» — содержательно только рядом с заголовком-родителем;
                # одинокие «или», «Да.», «404» не требование
                if not parent or not _MIN_WORD.search(block.text):
                    continue
                text = f"{parent}: {block.text}"
            if block.kind == BlockKind.table_row:
                if is_metadata_row(block):
                    kind = "metadata"
                if block.header:
                    text = "; ".join(
                        f"{h}: {c}" for h, c in zip(block.header, block.cells or [], strict=False) if c
                    )
        reqs.append(
            Requirement(
                id=f"R-{len(reqs) + 1:03d}",
                source_id=source_id,
                block_id=block.id,
                text=text,
                section=block.section,
                location=block.location,
                raw_text=raw,
                line_start=start if block.line else None,
                line_end=end if block.line else None,
                kind=kind,  # type: ignore[arg-type]
            )
        )
        covered += size
    products = sum(1 for r in reqs if r.kind == "requirement")
    return reqs, Coverage(products, covered, total)
