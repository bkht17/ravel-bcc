"""Детерминированные проверки без LLM: одинаковый результат на каждом прогоне.

Каждое правило — функция (doc) -> list[Finding]. Чтобы добавить правило, напишите
функцию и добавьте её в RULES.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

from ravel.models import Block, BlockKind, Criterion, DocumentIR, Finding, Quote, Severity

Rule = Callable[[DocumentIR], list[Finding]]


def _f(
    rule: str,
    criterion: Criterion,
    severity: Severity,
    title: str,
    explanation: str,
    quotes: list[str],
    suggestion: str = "",
    scope: str = "set",
) -> Finding:
    return Finding(
        scope=scope,
        criterion=criterion,
        severity=severity,
        title=title,
        explanation=explanation,
        suggestion=suggestion,
        quotes=[Quote(text=q) for q in quotes],
        source=f"lint:{rule}",
    )


def _sentence(text: str, word: str) -> str:
    for s in re.split(r"(?<=[.;!?])\s+", text):
        if word.lower() in s.lower():
            return s.strip()
    return text


# --------------------------------------------------------------------------- правила

_PLACEHOLDER = re.compile(
    r"^(заполня\w+ .*|tbd|todo|tba|н/д|—|-|\.\.\.|…)$|не заполнен|будет уточнено|уточняется", re.IGNORECASE
)


def placeholders(doc: DocumentIR) -> list[Finding]:
    out = []
    for b in doc.blocks:
        if b.kind in (BlockKind.paragraph, BlockKind.list_item) and _PLACEHOLDER.search(b.text):
            out.append(
                _f(
                    "placeholder",
                    Criterion.set_complete,
                    Severity.major,
                    f"Раздел «{b.section[-1] if b.section else '—'}» не заполнен",
                    "Вместо содержания стоит заглушка; по разделу нельзя ни разрабатывать, ни тестировать.",
                    [b.text],
                    "Заполнить раздел или явно указать, что он не требуется и почему.",
                )
            )
    return out


def empty_sections(doc: DocumentIR) -> list[Finding]:
    out = []
    blocks = doc.blocks
    for i, b in enumerate(blocks):
        if b.kind != BlockKind.heading:
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if nxt is None or (nxt.kind == BlockKind.heading and nxt.level <= b.level):
            out.append(
                _f(
                    "empty_section",
                    Criterion.set_complete,
                    Severity.minor,
                    f"Пустой раздел «{b.text}»",
                    "Под заголовком нет содержания.",
                    [b.text],
                    "Заполнить раздел или удалить заголовок.",
                )
            )
    return out


def unfilled_checklists(doc: DocumentIR) -> list[Finding]:
    by_section: dict[tuple[str, ...], list[Block]] = defaultdict(list)
    for b in doc.blocks:
        if b.kind == BlockKind.table_row and re.search(r"да\s*/\s*нет", b.text, re.IGNORECASE):
            by_section[tuple(b.section)].append(b)
    out = []
    for section, rows in by_section.items():
        out.append(
            _f(
                "checklist_unfilled",
                Criterion.set_validatable,
                Severity.major,
                f"Чек-лист «{section[-1] if section else '—'}» не заполнен",
                f"В {len(rows)} строках остались варианты «Да / Нет» без выбранного ответа: ревью требований не проведено.",
                [rows[0].cells[0] if rows[0].cells else rows[0].text],
                "Провести ревью и проставить отметки до передачи в разработку.",
            )
        )
    return out


def empty_tables(doc: DocumentIR) -> list[Finding]:
    out = []
    seen: set[tuple[str, ...]] = set()
    for b in doc.blocks:
        if b.kind == BlockKind.table_row and b.cells is not None and not any(c.strip() for c in b.cells):
            key = tuple(b.section)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                _f(
                    "empty_table",
                    Criterion.set_complete,
                    Severity.minor,
                    f"Таблица «{b.section[-1] if b.section else '—'}» пустая",
                    "В таблице нет ни одной заполненной строки.",
                    [b.section[-1] if b.section else ""],
                    "Заполнить таблицу (например, согласующих) или удалить её.",
                )
            )
    return out


_VAGUE = [
    "быстро",
    "своевременно",
    "удобно",
    "удобный",
    "интуитивн",
    "при необходимости",
    "по возможности",
    "оптимальн",
    "достаточно",
    "как правило",
    "и т.д.",
    "и т. д.",
    "и тому подобное",
    "эффективно",
    "надёжно",
    "надежно",
    "гибко",
    "современн",
    "user-friendly",
    "в разумные сроки",
]


def vague_words(doc: DocumentIR) -> list[Finding]:
    out = []
    for b in doc.blocks:
        if b.kind not in (BlockKind.paragraph, BlockKind.list_item, BlockKind.table_row):
            continue
        low = b.text.lower()
        for w in _VAGUE:
            if w in low:
                out.append(
                    _f(
                        "vague_word",
                        Criterion.unambiguous,
                        Severity.major,
                        f"Размытая формулировка «{w}»",
                        "Слово допускает разное толкование и не даёт измеримого критерия приёмки.",
                        [_sentence(b.text, w)],
                        "Заменить на измеримое значение (время, количество, условие).",
                        scope="requirement",
                    )
                )
    return out


def http_get_for_write(doc: DocumentIR) -> list[Finding]:
    method_row = None
    for b in doc.blocks:
        if (
            b.kind == BlockKind.table_row
            and b.cells
            and re.search(r"rest\s*type|method|метод", b.cells[0], re.I)
        ):
            if len(b.cells) > 1 and b.cells[1].strip().upper() == "GET":
                method_row = b
    if method_row is None:
        return []
    writes = [
        b
        for b in doc.blocks
        if b.section[:-1] == method_row.section[:-1]
        and re.search(r"\b(save|update|create|delete)\w*|сохран|измен|замен|удал", b.text, re.I)
    ]
    if not writes:
        return []
    return [
        _f(
            "http_get_write",
            Criterion.correct,
            Severity.critical,
            "Изменение данных через GET",
            "Метод меняет состояние (сохраняет настройку), но объявлен как GET. GET должен быть безопасным и "
            "идемпотентным: прокси и браузеры кэшируют и повторяют его, параметры попадают в логи.",
            [method_row.text, writes[0].text if len(writes[0].text) < 200 else writes[0].text[:200]],
            "Использовать POST или PUT с телом запроса.",
        )
    ]


_STATUS_HEAD = re.compile(r"^(\d{3})\.\s")
# «200. Ответ в случае …» — объявление кода в перечне статусов
_STATUS_DECL = re.compile(r"(?:^|\s)([1-5]\d{2})\.\s+(?:ответ|response)", re.IGNORECASE)
_STATUS_ANY = re.compile(r"\b([1-5]\d{2})\b")


def status_codes(doc: DocumentIR) -> list[Finding]:
    declared: dict[str, Block] = {}
    used: dict[str, Block] = {}
    out = []
    for b in doc.blocks:
        if b.kind in (BlockKind.paragraph, BlockKind.list_item):
            for m in _STATUS_DECL.finditer(b.text):
                declared.setdefault(m.group(1), b)
    for b in doc.blocks:
        in_statuses = any(re.search(r"статус|status", s, re.I) for s in b.section)
        if not in_statuses:
            continue
        if b.kind == BlockKind.paragraph and not _STATUS_DECL.search(b.text):
            for m in _STATUS_ANY.finditer(b.text):
                used.setdefault(m.group(1), b)
        elif b.kind == BlockKind.heading and (m := _STATUS_HEAD.match(b.text)):
            used.setdefault(m.group(1), b)
            if m.group(1) == "200" and "created" in b.text.lower():
                out.append(
                    _f(
                        "status_200_created",
                        Criterion.correct,
                        Severity.major,
                        "Код 200 назван «Created»",
                        "«Created» — это 201. Непонятно, какой код вернёт сервис при успешном сохранении.",
                        [b.text],
                        "Выбрать один код (200 OK или 201 Created) и использовать его везде.",
                    )
                )
    if declared:
        for code, b in used.items():
            if code not in declared:
                out.append(
                    _f(
                        "status_undeclared",
                        Criterion.set_consistent,
                        Severity.major,
                        f"Код {code} не объявлен в списке статусов",
                        f"Код {code} встречается в описании ответов, но отсутствует в перечне статусов метода.",
                        [b.text],
                        f"Добавить {code} в перечень статусов или убрать его из примеров.",
                    )
                )
    return out


def content_type_without_body(doc: DocumentIR) -> list[Finding]:
    by_section: dict[tuple[str, ...], list[Block]] = defaultdict(list)
    for b in doc.blocks:
        by_section[tuple(b.section)].append(b)
    out = []
    for blocks in by_section.values():
        ct = next((b for b in blocks if re.search(r"content-type.*json", b.text, re.I)), None)
        nobody = next(
            (b for b in blocks if re.search(r"тело не используется|без тела|body is not used", b.text, re.I)),
            None,
        )
        if ct and nobody:
            out.append(
                _f(
                    "content_type_no_body",
                    Criterion.set_consistent,
                    Severity.minor,
                    "Content-Type указан для запроса без тела",
                    "Заголовок Content-Type: application/json задан, но тело запроса не используется.",
                    [ct.text, nobody.text],
                    "Убрать Content-Type или описать тело запроса.",
                )
            )
    return out


def param_type_mismatch(doc: DocumentIR) -> list[Finding]:
    seen: dict[str, tuple[str, Block]] = {}
    out = []
    for b in doc.blocks:
        if b.kind != BlockKind.table_row or not b.header or not b.cells:
            continue
        hdr = [h.lower() for h in b.header]
        if "parameter" not in hdr or "format" not in hdr:
            continue
        name = b.cells[hdr.index("parameter")].strip()
        fmt = b.cells[hdr.index("format")].strip().lower() if len(b.cells) > hdr.index("format") else ""
        if not name or not fmt:
            continue
        if name in seen and seen[name][0] != fmt:
            prev_fmt, prev = seen[name]
            out.append(
                _f(
                    "param_type_mismatch",
                    Criterion.set_consistent,
                    Severity.major,
                    f"Параметр {name}: тип {prev_fmt} в одном месте и {fmt} в другом",
                    "Один и тот же параметр описан с разными типами в запросе и ответе.",
                    [prev.text, b.text],
                    "Привести тип к одному значению во всех таблицах.",
                )
            )
        seen.setdefault(name, (fmt, b))
    return out


def mixed_units(doc: DocumentIR) -> list[Finding]:
    out = []
    for b in doc.blocks:
        low = b.text.lower()
        if "тиын" in low and ("тенге" in low or "₸" in low):
            out.append(
                _f(
                    "mixed_units",
                    Criterion.unambiguous,
                    Severity.major,
                    "Разные единицы сумм в одном требовании",
                    "Суммы хранятся в тиынах, а порог — в тенге; не описано, как и где они сравниваются и округляются.",
                    [b.text],
                    "Хранить все суммы в одной единице или явно описать преобразование.",
                    scope="requirement",
                )
            )
    return out


RULES: list[Rule] = [
    placeholders,
    empty_sections,
    unfilled_checklists,
    empty_tables,
    vague_words,
    http_get_for_write,
    status_codes,
    content_type_without_body,
    param_type_mismatch,
    mixed_units,
]


def run_lint(doc: DocumentIR) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        findings.extend(rule(doc))
    return findings
