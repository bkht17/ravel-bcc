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


def _unfilled_checklist_rows(doc: DocumentIR) -> dict[tuple[str, ...], list[Block]]:
    """Строки чек-листов с невыбранным «Да / Нет», по разделам."""
    by_section: dict[tuple[str, ...], list[Block]] = defaultdict(list)
    for b in doc.blocks:
        if b.kind == BlockKind.table_row and re.search(r"да\s*/\s*нет", b.text, re.IGNORECASE):
            by_section[tuple(b.section)].append(b)
    return by_section


def placeholders(doc: DocumentIR) -> list[Finding]:
    # Заглушка в разделе с незаполненным чек-листом описана одной находкой — checklist_unfilled.
    checklists = _unfilled_checklist_rows(doc)
    out = []
    for b in doc.blocks:
        if b.kind in (BlockKind.paragraph, BlockKind.list_item) and _PLACEHOLDER.search(b.text):
            if tuple(b.section) in checklists:
                continue
            name = b.section[-1] if b.section else "—"
            out.append(
                _f(
                    "placeholder",
                    Criterion.set_complete,
                    Severity.major,
                    f"Раздел «{name}» не заполнен",
                    f"В разделе «{name}» вместо содержания стоит заглушка «{b.text}»"
                    + (f" (строка {b.line})." if b.line else "."),
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
                    f"Заголовок «{b.text}»"
                    + (f" (строка {b.line})" if b.line else "")
                    + (
                        " последний в документе."
                        if nxt is None
                        else f" сразу сменяется заголовком «{nxt.text}»."
                    ),
                    [b.text],
                    "Заполнить раздел или удалить заголовок.",
                )
            )
    return out


def _rows_word(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "строки"
    return "строк"


def unfilled_checklists(doc: DocumentIR) -> list[Finding]:
    out = []
    for section, rows in _unfilled_checklist_rows(doc).items():
        name = section[-1] if section else "—"
        n = len(rows)
        quotes = [rows[0].cells[0] if rows[0].cells else rows[0].text]
        note = ""
        for b in doc_blocks_in(doc, section):
            if b.kind in (BlockKind.paragraph, BlockKind.list_item) and _PLACEHOLDER.search(b.text):
                quotes.insert(0, b.text)
                note = f" В том же разделе стоит пометка: «{b.text}»"
                break
        out.append(
            _f(
                "checklist_unfilled",
                Criterion.set_validatable,
                Severity.major,
                f"Чек-лист «{name}» не заполнен",
                f"Отметки в чек-листе не проставлены ни в одной из {n} {_rows_word(n)}: "
                f"во всех остались варианты «Да / Нет» без выбранного ответа.{note}{'.' if note else ''}",
                quotes,
                "Проставить отметки в чек-листе (или указать, что ревью не требуется) до передачи в разработку.",
            )
        )
    return out


def doc_blocks_in(doc: DocumentIR, section: tuple[str, ...]) -> list[Block]:
    return [b for b in doc.blocks if tuple(b.section) == section]


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
                    f"Все строки таблицы в разделе «{b.section[-1] if b.section else '—'}» пусты"
                    + (f" (первая — строка {b.line})." if b.line else "."),
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
                        "В тексте"
                        + (f" (строка {b.line})" if b.line else "")
                        + f" стоит слово «{w}» без числового значения, порога или условия.",
                        [_sentence(b.text, w)],
                        "Заменить на измеримое значение (время, количество, условие).",
                        scope="requirement",
                    )
                )
    return out


_HTTP_VERBS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_WRITE_WORDS = re.compile(r"\b(save|update|create|delete)\w*|сохран|измен|замен|удал", re.I)


def _method_rows(blocks: list[Block]) -> list[int]:
    """Индексы строк таблиц параметров, где указан HTTP-метод («REST type | GET»)."""
    rows = []
    for i, b in enumerate(blocks):
        if (
            b.kind == BlockKind.table_row
            and b.cells
            and len(b.cells) > 1
            and re.search(r"rest\s*type|method|метод", b.cells[0], re.I)
            and b.cells[1].strip().upper() in _HTTP_VERBS
        ):
            rows.append(i)
    return rows


def _method_start(blocks: list[Block], row: int) -> int:
    """Начало описания метода: первая строка его таблицы и заголовки/текст прямо над ней."""
    start = row
    while (
        start > 0
        and blocks[start - 1].kind == BlockKind.table_row
        and blocks[start - 1].section == blocks[row].section
    ):
        start -= 1
    target = blocks[row].section
    while start > 0:
        prev = blocks[start - 1]
        is_own_heading = (
            prev.kind == BlockKind.heading and (prev.section + [prev.text]) == target[: len(prev.section) + 1]
        )
        is_own_text = prev.kind != BlockKind.heading and prev.section == target
        if not (is_own_heading or is_own_text):
            break
        start -= 1
    return start


def http_get_for_write(doc: DocumentIR) -> list[Finding]:
    """GET, который меняет данные: метод и признак записи берутся из одного описания метода.

    Описание метода — от его таблицы параметров до таблицы следующего метода (в том же разделе).
    """
    blocks = doc.blocks
    rows = _method_rows(blocks)
    starts = [_method_start(blocks, r) for r in rows]
    out = []
    for n, row in enumerate(rows):
        method = blocks[row]
        if method.cells[1].strip().upper() != "GET":
            continue
        parent = method.section[:-1]
        end = starts[n + 1] if n + 1 < len(rows) else len(blocks)
        scope = [
            b
            for b in blocks[starts[n] : end]
            if b is not method and b.section[: len(parent)] == parent and b.kind != BlockKind.heading
        ]
        writes = [b for b in scope if _WRITE_WORDS.search(b.text)]
        if not writes:
            continue
        out.append(
            _f(
                "http_get_write",
                Criterion.correct,
                Severity.critical,
                "Изменение данных через GET",
                f"Метод объявлен как {method.cells[1].strip()}"
                + (f" (строка {method.line})" if method.line else "")
                + f", но в его описании сказано про изменение данных: «{writes[0].text[:120]}»"
                + (f" (строка {writes[0].line})." if writes[0].line else "."),
                [method.text, writes[0].text[:200]],
                "Использовать POST или PUT с телом запроса.",
            )
        )
    return out


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
                        # Заголовок раздела — не требование, находка всегда уровня набора: код 200
                        # назван именем кода 201, документ расходится сам с собой и с HTTP.
                        "status_200_created",
                        Criterion.set_consistent,
                        Severity.major,
                        "Код 200 назван «Created»",
                        f"Заголовок «{b.text}»"
                        + (f" (строка {b.line})" if b.line else "")
                        + " называет код 200 словом «Created», которым в HTTP назван код 201.",
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
                        f"Код {code} указан в тексте"
                        + (f" (строка {b.line})" if b.line else "")
                        + f", но среди объявленных кодов ({', '.join(sorted(declared))}) его нет.",
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
                    f"Заголовок «{ct.text}»"
                    + (f" (строка {ct.line})" if ct.line else "")
                    + f" задан, но в том же разделе указано: «{nobody.text}»"
                    + (f" (строка {nobody.line})." if nobody.line else "."),
                    [ct.text, nobody.text],
                    "Убрать Content-Type или описать тело запроса.",
                )
            )
    return out


# Заголовки колонок таблиц параметров: имя поля и его тип/формат (рус. и англ.)
_PARAM_NAME_COL = re.compile(
    r"^(?:parameter|param|field|attribute|name|параметр|поле|атрибут|имя|название)"
    r"(?: (?:name|поля|параметра))?$"
)
_PARAM_TYPE_COL = re.compile(r"^(?:format|type|data type|формат|тип|тип данных)$")


def _column(header: list[str], rx: re.Pattern[str]) -> int | None:
    """Номер единственной колонки, чей заголовок подходит под rx; None — нет или неоднозначно."""
    found = [i for i, h in enumerate(header) if rx.match(re.sub(r"\s+", " ", h.strip().lower()))]
    return found[0] if len(found) == 1 else None


def param_type_mismatch(doc: DocumentIR) -> list[Finding]:
    seen: dict[str, tuple[str, Block]] = {}
    out = []
    for b in doc.blocks:
        if b.kind != BlockKind.table_row or not b.header or not b.cells:
            continue
        name_col, type_col = _column(b.header, _PARAM_NAME_COL), _column(b.header, _PARAM_TYPE_COL)
        if name_col is None or type_col is None or name_col == type_col:
            continue
        name = b.cells[name_col].strip() if len(b.cells) > name_col else ""
        fmt = b.cells[type_col].strip().lower() if len(b.cells) > type_col else ""
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
                    f"Параметр {name} описан с форматом «{prev_fmt}»"
                    + (f" (строка {prev.line})" if prev.line else "")
                    + f" и с форматом «{fmt}»"
                    + (f" (строка {b.line})." if b.line else "."),
                    [prev.text, b.text],
                    "Привести тип к одному значению во всех таблицах.",
                )
            )
        seen.setdefault(name, (fmt, b))
    return out


# Словарь измерений: (измерение, единица, регэксп полной формы | None, сокращения | None).
# Сокращения («с», «ч», «Б») засчитываются только сразу после числа, иначе «с» — это предлог.
_UNITS: list[tuple[str, str, str | None, str | None]] = [
    ("деньги", "тенге", r"тенге", "₸"),
    ("деньги", "тиын", r"тиын\w*", None),
    ("деньги", "рубль", r"рубл\w*", r"руб\.?|₽"),
    ("деньги", "копейка", r"копе(?:йк\w*|ек)", r"коп\.?"),
    ("деньги", "доллар", r"доллар\w*|dollars?", r"\$"),
    ("деньги", "цент", r"цент(?:а|ов|ы|ам|ах)?", None),
    ("деньги", "евро", r"евро", r"€"),
    ("время", "миллисекунда", r"миллисекунд\w*|milliseconds?", r"мс|ms"),
    ("время", "секунда", r"секунд\w*|seconds?", r"сек\.?|с|s"),
    ("время", "минута", r"минут\w*|minutes?", r"мин\.?"),
    ("время", "час", r"час(?:а|ов|у|ы|ам|ах|е)?|hours?", r"ч"),
    ("время", "сутки", r"сут(?:ки|ок|кам\w*|ках|ками)|days?", r"сут\.?"),
    ("размер данных", "байт", r"байт\w*|bytes?", r"Б"),
    ("размер данных", "КБ", r"килобайт\w*", r"КБ|Кб|KB|Kb"),
    ("размер данных", "МБ", r"мегабайт\w*", r"МБ|Мб|MB|Mb"),
    ("размер данных", "ГБ", r"гигабайт\w*", r"ГБ|Гб|GB|Gb"),
]
_NUM = r"\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?"
_UNIT_RES: list[tuple[str, str, re.Pattern[str]]] = []  # слова — без учёта регистра, сокращения — с учётом
for _dim, _name, _word, _abbr in _UNITS:
    if _word:
        _UNIT_RES.append((_dim, _name, re.compile(rf"(?<!\w)(?:{_word})(?!\w)", re.I)))
    if _abbr:
        _UNIT_RES.append((_dim, _name, re.compile(rf"(?<![\w.]){_NUM}\s?(?:{_abbr})(?!\w)(?!\s+[a-zа-яё])")))

_COMPARE = re.compile(
    r"больш|меньш|превыша|достига|не менее|не более|не выше|не ниже|выше|ниже|порог|лимит|сравни|сопостав"
    r"|пересека|[≥≤<>]",
    re.I,
)
_CONVERSION = re.compile(
    r"×\s*\d|[*/]\s*\d|умнож|дел(?:ить|ит\w*|[её]н\w*|ени\w*)|пересчит|пересчёт|пересчет|конверт|курс"
    r"|\d[\d\s.,]*[^\d\s=]+\s*=\s*\d|\d[\d\s.,]*[^\d\s]+\s+как\s+\d",
    re.I,
)
_CLAUSE_END = re.compile(r"[.;,:!?\n]")
_LEAD_CONJ = re.compile(r"^(?:и|а|но|или|если|когда)\s+", re.I)


def _unit_mentions(text: str) -> list[tuple[int, int, str, str]]:
    """(начало, конец, измерение, единица) без пересечений, по порядку в тексте."""
    found = [(m.start(), m.end(), dim, name) for dim, name, rx in _UNIT_RES for m in rx.finditer(text)]
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    out: list[tuple[int, int, str, str]] = []
    for f in found:
        if out and f[0] < out[-1][1]:
            continue
        out.append(f)
    return out


def _mention_quote(text: str, mentions: list[tuple[int, int, str, str]], i: int) -> str:
    """Фрагмент блока от границы фразы (или конца предыдущей единицы) до самой единицы."""
    start_m, end_m = mentions[i][0], mentions[i][1]
    start = 0
    for m in _CLAUSE_END.finditer(text, 0, start_m):
        start = m.end()
    if i > 0:
        start = max(start, mentions[i - 1][1])
    seg = text[start:end_m].strip()
    seg = _LEAD_CONJ.sub("", seg)
    return seg


def _unit_groups(blocks: list[Block]) -> list[list[Block]]:
    """Связанные группы: блоки одного раздела; строки таблицы — отдельно от остального текста."""
    groups: dict[tuple, list[Block]] = {}
    run = 0
    prev_table = False
    for b in blocks:
        if b.kind in (BlockKind.heading, BlockKind.code):
            prev_table = False
            continue
        if b.kind == BlockKind.table_row:
            if not prev_table:
                run += 1
            prev_table = True
            key: tuple = ("t", tuple(b.section), run)
        else:
            prev_table = False
            key = ("s", tuple(b.section))
        groups.setdefault(key, []).append(b)
    return list(groups.values())


def mixed_units(doc: DocumentIR) -> list[Finding]:
    """Величины одного измерения в разных единицах сравниваются без правила пересчёта."""
    out = []
    for group in _unit_groups(doc.blocks):
        mentions = {b.id: _unit_mentions(b.text) for b in group}
        group_text = "\n".join(b.text for b in group)
        if _CONVERSION.search(group_text):
            continue
        # первое упоминание каждой единицы в группе: (блок, индекс упоминания)
        first: dict[tuple[str, str], tuple[Block, int]] = {}
        for b in group:
            for i, (_, _, dim, name) in enumerate(mentions[b.id]):
                first.setdefault((dim, name), (b, i))
        keys = list(first)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                (dim1, u1), (dim2, u2) = keys[x], keys[y]
                if dim1 != dim2:
                    continue
                (b1, i1), (b2, i2) = first[keys[x]], first[keys[y]]
                if not _COMPARE.search(b1.text + "\n" + b2.text):
                    continue
                q1 = _mention_quote(b1.text, mentions[b1.id], i1)
                q2 = _mention_quote(b2.text, mentions[b2.id], i2)
                out.append(
                    _f(
                        "mixed_units",
                        Criterion.unambiguous,
                        Severity.major,
                        f"Величины в разных единицах ({dim1}): {u1} и {u2}",
                        f"Сравниваются «{q1}» ({u1}) и «{q2}» ({u2}); правило пересчёта не указано.",
                        [q1, q2] if q1 != q2 else [q1],
                        "Привести величины к одной единице или явно описать пересчёт и округление.",
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
