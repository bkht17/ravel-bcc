"""Деление абзаца на независимо проверяемые утверждения (детерминированно, без LLM).

Граница — конец предложения: «.», «!», «?», «…», затем пробел и заглавная буква, цифра или
открывающая кавычка. Предложение остаётся целым вместе с условием внутри него («Если X, то Y»,
«Когда X, клиент получает Y»). Не режем:

- после сокращения («т. д.», «см.», «стр.», «e.g.») и одиночной буквы («Часть A.», инициалы);
- метку без двух полноценных слов («FR-1.», «201.», «Нет.»): она приклеивается к соседнему
  предложению (а «Проверить сессию.» — уже утверждение);
- вводную фразу, которая кончается двоеточием: она относится к следующему предложению;
- предложение-продолжение («Иначе …», «В этом случае …», «При этом …»): это последствие или
  альтернатива условия из предыдущего предложения, оно остаётся вместе с ним.
"""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"[.!?…]+[»\"”')\]]*\s+(?=[«\"“'(\[]?[A-ZА-ЯЁ0-9])")
_PREV_TOKEN = re.compile(r"([\w.]+)[.!?…]*[»\"”')\]]*$")
_ABBREVIATIONS = frozenset(
    {
        "т.е", "т.к", "т.д", "т.п", "т.н", "и.о", "др", "пр", "см", "ср", "стр", "рис", "табл", "прим",
        "напр", "пп", "гг", "руб", "коп", "тыс", "млн", "млрд", "мин", "сек", "ок", "ул", "им", "англ",
        "рус", "вкл", "искл", "сут", "мес", "нед", "шт", "ед", "экз", "кв", "max", "min", "e.g", "i.e",
        "etc", "vs", "no", "fig", "approx", "resp",
    }
)  # fmt: skip
_DEPENDENT = re.compile(
    r"^(?:иначе|в противном случае|в этом случае|в таком случае|при этом|тогда|"
    r"otherwise|else|in this case|in that case|then)\b",
    re.IGNORECASE,
)
# Слово из двух и более букв: номера («FR-1», «201») и однобуквенные части словами не считаются
_WORD = re.compile(r"[^\W\d_]{2,}")
# Фрагмент, где таких слов меньше, — метка или реплика, а не отдельное утверждение
MIN_WORDS = 2


def _is_abbreviation(text: str, end: int) -> bool:
    m = _PREV_TOKEN.search(text[:end])
    if not m:
        return False
    token = m.group(1).lower().strip(".")
    return len(token) == 1 and token.isalpha() or token in _ABBREVIATIONS


def split_spans(text: str) -> list[tuple[int, int]]:
    """Границы утверждений в text: [(начало, конец)], без ведущих и хвостовых пробелов."""
    cuts = [0]
    for m in _BOUNDARY.finditer(text):
        if _is_abbreviation(text, m.start() + 1):
            continue
        cuts.append(m.end())
    spans = [(a, b) for a, b in zip(cuts, [*cuts[1:], len(text)], strict=True)]
    spans = [_trim(text, a, b) for a, b in spans]
    spans = [s for s in spans if s[0] < s[1]]
    return _merge(text, spans)


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _merge(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for i, (a, b) in enumerate(spans):
            frag = text[a:b]
            short = len(_WORD.findall(frag)) < MIN_WORDS
            if (short or frag.rstrip().endswith(":")) and i + 1 < len(spans):
                spans[i : i + 2] = [(a, spans[i + 1][1])]  # метка или вводная — к следующему
            elif (short or _DEPENDENT.match(frag)) and i > 0:
                spans[i - 1 : i + 1] = [(spans[i - 1][0], b)]  # продолжение — к предыдущему
            else:
                continue
            changed = True
            break
    return spans
