"""Проверка цитат: каждая находка должна ссылаться на реальный текст документа.

Цитата подтверждена, только если она целиком есть в тексте. Допускаются различия, которые
не меняют смысла: регистр, ё/е, пробелы, пунктуация, кавычки, markdown-разметка, опечатка
в окончании слова. Не допускаются: вставленное или удалённое слово (например, «не»),
другое число, другое слово с похожим написанием. Совпадения по началу цитаты нет: модель,
поменявшая смысл в хвосте длинной цитаты, должна быть поймана.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

from ravel.models import Block, BlockKind, DocumentIR

_MARKUP = re.compile(r"[*_`|#>]")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w]|_", re.UNICODE)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_QUOTES = str.maketrans(
    {"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "'": '"', "–": "-", "—": "-", "−": "-"}
)

# Опечатка допустима в одном слове из стольких
_TYPO_EVERY = 8
_MIN_TOKENS_FOR_TYPO = 5


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е").translate(_QUOTES)
    text = _MARKUP.sub(" ", text)
    return _WS.sub(" ", text).strip(" .;:,")


def _alnum(text: str) -> str:
    """Только слова через пробел: «REST type: GET» и «| REST type | GET |» совпадают.

    Границы слов сохраняются, иначе «не бумаги» нашлось бы внутри «стороне бумаги».
    """
    return " " + _WS.sub(" ", _NON_ALNUM.sub(" ", norm(text))).strip() + " "


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(norm(text))


def variants(block: Block) -> list[str]:
    """Как текст блока может выглядеть в цитате модели.

    Строки таблиц агент показывает модели в виде «Заголовок: значение; …» (см. segment.py),
    поэтому цитата может прийти в любом из двух видов.
    """
    out = [block.text]
    if block.kind == BlockKind.table_row and block.header and block.cells:
        out.append("; ".join(f"{h}: {c}" for h, c in zip(block.header, block.cells, strict=False) if c))
    return out


def _similar_word(a: str, b: str) -> bool:
    """Опечатка или другое окончание того же слова, но не другое слово и не другое число."""
    if a == b:
        return True
    if any(ch.isdigit() for ch in a + b):
        return False
    if min(len(a), len(b)) < 4 or a[:4] != b[:4]:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.8


def _token_match(q: list[str], t: list[str]) -> bool:
    """Окно той же длины в тексте блока, где различаются только опечатки."""
    n = len(q)
    if n < _MIN_TOKENS_FOR_TYPO or len(t) < n:
        return False
    allowed = max(1, n // _TYPO_EVERY)
    for start in range(len(t) - n + 1):
        diffs = 0
        for a, b in zip(q, t[start : start + n], strict=True):
            if a != b:
                if not _similar_word(a, b):
                    break
                diffs += 1
                if diffs > allowed:
                    break
        else:
            return True
    return False


def _order(doc: DocumentIR, prefer: Iterable[str] | None) -> list[Block]:
    """Сначала блоки, на которые ссылается находка: одинаковая фраза бывает в разных местах."""
    first = set(prefer or [])
    return [b for b in doc.blocks if b.id in first] + [b for b in doc.blocks if b.id not in first]


def _locate_in_parent(q_alnum: str, doc: DocumentIR, blocks: list[Block]) -> Block | None:
    """Цитата через несколько фрагментов одного абзаца: возвращается фрагмент, где она начинается."""
    groups: dict[str, list[Block]] = {}
    for b in doc.blocks:
        if b.parent_id and b.parent_text is not None and b.offset is not None:
            groups.setdefault(b.parent_id, []).append(b)
    seen: set[str] = set()
    for b in blocks:
        if not b.parent_id or b.parent_id in seen or b.parent_id not in groups:
            continue
        seen.add(b.parent_id)
        parts = groups[b.parent_id]
        parent = parts[0].parent_text or ""
        pos = _alnum(parent).find(q_alnum)
        if pos < 0:
            continue
        for f in parts:
            # длина «слов через пробел» до конца фрагмента: цитата начинается в нём, если раньше
            if pos < len(_alnum(parent[: (f.offset or 0) + len(f.text)])) - 1:
                return f
        return parts[-1]
    return None


def locate(
    quote: str, doc: DocumentIR, fuzzy: bool = True, prefer: Iterable[str] | None = None
) -> Block | None:
    """Найти блок, в котором стоит цитата. None — цитата не подтверждена.

    prefer — id блоков, которые проверяются первыми (блоки заявленных требований).
    Цитата на стыке двух соседних блоков (абзац + пункт списка) возвращает первый из них.
    """
    q_alnum = _alnum(quote)
    if len(q_alnum.strip()) < 3:
        return None
    blocks = _order(doc, prefer)

    for b in blocks:
        if any(q_alnum in _alnum(v) for v in variants(b)):
            return b

    by_pos = {b.id: i for i, b in enumerate(doc.blocks)}
    for b in blocks:
        i = by_pos[b.id]
        if i + 1 < len(doc.blocks) and q_alnum in _alnum(b.text + " " + doc.blocks[i + 1].text):
            return b

    if found := _locate_in_parent(q_alnum, doc, blocks):
        return found

    if fuzzy:
        q_tokens = _tokens(quote)
        for b in blocks:
            if any(_token_match(q_tokens, _tokens(v)) for v in variants(b)):
                return b
    return None
