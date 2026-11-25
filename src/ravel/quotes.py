"""Проверка цитат: каждая находка должна ссылаться на реальный текст документа."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from ravel.models import Block, DocumentIR

_MARKUP = re.compile(r"[*_`|#>]")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans(
    {"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "'": '"', "–": "-", "—": "-", "−": "-"}
)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е").translate(_QUOTES)
    text = _MARKUP.sub(" ", text)
    return _WS.sub(" ", text).strip(" .;:,")


def locate(quote: str, doc: DocumentIR, fuzzy: float = 0.9) -> Block | None:
    """Найти блок, в котором стоит цитата. None — цитата не подтверждена."""
    q = norm(quote)
    if len(q) < 3:
        return None
    normed = [(b, norm(b.text)) for b in doc.blocks]
    for b, t in normed:
        if q in t:
            return b
    # цитата могла пересечь границу блоков (например, абзац + список)
    head = q[: min(len(q), 40)]
    for b, t in normed:
        if head in t:
            return b
    # нечёткое совпадение: модель могла поправить пробел или кавычку
    for b, t in normed:
        if len(t) < len(q) * 0.5:
            continue
        m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
        if m.size >= len(q) * fuzzy:
            return b
    return None
