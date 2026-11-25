"""Определение формата по содержимому и оценка качества извлечённого текста."""

from __future__ import annotations

import io
import re
import zipfile
from typing import Literal

Format = Literal["pdf", "docx", "text"]

_EXT = {".pdf": "pdf", ".docx": "docx", ".md": "text", ".markdown": "text", ".txt": "text"}


class UnsupportedFormat(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def sniff(data: bytes) -> Format:
    """Формат по сигнатуре байтов, а не по расширению."""
    head = data[:8]
    if head.startswith(b"%PDF-") or b"%PDF-" in data[:1024]:
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            # Начало как у zip, но архив битый — пробуем как docx с восстановлением
            return "docx"
        if "word/document.xml" in names:
            return "docx"
        raise UnsupportedFormat("unsupported_zip", "ZIP-архив не является документом Word (.docx)")
    if head.startswith(b"{\\rtf"):
        raise UnsupportedFormat("unsupported_rtf", "Формат RTF не поддерживается, сохраните как DOCX или PDF")
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise UnsupportedFormat("unsupported_doc", "Старый формат .doc не поддерживается, сохраните как DOCX")
    return "text"


def ext_format(suffix: str) -> str | None:
    return _EXT.get(suffix.lower())


_CID = re.compile(r"\(cid:\d+\)")


def garbage_ratio(text: str) -> float:
    """Доля «мусора» в тексте: U+FFFD, (cid:N), управляющие символы, не-буквы."""
    if not text.strip():
        return 1.0
    bad = text.count("�") + 5 * len(_CID.findall(text))
    bad += sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t")
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 1.0
    letters = sum(1 for ch in visible if ch.isalpha() or ch.isdigit())
    non_letter_share = 1 - letters / len(visible)
    # пунктуации и таблиц в требованиях много; тревожно, когда не-букв больше 60 %
    return min(1.0, bad / len(visible) + max(0.0, non_letter_share - 0.6))
