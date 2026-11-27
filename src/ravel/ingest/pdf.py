"""PDF → markdown постранично. Битые и зашифрованные файлы не роняют прогон.

Страницы без текстового слоя (сканы) распознаются Tesseract через pymupdf; результат OCR
проверяется по словарю частых слов (ingest/quality.py), мусор не попадает в требования.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
from dataclasses import dataclass, field

import pymupdf

from ravel.ingest.quality import assess, native_is_suspicious, ocr_is_poor
from ravel.ingest.sniff import garbage_ratio
from ravel.models import IngestIssue, IssueLevel

GARBAGE_THRESHOLD = 0.3

# Меньше стольких непробельных символов родного текста на странице с картинкой — считаем страницу
# сканом (текстовый слой отсутствует или это колонтитул/номер страницы поверх изображения).
SCAN_MAX_NATIVE_CHARS = 20

# Разрешение рендера страницы для OCR: 300 dpi — стандартная рекомендация Tesseract.
OCR_DPI = 300

# Ширина, с которой строка считается «упёрлась в край» при печати моноширинным шрифтом (см. _unwrap).
_MIN_FIXED_WIDTH = 40


@dataclass
class PdfRead:
    pages: list[tuple[int, str]] = field(default_factory=list)  # markdown от pymupdf4llm
    plain: list[tuple[int, str]] = field(
        default_factory=list
    )  # простой текстовый слой, строки как напечатаны
    issues: list[IngestIssue] = field(default_factory=list)


def pdf_pages(data: bytes) -> tuple[list[tuple[int, str]], list[IngestIssue]]:
    """Вернуть [(номер страницы, markdown)] и список проблем чтения."""
    r = read_pdf(data)
    return r.pages, r.issues


def read_pdf(data: bytes) -> PdfRead:
    issues: list[IngestIssue] = []
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # FileDataError и прочие ошибки MuPDF
        return PdfRead(
            issues=[
                IngestIssue(level=IssueLevel.fatal, code="broken_pdf", message=f"PDF не открывается: {e}")
            ]
        )

    with doc:
        if doc.needs_pass:
            return PdfRead(
                issues=[
                    IngestIssue(
                        level=IssueLevel.fatal,
                        code="encrypted_pdf",
                        message="PDF защищён паролем. Снимите защиту и запустите проверку снова.",
                    )
                ]
            )
        if doc.is_repaired:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="repaired_pdf",
                    message="Структура PDF повреждена и восстановлена при чтении; часть текста может быть потеряна.",
                )
            )
        if doc.page_count == 0:
            return PdfRead(
                issues=[IngestIssue(level=IssueLevel.fatal, code="empty_pdf", message="В PDF нет страниц.")]
            )

        scan_pages = _scan_pages(doc)
        pages = _pages_via_pymupdf4llm(doc, issues) or _pages_plain(doc, issues)
        plain = _pages_plain(doc, issues, unwrap=True)
        if scan_pages:
            ocr = _ocr_pages(doc, scan_pages, issues)
            pages = _replace_pages(pages, ocr)
            plain = _replace_pages(plain, ocr)

    good = _filter_pages(pages, issues)
    good_plain = _filter_pages(plain, [])
    if not good:
        if not any(i.code == "ocr_low_quality" for i in issues):
            issues.append(
                IngestIssue(
                    level=IssueLevel.fatal,
                    code="no_text_layer",
                    message="В PDF нет читаемого текстового слоя (скан?), а распознать страницы не удалось: "
                    "Tesseract не найден или не нашёл текст.",
                )
            )
    else:
        q = assess("\n".join(t for _, t in good))
        if native_is_suspicious(q):
            issues.append(
                IngestIssue(
                    level=IssueLevel.fatal,
                    code="ocr_low_quality",
                    message=f"Текстовый слой PDF подозрителен: {q.mixed_share:.0%} слов смешивают кириллицу и "
                    "латиницу (битая кодировка шрифтов?). Требования из такого текста не выдаются.",
                )
            )
    return PdfRead(good, good_plain, issues)


def _filter_pages(pages: list[tuple[int, str]], issues: list[IngestIssue]) -> list[tuple[int, str]]:
    good: list[tuple[int, str]] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        ratio = garbage_ratio(text)
        if ratio > GARBAGE_THRESHOLD:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="low_text_quality",
                    message=f"Текст страницы извлечён с искажениями (доля мусора {ratio:.0%}); страница пропущена.",
                    location=f"стр. {page_no}",
                )
            )
            continue
        good.append((page_no, text))
    return good


def _replace_pages(pages: list[tuple[int, str]], new: dict[int, str]) -> list[tuple[int, str]]:
    out = {n: t for n, t in pages}
    out.update(new)
    return sorted(out.items())


def _scan_pages(doc: pymupdf.Document) -> list[int]:
    """Номера (1-based) страниц без родного текста, но с изображением: кандидаты на OCR."""
    result = []
    for idx in range(doc.page_count):
        try:
            page = doc[idx]
            native = "".join(page.get_text("text").split())
            if len(native) <= SCAN_MAX_NATIVE_CHARS and page.get_images():
                result.append(idx + 1)
        except Exception:
            continue
    return result


def tessdata_dir() -> str | None:
    """Каталог tessdata, если Tesseract доступен (иначе OCR невозможен)."""
    if shutil.which("tesseract") is None:
        return None
    try:
        path = os.environ.get("TESSDATA_PREFIX") or pymupdf.get_tessdata()
    except Exception:
        return None
    return path if path and os.path.isdir(path) else None


def _ocr_languages(tessdata: str) -> str:
    has_rus = os.path.exists(os.path.join(tessdata, "rus.traineddata"))
    has_eng = os.path.exists(os.path.join(tessdata, "eng.traineddata"))
    langs = [lang for lang, ok in (("rus", has_rus), ("eng", has_eng)) if ok]
    return "+".join(langs)


def _ocr_pages(doc: pymupdf.Document, scan_pages: list[int], issues: list[IngestIssue]) -> dict[int, str]:
    """Распознать страницы-сканы. Плохой результат отбрасывается с fatal-проблемой ocr_low_quality."""
    tessdata = tessdata_dir()
    if tessdata is None:
        return {}
    langs = _ocr_languages(tessdata)
    if not langs:
        return {}
    texts: dict[int, str] = {}
    for page_no in scan_pages:
        try:
            page = doc[page_no - 1]
            tp = page.get_textpage_ocr(language=langs, dpi=OCR_DPI, full=True, tessdata=tessdata)
            texts[page_no] = page.get_text("text", textpage=tp)
        except Exception as e:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="page_unreadable",
                    message=f"OCR страницы не удался: {e}",
                    location=f"стр. {page_no}",
                )
            )
    if not texts:
        return {}
    q = assess("\n".join(texts.values()))
    if ocr_is_poor(q):
        note = ""
        if "rus" not in langs.split("+"):
            note = (
                " Языковой пакет rus для Tesseract не установлен (распознавание шло только по "
                f"«{langs}»); установите tessdata rus и повторите."
            )
        issues.append(
            IngestIssue(
                level=IssueLevel.fatal,
                code="ocr_low_quality",
                message=f"Скан распознан плохо (языки OCR: {langs}; слов: {q.tokens}, "
                f"узнаваемых {q.known_share:.0%}, со смесью алфавитов {q.mixed_share:.0%}). "
                f"Требования из такого текста не выдаются.{note}",
                location="стр. " + ", ".join(str(n) for n in sorted(texts)),
            )
        )
        return {n: "" for n in texts}
    if "rus" not in langs.split("+"):
        issues.append(
            IngestIssue(
                level=IssueLevel.warning,
                code="ocr_no_rus",
                message=f"Скан распознан без языкового пакета rus (языки OCR: {langs}); русский текст мог исказиться.",
            )
        )
    return texts


def _pages_via_pymupdf4llm(doc: pymupdf.Document, issues: list[IngestIssue]) -> list[tuple[int, str]]:
    try:
        import pymupdf4llm

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            # OCR выполняем сами (см. _ocr_pages): с проверкой качества и языками rus+eng
            chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False, use_ocr=False)
    except Exception as e:
        issues.append(
            IngestIssue(
                level=IssueLevel.warning,
                code="layout_fallback",
                message=f"Разметка PDF не распознана ({type(e).__name__}); заголовки и таблицы могут потеряться.",
            )
        )
        return []
    pages = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        page_no = meta.get("page_number") or meta.get("page") or idx
        pages.append((int(page_no), chunk.get("text") or ""))
    return pages


def _pages_plain(
    doc: pymupdf.Document, issues: list[IngestIssue], unwrap: bool = False
) -> list[tuple[int, str]]:
    pages = []
    for idx in range(doc.page_count):
        try:
            text = doc[idx].get_text("text")
            pages.append((idx + 1, _unwrap_fixed_width(text) if unwrap else text))
        except Exception as e:
            issues.append(
                IngestIssue(
                    level=IssueLevel.warning,
                    code="page_unreadable",
                    message=f"Страница не читается: {e}",
                    location=f"стр. {idx + 1}",
                )
            )
    return pages


def _unwrap_fixed_width(text: str) -> str:
    """Склеить жёсткие переносы документа, напечатанного моноширинным шрифтом с дополнением до ширины.

    Если все строки страницы дополнены пробелами до одной ширины W, то строка длиной ровно W,
    оканчивающаяся не пробелом, оборвана на краю посреди слова: следующая строка — её продолжение
    без пробела. Строка длиной W с пробелом в конце оборвана на границе слов: продолжение через
    этот пробел. Остальные строки — настоящие концы строк. Если ширины нет, текст возвращается как есть (с обрезанными хвостами).
    """
    lines = text.split("\n")
    widths = [len(ln) for ln in lines if ln.strip()]
    if not widths:
        return text
    width = max(widths)
    full = sum(1 for w in widths if w == width)
    if width < _MIN_FIXED_WIDTH or full < 3:
        return text
    out: list[str] = []
    i = 0
    while i < len(lines):
        seg = lines[i]
        cur = seg
        while i + 1 < len(lines) and lines[i + 1].strip() and _wrapped(seg, lines[i + 1], width):
            i += 1
            seg = lines[i]
            cur += seg
        out.append(cur.rstrip())
        i += 1
    return "\n".join(out)


_STRUCTURAL = re.compile(r"^\s*(#{1,6}\s|\||[-*+•]\s|\d+[.)]\s|---)")


def _wrapped(line: str, nxt: str, width: int) -> bool:
    """Строка упёрлась в ширину печати и продолжается на следующей (кроме строк таблицы и новых блоков).

    Строка с одним пробелом в конце — перенос на границе слов; несколько пробелов — обычное
    дополнение пустой строки до ширины.
    """
    s = line.strip()
    if len(line) != width or not s or (line[-1] == " " and line[-2] == " "):
        return False
    if s.startswith("|"):
        # строка таблицы закончилась на краю, если закрыта «|» и дальше начинается новая строка
        n = nxt.strip()
        if s.endswith("|") and n.startswith("|") and n != "|":
            return False
        return True
    elif _STRUCTURAL.match(nxt):
        return False
    return True
