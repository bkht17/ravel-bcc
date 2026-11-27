"""Доводка разметки PDF после pymupdf4llm: таблицы по линиям сетки, заголовки, маркеры списков.

pymupdf4llm в режиме layout (модель разметки) делит страницу на боксы (``page_boxes``: класс,
bbox и диапазон ``pos`` в тексте страницы) и сам собирает таблицы. Аудит табличного PDF показал,
что сборка таблиц теряет данные молча:

- граница колонки проходит сквозь слово заголовка: слово делится между двумя ячейками, и
  правило о типах полей больше не находит колонку типа;
- перенос строки внутри ячейки становится отдельной строкой таблицы с пустыми ячейками;
- пробелы между словами в ячейке пропадают (два слова склеиваются в одно);
- у продолжения таблицы на следующей странице первая строка данных становится заголовком;
- пустая таблица распознаётся как заголовок раздела, а заголовок вверху страницы — как колонтитул.

Здесь таблица с линиями сетки перестраивается заново: ячейки — по линиям (``find_tables`` без
модели разметки), текст ячейки — из слов текстового слоя, попавших в её прямоугольник. Строка
без верхней линии в начале страницы (продолжение таблицы) восстанавливается по колонкам сетки.
Слова, которые были в заменённом фрагменте, но не попали ни в одну ячейку, дают предупреждение
``pdf_table_degraded`` — текст не теряется молча. Таблицу без линий перестроить нечем; если в её
ячейках есть слова, которых нет в текстовом слое (разрезанные или склеенные), — то же предупреждение.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pymupdf

from ravel.models import IngestIssue, IssueLevel

# Совпадение границ колонок у продолжения таблицы на следующей странице, пунктов
COLUMN_TOLERANCE = 2.0
# Доля площади бокса разметки внутри сетки, начиная с которой бокс считается частью таблицы
INSIDE_SHARE = 0.6
# Колонтитул длиннее этого — не заголовок
HEADING_MAX_WORDS = 15

_BODY_EXCLUDED = ("page-header", "page-footer")
_LIST_BULLET_DUP = re.compile(r"(?m)^(\s*[-*+]\s+)[•◦▪●‣⁃·]\s+")
_MD_MARKUP = re.compile(r"\*\*|__|`|<br\s*/?>", re.I)
_TOKEN_EDGE = re.compile(r"^[^\w]+|[^\w]+$")


@dataclass
class Grid:
    """Таблица, найденная по линиям сетки."""

    bbox: pymupdf.Rect
    xs: list[float]  # границы колонок слева направо (колонок на одну меньше)
    rows: list[list[str]]  # строки сетки сверху вниз
    used: set[int] = field(default_factory=set)  # id() слов страницы, попавших в ячейки
    row_height: float = 0.0  # наибольшая высота строки сетки
    # заголовок продолжения таблицы со страницы раньше; None — заголовок это rows[0]
    carried_header: list[str] | None = None

    @property
    def header(self) -> list[str]:
        return self.carried_header if self.carried_header is not None else self.rows[0]


@dataclass
class _Box:
    cls: str
    bbox: pymupdf.Rect
    start: int
    end: int
    text: str  # что подставить вместо исходного фрагмента страницы
    grid: Grid | None = None
    heading_level: int = 0


# --------------------------------------------------------------------------- таблицы


def join_cell_lines(lines: list[str]) -> str:
    """Строки одной ячейки → одна строка.

    Строка, оборванная после «/» или «-» без пробела перед знаком («клиент/», «Wi-»),
    продолжается без пробела; знак не удаляется (это не обязательно перенос по слогам).
    Остальные строки соединяются пробелом («Дата» + «платежа»).
    """
    out = ""
    for line in (ln.strip() for ln in lines):
        if not line:
            continue
        if out and out[-1] in "/-" and len(out) > 1 and not out[-2].isspace():
            out += line
        else:
            out = f"{out} {line}" if out else line
    return out


def _words_in(words: list[tuple], rect: pymupdf.Rect) -> list[tuple]:
    return [w for w in words if rect.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))]


def _cell_text(words: list[tuple]) -> str:
    """Слова ячейки по строкам текстового слоя, строки сверху вниз, слова слева направо."""
    lines: dict[tuple[int, int], list[tuple]] = {}
    for w in words:
        lines.setdefault((w[5], w[6]), []).append(w)
    ordered = sorted(lines.values(), key=lambda ws: (min(w[1] for w in ws), min(w[0] for w in ws)))
    return join_cell_lines([" ".join(w[4] for w in sorted(ws, key=lambda w: w[0])) for ws in ordered])


def ruled_tables(page: pymupdf.Page, words: list[tuple]) -> list[Grid]:
    """Таблицы страницы по линиям сетки (без модели разметки), текст ячеек — из слов words."""
    try:
        found = page.find_tables(use_layout=False).tables
    except TypeError:  # PyMuPDF без параметра use_layout
        found = page.find_tables().tables
    except Exception:
        return []
    grids: list[Grid] = []
    for t in found:
        if t.col_count < 2:
            continue  # рамка вокруг абзаца, а не таблица
        # Заголовок — первая строка сетки. «Внешний» заголовок find_tables (строка текста над
        # линиями) не берём: это эвристика, и она принимает за заголовок абзац или раздел над таблицей.
        row_cells = [list(r.cells) for r in t.rows]
        used: set[int] = set()
        rows: list[list[str]] = []
        for cells in row_cells:
            row = []
            for c in cells:
                if c is None:
                    row.append("")
                    continue
                inside = _words_in(words, pymupdf.Rect(c))
                used.update(id(w) for w in inside)
                row.append(_cell_text(inside))
            rows.append(row)
        bbox = pymupdf.Rect(t.bbox)
        first = next((cells for cells in row_cells if all(c is not None for c in cells)), row_cells[0])
        xs = [c[0] for c in first if c is not None] + [first[-1][2] if first[-1] is not None else bbox.x1]
        height = max((c[3] - c[1] for cells in row_cells for c in cells if c is not None), default=0.0)
        grids.append(Grid(bbox=bbox, xs=xs, rows=rows, used=used, row_height=height))
    return grids


def _open_row(grid: Grid, band: list[tuple]) -> list[str] | None:
    """Строка таблицы без линии сверху (или снизу): так рвётся строка на границе страницы.

    Слова полосы над (под) сеткой берутся строкой, только если каждое слово целиком лежит в одной
    колонке сетки и полоса не выше строки сетки. Подпись над таблицей, пересекающая колонки, строкой
    не становится — её слова останутся «вне ячеек» и дадут предупреждение.
    """
    if not band:
        return None
    top, bottom = min(w[1] for w in band), max(w[3] for w in band)
    if grid.row_height and bottom - top > 1.5 * grid.row_height:
        return None
    cols: list[list[tuple]] = [[] for _ in grid.xs[:-1]]
    for w in band:
        k = next(
            (
                i
                for i in range(len(cols))
                if w[0] >= grid.xs[i] - COLUMN_TOLERANCE and w[2] <= grid.xs[i + 1] + COLUMN_TOLERANCE
            ),
            None,
        )
        if k is None:
            return None
        cols[k].append(w)
    return [_cell_text(c) for c in cols]


def escape_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def table_markdown(header: list[str], rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(escape_cell(c) for c in cells) + " |"

    out = [line(header), "| " + " | ".join("---" for _ in header) + " |"]
    out += [line(r) for r in rows]
    return "\n".join(out) + "\n\n"


def _same_columns(a: Grid, b: Grid) -> bool:
    return len(a.xs) == len(b.xs) and all(
        abs(x - y) <= COLUMN_TOLERANCE for x, y in zip(a.xs, b.xs, strict=True)
    )


def _inside_share(box: pymupdf.Rect, grid: pymupdf.Rect) -> float:
    area = box.get_area()
    return (box & grid).get_area() / area if area else 0.0


def _layout_table_cells(md: str) -> list[list[str]]:
    rows = []
    for line in md.splitlines():
        s = line.strip()
        if not s.startswith("|") or re.fullmatch(r"\|?[\s:|-]+\|?", s):
            continue
        rows.append([_MD_MARKUP.sub(" ", c) for c in s.strip("|").split("|")])
    return rows


def unverified_tokens(md: str, words: list[str]) -> list[str]:
    """Слова ячеек таблицы, которых нет в текстовом слое: разрезаны или склеены при сборке."""
    known = {_TOKEN_EDGE.sub("", w) for w in words}
    bad = []
    for row in _layout_table_cells(md):
        for cell in row:
            for tok in cell.split():
                core = _TOKEN_EDGE.sub("", tok)
                if core and core not in known:
                    bad.append(tok)
    return bad


# --------------------------------------------------------------------------- заголовки


def _signature(page: pymupdf.Page, rect: pymupdf.Rect) -> tuple[float, bool] | None:
    """(кегль, жирный) текста в прямоугольнике, если он однороден."""
    d = page.get_text("dict", clip=rect)
    spans = [
        s
        for b in d.get("blocks", [])
        for ln in b.get("lines", [])
        for s in ln.get("spans", [])
        if s.get("text", "").strip()
    ]
    if not spans:
        return None
    sizes = {round(s["size"], 1) for s in spans}
    bold = {bool(s["flags"] & 16) or "bold" in s.get("font", "").lower() for s in spans}
    if len(sizes) != 1 or len(bold) != 1:
        return None
    return sizes.pop(), bold.pop()


def _plain(md: str) -> str:
    return re.sub(r"\s+", " ", _MD_MARKUP.sub("", md).replace("#", "")).strip()


def _running_key(text: str) -> str:
    return re.sub(r"\d+", "#", _plain(text).lower())


# --------------------------------------------------------------------------- сборка


def repair_pages(doc: pymupdf.Document, chunks: list[dict[str, Any]], issues: list[IngestIssue]) -> list[str]:
    """Тексты страниц pymupdf4llm с перестроенными таблицами и восстановленными заголовками.

    Без ``page_boxes`` (pymupdf4llm без модели разметки) тексты возвращаются как есть.
    """
    texts = [str(c.get("text") or "") for c in chunks]
    if len(chunks) != doc.page_count or not all(c.get("page_boxes") for c in chunks):
        return [_LIST_BULLET_DUP.sub(r"\1", t) for t in texts]

    pages: list[list[_Box]] = []
    grids_by_page: list[list[Grid]] = []
    words_by_page: list[list[tuple]] = []
    for idx, chunk in enumerate(chunks):
        page = doc[idx]
        boxes = []
        for b in chunk["page_boxes"]:
            pos = b.get("pos") or (0, 0)
            s, e = int(pos[0]), int(pos[1])
            boxes.append(_Box(str(b.get("class", "")), pymupdf.Rect(b["bbox"]), s, e, texts[idx][s:e]))
        boxes.sort(key=lambda bx: bx.start)
        pages.append(boxes)
        words = page.get_text("words")
        words_by_page.append(words)
        grids_by_page.append(ruled_tables(page, words))

    _attach_grids(pages, grids_by_page, words_by_page, issues)
    _restore_headings(doc, pages)
    _join_continuations(pages)

    out = []
    for idx, boxes in enumerate(pages):
        text = texts[idx]
        for bx in sorted(boxes, key=lambda b: b.start, reverse=True):
            text = text[: bx.start] + bx.text + text[bx.end :]
        out.append(_LIST_BULLET_DUP.sub(r"\1", text))
    return out


def _attach_grids(
    pages: list[list[_Box]],
    grids_by_page: list[list[Grid]],
    words_by_page: list[list[tuple]],
    issues: list[IngestIssue],
) -> None:
    for idx, (boxes, grids, words) in enumerate(zip(pages, grids_by_page, words_by_page, strict=True)):
        page_no = idx + 1
        claimed: set[int] = set()
        for grid in grids:
            inside = [bx for bx in boxes if _inside_share(bx.bbox, grid.bbox) >= INSIDE_SHARE]
            if not inside:
                # таблица есть на странице, но модель разметки её не вывела: вставляем перед
                # первым боксом ниже неё, чтобы текст таблицы не потерялся
                after = next((bx for bx in boxes if bx.bbox.y0 >= grid.bbox.y0), None)
                anchor = after.start if after else (boxes[-1].end if boxes else 0)
                new = _Box("table", grid.bbox, anchor, anchor, "", grid)
                boxes.append(new)
                boxes.sort(key=lambda bx: (bx.start, bx.end))
                inside = [new]
            first, *rest = sorted(inside, key=lambda bx: bx.start)
            region = pymupdf.Rect(grid.bbox)
            for bx in inside:
                region |= bx.bbox
            _recover_open_rows(grid, words, region)
            first.cls, first.grid = "table", grid
            first.text = table_markdown(grid.rows[0], grid.rows[1:])
            for bx in rest:
                bx.text, bx.cls = "", "merged"
            claimed.update(id(bx) for bx in inside)
            # текст боксов заменён таблицей: всё, что в них было и не попало в ячейку, — потеря
            lost = [w[4] for w in _words_in(words, region) if id(w) not in grid.used]
            if lost:
                issues.append(
                    IngestIssue(
                        level=IssueLevel.warning,
                        code="pdf_table_degraded",
                        message="Часть слов таблицы не попала ни в одну ячейку при чтении PDF "
                        f"(например, «{' '.join(lost[:6])}»); содержимое таблицы может быть неполным.",
                        location=f"стр. {page_no}",
                    )
                )
        for bx in boxes:
            if bx.cls == "table" and id(bx) not in claimed:
                _check_layout_table(bx, [w[4] for w in _words_in(words, bx.bbox)], page_no, issues)


def _recover_open_rows(grid: Grid, words: list[tuple], region: pymupdf.Rect) -> None:
    """Строки без верхней (нижней) линии над (под) сеткой внутри бокса таблицы — в начало (конец)."""
    x0, x1 = grid.xs[0] - COLUMN_TOLERANCE, grid.xs[-1] + COLUMN_TOLERANCE
    free = [
        w
        for w in _words_in(words, region)
        if id(w) not in grid.used and (w[0] + w[2]) / 2 >= x0 and (w[0] + w[2]) / 2 <= x1
    ]
    above = [w for w in free if (w[1] + w[3]) / 2 < grid.bbox.y0]
    below = [w for w in free if (w[1] + w[3]) / 2 > grid.bbox.y1]
    if (row := _open_row(grid, above)) is not None:
        grid.rows.insert(0, row)
        grid.used.update(id(w) for w in above)
    if (row := _open_row(grid, below)) is not None:
        grid.rows.append(row)
        grid.used.update(id(w) for w in below)


def _check_layout_table(bx: _Box, words: list[str], page_no: int, issues: list[IngestIssue]) -> None:
    """Таблица без линий сетки: перестроить нельзя, но порчу слов видно по текстовому слою."""
    bad = unverified_tokens(bx.text, words)
    if bad:
        issues.append(
            IngestIssue(
                level=IssueLevel.warning,
                code="pdf_table_degraded",
                message="Таблица PDF без линий сетки собрана с искажениями: в ячейках есть слова, которых "
                f"нет в тексте страницы (разрезаны или склеены: «{'», «'.join(bad[:5])}»). Колонки и "
                "значения таблицы могут быть сдвинуты; сверьте таблицу с документом.",
                location=f"стр. {page_no}",
            )
        )


def _restore_headings(doc: pymupdf.Document, pages: list[list[_Box]]) -> None:
    """Колонтитул, который не повторяется на других страницах и набран шрифтом заголовка, — заголовок."""
    levels: dict[tuple[float, bool], set[int]] = {}
    margins: dict[str, int] = {}
    for idx, boxes in enumerate(pages):
        for bx in boxes:
            if bx.cls == "section-header":
                m = re.match(r"\s*(#{1,6})\s", bx.text)
                sig = _signature(doc[idx], bx.bbox)
                if m and sig:
                    levels.setdefault(sig, set()).add(len(m.group(1)))
            elif bx.cls in _BODY_EXCLUDED:
                key = _running_key(bx.text)
                margins[key] = margins.get(key, 0) + 1
    for idx, boxes in enumerate(pages):
        for bx in boxes:
            if bx.cls not in _BODY_EXCLUDED:
                continue
            text = _plain(bx.text)
            if not text or margins.get(_running_key(bx.text), 0) > 1:
                continue  # пусто или настоящий колонтитул, повторяется на страницах
            if len(text.split()) > HEADING_MAX_WORDS or text.endswith((".", ";", ",")):
                continue
            sig = _signature(doc[idx], bx.bbox)
            found = levels.get(sig) if sig else None
            if found and len(found) == 1:
                level = next(iter(found))
                bx.cls, bx.heading_level = "section-header", level
                bx.text = f"{'#' * level} {text}\n\n"


def _body(boxes: list[_Box]) -> list[_Box]:
    return [bx for bx in boxes if bx.cls not in _BODY_EXCLUDED and bx.cls != "merged" and bx.text.strip()]


def _join_continuations(pages: list[list[_Box]]) -> None:
    """Таблица в начале страницы с теми же колонками, что таблица в конце предыдущей, — её продолжение.

    Если первая строка не повторяет заголовок, она — данные: подставляем заголовок исходной таблицы.
    """
    for idx in range(1, len(pages)):
        prev, cur = _body(pages[idx - 1]), _body(pages[idx])
        if not prev or not cur:
            continue
        last, first = prev[-1], cur[0]
        if not (last.grid and first.grid and _same_columns(last.grid, first.grid)):
            continue
        header = last.grid.header  # у продолжения продолжения — заголовок самой первой части
        if [c.strip() for c in first.grid.rows[0]] == [c.strip() for c in header]:
            continue  # заголовок повторён на новой странице
        first.grid.carried_header = header
        first.text = table_markdown(header, first.grid.rows)
