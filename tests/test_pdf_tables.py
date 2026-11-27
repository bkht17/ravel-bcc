"""Таблицы PDF: ячейки, заголовки колонок, страницы и предупреждения о порче (аудит, табличный PDF).

sample_typeset_tables.pdf — samples/requirements.md, свёрстанный аудитором в PDF с таблицами (4 страницы).
До исправления этот PDF давал то же число требований, что и MD, но заголовок колонки «Format» таблицы
ответа делился на «Forma» и «t Logic», и две находки о типах полей молча пропадали.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ravel import orchestrator
from ravel.ingest import load_document
from ravel.ingest.markdown import parse_markdown
from ravel.ingest.pdf_layout import join_cell_lines, repair_pages, unverified_tokens
from ravel.models import BlockKind, IssueLevel
from ravel.quotes import locate
from ravel.segment import segment

FIXTURES = Path(__file__).parent / "fixtures"
TYPESET = FIXTURES / "sample_typeset_tables.pdf"


def _rows(doc):
    return [
        (b.section[-1], tuple(b.header or ()), tuple(b.cells or ()))
        for b in doc.blocks
        if b.kind == BlockKind.table_row
    ]


def _row(doc, first_cell: str, section: str):
    return next(
        b
        for b in doc.blocks
        if b.kind == BlockKind.table_row and b.cells and b.cells[0] == first_cell and b.section[-1] == section
    )


@pytest.fixture(scope="module")
def typeset():
    return load_document(TYPESET)


@pytest.fixture(scope="module")
def sample_doc():
    return load_document(Path(__file__).resolve().parents[1] / "samples" / "requirements.md")


# --------------------------------------------------------------------------- тот же источник: MD и PDF


def test_typeset_pdf_header_cells_are_whole(typeset):
    """Первая точка расхождения: граница колонки резала слово заголовка «Format»."""
    hdr = _row(typeset, "amount", "Ответ").header
    assert hdr == ["Parameter", "Format", "Logic", "Description", "DB field"]
    assert _row(typeset, "amount", "Query-параметры").header == [
        "Parameter", "Format", "Size", "Mandatory", "Logic", "Description", "DB field",
    ]  # fmt: skip


def test_typeset_pdf_cells_values_and_pages(typeset):
    resp = _row(typeset, "enabled", "Ответ")
    # перенос строки внутри ячейки — та же ячейка, а не новая строка таблицы с пустыми ячейками
    assert resp.cells == ["enabled", "string", "«0» или «1»", "Признак включения", "enabled"]
    assert resp.page == 4
    query = _row(typeset, "amount", "Query-параметры")
    assert query.cells[4] == "1 000–1 000 000"  # число, разорванное переносом в ячейке
    assert query.page == 4
    # пробелы между словами в ячейке сохранены
    assert (
        _row(typeset, "Наименование задачи", "Общие сведения").cells[1] == "Месячный порог расходов по карте"
    )
    assert not [r for r in _rows(typeset) if any("<br>" in c for c in r[2])]


def test_typeset_pdf_heading_at_page_top_is_section(typeset):
    """«Query-параметры» вверху стр. 4 модель разметки сочла колонтитулом: таблица уходила в чужой раздел."""
    row = _row(typeset, "cardId", "Query-параметры")
    assert row.section[-2:] == ["Параметры запроса", "Query-параметры"]


def test_typeset_pdf_table_continues_across_page_break(typeset):
    """Продолжение таблицы на стр. 3 без повтора заголовка: первая строка — данные, заголовок — исходный."""
    rows = [
        b for b in typeset.blocks if b.kind == BlockKind.table_row and b.section[-1] == "Ревью требований"
    ]
    assert {b.page for b in rows} == {2, 3}
    on_3 = [b for b in rows if b.page == 3]
    assert (
        on_3[0].cells[0]
        == "В требовании присутствует схема полного бизнес-процесса и часть реализуемой доработки"
    )
    assert all(b.header == ["Проверка", "Создано или доступно", "Проверено", "Примечания"] for b in rows)


def test_typeset_pdf_tables_equal_markdown_tables(typeset, sample_doc):
    md, pdf = _rows(sample_doc), _rows(typeset)
    missing = [r for r in md if r not in pdf]
    # единственное отличие: пустая строка пустой таблицы — в PDF её высота 0,5 пт, линии сливаются
    assert missing == [("Блок согласующих", ("Подразделение", "ФИО ответственного", "Подпись"), ("", "", ""))]
    assert [r for r in pdf if r not in md] == []
    assert all(b.page for b in typeset.blocks if b.kind == BlockKind.table_row)
    assert not [i for i in typeset.issues if i.level != IssueLevel.info]


def test_typeset_pdf_requirements_equal_markdown(typeset, sample_doc):
    """Содержание, происхождение и связи, а не количество: текст, номер пункта, раздел, вид."""

    def key(r):
        return (r.kind, r.source_id, r.text, tuple(r.section))

    md, pdf = segment(sample_doc), segment(typeset)
    assert [key(r) for r in md] == [key(r) for r in pdf]
    assert all("стр." in r.location for r in pdf)
    f4 = [r for r in pdf if r.source_id == "Ф-4"]  # маркер «•» больше не прячет номер пункта
    assert f4 and f4[0].text.startswith("Ф-4. ")


def test_typeset_pdf_type_findings_match_markdown():
    """Находки о типах полей amount и enabled есть на обоих входах, с теми же требованиями и страницей."""

    def type_findings(path):
        report = orchestrator.run(path, llm=None)
        by_id = {r.id: r for r in report.requirements}
        out = {}
        for f in report.findings:
            if f.source == "lint:param_type_mismatch":
                out[f.title] = (
                    sorted(by_id[i].text for i in f.requirement_ids),
                    [q.location for q in f.quotes],
                )
        return report, out

    md_report, md = type_findings(Path(__file__).resolve().parents[1] / "samples" / "requirements.md")
    pdf_report, pdf = type_findings(TYPESET)
    assert (
        set(md)
        == set(pdf)
        == {
            "Параметр amount: тип integer в одном месте и string в другом",
            "Параметр enabled: тип integer в одном месте и string в другом",
        }
    )
    for title in md:
        assert md[title][0] == pdf[title][0]
        assert all(loc.endswith("стр. 4") for loc in pdf[title][1])
    assert {f.title for f in md_report.findings} - {f.title for f in pdf_report.findings} == {
        "Таблица «Блок согласующих» пустая"
    }


# --------------------------------------------------------------------------- независимая таблица


def _story_pdf(path: Path, html: str) -> None:
    css = "table{border-collapse:collapse} td,th{border:1px solid #000;padding:2px;font-size:10px}"
    story = pymupdf.Story(html=html, user_css=css)
    writer = pymupdf.DocumentWriter(str(path))
    media = pymupdf.paper_rect("a5")
    more = True
    while more:
        dev = writer.begin_page(media)
        more, _ = story.place(media + (36, 36, -36, -36))
        story.draw(dev)
        writer.end_page()
    writer.close()


LONG_ROWS = [(f"код_услуги_{i:02d}", "string", f"Тариф {i * 150} ₸ за операцию") for i in range(1, 56)]
REQUEST = [
    ("сумма_платежа", "integer", "Сумма в ₸, от 100 до 5 000 000 ₸ включительно, без копеек и тиынов"),
    ("признак_активности", "boolean", "Флаг включения"),
]
RESPONSE = [("сумма_платежа", "string", "Сумма в ₸ строкой")]


def _html_table(rows) -> str:
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return f"<table><tr><th>Поле</th><th>Тип</th><th>Описание</th></tr>{body}</table>"


def _md_table(rows) -> str:
    return "| Поле | Тип | Описание |\n| --- | --- | --- |\n" + "".join(
        f"| {a} | {b} | {c} |\n" for a, b, c in rows
    )


@pytest.fixture(scope="module")
def independent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("independent")
    pdf = tmp / "платежи.pdf"
    _story_pdf(
        pdf,
        "<h1>Платежи</h1><h2>Запрос</h2>" + _html_table(REQUEST) + "<h2>Тарифы</h2>" + _html_table(LONG_ROWS)
        + "<h2>Ответ</h2>" + _html_table(RESPONSE),
    )  # fmt: skip
    md = tmp / "платежи.md"
    md.write_text(
        "# Платежи\n\n## Запрос\n\n" + _md_table(REQUEST) + "\n## Тарифы\n\n" + _md_table(LONG_ROWS)
        + "\n## Ответ\n\n" + _md_table(RESPONSE),
        encoding="utf-8",
    )  # fmt: skip
    return md, pdf


def test_independent_table_cells_headers_and_pages(independent):
    md_path, pdf_path = independent
    md, pdf = load_document(md_path), load_document(pdf_path)
    assert pymupdf.open(pdf_path).page_count >= 2  # таблица тарифов переходит на следующую страницу
    assert _rows(pdf) == _rows(md)  # раздел, заголовок и каждая ячейка — кириллица, «₸», «_»
    tariffs = [b for b in pdf.blocks if b.kind == BlockKind.table_row and b.section[-1] == "Тарифы"]
    assert len({b.page for b in tariffs}) >= 2
    assert all(b.page and b.header == ["Поле", "Тип", "Описание"] for b in tariffs)
    # страница строки — та, где строка напечатана
    for b in tariffs:
        assert pymupdf.open(pdf_path)[b.page - 1].search_for(b.cells[0])
    assert not [i for i in pdf.issues if i.level != IssueLevel.info]


def test_independent_table_type_conflict_found_in_md_and_pdf(independent):
    """Колонки «Поле»/«Тип» — те же правила, что для «Parameter»/«Format»; находка на обоих входах."""
    for path in independent:
        report = orchestrator.run(path, llm=None)
        found = [f for f in report.findings if f.source == "lint:param_type_mismatch"]
        assert [f.title for f in found] == [
            "Параметр сумма_платежа: тип integer в одном месте и string в другом"
        ]
        assert len(found[0].requirement_ids) == 2


# --------------------------------------------------------------------------- деградация и мелочи


def test_layout_table_with_broken_words_warns(tmp_path):
    """Таблица без линий сетки, собранная с разрезанным словом: явное предупреждение, не молчание."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Parameter Format Logic")
    page.insert_text((72, 90), "amount string yes")
    chunk = {
        "text": "|Parameter|Forma|t Logic|\n|---|---|---|\n|amount|string|yes|\n\n",
        "page_boxes": [{"class": "table", "bbox": (60, 50, 400, 100), "pos": (0, 70)}],
    }
    issues = []
    texts = repair_pages(doc, [chunk], issues)
    assert texts[0] == chunk["text"]  # перестроить без сетки нечем — текст как был
    warn = [i for i in issues if i.code == "pdf_table_degraded"]
    assert warn and warn[0].level == IssueLevel.warning and warn[0].location == "стр. 1"
    assert "Forma" in warn[0].message
    assert unverified_tokens("|Parameter|Format|Logic|", ["Parameter", "Format", "Logic"]) == []


def test_page_header_heading_restored_but_running_header_kept(tmp_path):
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page()
    doc[0].insert_text((72, 40), "Отчётная форма 7", fontsize=8)  # колонтитул, повторяется
    doc[0].insert_text((72, 100), "Раздел запроса", fontname="hebo", fontsize=10)
    doc[1].insert_text((72, 40), "Отчётная форма 7", fontsize=8)
    doc[1].insert_text((72, 60), "Параметры ответа", fontname="hebo", fontsize=10)
    chunks = [
        {
            "text": "Отчётная форма 7\n\n#### Раздел запроса\n\n",
            "page_boxes": [
                {"class": "page-header", "bbox": (70, 30, 300, 45), "pos": (0, 18)},
                {"class": "section-header", "bbox": (70, 90, 300, 105), "pos": (18, 40)},
            ],
        },
        {
            "text": "Отчётная форма 7\n\n**Параметры ответа**\n\n",
            "page_boxes": [
                {"class": "page-header", "bbox": (70, 30, 300, 45), "pos": (0, 18)},
                {"class": "page-header", "bbox": (70, 50, 300, 65), "pos": (18, 42)},
            ],
        },
    ]
    texts = repair_pages(doc, chunks, [])
    assert "#### Параметры ответа" in texts[1]
    assert "#" not in texts[1].split("\n")[0]  # повторяющийся колонтитул заголовком не стал


def test_cell_lines_join_keeps_slash_and_hyphen_words():
    assert join_cell_lines(["Признак", "включения"]) == "Признак включения"
    assert join_cell_lines(["иные продукты/", "системы"]) == "иные продукты/системы"
    assert join_cell_lines(["документа /", "зафиксирована"]) == "документа / зафиксирована"
    assert join_cell_lines(["без Jira-", "задачи"]) == "без Jira-задачи"
    assert join_cell_lines(["1 000–1 000", "000 ₸"]) == "1 000–1 000 000 ₸"


def test_markdown_cell_escaped_pipe_and_br():
    _, blocks = parse_markdown("| Поле | Значение |\n| --- | --- |\n| A \\| B | строка 1<br>строка 2 |\n")
    row = next(b for b in blocks if b.kind == BlockKind.table_row)
    assert row.cells == ["A | B", "строка 1 строка 2"]


def test_example_from_pdf_stays_evidence(typeset):
    """Демонстрационный curl — пример (E-), не требование, но цитата из него находится в документе."""
    ex = [r for r in segment(typeset) if r.kind == "example" and "curl" in r.text]
    assert ex
    assert locate('curl -G "$BASE/alerts/saveLimit"', typeset).id == ex[0].block_id
