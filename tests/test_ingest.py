"""Повреждённые и необычные файлы: прогон никогда не падает с исключением."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pymupdf
import pytest

from ravel.ingest import load_document
from ravel.models import IssueLevel
from ravel.segment import segment

from .conftest import MINI_MD, make_docx, make_pdf


def codes(doc):
    return {i.code for i in doc.issues}


def is_fatal(doc):
    return any(i.level == IssueLevel.fatal for i in doc.issues)


def test_sample_reads_clean(sample):
    doc = load_document(sample)
    assert not doc.issues
    assert doc.title == "Уведомление о расходах по карте"
    assert any(b.text.startswith("Ф-3.") for b in doc.blocks)


@pytest.mark.parametrize(
    ("name", "data", "code"),
    [
        ("empty.md", b"", "empty_file"),
        ("spaces.txt", b"   \n\n  ", "empty_file"),
        ("noise.md", random.Random(0).randbytes(4096), None),
        ("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100, "unsupported_doc"),
        ("doc.rtf", b"{\\rtf1\\ansi hello}", "unsupported_rtf"),
        ("arch.zip", b"PK\x03\x04" + b"\x00" * 64, "broken_docx"),
    ],
)
def test_broken_inputs_are_fatal_not_crash(tmp_path, name, data, code):
    p = tmp_path / name
    p.write_bytes(data)
    doc = load_document(p)
    assert is_fatal(doc)
    if code:
        assert code in codes(doc)
    assert doc.blocks == []


def test_missing_file_and_directory(tmp_path):
    assert "not_found" in codes(load_document(tmp_path / "nope.md"))
    assert "is_directory" in codes(load_document(tmp_path))


def test_encrypted_pdf(tmp_path):
    p = tmp_path / "secret.pdf"
    p.write_bytes(
        make_pdf(["F-1. Secret."], encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    )
    assert "encrypted_pdf" in codes(load_document(p))


def test_truncated_pdf_does_not_crash(tmp_path):
    data = make_pdf(["F-1. The user sets a monthly limit between 1000 and 10000."])
    p = tmp_path / "cut.pdf"
    p.write_bytes(data[: len(data) // 2])
    doc = load_document(p)  # либо восстановлен с предупреждением, либо понятная ошибка
    assert doc.issues


def test_pdf_without_headings_still_segments(tmp_path):
    p = tmp_path / "plain.pdf"
    p.write_bytes(
        make_pdf(
            [
                "F-1. The user sets a monthly limit between 1000 and 10000.",
                "",
                "F-2. A push is sent when the limit is reached.",
            ]
        )
    )
    doc = load_document(p)
    assert not is_fatal(doc), doc.issues
    ids = {r.source_id for r in segment(doc)}
    assert {"F-1", "F-2"} <= ids


def test_blank_pdf_reports_no_text_layer(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    p = tmp_path / "scan.pdf"
    p.write_bytes(doc.tobytes())
    assert "no_text_layer" in codes(load_document(p))


def test_extension_mismatch_is_warning(tmp_path):
    p = tmp_path / "really_markdown.pdf"
    p.write_text(MINI_MD, encoding="utf-8")
    doc = load_document(p)
    assert "ext_mismatch" in codes(doc)
    assert not is_fatal(doc)
    assert len(segment(doc)) == 3


def test_cp1251_text(tmp_path):
    p = tmp_path / "legacy.md"
    p.write_bytes(MINI_MD.encode("cp1251"))
    doc = load_document(p)
    assert "non_utf8" in codes(doc)
    assert any("Пользователь задаёт порог" in b.text for b in doc.blocks)


def test_docx_with_damaged_xml_is_salvaged(tmp_path):
    import io
    import zipfile

    good = make_docx(MINI_MD)
    src = zipfile.ZipFile(io.BytesIO(good))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                data = data.replace(b"</w:body>", b"")  # XML больше не валиден
            out.writestr(item, data)
    p = tmp_path / "damaged.docx"
    p.write_bytes(buf.getvalue())
    doc = load_document(p)
    assert "docx_fallback" in codes(doc)
    assert any("Пользователь задаёт порог" in b.text for b in doc.blocks)


# --- контроль покрытия, резервная нарезка, OCR ---------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
# число требований в табличном PDF до правок (прогон текущего кода перед изменением)
TABLES_PDF_BASELINE = 51
SAMPLE_REQS = 49


def test_md_sample_requirement_count_unchanged(sample):
    reqs = segment(load_document(sample))
    assert sum(r.kind == "requirement" for r in reqs) == SAMPLE_REQS
    # 5 служебных строк из «Часть B» + 7 строк таблицы «Общие сведения» (раньше пропускались)
    assert sum(r.kind == "metadata" for r in reqs) == 12
    assert sum(r.kind == "example" for r in reqs) == 6


def test_printed_pdf_is_not_silently_lost(sample):
    doc = load_document(FIXTURES / "requirements_printed.pdf")
    assert not is_fatal(doc), doc.issues
    n = len(segment(doc))
    n_md = len(segment(load_document(sample)))
    assert n > 0
    assert n >= 0.7 * n_md
    assert "low_coverage" not in codes(doc) or "fallback_segmentation" in codes(doc)


def test_tables_pdf_not_worse():
    doc = load_document(FIXTURES / "requirements_tables.pdf")
    # baseline — число именно требований; строки «Общие сведения» теперь метаданные и не в счёте
    n = sum(r.kind == "requirement" for r in segment(doc))
    assert abs(n - TABLES_PDF_BASELINE) <= 0.1 * TABLES_PDF_BASELINE
    assert "low_coverage" not in codes(doc)


def test_scan_is_ocr_low_quality_without_requirements():
    from ravel.ingest.pdf import _ocr_languages, tessdata_dir

    tessdata = tessdata_dir()
    if tessdata is None:
        pytest.skip("Tesseract не установлен")
    if "rus" in _ocr_languages(tessdata).split("+"):
        pytest.skip("есть tessdata rus: скан может прочитаться нормально")
    doc = load_document(FIXTURES / "scan.pdf")
    fatal = [i for i in doc.issues if i.code == "ocr_low_quality"]
    assert fatal and fatal[0].level == IssueLevel.fatal
    assert "rus" in fatal[0].message  # причина названа: нет языкового пакета
    assert doc.blocks == [] and segment(doc) == []


def test_headingless_text_with_six_numbered_items(tmp_path):
    p = tmp_path / "plain.txt"
    p.write_text(
        "\n".join(
            f"{i}. Система должна выполнять действие номер {i} за время не более {i} секунд."
            for i in range(1, 7)
        ),
        encoding="utf-8",
    )
    doc = load_document(p)
    assert len(segment(doc)) == 6
    assert "low_coverage" not in codes(doc)


def test_mostly_fragments_reports_low_coverage(tmp_path):
    p = tmp_path / "frag.md"
    p.write_text(
        "\n\n".join(["Да."] * 80) + "\n\nСистема должна отправлять уведомление клиенту.\n", encoding="utf-8"
    )
    doc = load_document(p)
    issue = next(i for i in doc.issues if i.code == "low_coverage")
    assert issue.level == IssueLevel.warning
    assert "%" in issue.message and "символов" in issue.message


def test_fallback_blocks_keep_lines_and_pages():
    from ravel.ingest.fallback import parse_fallback

    md = "Ф-1. Первое требование документа.\nпродолжение первого.\n\n- второй пункт списка длинный\n3) третий пункт нумерации\n| a | b |\n"
    blocks = parse_fallback(md, {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
    assert [b.line for b in blocks] == [1, 4, 5, 6]
    assert blocks[0].text == "Ф-1. Первое требование документа. продолжение первого."
    assert blocks[0].location == "начало документа, стр. 1"
    assert blocks[1].location.endswith("стр. 2")
    assert blocks[3].kind.value == "table_row"


def test_pdf_locations_have_page_numbers():
    doc = load_document(FIXTURES / "requirements_printed.pdf")
    reqs = segment(doc)
    assert reqs and all("стр. " in r.location for r in reqs)
    assert {r.location.rsplit("стр. ", 1)[1] for r in reqs} >= {"1", "4"}


# --------------------------------------------------------------------------- нарезка: классы и строки


def _segment_text(tmp_path, md: str):
    p = tmp_path / "doc.md"
    p.write_text(md, encoding="utf-8")
    return segment(load_document(p))


def test_passport_table_is_metadata_not_requirements(tmp_path):
    reqs = _segment_text(
        tmp_path,
        "# Док\n\n## Часть B\n\n### Общие параметры\n\n| Поле | Значение |\n| :--- | :--- |\n"
        "| **Author** | Школа |\n| **Developer** | Команда |\n| **Gitlab project** | Учебный |\n\n"
        "### Цель\nСервис должен уведомлять клиента о достижении порога.\n",
    )
    # порядок выдачи: требования, затем метаданные; ID по видам раздельные
    assert [r.kind for r in reqs] == ["requirement"] + ["metadata"] * 3
    assert [r.id for r in reqs] == ["R-001", "M-001", "M-002", "M-003"]
    assert [r.section[-1] for r in reqs if r.kind == "requirement"] == ["Цель"]  # из паспорта — ни одного
    assert reqs[1].raw_text == "| **Author** | Школа |"
    assert (reqs[1].line_start, reqs[1].line_end) == (9, 9)


def test_numbered_lists_keep_numbers_in_source_id(tmp_path):
    reqs = _segment_text(
        tmp_path,
        "# Док\n\n## A\n\n### Логика\n1. Проверить сессию клиента.\n2. Проверить карту клиента.\n\n"
        "### Правила\n1. Целое число тенге.\n2. Границы включены.\n",
    )
    ids = [r.source_id for r in reqs]
    assert ids == ["Логика п. 1", "Логика п. 2", "Правила п. 1", "Правила п. 2"]
    assert len(set(ids)) == len(ids)


def test_status_lines_in_one_paragraph_split_into_requirements(tmp_path):
    reqs = _segment_text(
        tmp_path,
        "# Док\n\n## A\n\n### Статусы\n"
        "**400.** Ответ, если параметры неверны.  \n"
        "**401.** Ответ, если нет авторизации.  \n"
        "**403.** Ответ, если доступ запрещён.\n",
    )
    assert len(reqs) == 3
    assert [(r.line_start, r.line_end) for r in reqs] == [(6, 6), (7, 7), (8, 8)]
    assert reqs[1].text.startswith("401.")
    assert reqs[1].raw_text.startswith("**401.**")


def test_json_block_is_one_example_with_lines(tmp_path):
    reqs = _segment_text(
        tmp_path,
        '# Док\n\n## A\n\n#### 400. Bad request\n```json\n{"message": "x"}\n```\n',
    )
    assert len(reqs) == 1
    ex = reqs[0]
    assert ex.kind == "example"
    assert (ex.line_start, ex.line_end) == (6, 8)
    assert ex.raw_text == '```json\n{"message": "x"}\n```'
    assert ex.text.startswith("400. Bad request:") and '"message"' in ex.text


def test_schema_block_is_requirement(tmp_path):
    reqs = _segment_text(
        tmp_path, "# Док\n\n## A\n\n### Схема\n```text\nПриложение → Сервис → приложение.\n```\n"
    )
    assert [r.kind for r in reqs] == ["requirement"]
    assert reqs[0].text == "Схема: Приложение → Сервис → приложение."


def test_short_statement_under_heading_is_kept(tmp_path):
    reqs = _segment_text(
        tmp_path, "# Док\n\n## A\n\n#### Path-параметры\nНе используются.\n\n```json\n{}\n```\nили\n"
    )
    short = [r for r in reqs if r.kind == "requirement"]
    assert [r.text for r in short] == ["Path-параметры: Не используются."]
    assert short[0].raw_text == "Не используются."
    assert short[0].line_start == 6


def test_table_row_keeps_raw_text(tmp_path):
    reqs = _segment_text(
        tmp_path,
        "# Док\n\n## A\n\n### Критерии\n| Условие | Результат |\n| --- | --- |\n| Порог 5 | Один push |\n",
    )
    assert reqs[0].text == "Условие: Порог 5; Результат: Один push"
    assert reqs[0].raw_text == "| Порог 5 | Один push |"


def test_info_issue_is_not_an_undecided_reason(sample):
    from ravel.models import IngestIssue
    from ravel.orchestrator import run, undecided_reasons

    report = run(sample, None)
    report.issues = [IngestIssue(level=IssueLevel.info, code="fallback_segmentation", message="x")]
    assert not any("прочитан частично" in r for r in undecided_reasons(report))


def test_sample_raw_text_is_verbatim_for_every_requirement(sample):
    from ravel.ingest.markdown import normalize_text

    lines = normalize_text(sample.read_text(encoding="utf-8")).split("\n")
    reqs = segment(load_document(sample))
    for r in reqs:
        assert r.raw_text and r.line_start and r.line_end, r
        assert "\n".join(lines[r.line_start - 1 : r.line_end]).count(r.raw_text) >= 1, r


def test_sample_expected_classes(sample):
    reqs = segment(load_document(sample))

    def at(line: int):
        return next(r for r in reqs if r.line_start == line)

    for line in (110, 111, 112, 113, 114):  # Swagger, Gitlab project, Framework, Author, Developer
        assert at(line).kind == "metadata"
    assert at(103).kind == "requirement"  # Description остаётся описанием метода
    assert at(129).kind == "requirement" and at(129).text.startswith("Схема:")
    for start in (166, 170, 179, 184, 191, 196):
        assert at(start).kind == "example"
    for line in (162, 163, 174, 175, 176):  # статусы 200, 400, 401, 403, 500 — по отдельному требованию
        assert at(line).kind == "requirement" and at(line).raw_text.startswith("**")
    assert at(136).text == "Path-параметры: Не используются."
    nums = [r.source_id for r in reqs if r.source_id and " п. " in r.source_id]
    assert len(nums) == 8 and len(set(nums)) == 8


def test_sample_ids_are_separate_series_without_gaps(sample):
    reqs = segment(load_document(sample))
    for kind, prefix in (("requirement", "R"), ("metadata", "M"), ("example", "E")):
        ids = [r.id for r in reqs if r.kind == kind]
        assert ids, kind
        assert ids == [f"{prefix}-{i:03d}" for i in range(1, len(ids) + 1)], kind
    assert len({r.id for r in reqs}) == len(reqs)
    assert sum(r.id.startswith("R-") for r in reqs) == sum(r.kind == "requirement" for r in reqs)


def test_general_info_rows_are_metadata_not_skipped(sample):
    reqs = segment(load_document(sample))
    meta = [r.text for r in reqs if r.kind == "metadata"]
    for field in (
        "Наименование задачи",
        "Продукт",
        "Бизнес-процесс",
        "Тип работы",
        "Влияние на ИС",
        "Задача в JIRA",
    ):
        assert any(field in t for t in meta), field
    assert not any(r.kind == "requirement" and "Общие сведения" in r.section for r in reqs)
    # прочие служебные разделы остаются пропущенными
    assert not any(
        re.search(r"ревью требований|согласующ|согласован|история изменений", " ".join(r.section), re.I)
        for r in reqs
    )
