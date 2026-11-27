"""Повреждённые и необычные файлы: прогон никогда не падает с исключением."""

from __future__ import annotations

import random
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
    assert len(segment(load_document(sample))) == SAMPLE_REQS


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
    n = len(segment(doc))
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
