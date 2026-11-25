"""Повреждённые и необычные файлы: прогон никогда не падает с исключением."""

from __future__ import annotations

import random

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
