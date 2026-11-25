"""Один документ в разных форматах и на разных прогонах даёт одинаковый результат."""

from __future__ import annotations

import json

from ravel import orchestrator
from ravel.ingest import load_document
from ravel.render import render_markdown, write_outputs
from ravel.segment import segment

from .conftest import MINI_MD, make_docx


def _shape(path):
    return [(r.source_id, r.text, r.section) for r in segment(load_document(path))]


def test_markdown_and_docx_segment_identically(tmp_path):
    md = tmp_path / "spec.md"
    md.write_text(MINI_MD, encoding="utf-8")
    dx = tmp_path / "spec.docx"
    dx.write_bytes(make_docx(MINI_MD))
    assert _shape(md) == _shape(dx)


def test_crlf_bom_and_nbsp_do_not_change_segmentation(tmp_path):
    a = tmp_path / "a.md"
    a.write_text(MINI_MD, encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_bytes(b"\xef\xbb\xbf" + MINI_MD.replace("\n", "\r\n").replace("1 000", "1 000").encode())
    assert _shape(a) == _shape(b)


def test_two_runs_give_identical_report(sample, tmp_path):
    r1 = orchestrator.run(sample, llm=None)
    r2 = orchestrator.run(sample, llm=None)
    d1 = json.loads(r1.model_dump_json(exclude={"passport": {"started_at"}, "trace": True}))
    d2 = json.loads(r2.model_dump_json(exclude={"passport": {"started_at"}, "trace": True}))
    assert d1 == d2
    r2.passport.started_at = r1.passport.started_at
    assert render_markdown(r1) == render_markdown(r2)


def test_report_has_all_required_sections(sample, tmp_path):
    report = orchestrator.run(sample, llm=None)
    paths = write_outputs(report, tmp_path / "out" / "report.md")
    md = paths["report"].read_text(encoding="utf-8")
    for heading in [
        "Паспорт прогона",
        "нарезал требования",
        "Проверка каждого требования",
        "Проверка всего набора",
        "Уточняющие вопросы",
        "Сводка",
        "Итог",
    ]:
        assert heading in md
    assert all(p.exists() for p in paths.values())
    assert not list((tmp_path / "out").glob(".*.tmp"))
    # счётчики в сводке совпадают с данными
    assert f"| Нарушений всего | {len(report.findings)} |" in md


def test_every_finding_has_verified_quote(sample):
    report = orchestrator.run(sample, llm=None)
    assert report.findings
    for f in report.findings:
        assert f.verified and f.quotes and all(q.block_id for q in f.quotes)
