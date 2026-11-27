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


# --- итог прогона: ready / not_ready / undecided -----------------------------------------------


def _mini(tmp_path):
    p = tmp_path / "mini.md"
    p.write_text(MINI_MD, encoding="utf-8")
    return p


def _facts_ok():
    from .test_agents import _facts

    return _facts(
        ("frequency_limits", "порог", "Пользователь задаёт порог от 1 000 до 10 000 тенге включительно")
    )


def test_all_batches_ok_without_defects_is_ready(tmp_path):
    from .test_agents import FakeLLM

    report = orchestrator.run(_mini(tmp_path), FakeLLM({"LLMFacts": [_facts_ok()]}))
    assert report.verdict.status == "ready" and report.verdict.ready
    assert all(r.checked for r in report.requirements)
    assert orchestrator.exit_code(report) == 0
    assert "Покрытие: проверено 3 из 3 требований." in render_markdown(report)


def test_all_batches_ok_with_critical_is_not_ready(tmp_path):
    from .test_agents import FakeLLM, _finding, _findings

    crit = _finding("R-001", "Пользователь задаёт порог от 1 000 до 10 000 тенге включительно")
    llm = FakeLLM({"LLMFacts": [_facts_ok()], "LLMFindings": [_findings(crit)]})
    report = orchestrator.run(_mini(tmp_path), llm)
    assert report.verdict.status == "not_ready" and not report.verdict.ready
    assert "можно начинать" not in render_markdown(report)
    assert orchestrator.exit_code(report) == 0  # проверка полная, дефекты — это результат


def test_second_batch_fails_gives_partial_coverage(sample):
    from ravel.llm.base import LLMUnavailable

    from .test_agents import FakeLLM

    class DiesOnSecond(FakeLLM):
        def _raw(self, system, user, schema):
            if schema.__name__ == "LLMFindings" and len(self.users("LLMFindings")) >= 1:
                raise LLMUnavailable("503")
            return super()._raw(system, user, schema)

    report = orchestrator.run(sample, DiesOnSecond({}))
    total = len(report.requirements)
    done = sum(r.checked for r in report.requirements)
    assert 0 < done < total
    assert report.verdict.status == "undecided"
    assert orchestrator.exit_code(report) == 2
    md = render_markdown(report)
    assert f"проверено {done} из {total} требований" in md
    assert f"Покрытие: проверено {done} из {total} требований." in md
    assert md.count("**Не проверено (LLM недоступен или ответ отклонён).**") == total - done
    assert "можно начинать" not in md
    assert "Уже найдено критичных:" in md  # линтеры нашли критичное, оно не теряется


def test_single_heading_document_is_undecided(tmp_path):
    from .test_agents import FakeLLM

    src = tmp_path / "h.md"
    src.write_text("# Только заголовок\n", encoding="utf-8")
    report = orchestrator.run(src, FakeLLM({}))
    assert report.requirements == []
    assert report.verdict.status == "undecided"
    assert orchestrator.exit_code(report) == 2
    assert "можно начинать" not in render_markdown(report)


def test_partial_read_warning_makes_undecided(tmp_path):
    from ravel.models import IngestIssue, IssueLevel

    from .test_agents import FakeLLM

    report = orchestrator.run(_mini(tmp_path), FakeLLM({"LLMFacts": [_facts_ok()]}))
    assert report.verdict.status == "ready"
    for code, expected in [("low_text_quality", "undecided"), ("non_utf8", "ready")]:
        report.issues = [IngestIssue(level=IssueLevel.warning, code=code, message="m")]
        assert orchestrator.verdict(report).status == expected


def test_ready_flag_is_computed_and_dumped():
    from ravel.models import Verdict

    v = Verdict(status="undecided", headline="h", fix_first=[])
    assert v.ready is False and v.model_dump()["ready"] is False
