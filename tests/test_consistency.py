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


# --------------------------------------------------------------------------- карточки и находки набора


def _synthetic_report(**kw):
    from datetime import datetime

    from ravel.models import Criterion, Finding, Passport, Quote, Report, Requirement, Severity, Verdict

    reqs = [
        Requirement(id=f"R-{i}", block_id=f"B-{i}", text=f"Требование {i}", location="строка 1", checked=True)
        for i in (1, 2, 3)
    ]
    set_f = Finding(
        id="F-001",
        scope="set",
        criterion=Criterion.set_consistent,
        severity=Severity.critical,
        title="Противоречие порогов",
        explanation="Пороги в R-1 и R-2 различаются.",
        quotes=[Quote(text="Требование 1", location="строка 1")],
        requirement_ids=["R-1"],
        source="test",
    )
    own_f = Finding(
        id="F-002",
        scope="requirement",
        criterion=Criterion.unambiguous,
        severity=Severity.minor,
        title="Размыто",
        explanation="e",
        quotes=[Quote(text="Требование 2", location="строка 2")],
        requirement_ids=["R-2"],
        source="test",
    )
    passport = Passport(
        document_title="T",
        source_name="t.md",
        source_sha256="0" * 64,
        started_at=datetime.now().astimezone(),
        provider="fake",
        model="m",
        agent_version="v",
        prompt_version="p",
    )
    return Report(
        passport=passport,
        requirements=reqs,
        findings=[set_f, own_f],
        verdict=Verdict(status="not_ready", headline="h", fix_first=[]),
        **kw,
    )


def _card(md: str, rid: str) -> str:
    return md.split(f"### {rid}\n")[1].split("### ")[0].split("## 4.")[0]


def test_set_finding_appears_in_requirement_card():
    md = render_markdown(_synthetic_report())
    card = _card(md, "R-1")
    assert "Нарушений не найдено" not in card
    assert "Затрагивающие находки уровня набора" in card
    assert "F-001" in card and "Критично" in card and "Противоречие порогов" in card
    assert "раздел 4" in card
    assert "Нарушений не найдено" not in _card(md, "R-2")  # своя находка
    assert "Нарушений не найдено." in _card(md, "R-3")  # проверено, находок нет


def test_summary_counts_requirements_touched_by_set_findings():
    report = _synthetic_report()
    md = render_markdown(report)
    # вручную: R-1 (набор) и R-2 (своя) затронуты, R-3 нет
    touched = {rid for f in json.loads(report.model_dump_json())["findings"] for rid in f["requirement_ids"]}
    assert touched == {"R-1", "R-2"}
    assert "| Требований с нарушениями | 2 |" in md


def test_unchecked_texts_kept_next_to_set_findings():
    report = _synthetic_report()
    report.requirements[0].checked = False
    md = render_markdown(report)
    card = _card(md, "R-1")
    assert "Не проверено (LLM недоступен или ответ отклонён)." in card
    assert "Затрагивающие находки уровня набора" in card


def test_run_id_rendered_by_template():
    md = render_markdown(_synthetic_report(), "abc-123")
    assert "| Версия промптов | p |\n| ID прогона | `abc-123` |\n" in md
    assert "ID прогона" not in render_markdown(_synthetic_report())


def test_checklist_texts_do_not_claim_review_not_done(sample):
    report = orchestrator.run(sample, llm=None)
    md = render_markdown(report)
    assert "не проведено" not in md
    checklist = [f for f in report.findings if f.source == "lint:checklist_unfilled"]
    assert (
        checklist and "Отметки в чек-листе не проставлены ни в одной из 14 строк" in checklist[0].explanation
    )
    # пустой раздел ревью и пустой чек-лист в нём — одна находка, а не две
    assert not [
        f for f in report.findings if f.source == "lint:placeholder" and "Ревью требований" in f.title
    ]
    assert len(checklist[0].quotes) == 2


FORBIDDEN_PHRASES = [
    "догадки",
    "опираться на догад",
    "нельзя ни разрабатывать, ни тестировать",
    "ревью требований не проведено",
    "не проведено",
    "противоречия, непроверяемые или неполные требования",
]


def test_report_linter_on_sample(sample, tmp_path):
    """Каждая находка в MD имеет цитату со строкой; общих фраз без привязки к тексту нет."""
    import re

    from typer.testing import CliRunner

    from ravel.cli import app

    out = tmp_path / "r.md"
    CliRunner().invoke(app, ["check", str(sample), "-o", str(out), "--provider", "none"])
    md = out.read_text(encoding="utf-8")
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in md, phrase
    findings = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["findings"]
    assert findings
    # блок находки: от «F-NNN ·» до следующей находки/заголовка
    parts = re.split(r"(?m)^(?:### |- \*\*)(?=F-\d{3} ·)", md)[1:]
    by_id = {p[:5]: p for p in parts}
    for f in findings:
        block = by_id[f["id"]]
        assert re.search(r"Цитата: «.+» \(.*строка \d+\)", block), f["id"]
