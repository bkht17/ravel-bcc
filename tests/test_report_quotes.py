"""Отчёт: краткая формулировка в таблице нарезки, полный текст в самом Markdown, экранирование, счётчики."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from ravel import orchestrator
from ravel.models import (
    SET_CRITERIA,
    Criterion,
    Finding,
    Passport,
    Quote,
    Report,
    Requirement,
    Severity,
    Verdict,
)
from ravel.render import CELL_LIMIT, render_markdown

LONG = "Сервис " + " ".join(f"проверяет условие {i} и пишет запись в журнал" for i in range(1, 12)) + "."
PIPE = "Формат строки: код | сумма | валюта; разделитель — «|»."
MULTILINE = "Первая строка требования\nвторая строка с <cardId> и тегом <b>жирно</b>\n\nпосле пустой строки"
QUOTE_400 = ("Цитата " + "очень длинная " * 40).strip()


def _passport() -> Passport:
    return Passport(
        document_title="Т",
        source_name="t.md",
        source_sha256="0" * 64,
        started_at=datetime(2026, 1, 1).astimezone(),
        provider="none",
        model="—",
        agent_version="test",
        prompt_version="test",
    )


def _req(rid: str, text: str, kind: str = "requirement", **kw) -> Requirement:
    return Requirement(
        id=rid, block_id=f"B-{rid}", text=text, location="Раздел › Подраздел, строка 1", kind=kind, **kw
    )


def _report() -> Report:
    parent = "П-1. Первое утверждение пункта. Второе утверждение пункта."
    reqs = [
        _req("R-001", LONG),
        _req("R-002", PIPE, source_id="П|2"),
        _req("R-003", MULTILINE),
        _req("R-004", "П-1. Первое утверждение пункта.", source_id="П-1", parent_id="R-004",
             parent_text=parent, part=1, parts=2, offset=0),
        _req("R-005", "Второе утверждение пункта.", source_id="П-1", parent_id="R-004",
             parent_text=parent, part=2, parts=2, offset=32),
        _req("M-001", "Автор: Отдел", kind="metadata"),
        _req("E-001", "curl -X POST " + "-d 'x=1' " * 30, kind="example"),
    ]  # fmt: skip
    findings = [
        Finding(id="F-001", scope="requirement", requirement_ids=["R-001"], criterion=Criterion.verifiable,
                severity=Severity.major, title="t1", explanation="e", quotes=[Quote(text=QUOTE_400, location="Раздел")],
                source="lint:x", verified=True),
        Finding(id="F-002", scope="requirement", requirement_ids=["R-001", "M-001"], criterion=Criterion.unambiguous,
                severity=Severity.minor, title="t2", explanation="e", quotes=[Quote(text=PIPE, location="Раздел")],
                source="lint:y", verified=True),
        Finding(id="F-003", scope="set", requirement_ids=["R-001", "E-001"], criterion=Criterion.set_consistent,
                severity=Severity.minor, title="t3", explanation="e", quotes=[Quote(text=MULTILINE, location="Раздел")],
                source="lint:z", verified=True),
    ]  # fmt: skip
    r = Report(passport=_passport(), requirements=reqs, findings=findings)
    r.verdict = Verdict(status="undecided", headline="h", fix_first=[])
    return r


def _cells(line: str) -> list[str]:
    """Ячейки строки GFM-таблицы: делит только по неэкранированным «|»."""
    return re.split(r"(?<!\\)\|", line.strip().strip("|"))


def _unquote(md: str, start: str) -> str:
    """Текст цитаты markdown (блок «> …»), идущей сразу после строки start."""
    lines = md.split("\n")
    i = lines.index(start) + 1
    while not lines[i].startswith(">"):
        i += 1
    out = []
    while i < len(lines) and lines[i].startswith(">"):
        out.append(lines[i][2:].rstrip() if len(lines[i]) > 1 else "")
        i += 1
    return "\n".join(out).replace("\\<", "<")


@pytest.fixture(scope="module")
def md() -> str:
    return render_markdown(_report())


def test_slicing_table_rows_stay_intact(md):
    rows = [ln for ln in md.split("\n") if ln.startswith("| R-0")]
    assert len(rows) == 5
    assert {len(_cells(r)) for r in rows} == {4}  # «|» и переносы строк не ломают таблицу
    long_row = next(r for r in rows if r.startswith("| R-001 "))
    assert _cells(long_row)[3].strip().endswith("…") and len(_cells(long_row)[3].strip()) <= CELL_LIMIT
    assert "П\\|2" in next(r for r in rows if r.startswith("| R-002 "))
    assert "\\<cardId>" in next(r for r in rows if r.startswith("| R-003 "))  # «тег» не прячется


def test_full_text_is_in_requirement_card(md):
    assert _unquote(md, "### R-001") == LONG
    assert _unquote(md, "### R-002 (П|2)") == PIPE
    assert _unquote(md, "### R-003") == MULTILINE  # переносы и пустая строка сохранены
    assert "Место: Раздел › Подраздел, строка 1." in md


def test_fragment_card_shows_parent_once(md):
    card4 = md.split("### R-004 (П-1)")[1].split("### R-005")[0]
    assert "Фрагмент 1 из 2 пункта П-1. Исходный текст целиком:" in card4
    assert "> П-1. Первое утверждение пункта. Второе утверждение пункта." in card4
    card5 = md.split("### R-005 (П-1)")[1].split("## 4.")[0]
    assert "Фрагмент 2 из 2 пункта П-1; исходный текст целиком — в карточке R-004." in card5
    assert "| R-005 | П-1 · ч. 2/2 |" in md


def test_finding_quotes_are_full(md):
    assert f"Цитата: «{QUOTE_400}»" in md
    assert f"Цитата: «{PIPE}»" in md  # вне таблицы «|» не экранируется
    assert (
        "Цитата: «Первая строка требования вторая строка с \\<cardId> и тегом \\<b>жирно\\</b> после пустой строки»"
        in md
    )


def test_long_example_full_text_below_table(md):
    full = _report().requirements[-1].text
    assert "Полный текст сокращённых строк:" in md
    assert _unquote(md, "**E-001** (Раздел › Подраздел, строка 1):") == full.strip()


def test_requirements_with_defects_counts_unique_product_ids(md):
    assert "| Требований с нарушениями | 1 |" in md  # R-001 трижды, M-001 и E-001 не требования


# --------------------------------------------------------------------------- уровни критериев


@pytest.mark.parametrize("name", ["samples/requirements.md", "tests/fixtures/sample_typeset_tables.pdf"])
def test_set_findings_have_set_level_criteria(name):
    root = Path(__file__).resolve().parents[1]
    report = orchestrator.run(root / name, llm=None)
    set_findings = [f for f in report.findings if f.scope == "set"]
    assert set_findings
    assert all(f.criterion in SET_CRITERIA for f in set_findings), [
        (f.title, f.criterion) for f in set_findings if f.criterion not in SET_CRITERIA
    ]
    created = next(f for f in report.findings if f.source == "lint:status_200_created")
    assert created.scope == "set" and created.criterion == Criterion.set_consistent
    md = render_markdown(report)
    assert f"### {created.id} · Существенно · Непротиворечивость набора (29148 5.2.6)" in md
    product = {r.id for r in report.requirements if r.kind == "requirement"}
    touched = {rid for f in report.findings for rid in f.requirement_ids}
    assert touched <= product
    assert f"| Требований с нарушениями | {len(touched)} |" in md
