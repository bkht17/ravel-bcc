"""Оркестратор: порядок агентов, трасса шагов, итоговый Report.

Ingest → Segmenter → [Linter, Checker, SetChecker] → Critic → Clarifier → Verdict
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

from ravel import __version__
from ravel.agents import (
    Agent,
    CheckerAgent,
    ClarifierAgent,
    CriticAgent,
    LintAgent,
    RunContext,
    SegmenterAgent,
    SetCheckerAgent,
)
from ravel.ingest import load_document
from ravel.llm import LLMClient, LLMUnavailable
from ravel.models import (
    IssueLevel,
    Passport,
    Question,
    Report,
    Severity,
    TraceStep,
    Verdict,
)
from ravel.prompts import PROMPT_VERSION


def run(
    input_path: Path, llm: LLMClient | None, answers: list[Question] | None = None, on_step=None
) -> Report:
    started = datetime.now().astimezone()
    doc = load_document(input_path)
    passport = Passport(
        document_title=doc.title,
        source_name=doc.source_name,
        source_sha256=doc.sha256 or "—",
        started_at=started,
        provider=llm.provider if llm else "none",
        model=llm.model if llm else "— (только проверки без LLM)",
        agent_version=agent_version(),
        prompt_version=PROMPT_VERSION,
    )
    report = Report(passport=passport, issues=doc.issues)
    if any(i.level == IssueLevel.fatal for i in doc.issues):
        report.fatal = True
        report.verdict = Verdict(
            ready=False,
            headline="Документ не удалось прочитать, проверка не выполнена.",
            fix_first=[i.message for i in doc.issues if i.level == IssueLevel.fatal],
        )
        return report

    ctx = RunContext(doc=doc, answers=answers or [])
    client = llm  # llm обнуляется при сбое провайдера, а паспорту нужна итоговая модель
    pipeline: list[Agent] = [
        SegmenterAgent(),
        LintAgent(),
        CheckerAgent(llm),
        SetCheckerAgent(llm),
        CriticAgent(),
        ClarifierAgent(llm),
    ]
    for agent in pipeline:
        if not agent.enabled:
            report.trace.append(
                TraceStep(
                    agent=agent.name,
                    started_at=datetime.now().astimezone(),
                    seconds=0,
                    note="пропущен: LLM отключён",
                )
            )
            continue
        if on_step:
            on_step(agent.name)
        calls0, hits0 = (llm.calls, llm.cache_hits) if llm else (0, 0)
        t0, at = time.perf_counter(), datetime.now().astimezone()
        try:
            produced = agent.run(ctx)
            note = ""
        except LLMUnavailable as e:
            # провайдер лёг посреди прогона — дальше работаем без LLM
            produced, note = 0, f"LLM недоступен: {e}"
            ctx.llm_failures.append(f"{agent.name}: {e}")
            llm = None
            for a in pipeline:
                a.llm = None
            if isinstance(agent, ClarifierAgent):
                produced = agent.run(ctx)
        report.trace.append(
            TraceStep(
                agent=agent.name,
                started_at=at,
                seconds=round(time.perf_counter() - t0, 2),
                calls=(llm.calls - calls0) if llm else 0,
                cache_hits=(llm.cache_hits - hits0) if llm else 0,
                produced=produced,
                note=note,
            )
        )

    if client is not None:
        passport.model = client.models_label  # если основная модель была перегружена — видно в паспорте
    report.requirements = ctx.requirements
    report.findings = ctx.findings
    report.questions = ctx.questions
    report.dropped_unverified = ctx.dropped_unverified
    report.llm_failures = ctx.llm_failures
    report.verdict = verdict(report)
    return report


def verdict(report: Report) -> Verdict:
    crit = [f for f in report.findings if f.severity == Severity.critical]
    major = [f for f in report.findings if f.severity == Severity.major]
    fix_first = [f"{f.id}: {f.title}" for f in (crit + major)[:5]]
    if crit:
        headline = (
            "Начинать разработку и тестирование по документу нельзя: критичных дефектов — "
            f"{len(crit)} (противоречия, непроверяемые или неполные требования)."
        )
        return Verdict(ready=False, headline=headline, fix_first=fix_first)
    if major:
        return Verdict(
            ready=False,
            headline=f"Документ требует доработки: существенных дефектов — {len(major)}.",
            fix_first=fix_first,
        )
    return Verdict(
        ready=True,
        headline="Критичных и существенных дефектов не найдено; можно начинать.",
        fix_first=fix_first,
    )


def exit_code(report: Report) -> int:
    """0 — всё прочитано и проверено; 2 — частично; 1 — документ не прочитан."""
    if report.fatal:
        return 1
    if report.issues or report.llm_failures:
        return 2
    return 0


def agent_version() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty", "--tags"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{__version__} ({out})" if out else __version__
    except Exception:
        return f"{__version__} (git: нет коммита)"
