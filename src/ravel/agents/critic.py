"""Critic: проверяет каждую находку по тексту документа, убирает дубли, сортирует.

Правила для находок LLM строже, чем для правил линтера, потому что модель может ошибиться:
- каждая цитата должна найтись в документе целиком; одна выдуманная цитата — находка снята;
- находка по отдельному требованию держится только на тех ID, в тексте которых стоит цитата;
- противоречие (set_consistent) требует двух подтверждённых цитат из разных мест.

TODO(B): LLM-режим с инструментами search_document/get_section, который перепроверяет
спорные находки и калибрует серьёзность.
"""

from __future__ import annotations

from ravel.agents.base import Agent, RunContext
from ravel.models import REQUIREMENT_CRITERIA, SEVERITY_ORDER, Criterion, Finding
from ravel.quotes import locate, norm


class CriticAgent(Agent):
    name = "critic"

    def run(self, ctx: RunContext) -> int:
        block_of = {r.id: r.block_id for r in ctx.requirements}
        rid_of = {r.block_id: r.id for r in ctx.requirements}
        kept: list[Finding] = []
        for f in ctx.findings:
            ok = (
                _verify_llm(f, ctx, block_of, rid_of)
                if f.source.startswith("llm:")
                else _verify_rule(f, ctx, rid_of)
            )
            if not ok:
                ctx.dropped_unverified += 1
                continue
            f.verified = True
            # находка с критерием отдельного требования и известными ID — это находка по
            # требованиям: она должна появиться в их карточках, а не только в разделе набора
            if f.criterion in REQUIREMENT_CRITERIA and f.requirement_ids:
                f.scope = "requirement"
            if f.scope == "requirement" and not f.requirement_ids:
                f.scope = "set"
            kept.append(f)

        kept = _dedupe(kept)
        kept.sort(
            key=lambda f: (
                SEVERITY_ORDER[f.severity],
                min(f.requirement_ids, default="R-999"),
                f.criterion.value,
                norm(f.quotes[0].text),
            )
        )
        for i, f in enumerate(kept, start=1):
            f.id = f"F-{i:03d}"
        ctx.findings = kept
        return len(kept)


def _verify_rule(f: Finding, ctx: RunContext, rid_of: dict[str, str]) -> bool:
    """Находки линтера: цитаты взяты из блоков кодом, проверяем и привязываем к требованиям."""
    ok = []
    for q in f.quotes:
        if block := locate(q.text, ctx.doc):
            q.block_id, q.location = block.id, block.location
            ok.append(q)
    if not ok:
        return False
    f.quotes = ok
    for q in ok:
        rid = rid_of.get(q.block_id or "")
        if rid and rid not in f.requirement_ids:
            f.requirement_ids.append(rid)
    return True


def _verify_llm(f: Finding, ctx: RunContext, block_of: dict[str, str], rid_of: dict[str, str]) -> bool:
    claimed = [r for r in f.requirement_ids if r in block_of]
    prefer = [block_of[r] for r in claimed]
    for q in f.quotes:
        block = locate(q.text, ctx.doc, prefer=prefer)
        if block is None:
            return False  # выдуманная или искажённая цитата — не доверяем всей находке
        q.block_id, q.location = block.id, block.location
    if not f.quotes:
        return False
    quoted = [rid_of[q.block_id] for q in f.quotes if q.block_id in rid_of]

    if f.scope == "requirement":
        supported = [r for r in claimed if r in quoted] if claimed else list(dict.fromkeys(quoted))
        if not supported:
            return False
        f.requirement_ids = supported
        return True

    if f.criterion == Criterion.set_consistent and len({q.block_id for q in f.quotes}) < 2:
        return False  # противоречию нужны обе стороны
    f.requirement_ids = sorted(set(claimed) | set(quoted))
    return True


def _same_defect(a: Finding, b: Finding) -> bool:
    """Тот же критерий, общие требования (или обе без требований) и пересекающиеся цитаты.

    Разные дефекты одного требования по одному критерию (например, «не описан возврат» и
    «не описана валюта» в Ф-2) цитируют разные фрагменты и не сливаются.
    """
    if a.criterion != b.criterion:
        return False
    ra, rb = set(a.requirement_ids), set(b.requirement_ids)
    if (ra or rb) and not ra & rb:
        return False
    qa = [norm(q.text) for q in a.quotes]
    qb = [norm(q.text) for q in b.quotes]
    return any(x in y or y in x for x in qa for y in qb)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Один дефект — одна находка. Правило линтера важнее догадки LLM."""
    result: list[Finding] = []
    order = sorted(findings, key=lambda f: (not f.source.startswith("lint:"), SEVERITY_ORDER[f.severity]))
    for f in order:
        twin = next((other for other in result if _same_defect(f, other)), None)
        if twin:
            twin.requirement_ids = sorted(set(twin.requirement_ids) | set(f.requirement_ids))
            continue
        result.append(f)
    return result
