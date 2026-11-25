"""Critic: проверяет каждую находку по тексту документа, убирает дубли, сортирует.

Сейчас детерминированный. TODO(B): LLM-режим с инструментами search_document/get_section,
который перепроверяет спорные находки и калибрует серьёзность.
"""

from __future__ import annotations

from ravel.agents.base import Agent, RunContext
from ravel.models import SEVERITY_ORDER, Finding
from ravel.quotes import locate, norm


class CriticAgent(Agent):
    name = "critic"

    def run(self, ctx: RunContext) -> int:
        by_block = {r.block_id: r.id for r in ctx.requirements}
        kept: list[Finding] = []
        for f in ctx.findings:
            ok = []
            for q in f.quotes:
                block = locate(q.text, ctx.doc)
                if block:
                    q.block_id, q.location = block.id, block.location
                    ok.append(q)
            if not ok:
                ctx.dropped_unverified += 1
                continue
            f.quotes = ok
            f.verified = True
            # привязать к требованиям по месту цитаты, если агент не указал
            for q in ok:
                rid = by_block.get(q.block_id or "")
                if rid and rid not in f.requirement_ids:
                    f.requirement_ids.append(rid)
            if f.scope == "requirement" and not f.requirement_ids:
                f.scope = "set"
            kept.append(f)

        kept = _dedupe(kept)
        kept.sort(
            key=lambda f: (
                SEVERITY_ORDER[f.severity],
                min((r for r in f.requirement_ids), default="R-999"),
                f.criterion.value,
                norm(f.quotes[0].text),
            )
        )
        for i, f in enumerate(kept, start=1):
            f.id = f"F-{i:03d}"
        ctx.findings = kept
        return len(kept)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Один дефект — одна находка. Правило линтера важнее догадки LLM."""
    result: dict[tuple[str, str], Finding] = {}
    for f in sorted(findings, key=lambda f: (not f.source.startswith("lint:"), SEVERITY_ORDER[f.severity])):
        key = (f.criterion.value, norm(f.quotes[0].text)[:80])
        if key in result:
            other = result[key]
            other.requirement_ids = sorted(set(other.requirement_ids) | set(f.requirement_ids))
            continue
        result[key] = f
    return list(result.values())
