"""Агенты без LLM: нарезка по правилам и линтеры."""

from __future__ import annotations

from ravel.agents.base import Agent, RunContext
from ravel.lint import run_lint
from ravel.segment import segment


class SegmenterAgent(Agent):
    name = "segmenter"

    def run(self, ctx: RunContext) -> int:
        ctx.requirements = segment(ctx.doc)
        return len(ctx.requirements)


class LintAgent(Agent):
    name = "linter"

    def run(self, ctx: RunContext) -> int:
        found = run_lint(ctx.doc)
        ctx.findings.extend(found)
        return len(found)
