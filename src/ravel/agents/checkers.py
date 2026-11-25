"""LLM-агенты проверки: по одному требованию (Checker) и набора целиком (SetChecker)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ravel.agents.base import Agent, RunContext, format_requirements
from ravel.llm import LLMError, LLMUnavailable
from ravel.models import Criterion, Finding, Quote, Severity
from ravel.prompts import load

MAX_DOC_CHARS = 60_000


class LLMFinding(BaseModel):
    """То, что возвращает модель. Внутренний Finding собирается из этого кодом."""

    requirement_ids: list[str] = Field(description="Идентификаторы требований R-xxx из списка")
    criterion: Criterion
    severity: Severity
    title: str = Field(description="Суть дефекта, до 12 слов")
    explanation: str = Field(description="Почему это дефект, со ссылкой на текст, 1–3 предложения")
    suggestion: str = Field(description="Как исправить формулировку, 1 предложение")
    quotes: list[str] = Field(description="Дословные цитаты из документа")


class LLMFindings(BaseModel):
    findings: list[LLMFinding]


def _to_finding(f: LLMFinding, scope: str, source: str, valid_ids: set[str]) -> Finding:
    return Finding(
        scope=scope,
        requirement_ids=[r for r in f.requirement_ids if r in valid_ids],
        criterion=f.criterion,
        severity=f.severity,
        title=f.title.strip(),
        explanation=f.explanation.strip(),
        suggestion=f.suggestion.strip(),
        quotes=[Quote(text=q) for q in f.quotes if q.strip()],
        source=source,
    )


class CheckerAgent(Agent):
    """Каждое требование отдельно, пачками по batch_size (экономит квоту бесплатного тарифа)."""

    name = "checker"
    uses_llm = True

    def __init__(self, llm, batch_size: int = 6):
        super().__init__(llm)
        self.batch_size = batch_size

    def run(self, ctx: RunContext) -> int:
        system = load("common") + "\n\n" + load("checker")
        valid = {r.id for r in ctx.requirements}
        produced = 0
        for i in range(0, len(ctx.requirements), self.batch_size):
            batch = ctx.requirements[i : i + self.batch_size]
            user = (
                f"Документ: «{ctx.doc.title}».\n\n"
                + (ctx.answers_block() + "\n\n" if ctx.answers else "")
                + "Требования для проверки:\n\n"
                + format_requirements(batch)
            )
            try:
                out = self.llm.complete(system, user, LLMFindings)
            except LLMUnavailable:
                raise
            except LLMError as e:
                ctx.llm_failures.append(f"{self.name}: {batch[0].id}–{batch[-1].id} не проверены ({e})")
                continue
            for f in out.findings:
                finding = _to_finding(f, "requirement", "llm:checker", valid)
                if finding.quotes:
                    ctx.findings.append(finding)
                    produced += 1
        return produced


class SetCheckerAgent(Agent):
    """Набор целиком: противоречия между разделами, пропуски, трассируемость.

    TODO(B): сначала извлекать таблицу фактов (порог, месяц, часовой пояс, единицы, коды
    ответов), потом сравнивать группы — точнее и дешевле, чем весь документ в одном запросе.
    """

    name = "set_checker"
    uses_llm = True

    def run(self, ctx: RunContext) -> int:
        system = load("common") + "\n\n" + load("set_checker")
        doc_text = ctx.doc.markdown[:MAX_DOC_CHARS]
        user = (
            f"Полный текст документа «{ctx.doc.title}»:\n<<<\n{doc_text}\n>>>\n\n"
            + (ctx.answers_block() + "\n\n" if ctx.answers else "")
            + "Требования, выделенные из документа:\n\n"
            + format_requirements(ctx.requirements)
        )
        try:
            out = self.llm.complete(system, user, LLMFindings)
        except LLMUnavailable:
            raise
        except LLMError as e:
            ctx.llm_failures.append(f"{self.name}: проверка набора не выполнена ({e})")
            return 0
        valid = {r.id for r in ctx.requirements}
        produced = 0
        for f in out.findings:
            finding = _to_finding(f, "set", "llm:set", valid)
            if finding.quotes:
                ctx.findings.append(finding)
                produced += 1
        return produced
