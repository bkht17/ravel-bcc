"""Базовый класс агента и общий контекст прогона.

Агент = роль: свой промпт, своя схема ответа, свой шаг в оркестраторе.
Агенты не вызывают друг друга напрямую — только через RunContext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ravel.llm import LLMClient
from ravel.models import DocumentIR, Fact, Finding, Question, Requirement


@dataclass
class RunContext:
    doc: DocumentIR
    requirements: list[Requirement] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)  # таблица фактов SetChecker
    findings: list[Finding] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    answers: list[Question] = field(default_factory=list)  # вопросы прошлого прогона с ответами
    dropped_unverified: int = 0
    llm_failures: list[str] = field(default_factory=list)

    def req(self, rid: str) -> Requirement | None:
        return next((r for r in self.requirements if r.id == rid), None)

    def answer_for(self, q: Question) -> str | None:
        from ravel.quotes import norm

        for a in self.answers:
            if a.answer and (a.id == q.id or norm(a.question) == norm(q.question)):
                return a.answer
        return None

    def answers_block(self) -> str:
        """Ответы автора на прошлые вопросы — контекст для LLM-агентов."""
        lines = [f"- {a.question}\n  Ответ автора: {a.answer}" for a in self.answers if a.answer]
        if not lines:
            return ""
        return "Уточнения, полученные от автора требований (считай их частью документа):\n" + "\n".join(lines)


class Agent(ABC):
    name: str = "agent"
    uses_llm: bool = False

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    @property
    def enabled(self) -> bool:
        return not self.uses_llm or self.llm is not None

    @abstractmethod
    def run(self, ctx: RunContext) -> int:
        """Выполнить шаг, изменить ctx. Вернуть число созданных объектов (для трассы)."""


def format_requirements(reqs: list[Requirement]) -> str:
    return "\n".join(
        f"[{r.id}] ({r.source_id or 'без номера'}; {' › '.join(r.section[-2:])})\n{r.text}" for r in reqs
    )
