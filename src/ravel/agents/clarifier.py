"""Clarifier: уточняющие вопросы автору требований.

С LLM — объединяет дефекты в решения, которые должен принять автор. Без LLM — по шаблону
из критичных и существенных находок, чтобы раздел вопросов был всегда.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ravel.agents.base import Agent, RunContext
from ravel.llm import LLMError, LLMUnavailable
from ravel.models import SEVERITY_ORDER, Question, Quote, Severity
from ravel.prompts import load
from ravel.quotes import locate

MAX_QUESTIONS = 12


class LLMQuestion(BaseModel):
    question: str = Field(description="Вопрос автору, одно решение")
    why: str = Field(description="Что сломается без ответа")
    options: list[str] = Field(description="2–3 варианта ответа или пустой список")
    finding_ids: list[str] = Field(description="F-xxx, которые закрывает ответ")
    requirement_ids: list[str]
    quotes: list[str] = Field(description="Дословные цитаты из документа")
    priority: Severity


class LLMQuestions(BaseModel):
    questions: list[LLMQuestion]


class ClarifierAgent(Agent):
    name = "clarifier"

    def run(self, ctx: RunContext) -> int:
        important = [f for f in ctx.findings if f.severity in (Severity.critical, Severity.major)]
        if not important:
            ctx.questions = []
            return 0
        questions = self._with_llm(ctx, important) if self.llm else None
        if questions is None:
            questions = self._from_templates(important)

        questions.sort(key=lambda q: SEVERITY_ORDER[q.priority])
        questions = questions[:MAX_QUESTIONS]
        for i, q in enumerate(questions, start=1):
            q.id = f"Q-{i:02d}"
            q.answer = ctx.answer_for(q)
        ctx.questions = questions
        return len(questions)

    def _with_llm(self, ctx: RunContext, important) -> list[Question] | None:
        system = load("common") + "\n\n" + load("clarifier")
        listing = "\n\n".join(
            f"[{f.id}] {f.severity.value} / {f.criterion.value} / {', '.join(f.requirement_ids) or 'набор'}\n"
            f"{f.title}. {f.explanation}\nЦитаты: " + " | ".join(f"«{q.text}»" for q in f.quotes)
            for f in important
        )
        answers = ctx.answers_block()
        user = f"Документ: «{ctx.doc.title}».\n\n" + (answers + "\n\n" if answers else "")
        user += f"Найденные дефекты:\n\n{listing}"
        try:
            out = self.llm.complete(system, user, LLMQuestions, "вопросы")
        except LLMUnavailable:
            raise
        except LLMError as e:
            ctx.llm_failures.append(f"{self.name}: вопросы составлены по шаблону ({e})")
            return None
        valid_f = {f.id for f in ctx.findings}
        product = [r for r in ctx.requirements if r.kind == "requirement"]
        valid_r = {r.id for r in product}
        block_of = {r.id: r.block_id for r in product}
        result = []
        for q in out.questions:
            prefer = [block_of[x] for x in q.requirement_ids if x in block_of]
            quotes = []
            for text in q.quotes:
                if block := locate(text, ctx.doc, prefer=prefer):
                    quotes.append(Quote(text=text, block_id=block.id, location=block.location))
            # вопрос без подтверждённой цитаты или с выдуманной цитатой не задаём: автор должен
            # видеть, к какому месту документа он относится
            if not quotes or len(quotes) < len([t for t in q.quotes if t.strip()]):
                continue
            result.append(
                Question(
                    question=q.question.strip(),
                    why=q.why.strip(),
                    options=[o for o in q.options if o.strip()],
                    finding_ids=[x for x in q.finding_ids if x in valid_f],
                    requirement_ids=[x for x in q.requirement_ids if x in valid_r],
                    quotes=quotes,
                    priority=q.priority,
                )
            )
        if not result:
            if out.questions:
                ctx.llm_failures.append(
                    f"{self.name}: ни один вопрос модели не подтверждён цитатой, вопросы составлены по шаблону"
                )
            return None
        return result

    @staticmethod
    def _from_templates(important) -> list[Question]:
        return [
            Question(
                question=f"{f.title}: как должно быть?"
                if not f.suggestion
                else f"{f.title}. Подтвердите решение: {f.suggestion}",
                why=f.explanation,
                finding_ids=[f.id],
                requirement_ids=f.requirement_ids,
                quotes=f.quotes[:2],
                priority=f.severity,
            )
            for f in important
        ]
