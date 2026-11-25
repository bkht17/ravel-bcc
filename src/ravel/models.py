"""Контракт между модулями: всё, что ходит по конвейеру, описано здесь.

Меняйте эти модели только вдвоём — от них зависят и чтение файлов, и LLM-часть,
и шаблон отчёта.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- документ


class BlockKind(StrEnum):
    heading = "heading"
    paragraph = "paragraph"
    list_item = "list_item"
    table_row = "table_row"
    code = "code"


class Block(BaseModel):
    """Минимальная единица документа с адресом, на который ссылается отчёт."""

    id: str  # B-0001, по порядку в документе
    kind: BlockKind
    text: str  # текст как в документе (без markdown-разметки списков/заголовков)
    level: int = 0  # уровень заголовка; для остальных 0
    section: list[str] = Field(default_factory=list)  # путь заголовков над блоком
    line: int | None = None  # строка в нормализованном markdown (1-based)
    page: int | None = None  # страница PDF (1-based), если известна
    cells: list[str] | None = None  # для table_row: ячейки
    header: list[str] | None = None  # для table_row: заголовок таблицы

    @property
    def location(self) -> str:
        """«Часть A › Функциональные требования, стр. 2» — без названия документа (H1)."""
        path = self.section[1:] if len(self.section) > 1 else self.section
        where = " › ".join(path) if path else "начало документа"
        if self.page:
            return f"{where}, стр. {self.page}"
        if self.line:
            return f"{where}, строка {self.line}"
        return where


class IssueLevel(StrEnum):
    warning = "warning"  # прочитали частично, прогон продолжается
    fatal = "fatal"  # прочитать нельзя, отчёт содержит только причину


class IngestIssue(BaseModel):
    level: IssueLevel
    code: str  # empty_file, encrypted_pdf, ext_mismatch, low_text_quality, ...
    message: str  # по-русски, попадает в отчёт
    location: str | None = None


class DocumentIR(BaseModel):
    title: str
    source_name: str
    source_format: Literal["markdown", "text", "pdf", "docx"]
    sha256: str
    markdown: str  # нормализованный текст, по нему проверяются цитаты
    blocks: list[Block]
    issues: list[IngestIssue] = Field(default_factory=list)

    def block(self, block_id: str) -> Block | None:
        return next((b for b in self.blocks if b.id == block_id), None)


# --------------------------------------------------------------------------- требования


class Requirement(BaseModel):
    id: str  # R-001 — наш идентификатор в отчёте
    source_id: str | None = None  # Ф-1, AC-2 и т.п. — как в документе (или None)
    block_id: str
    text: str  # дословная цитата
    section: list[str] = Field(default_factory=list)
    location: str

    @property
    def short(self) -> str:
        return self.text if len(self.text) <= 120 else self.text[:117] + "…"


# --------------------------------------------------------------------------- находки


class Criterion(StrEnum):
    # ISO/IEC/IEEE 29148:2018, 5.2.5 — отдельное требование
    necessary = "necessary"
    appropriate = "appropriate"
    unambiguous = "unambiguous"
    complete = "complete"
    singular = "singular"
    feasible = "feasible"
    verifiable = "verifiable"
    correct = "correct"
    conforming = "conforming"
    # 29148, 5.2.6 — набор требований
    set_complete = "set_complete"
    set_consistent = "set_consistent"
    set_feasible = "set_feasible"
    set_comprehensible = "set_comprehensible"
    set_validatable = "set_validatable"
    # IEEE 830-1998, 4.3
    ranked = "ranked"
    modifiable = "modifiable"
    traceable = "traceable"
    # Прочее, что важно банку
    security = "security"


CRITERION_LABELS: dict[Criterion, str] = {
    Criterion.necessary: "Необходимость (29148 5.2.5)",
    Criterion.appropriate: "Уместность уровня (29148 5.2.5)",
    Criterion.unambiguous: "Однозначность (29148 5.2.5)",
    Criterion.complete: "Полнота (29148 5.2.5)",
    Criterion.singular: "Единичность (29148 5.2.5)",
    Criterion.feasible: "Выполнимость (29148 5.2.5)",
    Criterion.verifiable: "Проверяемость (29148 5.2.5)",
    Criterion.correct: "Корректность (29148 5.2.5)",
    Criterion.conforming: "Соответствие шаблону (29148 5.2.5)",
    Criterion.set_complete: "Полнота набора (29148 5.2.6)",
    Criterion.set_consistent: "Непротиворечивость набора (29148 5.2.6)",
    Criterion.set_feasible: "Выполнимость набора (29148 5.2.6)",
    Criterion.set_comprehensible: "Понятность набора (29148 5.2.6)",
    Criterion.set_validatable: "Валидируемость набора (29148 5.2.6)",
    Criterion.ranked: "Ранжированность (IEEE 830 4.3)",
    Criterion.modifiable: "Модифицируемость (IEEE 830 4.3)",
    Criterion.traceable: "Трассируемость (IEEE 830 4.3)",
    Criterion.security: "Риск ИБ",
}


class Severity(StrEnum):
    critical = "critical"  # непроверяемо / противоречиво / неполно так, что нельзя разрабатывать
    major = "major"
    minor = "minor"


SEVERITY_LABELS = {
    Severity.critical: "Критично",
    Severity.major: "Существенно",
    Severity.minor: "Незначительно",
}
SEVERITY_ORDER = {Severity.critical: 0, Severity.major: 1, Severity.minor: 2}


class Quote(BaseModel):
    text: str
    block_id: str | None = None  # заполняет verifier
    location: str | None = None  # заполняет verifier


class Finding(BaseModel):
    id: str = ""  # F-001, присваивается после сортировки
    scope: Literal["requirement", "set"]
    requirement_ids: list[str] = Field(default_factory=list)  # R-001…
    criterion: Criterion
    severity: Severity
    title: str
    explanation: str
    suggestion: str = ""
    quotes: list[Quote]
    source: str  # lint:<rule> | llm:checker | llm:set
    verified: bool = False


# --------------------------------------------------------------------------- отчёт


class Passport(BaseModel):
    document_title: str
    source_name: str
    source_sha256: str
    started_at: datetime
    provider: str
    model: str
    agent_version: str
    prompt_version: str


class Question(BaseModel):
    """Уточняющий вопрос к автору требований (агент Clarifier)."""

    id: str = ""  # Q-01
    question: str
    why: str  # что сломается в разработке/тестировании без ответа
    options: list[str] = Field(default_factory=list)  # варианты ответа, если очевидны
    requirement_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)
    priority: Severity = Severity.major
    answer: str | None = None  # заполняется из --answers / --interactive


class TraceStep(BaseModel):
    """Шаг оркестратора: какой агент что сделал. Пишется в trace.jsonl."""

    agent: str
    started_at: datetime
    seconds: float
    calls: int = 0  # обращений к LLM
    cache_hits: int = 0
    produced: int = 0  # сколько объектов вернул
    note: str = ""


class Verdict(BaseModel):
    ready: bool
    headline: str
    fix_first: list[str]


class Report(BaseModel):
    passport: Passport
    issues: list[IngestIssue] = Field(default_factory=list)  # ограничения прогона
    requirements: list[Requirement] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    dropped_unverified: int = 0  # находки, чьи цитаты не нашлись в тексте
    llm_failures: list[str] = Field(default_factory=list)
    verdict: Verdict | None = None
    fatal: bool = False
