"""Агенты с поддельной LLM: проверка цитат, кэш, починка ответа, вопросы."""

from __future__ import annotations

import json

import pytest

from ravel import orchestrator
from ravel.ingest import load_document
from ravel.llm.base import LLMClient, LLMUnavailable
from ravel.models import Question
from ravel.quotes import locate


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, replies: dict[str, list[str]], cache_dir=None):
        super().__init__("fake-1", cache_dir)
        self.replies = replies  # имя схемы → очередь ответов
        self.prompts: list[str] = []

    def _raw(self, system, user, schema):
        self.prompts.append(user)
        queue = self.replies.get(schema.__name__, [])
        return (
            queue.pop(0)
            if queue
            else json.dumps({schema.__name__ == "LLMQuestions" and "questions" or "findings": []})
        )


def _finding(rid, quote, criterion="verifiable", severity="critical"):
    return {
        "requirement_ids": [rid],
        "criterion": criterion,
        "severity": severity,
        "title": "t",
        "explanation": "e",
        "suggestion": "s",
        "quotes": [quote],
    }


def test_quote_locate(sample):
    doc = load_document(sample)
    assert locate("Переводы и снятие наличных исключаются", doc)
    assert locate("переводы  и снятие наличных исключаются.", doc)  # регистр и пробелы
    assert locate("**REST type** | GET", doc)  # разметка игнорируется
    assert locate("Клиент получает кэшбэк 5 %", doc) is None


def test_hallucinated_quotes_are_dropped(sample):
    good = _finding("R-006", "Месяц определяется по дате совершения покупки в Алматы")
    bad = _finding("R-006", "Порог хранится в долларах США")
    llm = FakeLLM({"LLMFindings": [json.dumps({"findings": [good, bad]})]})
    report = orchestrator.run(sample, llm)
    llm_findings = [f for f in report.findings if f.source == "llm:checker"]
    assert len(llm_findings) == 1
    assert report.dropped_unverified == 1


def test_cache_makes_second_run_free(sample, tmp_path):
    reply = json.dumps({"findings": [_finding("R-006", "Переводы и снятие наличных исключаются")]})
    llm = FakeLLM({"LLMFindings": [reply] * 50}, cache_dir=tmp_path)
    orchestrator.run(sample, llm)
    calls_first = llm.calls
    llm2 = FakeLLM({"LLMFindings": []}, cache_dir=tmp_path)
    orchestrator.run(sample, llm2)
    assert calls_first > 0
    assert llm2.calls == 0 and llm2.cache_hits == calls_first


def test_invalid_json_is_repaired_once(sample):
    ok = json.dumps({"findings": [_finding("R-006", "Переводы и снятие наличных исключаются")]})
    llm = FakeLLM({"LLMFindings": ["{not json", ok]})
    report = orchestrator.run(sample, llm)
    assert any(f.source == "llm:checker" for f in report.findings)
    assert "не прошёл проверку схемы" in llm.prompts[1]


def test_provider_down_falls_back_to_rules(sample):
    class Down(FakeLLM):
        def _raw(self, system, user, schema):
            raise LLMUnavailable("сервер не отвечает")

    report = orchestrator.run(sample, Down({}))
    assert report.findings  # линтеры отработали
    assert report.questions  # вопросы по шаблону
    assert report.llm_failures
    assert orchestrator.exit_code(report) == 2


def test_answers_reach_llm_prompts(sample):
    llm = FakeLLM({})
    answers = [
        Question(id="Q-01", question="По какой дате считать месяц?", why="", answer="По дате проведения")
    ]
    orchestrator.run(sample, llm, answers)
    assert any("По дате проведения" in p for p in llm.prompts)


@pytest.mark.parametrize("severity", ["critical", "major"])
def test_clarifier_template_without_llm(sample, severity):
    report = orchestrator.run(sample, llm=None)
    assert report.questions
    assert all(q.id.startswith("Q-") and q.why for q in report.questions)
