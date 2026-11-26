"""Агенты с поддельной LLM: проверка цитат, кэш, починка ответа, факты, вопросы."""

from __future__ import annotations

import json

import pytest

from ravel import orchestrator
from ravel.agents.facts import chunks
from ravel.ingest import load_document
from ravel.llm.base import LLMClient, LLMUnavailable, TransientError
from ravel.models import Question
from ravel.quotes import locate

F2 = "Месяц определяется по дате совершения покупки в Алматы"
F3 = "На карту приходится максимум один такой push"
F4 = "следующее достижение вызывает push независимо от истории"
EXCL = "Переводы и снятие наличных исключаются"


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, replies: dict[str, list[str]], cache_dir=None):
        super().__init__("fake-1", cache_dir)
        self.replies = replies  # имя схемы → очередь ответов
        self.prompts: list[tuple[str, str]] = []  # (имя схемы, user)

    def _raw(self, system, user, schema):
        self.prompts.append((schema.__name__, user))
        queue = self.replies.get(schema.__name__, [])
        if queue:
            return queue.pop(0)
        return json.dumps({next(iter(schema.model_fields)): []})

    def users(self, schema_name: str) -> list[str]:
        return [u for s, u in self.prompts if s == schema_name]


def _finding(rid, quote, criterion="verifiable", severity="critical"):
    quotes = quote if isinstance(quote, list) else [quote]
    rids = rid if isinstance(rid, list) else [rid]
    return {
        "requirement_ids": rids,
        "criterion": criterion,
        "severity": severity,
        "title": "t",
        "explanation": "e",
        "suggestion": "s",
        "quotes": quotes,
    }


def _findings(*items) -> str:
    return json.dumps({"findings": list(items)})


def _facts(*items) -> str:
    return json.dumps({"facts": [{"topic": t, "statement": s, "quote": q} for t, s, q in items]})


GOOD_FACTS = _facts(
    ("frequency_limits", "Максимум один push в месяц", F3),
    ("trigger_conditions", "После смены порога push независимо от истории", F4),
)


# --------------------------------------------------------------------------- цитаты


def test_quote_locate(sample):
    doc = load_document(sample)
    assert locate(EXCL, doc)
    assert locate("переводы  и снятие наличных исключаются.", doc)  # регистр и пробелы
    assert locate("**REST type** | GET", doc)  # разметка игнорируется
    assert locate("REST type: GET", doc)  # строка таблицы в виде «Заголовок: значение»
    assert locate("Клиент получает кэшбэк 5 %", doc) is None


def test_quote_must_match_entirely(sample):
    """Аудит: изменённый хвост длинной цитаты раньше проходил по первым 40 символам."""
    doc = load_document(sample)
    tail_changed = "Месяц определяется по дате совершения покупки в Астане, а не в Алматы"
    assert locate(tail_changed, doc) is None
    assert locate("Месяц определяется по дате совершения покупки не в Алматы", doc) is None
    assert locate("Месяц определяется по дате проведения покупки в Алматы", doc) is None


def test_quote_tolerates_typo_but_not_numbers(sample):
    doc = load_document(sample)
    assert locate("Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно", doc)
    assert locate("Держатель активной карты KZT задает целые порог от 1 000 до 1 000 000 ₸ включительно", doc)
    assert (
        locate("Держатель активной карты KZT задаёт целый порог от 2 000 до 1 000 000 ₸ включительно", doc)
        is None
    )


def test_quote_keeps_word_boundaries(sample):
    doc = load_document(sample)
    # «...исключаются» есть, «не исключаются» — нет: склейка слов без пробелов дала бы совпадение
    assert locate("наличных не исключаются", doc) is None


# --------------------------------------------------------------------------- Checker / Critic


def test_hallucinated_quotes_are_dropped(sample):
    good = _finding("R-006", F2)
    bad = _finding("R-006", "Порог хранится в долларах США")
    llm = FakeLLM({"LLMFindings": [_findings(good, bad)]})
    report = orchestrator.run(sample, llm)
    llm_findings = [f for f in report.findings if f.source == "llm:checker"]
    assert len(llm_findings) == 1
    assert report.dropped_unverified == 1


def test_one_bad_quote_drops_whole_finding(sample):
    mixed = _finding("R-006", [F2, "Порог хранится в долларах США"])
    llm = FakeLLM({"LLMFindings": [_findings(mixed)]})
    report = orchestrator.run(sample, llm)
    assert not [f for f in report.findings if f.source == "llm:checker"]
    assert report.dropped_unverified == 1


def test_requirement_id_must_match_quote(sample):
    """ID из другой пачки отбрасывается; ID, в тексте которого нет цитаты, не засчитывается."""
    other_batch = _finding("R-040", F2)  # R-040 не в первой пачке
    wrong_place = _finding("R-005", F2)  # в пачке, но цитата из R-006
    right = _finding(["R-005", "R-006"], F2, criterion="unambiguous")
    llm = FakeLLM({"LLMFindings": [_findings(other_batch, wrong_place, right)]})
    report = orchestrator.run(sample, llm)
    llm_findings = [f for f in report.findings if f.source == "llm:checker"]
    assert [f.requirement_ids for f in llm_findings] == [["R-006"]]
    assert report.dropped_unverified == 2  # other_batch и wrong_place


def test_checked_flags_follow_batches(sample):
    llm = FakeLLM({"LLMFindings": ["{bad", "{still bad"]})  # первая пачка падает
    report = orchestrator.run(sample, llm)
    reqs = report.requirements
    assert not any(r.checked for r in reqs[:6])
    assert all(r.checked for r in reqs[6:])
    assert any("R-001–R-006 не проверены" in m for m in report.llm_failures)


def test_requirement_criterion_rejected_for_set(sample):
    """Схема набора не принимает критерий отдельного требования — даже после починки."""
    bad = _finding("R-007", [F3, F4], criterion="verifiable")
    llm = FakeLLM({"LLMFacts": [GOOD_FACTS], "LLMSetFindings": [_findings(bad), _findings(bad)]})
    report = orchestrator.run(sample, llm)
    assert not [f for f in report.findings if f.source == "llm:set"]
    assert any("сравнение фактов не выполнено" in m for m in report.llm_failures)


def test_contradiction_needs_both_sides(sample):
    one_side = _finding(["R-007", "R-008"], [F3], criterion="set_consistent")
    both = _finding(["R-007", "R-008"], [F3, F4], criterion="set_consistent", severity="major")
    llm = FakeLLM({"LLMFacts": [GOOD_FACTS], "LLMSetFindings": [_findings(one_side, both)]})
    report = orchestrator.run(sample, llm)
    kept = [f for f in report.findings if f.source == "llm:set"]
    assert len(kept) == 1 and len(kept[0].quotes) == 2
    assert kept[0].requirement_ids == ["R-007", "R-008"]
    assert report.dropped_unverified == 1


def test_lint_finding_with_requirements_shows_in_cards(sample):
    """Аудит: F-001 (GET) касался R-017/R-018, а их карточки писали «нарушений не найдено»."""
    report = orchestrator.run(sample, llm=None)
    get = next(f for f in report.findings if f.source == "lint:http_get_write")
    assert get.scope == "requirement" and get.requirement_ids


# --------------------------------------------------------------------------- факты


def test_facts_feed_set_checker(sample):
    facts = _facts(
        ("frequency_limits", "Максимум один push в месяц", F3),
        ("trigger_conditions", "Push независимо от истории", F4),
        ("units_currency", "Порог в долларах", "Порог хранится в долларах США"),  # выдуман
    )
    llm = FakeLLM({"LLMFacts": [facts]})
    orchestrator.run(sample, llm)
    compare = llm.users("LLMSetFindings")
    assert len(compare) == 1
    assert F3 in compare[0] and F4 in compare[0]
    assert "долларах" not in compare[0]  # факт без подтверждённой цитаты не участвует


def test_chunks_cover_whole_document(sample):
    doc = load_document(sample)
    big = (doc.markdown + "\n\n") * 6  # ~80 000 символов
    parts = chunks(big, limit=24_000)
    assert len(parts) > 1 and all(len(p) <= 24_000 for p in parts)
    assert "".join(parts).replace("\n", "") == big.replace("\n", "")  # ничего не обрезано


def test_document_text_is_marked_as_data(sample):
    llm = FakeLLM({})
    orchestrator.run(sample, llm)
    assert all("<<<" in u for u in llm.users("LLMFindings") + llm.users("LLMFacts"))


# --------------------------------------------------------------------------- инфраструктура


def test_cache_makes_second_run_free(sample, tmp_path):
    reply = _findings(_finding("R-006", F2))
    llm = FakeLLM({"LLMFindings": [reply] * 50, "LLMFacts": [GOOD_FACTS]}, cache_dir=tmp_path)
    orchestrator.run(sample, llm)
    calls_first = llm.calls
    llm2 = FakeLLM({}, cache_dir=tmp_path)
    orchestrator.run(sample, llm2)
    assert calls_first > 0
    assert llm2.calls == 0 and llm2.cache_hits == calls_first


def test_invalid_json_is_repaired_once(sample):
    ok = _findings(_finding("R-006", F2))
    llm = FakeLLM({"LLMFindings": ["{not json", ok]})
    report = orchestrator.run(sample, llm)
    assert any(f.source == "llm:checker" for f in report.findings)
    assert "не прошёл проверку схемы" in llm.users("LLMFindings")[1]


def test_provider_down_falls_back_to_rules(sample):
    class Down(FakeLLM):
        def _raw(self, system, user, schema):
            raise LLMUnavailable("сервер не отвечает")

    report = orchestrator.run(sample, Down({}))
    assert report.findings  # линтеры отработали
    assert report.questions  # вопросы по шаблону
    assert report.llm_failures
    assert not any(r.checked for r in report.requirements)
    assert orchestrator.exit_code(report) == 2


def test_overloaded_primary_model_hands_over_to_fallback(sample):
    """Основная модель отвечает 503: прогон доходит до конца на запасной, паспорт это показывает."""

    class Busy(FakeLLM):
        def _raw(self, system, user, schema):
            if self.model == "fake-1":
                raise TransientError("503 high demand", final=LLMUnavailable, model_issue="перегружена (503)")
            return super()._raw(system, user, schema)

    llm = Busy({"LLMFacts": [GOOD_FACTS]})
    llm.fallback_models = ["fake-lite"]
    llm.retry_sleep = lambda s: None
    report = orchestrator.run(sample, llm)
    assert not report.llm_failures
    assert all(r.checked for r in report.requirements)
    assert report.passport.model.startswith("fake-1 → fake-lite")


def test_answers_reach_llm_prompts(sample):
    llm = FakeLLM({})
    answers = [
        Question(id="Q-01", question="По какой дате считать месяц?", why="", answer="По дате проведения")
    ]
    orchestrator.run(sample, llm, answers)
    assert any("По дате проведения" in u for _, u in llm.prompts)


# --------------------------------------------------------------------------- вопросы


def test_clarifier_drops_unquoted_questions(sample):
    finding = _findings(_finding("R-006", F2, criterion="unambiguous"))
    q_bad = {
        "question": "Какой порог?",
        "why": "w",
        "options": [],
        "finding_ids": ["F-001"],
        "requirement_ids": ["R-006"],
        "quotes": ["Порог хранится в долларах США"],
        "priority": "critical",
    }
    q_good = dict(q_bad, question="По какой дате считать месяц?", quotes=[F2])
    llm = FakeLLM({"LLMFindings": [finding], "LLMQuestions": [json.dumps({"questions": [q_bad, q_good]})]})
    report = orchestrator.run(sample, llm)
    assert [q.question for q in report.questions] == ["По какой дате считать месяц?"]


@pytest.mark.parametrize("severity", ["critical", "major"])
def test_clarifier_template_without_llm(sample, severity):
    report = orchestrator.run(sample, llm=None)
    assert report.questions
    assert all(q.id.startswith("Q-") and q.why for q in report.questions)
