"""Деление абзаца на независимо проверяемые утверждения: ID пункта, родитель, дословный фрагмент, место.

Аудит: в абзаце пункта стояло несколько самостоятельных правил, а отчёт сводил их к одной записи.
"""

from __future__ import annotations

import json

from ravel import orchestrator
from ravel.ingest import load_document
from ravel.ingest.sentences import split_spans
from ravel.quotes import locate
from ravel.segment import segment
from tests.test_agents import FakeLLM, _facts, _finding, _findings


def _doc(tmp_path, text: str, name: str = "req.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p, load_document(p)


def _fragments(text: str) -> list[str]:
    return [text[a:b] for a, b in split_spans(text)]


COMPOUND = """# Платёжный шлюз

## Функциональные требования

- П-7. Клиент оплачивает счёт до 500 000 ₸ за одну операцию. Если сумма больше лимита, платёж
  отклоняется с кодом LIMIT. Иначе платёж проводится не дольше 5 секунд. По умолчанию лимит включён.
- П-8. Комиссия не взимается.
"""


def test_compound_item_split_keeps_id_parent_fragment_and_position(tmp_path):
    path, doc = _doc(tmp_path, COMPOUND)
    reqs = [r for r in segment(doc) if r.kind == "requirement"]
    p7 = [r for r in reqs if r.source_id == "П-7"]
    assert [r.text for r in p7] == [
        "П-7. Клиент оплачивает счёт до 500 000 ₸ за одну операцию.",
        # условие и его последствие, и альтернатива «Иначе …» — одно утверждение
        "Если сумма больше лимита, платёж отклоняется с кодом LIMIT. Иначе платёж проводится не дольше 5 секунд.",
        "По умолчанию лимит включён.",
    ]
    parent = p7[0].parent_text
    assert parent and all(r.parent_text == parent for r in p7)
    assert all(parent[r.offset : r.offset + len(r.text)] == r.text for r in p7)  # дословно и на месте
    assert [(r.part, r.parts) for r in p7] == [(1, 3), (2, 3), (3, 3)]
    assert all(r.parent_id == p7[0].id for r in p7)
    # строки исходника: второе утверждение начинается в строке 5 и заканчивается в строке 6
    lines = path.read_text(encoding="utf-8").split("\n")
    assert [(r.line_start, r.line_end) for r in p7] == [(5, 5), (5, 6), (6, 6)]
    for r in p7:
        assert r.raw_text == "\n".join(lines[r.line_start - 1 : r.line_end])
        assert f"строка {r.line_start}" in r.location
    # одно утверждение — цельный пункт без полей фрагмента
    p8 = next(r for r in reqs if r.source_id == "П-8")
    assert (p8.part, p8.parts, p8.parent_id, p8.parent_text) == (None, None, None, None)


def test_compound_fragments_have_own_blocks_for_findings(tmp_path):
    """Находка правил по одному утверждению привязывается к его фрагменту, а не к соседнему."""
    _, doc = _doc(tmp_path, COMPOUND.replace("По умолчанию лимит включён.", "Отказ показывается быстро."))
    report = orchestrator.run(tmp_path / "req.md", llm=None)
    vague = next(f for f in report.findings if f.source == "lint:vague_word")
    target = next(r for r in report.requirements if "быстро" in r.text)
    assert vague.requirement_ids == [target.id]
    assert len({r.block_id for r in report.requirements}) == len(report.requirements)


def test_two_numbered_lists_with_repeated_numbers(tmp_path):
    _, doc = _doc(
        tmp_path,
        "# Д\n\n## Проверки\n\n1. Проверить сессию. Сессия старше 30 минут отклоняется.\n"
        "2. Проверить карту.\n\n1. Сохранить настройку.\n2. Вернуть результат клиенту.\n\n"
        "## Правила\n\n1. Порог целый.\n",
    )
    reqs = [r for r in segment(doc) if r.kind == "requirement"]
    assert [(r.source_id, r.text) for r in reqs] == [
        ("Проверки п. 1", "Проверить сессию."),
        ("Проверки п. 1", "Сессия старше 30 минут отклоняется."),
        ("Проверки п. 2", "Проверить карту."),
        # второй список того же раздела: нумерация началась заново, номера не сливаются
        ("Проверки п. 1 (2)", "Сохранить настройку."),
        ("Проверки п. 2 (2)", "Вернуть результат клиенту."),
        ("Правила п. 1", "Порог целый."),
    ]
    first = [r for r in reqs if r.source_id == "Проверки п. 1"]
    assert [(r.part, r.parts) for r in first] == [(1, 2), (2, 2)] and first[1].parent_id == first[0].id
    assert len({r.id for r in reqs}) == len(reqs)


def test_unnumbered_document_gets_own_ids_and_positions(tmp_path):
    text = (
        "Сервис принимает заявку на кредит. Заявка без паспорта отклоняется.\n"
        "\n"
        "Решение отправляется клиенту в течение 2 минут. Если скоринг недоступен, заявка ставится в очередь.\n"
    )
    path, doc = _doc(tmp_path, text, "plain.txt")
    reqs = [r for r in segment(doc) if r.kind == "requirement"]
    assert [r.text for r in reqs] == [
        "Сервис принимает заявку на кредит.",
        "Заявка без паспорта отклоняется.",
        "Решение отправляется клиенту в течение 2 минут.",
        "Если скоринг недоступен, заявка ставится в очередь.",
    ]
    assert [r.id for r in reqs] == ["R-001", "R-002", "R-003", "R-004"]
    assert all(r.source_id is None for r in reqs)
    assert [r.line_start for r in reqs] == [1, 1, 3, 3]
    assert reqs[1].parent_id == reqs[0].id and reqs[3].parent_id == reqs[2].id
    assert reqs[0].parent_text == text.split("\n")[0]


def test_sentence_split_keeps_labels_abbreviations_and_conditions():
    assert _fragments("200. Ответ при успехе. 400. Ответ при ошибке параметров.") == [
        "200. Ответ при успехе.",
        "400. Ответ при ошибке параметров.",
    ]
    assert _fragments("Срок — 3 сут., см. п. 4.2 и т. д. Отчёт формируется ежедневно.") == [
        "Срок — 3 сут., см. п. 4.2 и т. д. Отчёт формируется ежедневно."
    ]
    assert _fragments(
        "Если карта заблокирована, push не отправляется. В этом случае пишется событие в журнал."
    ) == ["Если карта заблокирована, push не отправляется. В этом случае пишется событие в журнал."]
    assert _fragments("Поддерживаются статусы: Новый, Оплачен.") == [
        "Поддерживаются статусы: Новый, Оплачен."
    ]
    assert _fragments("Для чужой карты возвращается 404. 404") == ["Для чужой карты возвращается 404. 404"]


def test_service_reference_and_demo_command_are_context(tmp_path):
    _, doc = _doc(
        tmp_path,
        "# Д\n\n## Интеграция\n\nРодительский документ: «ТЗ на платежи».\n\n"
        "Сервис хранит документ клиента: «паспорт».\n\n"
        "| Поле | Значение |\n| --- | --- |\n| Пример | curl -X POST \"$URL/pay\" -d 'sum=1' |\n"
        "| Метод | POST |\n",
    )
    by_text = {r.text: r for r in segment(doc)}
    assert by_text["Родительский документ: «ТЗ на платежи»."].kind == "metadata"
    assert by_text["Сервис хранит документ клиента: «паспорт»."].kind == "requirement"
    assert by_text["Поле: Пример; Значение: curl -X POST \"$URL/pay\" -d 'sum=1'"].kind == "example"
    assert by_text["Поле: Метод; Значение: POST"].kind == "requirement"


def test_example_remains_evidence_for_set_contradiction(sample):
    """Пример (curl) не проверяется как требование, но цитата из него подтверждает противоречие набора."""
    reqs = segment(load_document(sample))
    curl = next(r for r in reqs if r.kind == "example" and "curl" in r.text)
    rest = next(r for r in reqs if r.kind == "requirement" and r.text.endswith("REST type; Значение: GET"))
    quote_ex = 'curl -G "$BASE/alerts/saveLimit"'
    quote_req = "REST type | GET"
    finding = _finding(
        [curl.id, rest.id], [quote_ex, quote_req], criterion="set_consistent", severity="major"
    )
    facts = _facts(("api_method_url", "Метод GET", quote_req))
    llm = FakeLLM({"LLMFacts": [facts], "LLMSetFindings": [_findings(finding)]})
    report = orchestrator.run(sample, llm)
    kept = [f for f in report.findings if f.source == "llm:set"]
    assert len(kept) == 1
    assert kept[0].requirement_ids == [rest.id]  # E- в затронутые требования не попадает
    assert [q.block_id for q in kept[0].quotes] == [curl.block_id, rest.block_id]
    assert "[E-" not in "\n".join(llm.users("LLMFindings"))  # пример не уходит в проверку требований
    assert json.dumps(report.model_dump(mode="json"), ensure_ascii=False)  # контракт сериализуется


def test_quote_across_several_fragments_is_found(sample):
    doc = load_document(sample)
    quote = (
        "В расходы входят проведённые покупки. Возврат уменьшает расходы месяца исходной покупки. Переводы"
    )
    block = locate(quote, doc)
    assert block is not None and block.text == "В расходы входят проведённые покупки."


def test_explicit_prose_example_fragments_remain_context(tmp_path):
    _, doc = _doc(
        tmp_path,
        "# Бронирование\n\n"
        "Пример: сотрудник бронирует комнату с 10:00 до 13:00. Бронь создана.\n\n"
        "Сервис должен показывать пример бронирования.\n",
    )
    reqs = segment(doc)
    assert [r.kind for r in reqs] == ["requirement", "example", "example"]
    assert reqs[0].text == "Сервис должен показывать пример бронирования."
    examples = [r for r in reqs if r.kind == "example"]
    assert examples[1].parent_id == examples[0].id
    assert all(locate(r.text, doc) for r in examples)


def test_prose_example_can_prove_set_contradiction(tmp_path):
    path, doc = _doc(
        tmp_path,
        "# Бронирование\n\n"
        "Одна бронь длится не дольше 2 часов.\n\n"
        "Пример: бронь с 10:00 до 13:00 успешно создаётся.\n",
    )
    reqs = segment(doc)
    product = next(r for r in reqs if r.kind == "requirement")
    example = next(r for r in reqs if r.kind == "example")
    found = _finding([product.id, example.id], [product.text, example.text], criterion="set_consistent")
    model = FakeLLM(
        {
            "LLMFacts": [_facts(("timing_latency", "Длительность", product.text))],
            "LLMSetFindings": [json.dumps({"findings": [found]})],
        }
    )
    report = orchestrator.run(path, model)
    sets = [f for f in report.findings if f.source == "llm:set"]
    assert len(sets) == 1 and sets[0].requirement_ids == [product.id]
    assert len(sets[0].quotes) == 2
