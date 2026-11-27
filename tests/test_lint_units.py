"""Общее правило mixed_units и привязка http_get_write к конкретному методу."""

from __future__ import annotations

import pytest

from ravel.ingest import load_document
from ravel.lint import run_lint

from .conftest import SAMPLE


def _lint(tmp_path, text: str, rule: str):
    p = tmp_path / "doc.md"
    p.write_text(f"# Документ\n\n## Данные\n\n{text}\n", encoding="utf-8")
    return [f for f in run_lint(load_document(p)) if f.source == f"lint:{rule}"]


def test_conversion_statement_is_not_a_defect(tmp_path):
    assert not _lint(tmp_path, "Касса отображает 100 тиын как 1 тенге.", "mixed_units")


def test_conversion_by_division_is_not_a_defect(tmp_path):
    text = "Порог задаётся в тенге и сравнивается с суммой покупок в тиынах, делённой на 100."
    assert not _lint(tmp_path, text, "mixed_units")


def test_comparison_without_conversion_is_reported_with_both_quotes(tmp_path):
    text = "Хранить сумму покупки в тиынах и порог в тенге. Push отправляется, когда сумма достигает порога."
    (f,) = _lint(tmp_path, text, "mixed_units")
    assert [q.text for q in f.quotes] == ["Хранить сумму покупки в тиынах", "порог в тенге"]
    assert "Сравниваются «Хранить сумму покупки в тиынах» (тиын) и «порог в тенге» (тенге)" in f.explanation
    assert "правило пересчёта не указано" in f.explanation


def test_other_dimension_is_detected(tmp_path):
    text = "Таймаут задаётся в секундах; запрос прерывается, если длительность в миллисекундах превышает таймаут."
    (f,) = _lint(tmp_path, text, "mixed_units")
    assert "(секунда)" in f.explanation and "(миллисекунда)" in f.explanation


def test_units_without_comparison_are_not_reported(tmp_path):
    assert not _lint(tmp_path, "Цена указана в долларах. Скидка 10 центов.", "mixed_units")


def test_short_abbreviations_are_not_units_without_number(tmp_path):
    text = "Система работает с учётом лимита; ответ приходит за 5 с учётом сети, лимит 3 мин."
    assert not _lint(tmp_path, text, "mixed_units")


def test_cross_block_in_one_section(tmp_path):
    text = "Таймаут запроса: 30 секунд.\n\nЗапрос прерывается, если длительность в миллисекундах превышает таймаут."
    assert len(_lint(tmp_path, text, "mixed_units")) == 1


def test_get_write_is_bound_to_each_method(tmp_path):
    text = """## Метод чтения

| Поле | Значение |
| :--- | :--- |
| **Description** | Получение порога |
| **REST type** | GET |

Возвращает текущую настройку.

## Метод сохранения

| Поле | Значение |
| :--- | :--- |
| **Description** | Сохранение порога |
| **REST type** | GET |
"""
    p = tmp_path / "api.md"
    p.write_text("# API\n\n" + text, encoding="utf-8")
    found = [f for f in run_lint(load_document(p)) if f.source == "lint:http_get_write"]
    assert len(found) == 1
    assert "Сохранение порога" in [q.text for q in found[0].quotes][1]


def test_get_without_write_in_own_method_is_clean(tmp_path):
    text = """## Чтение

| Поле | Значение |
| :--- | :--- |
| **REST type** | GET |

## Сохранение

| Поле | Значение |
| :--- | :--- |
| **REST type** | POST |
| **Description** | Сохранение порога |
"""
    p = tmp_path / "api.md"
    p.write_text("# API\n\n" + text, encoding="utf-8")
    assert not [f for f in run_lint(load_document(p)) if f.source == "lint:http_get_write"]


def test_sample_keeps_mixed_units_and_http_get_write():
    sources = [f.source for f in run_lint(load_document(SAMPLE))]
    assert sources.count("lint:mixed_units") == 1
    assert sources.count("lint:http_get_write") == 1


def test_unrelated_time_intervals_do_not_require_conversion(tmp_path):
    text = (
        "Сотрудник задаёт время начала брони с шагом 15 минут.\n\n"
        "Одна бронь длится не дольше 2 часов; больший интервал отклоняется.\n\n"
        "Без подтверждения в течение 10 минут бронь снимается.\n\n"
        "Отменить бронь можно не позже чем за 5 минут до начала."
    )
    assert not _lint(tmp_path, text, "mixed_units")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("unrelated_first", [False, True])
def test_related_units_use_matching_blocks_in_any_order(tmp_path, reverse, unrelated_first):
    blocks = [
        "Таймаут запроса: 30 секунд.",
        "Запрос прерывается, если длительность в миллисекундах превышает таймаут.",
    ]
    if reverse:
        blocks.reverse()
    if unrelated_first:
        blocks.insert(0, "Заставка отображается 5 секунд.")
    (finding,) = _lint(tmp_path, "\n\n".join(blocks), "mixed_units")
    assert {quote.text for quote in finding.quotes} == {
        "Таймаут запроса: 30 секунд.",
        "Запрос прерывается, если длительность в миллисекундах превышает таймаут.",
    }
    assert "Заставка" not in finding.explanation
    # Сохраняем одну находку на пару единиц, даже если связанное сравнение повторено.
    repeated = _lint(tmp_path, "\n\n".join(blocks + blocks), "mixed_units")
    assert len(repeated) == 1
