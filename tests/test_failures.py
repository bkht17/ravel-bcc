"""Негативные сценарии аудита: сбои LLM, пустые и плохо прочитанные документы, вход = выход."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from ravel import orchestrator
from ravel.cli import app
from ravel.ingest import pdf as pdf_mod
from ravel.ingest.quality import (
    MAX_MIXED_SHARE,
    MIN_KNOWN_SHARE,
    MIN_TOKENS,
    assess,
    ocr_is_poor,
)
from ravel.render import render_markdown

from .conftest import FIXTURES, LLM_FAILURE_SCENARIOS
from .test_agents import FakeLLM, _facts, _finding, _findings

READY_PHRASE = "можно начинать"
runner = CliRunner()

CLEAR = FIXTURES / "clear_requirements.md"  # 5 ясных требований без дефектов
META = FIXTURES / "meta_example.md"  # метаданные (M-), примеры (E-) и требования
HEADING_ONLY = FIXTURES / "heading_only.md"
SCAN = FIXTURES / "scan.pdf"

# Пределы числа вызовов generate_content за весь прогон (на фикстуре из 5 требований: один запрос
# на каждый LLM-этап — checker, set_checker, clarifier). Сбой, после которого LLM бесполезен
# (403/404), стоит одного вызова: дальше модель не вызывается. Остальные сбои ограничены попытками
# одного запроса (MAX_ATTEMPTS = 3; пустой ответ и таймаут — по 2) на каждый из трёх этапов.
LIMIT_FATAL = 1  # 403, 404: один вызов, этапы после него модель не трогают
LIMIT_QUOTA = 3  # 429 после исчерпания повторов: один этап, дальше модель отключена
LIMIT_TRANSIENT = 3  # 503, нет сети: один этап с тремя попытками, дальше модель отключена
LIMIT_TIMEOUT = 2  # таймаут: две попытки, дальше модель отключена
LIMIT_BAD_REPLY = 6  # невалидный JSON/пустой ответ: по два вызова на каждый из трёх этапов

CALL_LIMITS = {
    "403": LIMIT_FATAL,
    "404": LIMIT_FATAL,
    "429": LIMIT_QUOTA,
    "503": LIMIT_TRANSIENT,
    "connect_error": LIMIT_TRANSIENT,
    "read_timeout": LIMIT_TIMEOUT,
    "invalid_json": LIMIT_BAD_REPLY,
    "empty": LIMIT_BAD_REPLY,
}

# --------------------------------------------------------------------------- сбои LLM


@pytest.mark.parametrize("gemini_failing", LLM_FAILURE_SCENARIOS, indirect=True)
def test_llm_failure_is_not_ready(request, gemini_failing):
    name = request.node.callspec.params["gemini_failing"]
    client, mock = gemini_failing
    report = orchestrator.run(CLEAR, client)  # исключение из run — провал теста
    assert orchestrator.exit_code(report) != 0
    assert READY_PHRASE not in report.verdict.headline
    assert READY_PHRASE not in render_markdown(report)
    assert mock.call_count <= CALL_LIMITS[name], f"{name}: вызовов {mock.call_count}"
    assert mock.call_count >= 1  # подмена действительно сработала


# --------------------------------------------------------------------------- «светофор»


GOOD_FACT = _facts(
    (
        "frequency_limits",
        "Повторное push-уведомление в том же месяце не отправляется",
        "Система не отправляет повторное push-уведомление по той же карте в том же календарном месяце",
    )
)


def test_clear_document_is_ready_with_success_fake():
    """Контроль: без дефектов и без сбоев итог зелёный, иначе остальные проверки ничего не значат."""
    report = orchestrator.run(CLEAR, FakeLLM({"LLMFacts": [GOOD_FACT]}))
    assert [r.kind for r in report.requirements] == ["requirement"] * 5
    assert report.verdict.status == "ready", report.verdict.headline
    assert orchestrator.exit_code(report) == 0
    assert READY_PHRASE in report.verdict.headline


# --------------------------------------------------------------------------- A2: документ без требований


@pytest.mark.parametrize("with_llm", [False, True], ids=["no_llm", "fake_llm"])
def test_heading_only_document_is_not_ready(with_llm):
    llm = FakeLLM({"LLMFacts": [GOOD_FACT]}) if with_llm else None
    report = orchestrator.run(HEADING_ONLY, llm)
    assert not [r for r in report.requirements if r.kind == "requirement"]
    assert orchestrator.exit_code(report) != 0
    assert READY_PHRASE not in report.verdict.headline
    assert READY_PHRASE not in render_markdown(report)


# --------------------------------------------------------------------------- A3: вход = выход


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_equal_to_input_is_rejected_and_input_untouched(tmp_path):
    src = tmp_path / "spec.md"
    src.write_bytes(CLEAR.read_bytes())
    before = _sha(src)
    res = runner.invoke(app, ["check", str(src), "-o", str(src), "-p", "none"])
    assert res.exit_code == 1, res.output
    assert _sha(src) == before


# --------------------------------------------------------------------------- A4: скан и OCR

# Записанный вывод Tesseract с английской моделью по русскому скану (tests/fixtures/scan.pdf)
GARBAGE_OCR = (
    "Het OGwme ceegenna Wz Ipxmv kro Lopa 4 ohbnn qwe Ttad Rnu "
    "Sdfg Hjkl Poiu Zxcv Mnbv Qazw Edcr Tgby Uhnj Ikmo Plok Ujmn\n"
    "Vfr Bgt Nhy Mju Kil Ol Pq Xsw Cde Zaq Wsx Edc Rfv Tgb Yhn"
)
GOOD_OCR = (
    "Система отправляет пользователю уведомление при достижении порога расходов. "
    "Пользователь может отключить уведомления в настройках, и система не будет их отправлять. "
    "Данные клиента хранятся в системе, и каждый клиент видит только свои данные."
)
MIXED_OCR = " ".join(["Пpивeт", "кapтa", "пopoг", "pacxoды", "cиcтeмa", "дaнныe", "клиeнт", "зaпpoc"] * 3)


def test_quality_garbage_is_poor():
    q = assess(GARBAGE_OCR)
    assert q.tokens >= MIN_TOKENS
    assert q.known_share < MIN_KNOWN_SHARE
    assert ocr_is_poor(q)


def test_quality_normal_text_is_not_poor():
    q = assess(GOOD_OCR)
    assert q.known_share >= MIN_KNOWN_SHARE and q.mixed_share <= MAX_MIXED_SHARE
    assert not ocr_is_poor(q)


def test_quality_mixed_alphabets_is_poor():
    q = assess(MIXED_OCR)
    assert q.mixed_share > MAX_MIXED_SHARE
    assert ocr_is_poor(q)


def test_quality_too_few_words_is_poor():
    assert ocr_is_poor(assess("Система отправляет"))
    assert ocr_is_poor(assess(""))


@pytest.fixture
def fake_ocr(monkeypatch):
    """Подменяет Tesseract: распознавание возвращает text из списка; от установленных языков не зависит."""
    state = {"text": GARBAGE_OCR}
    monkeypatch.setattr(pdf_mod, "tessdata_dir", lambda: "/fake/tessdata")
    monkeypatch.setattr(pdf_mod, "_ocr_languages", lambda tessdata: "rus+eng")
    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", lambda self, **kw: object())
    original = pymupdf.Page.get_text

    def get_text(self, *args, **kw):
        if kw.get("textpage") is not None:
            return state["text"]
        return original(self, *args, **kw)

    monkeypatch.setattr(pymupdf.Page, "get_text", get_text)
    return state


def test_scan_with_garbage_ocr_is_fatal_not_ready(fake_ocr):
    report = orchestrator.run(SCAN, None)
    assert report.fatal
    assert "ocr_low_quality" in {i.code for i in report.issues}
    assert not report.requirements
    assert orchestrator.exit_code(report) == 1
    assert READY_PHRASE not in render_markdown(report)


def test_scan_with_garbage_ocr_via_cli(fake_ocr, tmp_path):
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(SCAN), "-o", str(out), "-p", "none"])
    assert res.exit_code == 1, res.output
    assert READY_PHRASE not in out.read_text(encoding="utf-8")


def test_scan_with_good_ocr_is_read(fake_ocr):
    """Контроль подмены: нормальный текст OCR принимается, ошибки качества нет."""
    fake_ocr["text"] = GOOD_OCR
    report = orchestrator.run(SCAN, None)
    assert "ocr_low_quality" not in {i.code for i in report.issues}
    assert not report.fatal


def test_scan_without_tesseract_is_fatal_not_ready(monkeypatch):
    monkeypatch.setattr(pdf_mod, "tessdata_dir", lambda: None)
    report = orchestrator.run(SCAN, None)
    assert report.fatal
    assert "no_text_layer" in {i.code for i in report.issues}
    assert orchestrator.exit_code(report) == 1
    assert READY_PHRASE not in render_markdown(report)


# --------------------------------------------------------------------------- метаданные и примеры


def _first(report, kind):
    return next(r for r in report.requirements if r.kind == kind)


GOOD_FACT_META = GOOD_FACT
# цитаты берём из текста документа: выдуманная цитата отбрасывается как галлюцинация
QUOTE_R1 = (
    "Пользователь задаёт месячный порог расходов целым числом от 1 000 до 10 000 000 тенге включительно"
)
QUOTE_R3 = "Система не отправляет повторное push-уведомление по той же карте в том же календарном месяце"
QUOTE_OF = {
    "metadata": "Наименование задачи: Месячный порог расходов по карте",
    "example": '{"card": "4276", "threshold": 5000}',
}


@pytest.mark.xfail(
    strict=True,
    reason="ждёт B: фильтр по kind в агентах — находка с ID метаданных (M-) или примера (E-) попадает в report.findings",
)
@pytest.mark.parametrize("which", ["metadata", "example"])
@pytest.mark.parametrize("schema", ["LLMFindings", "LLMSetFindings"])
def test_findings_on_metadata_and_examples_are_not_counted(schema, which):
    base = orchestrator.run(META, None)  # ID берём из сегментации, а не из головы
    r1, r3 = base.requirements[0], base.requirements[2]
    target = _first(base, which)
    assert target.id.startswith(("M-", "E-"))
    kind = "set_consistent" if schema == "LLMSetFindings" else "verifiable"
    ok = _finding([r1.id, r3.id], [QUOTE_R1, QUOTE_R3], criterion=kind, severity="major")
    foreign = _finding([r1.id, target.id], [QUOTE_R1, QUOTE_OF[which]], criterion=kind, severity="major")

    def run(*items):
        replies = {schema: [_findings(*items)] * 5, "LLMFacts": [GOOD_FACT_META]}
        return orchestrator.run(META, FakeLLM(replies))

    # контроль: находка по настоящим требованиям проходит, иначе тест ничего не доказывает
    assert run(ok).findings, "корректная находка отброшена"
    report = run(ok, foreign)
    assert not [f for f in report.findings if target.id in f.requirement_ids]
    assert f"| Нарушений всего | {len(report.findings)} |" in render_markdown(report)
