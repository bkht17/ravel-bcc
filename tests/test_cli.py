from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ravel.cli import app
from ravel.fsutil import atomic_write

runner = CliRunner()


def test_cli_without_llm(sample, tmp_path):
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(sample), "-o", str(out), "-p", "none"])
    assert res.exit_code == 2, res.output
    assert out.exists() and (tmp_path / "r.questions.yaml").exists()


def test_cli_broken_file_writes_report_and_exits_1(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7\n garbage")
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(bad), "-o", str(out), "-p", "none"])
    assert res.exit_code == 1
    assert "не удалось прочитать" in out.read_text(encoding="utf-8")


def test_cli_missing_key_exits_3(sample, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.chdir(tmp_path)  # чтобы не подхватить реальный .env
    res = runner.invoke(app, ["check", str(sample), "-o", str(tmp_path / "r.md"), "-p", "gemini"])
    assert res.exit_code == 3


def test_answers_roundtrip(sample, tmp_path):
    out = tmp_path / "r.md"
    runner.invoke(app, ["check", str(sample), "-o", str(out), "-p", "none"])
    qfile = tmp_path / "r.questions.yaml"
    text = qfile.read_text(encoding="utf-8").replace("answer: ''", "answer: Используем POST", 1)
    qfile.write_text(text, encoding="utf-8")
    out2 = tmp_path / "r2.md"
    res = runner.invoke(app, ["check", str(sample), "-o", str(out2), "-p", "none", "--answers", str(qfile)])
    assert res.exit_code == 2
    assert "Ответ автора:** Используем POST" in out2.read_text(encoding="utf-8")


def test_atomic_write_keeps_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "report.md"
    target.write_text("old", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in Path(tmp_path).iterdir()] == ["report.md"]


# --- защита выходов и групповая запись ---------------------------------------------------------


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(src, out):
    return runner.invoke(app, ["check", str(src), "-o", str(out), "-p", "none"])


def test_output_equals_input_rejected(sample, tmp_path):
    src = tmp_path / "x.md"
    src.write_bytes(sample.read_bytes())
    before = _sha(src)
    res = _run(src, src)
    assert res.exit_code == 1
    assert "Выходной файл совпадает с входным" in res.output
    assert _sha(src) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.md"]


def test_output_symlink_to_input_rejected(sample, tmp_path):
    src = tmp_path / "x.md"
    src.write_bytes(sample.read_bytes())
    before = _sha(src)
    link = tmp_path / "link.md"
    link.symlink_to(src)
    res = _run(src, link)
    assert res.exit_code == 1
    assert _sha(src) == before and link.is_symlink()


def test_json_output_colliding_with_input_rejected(sample, tmp_path):
    src = tmp_path / "x.json"
    src.write_text('{"keep": 1}', encoding="utf-8")
    res = _run(src, tmp_path / "x.md")
    assert res.exit_code == 1
    assert src.read_text(encoding="utf-8") == '{"keep": 1}'
    assert not (tmp_path / "x.md").exists()


def test_output_is_directory_rejected(sample, tmp_path):
    d = tmp_path / "out.md"
    d.mkdir()
    assert _run(sample, d).exit_code == 1


def _snapshot(tmp_path, names):
    return {n: (tmp_path / n).read_bytes() for n in names}


NAMES = ["r.md", "r.json", "r.questions.yaml", "r.trace.jsonl"]


def test_failure_mid_group_keeps_old_files_and_no_tmpdir(sample, tmp_path, monkeypatch):
    from ravel import fsutil

    assert _run(sample, tmp_path / "r.md").exit_code == 2
    before = _snapshot(tmp_path, NAMES)

    real, calls = fsutil._stage, []

    def flaky(path, content):
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk full")
        real(path, content)

    monkeypatch.setattr(fsutil, "_stage", flaky)
    res = _run(sample, tmp_path / "r.md")
    assert res.exit_code != 0 and len(calls) == 3
    assert _snapshot(tmp_path, NAMES) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(NAMES)


def test_oserror_on_write_gives_clean_message(sample, tmp_path, monkeypatch):
    from ravel import fsutil

    def denied(path, content):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(fsutil, "_stage", denied)
    res = _run(sample, tmp_path / "r.md")
    assert res.exit_code == 1
    assert "Отчёт не записан" in res.output
    assert "Traceback" not in res.output
    assert isinstance(res.exception, SystemExit)


def _run_ids(tmp_path):
    import json

    import yaml  # noqa: F401

    ids = {
        "json": json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["run_id"],
        "trace": json.loads((tmp_path / "r.trace.jsonl").read_text(encoding="utf-8").splitlines()[0])[
            "run_id"
        ],
    }
    ids["yaml"] = (tmp_path / "r.questions.yaml").read_text(encoding="utf-8").splitlines()[0].split(": ")[1]
    row = next(
        ln
        for ln in (tmp_path / "r.md").read_text(encoding="utf-8").splitlines()
        if ln.startswith("| ID прогона")
    )
    ids["md"] = row.split("`")[1]
    return ids


def test_run_id_same_in_all_files_and_new_per_run(sample, tmp_path):
    from ravel.render import load_answers

    assert _run(sample, tmp_path / "r.md").exit_code == 2
    first = _run_ids(tmp_path)
    assert len(set(first.values())) == 1
    assert _run(sample, tmp_path / "r.md").exit_code == 2
    second = _run_ids(tmp_path)
    assert len(set(second.values())) == 1
    assert second["md"] != first["md"]
    assert load_answers(tmp_path / "r.questions.yaml") == []


def test_paths_with_spaces_and_cyrillic(sample, tmp_path):
    d = tmp_path / "папка с пробелом"
    d.mkdir()
    src = d / "требования 1.md"
    src.write_bytes(sample.read_bytes())
    out = d / "отчёт 1.md"
    assert _run(src, out).exit_code == 2
    assert out.exists() and (d / "отчёт 1.trace.jsonl").exists()
    assert [p.name for p in d.iterdir() if p.name.startswith(".")] == []


def test_missing_passport_row_writes_nothing(sample, tmp_path, monkeypatch):
    from ravel import render

    assert _run(sample, tmp_path / "r.md").exit_code == 2
    before = _snapshot(tmp_path, NAMES)

    monkeypatch.setattr(
        render, "render_markdown", lambda report, run_id=None: "# Отчёт\n\n| Поле | Значение |\n"
    )
    res = _run(sample, tmp_path / "r.md")
    assert res.exit_code == 1
    assert "Версия промптов" in res.output and "не записан" in res.output
    assert res.exception is None or isinstance(res.exception, SystemExit)
    assert _snapshot(tmp_path, NAMES) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(NAMES)


def test_write_outputs_missing_passport_row_raises_before_mkdir(tmp_path, monkeypatch):
    from ravel import render
    from ravel.models import Report

    monkeypatch.setattr(render, "render_markdown", lambda report, run_id=None: "без паспорта\n")
    out = tmp_path / "new" / "r.md"
    with pytest.raises(render.ReportRenderError, match="Версия промптов"):
        render.write_outputs(Report.model_construct(questions=[], trace=[]), out, "rid")
    assert not out.parent.exists()


def test_output_hardlink_to_input_rejected(sample, tmp_path):
    src = tmp_path / "x.md"
    src.write_bytes(sample.read_bytes())
    before = _sha(src)
    link = tmp_path / "link.md"
    try:
        os.link(src, link)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"ФС не поддерживает жёсткие ссылки: {e}")
    res = _run(src, link)
    assert res.exit_code == 1
    assert "Выходной файл совпадает с входным" in res.output
    assert _sha(src) == before and _sha(link) == before
    assert os.path.samefile(src, link)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.md", "x.md"]


# --- итог: undecided вместо ложного «можно начинать» -------------------------------------------

READY_PHRASE = "можно начинать"


def _use_llm(monkeypatch, llm):
    monkeypatch.setattr("ravel.llm.get_client", lambda provider, model=None: llm)


def _ok_facts():
    from .test_agents import _facts

    return _facts(
        ("frequency_limits", "порог", "Пользователь задаёт порог от 1 000 до 10 000 тенге включительно")
    )


def test_provider_none_is_undecided_exit_2(sample, tmp_path):
    out = tmp_path / "r.md"
    res = _run(sample, out)
    md = out.read_text(encoding="utf-8")
    assert res.exit_code == 2
    assert READY_PHRASE not in md
    assert "принять нельзя" in md and "только правилами" in md
    assert "Покрытие: проверено 0 из" in md


def test_provider_always_fails_exit_2(sample, tmp_path, monkeypatch):
    from ravel.llm.base import LLMUnavailable

    from .test_agents import FakeLLM

    class Down(FakeLLM):
        def _raw(self, system, user, schema):
            raise LLMUnavailable("403")

    _use_llm(monkeypatch, Down({}))
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(sample), "-o", str(out), "-p", "gemini"])
    md = out.read_text(encoding="utf-8")
    assert res.exit_code == 2
    assert READY_PHRASE not in md
    assert "Не проверено (LLM недоступен или ответ отклонён)" in md
    assert "Нарушений не найдено." not in md


def test_document_without_requirements_exit_2(tmp_path, monkeypatch):
    from .test_agents import FakeLLM

    src = tmp_path / "empty.md"
    src.write_text("# Только заголовок\n", encoding="utf-8")
    _use_llm(monkeypatch, FakeLLM({}))
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(src), "-o", str(out), "-p", "gemini"])
    md = out.read_text(encoding="utf-8")
    assert res.exit_code == 2
    assert READY_PHRASE not in md
    assert "проверено 0 из 0 требований" in md


def test_full_clean_run_exit_0(tmp_path, monkeypatch):
    from .conftest import MINI_MD
    from .test_agents import FakeLLM

    src = tmp_path / "mini.md"
    src.write_text(MINI_MD, encoding="utf-8")
    _use_llm(monkeypatch, FakeLLM({"LLMFacts": [_ok_facts()]}))
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(src), "-o", str(out), "-p", "gemini"])
    md = out.read_text(encoding="utf-8")
    assert res.exit_code == 0, res.output
    assert READY_PHRASE in md
    assert "Покрытие: проверено 3 из 3 требований." in md


def test_clarifier_failure_with_complete_check_exit_0(sample, tmp_path, monkeypatch):
    from ravel.llm.base import LLMUnavailable

    from .test_agents import GOOD_FACTS, FakeLLM

    class ClarifierDown(FakeLLM):
        def _raw(self, system, user, schema):
            if schema.__name__ == "LLMQuestions":
                raise LLMUnavailable("503")
            return super()._raw(system, user, schema)

    _use_llm(monkeypatch, ClarifierDown({"LLMFacts": [GOOD_FACTS]}))
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(sample), "-o", str(out), "-p", "gemini"])
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert any(m.startswith("clarifier") for m in report["llm_failures"])
    assert report["verdict"]["status"] != "undecided"
    assert res.exit_code == 0, res.output


def test_set_checker_failure_is_undecided_exit_2(tmp_path, monkeypatch):
    from ravel.llm.base import LLMUnavailable

    from .conftest import MINI_MD
    from .test_agents import FakeLLM

    class SetCheckerDown(FakeLLM):
        def _raw(self, system, user, schema):
            if schema.__name__ == "LLMFacts":
                raise LLMUnavailable("503")
            return super()._raw(system, user, schema)

    src = tmp_path / "mini.md"
    src.write_text(MINI_MD, encoding="utf-8")
    _use_llm(monkeypatch, SetCheckerDown({}))
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(src), "-o", str(out), "-p", "gemini"])
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert any(m.startswith("set_checker") for m in report["llm_failures"]), report["llm_failures"]
    assert report["verdict"]["status"] == "undecided"
    assert res.exit_code == 2, res.output


# --- частичное чтение: новые коды проблем чтения и итог ---------------------------------------


def test_low_coverage_with_successful_llm_is_undecided_exit_2(tmp_path, monkeypatch):
    from .conftest import MINI_MD
    from .test_agents import FakeLLM

    src = tmp_path / "mini.md"
    # три проверенных требования и много обрывков, не попадающих в требования: покрытие ниже порога
    src.write_text(MINI_MD + "\n" + "\n\n".join(["Да."] * 400) + "\n", encoding="utf-8")
    _use_llm(monkeypatch, FakeLLM({"LLMFacts": [_ok_facts()]}))
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(src), "-o", str(out), "-p", "gemini"])
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert any(i["code"] == "low_coverage" for i in report["issues"])
    assert report["verdict"]["status"] == "undecided"
    assert "low_coverage" in report["verdict"]["headline"] or "low_coverage" in out.read_text(
        encoding="utf-8"
    )
    assert res.exit_code == 2
    assert READY_PHRASE not in out.read_text(encoding="utf-8")


def test_fallback_segmentation_alone_is_not_partial_read():
    from pathlib import Path

    from ravel.models import IngestIssue, IssueLevel
    from ravel.orchestrator import run, undecided_reasons

    # печатный PDF проходит через резервный путь; покрытие нормальное, значит чтение полное
    report = run(Path(__file__).parent / "fixtures" / "requirements_printed.pdf", None)
    assert not any("прочитан частично" in r for r in undecided_reasons(report))
    assert not any(i.level == IssueLevel.warning for i in report.issues)
    # и наоборот: warning с кодом low_coverage или fallback_segmentation чтение частичным считает
    for code in ("low_coverage", "fallback_segmentation"):
        report.issues = [IngestIssue(level=IssueLevel.warning, code=code, message="x")]
        assert any("прочитан частично" in r for r in undecided_reasons(report))


def test_ocr_low_quality_is_fatal_exit_1(tmp_path):
    from ravel.ingest.pdf import _ocr_languages, tessdata_dir

    tessdata = tessdata_dir()
    if tessdata is None or "rus" in _ocr_languages(tessdata).split("+"):
        pytest.skip("нужен Tesseract без tessdata rus")
    from pathlib import Path

    scan = Path(__file__).parent / "fixtures" / "scan.pdf"
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(scan), "-o", str(out), "--provider", "none"])
    assert res.exit_code == 1
    assert "ocr_low_quality" in out.read_text(encoding="utf-8")


def test_cli_counts_are_split_by_kind(sample, tmp_path):
    out = tmp_path / "r.md"
    res = runner.invoke(app, ["check", str(sample), "-o", str(out), "-p", "none"])
    from ravel.ingest import load_document
    from ravel.segment import segment

    reqs = segment(load_document(sample))
    n = {k: sum(r.kind == k for r in reqs) for k in ("requirement", "metadata", "example")}
    assert n["requirement"] and n["metadata"] and n["example"]
    assert (
        f"Требований: {n['requirement']} (метаданных: {n['metadata']}, примеров: {n['example']}),"
        in res.output
    )
    md = out.read_text(encoding="utf-8")
    assert f"| Требований выделено | {n['requirement']} |" in md
    assert f"| Метаданных документа | {n['metadata']} |" in md
    assert f"| Примеров | {n['example']} |" in md
    assert "| E-001 | пример |" in md and "| M-001 | метаданные |" in md
