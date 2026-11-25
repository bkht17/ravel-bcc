from __future__ import annotations

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
    assert res.exit_code == 0, res.output
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
    assert res.exit_code == 0
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
