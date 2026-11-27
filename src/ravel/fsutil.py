"""Атомарная запись: файл либо записан целиком, либо не тронут."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _stage(path: Path, content: str) -> None:
    """Записать файл целиком и сбросить на диск (без переноса на место)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())


def atomic_write_group(files: dict[Path, str]) -> None:
    """Записать группу файлов: сначала все во временный каталог рядом, затем os.replace каждого.

    Если сбой случился до переноса, временный каталог удаляется, старые файлы не тронуты.
    Файлы группы должны лежать в одном каталоге (один том).
    """
    parents = {p.parent for p in files}
    if len(parents) != 1:
        raise ValueError("файлы группы должны лежать в одном каталоге")
    parent = parents.pop()
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(dir=parent, prefix=".ravel-write."))
    try:
        staged = []
        for i, (path, content) in enumerate(files.items()):
            tmp = tmpdir / f"{i}.tmp"
            _stage(tmp, content)
            staged.append((tmp, path))
        for tmp, path in staged:
            os.replace(tmp, path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
