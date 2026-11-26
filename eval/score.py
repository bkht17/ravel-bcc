"""Оценка отчёта ravel по эталону — по методике аудита жюри.

    uv run python eval/score.py reports/eval/requirements.json
    uv run python eval/score.py reports/eval/library.json --golden eval/foreign/library.golden.yaml
    uv run python eval/score.py reports/eval/requirements.json --judge eval/judgments.yaml --json

Полнота: запись эталона, найденная целиком, даёт свой вес, найденная частично — половину;
итог = набранное / сумма весов. Точность: оценки находок 1 / 0.5 / 0 из файла разметки
(--judge); находка, совпавшая с эталоном, без ручной оценки считается верной.
Формат эталона и методика — в eval/README.md. Код возврата 0 при любом счёте.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = EVAL_DIR / "golden.yaml"

_LINE = re.compile(r"строк[аеи]\s+(\d+)")
_RANGE = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$")
_NORM = str.maketrans(
    {
        "ё": "е",
        "Ё": "Е",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
    }
)
_VALID_SCORES = {0.0, 0.5, 1.0}


def rel(path: Path) -> str:
    """Путь относительно текущего каталога, если файл внутри него."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def norm(text: str) -> str:
    """Нижний регистр, «ё» → «е», дефисы и неразрывные пробелы — к обычным, пробелы схлопнуты."""
    return " ".join(text.translate(_NORM).lower().split())


# --------------------------------------------------------------------------- эталон


@dataclass(frozen=True)
class Term:
    pattern: str
    own_only: bool  # «@» — искать только в словах агента, не в цитатах
    rx: re.Pattern[str]


Alt = tuple[Term, ...]


@dataclass
class Entry:
    id: str
    weight: float
    what: str
    lines: list[int]
    anchors: set[int]
    full: list[Alt]
    partial: list[Alt]
    combine: int  # 0 — нет; N — столько разных partial-альтернатив дают полное совпадение
    note: str = ""


@dataclass
class Golden:
    path: Path
    document: str | None
    tolerance: int
    clean_lines: set[int]
    entries: list[Entry]


def _lines(spec: Any, where: str) -> list[int]:
    out: list[int] = []
    for item in spec or []:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and (m := _RANGE.match(item)):
            lo, hi = int(m.group(1)), int(m.group(2))
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
        else:
            raise ValueError(f"{where}: строка «{item}» — нужно число или диапазон «a-b»")
    return out


def _alts(spec: Any, where: str) -> list[Alt]:
    alts: list[Alt] = []
    for i, alt in enumerate(spec or []):
        if not isinstance(alt, list) or not alt:
            raise ValueError(f"{where}[{i}]: альтернатива — непустой список выражений")
        terms = []
        for raw in alt:
            own = raw.startswith("@")
            pattern = (raw[1:] if own else raw).replace("ё", "е")
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"{where}[{i}]: плохое выражение «{raw}»: {e}") from e
            terms.append(Term(raw, own, rx))
        alts.append(tuple(terms))
    return alts


def load_golden(path: Path) -> Golden:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидается словарь с ключом entries")
    entries: list[Entry] = []
    seen: set[str] = set()
    for raw in data.get("entries") or []:
        eid = str(raw["id"])
        if eid in seen:
            raise ValueError(f"{path}: повтор id {eid}")
        seen.add(eid)
        lines = _lines(raw.get("lines"), f"{eid}.lines")
        full = _alts(raw.get("full"), f"{eid}.full")
        partial = _alts(raw.get("partial"), f"{eid}.partial")
        combine = raw.get("combine", False)
        combine = len(partial) if combine is True else int(combine or 0)
        if combine > len(partial) or (combine and combine < 2):
            raise ValueError(f"{eid}: combine={combine} при {len(partial)} частичных альтернативах")
        if not full and not partial:
            raise ValueError(f"{eid}: нет ни full, ни partial")
        entries.append(
            Entry(
                id=eid,
                weight=float(raw["weight"]),
                what=str(raw.get("what", "")).strip(),
                lines=lines,
                anchors=set(lines) | set(_lines(raw.get("also_lines"), f"{eid}.also_lines")),
                full=full,
                partial=partial,
                combine=combine,
                note=str(raw.get("note", "")).strip(),
            )
        )
    return Golden(
        path=path,
        document=data.get("document"),
        tolerance=int(data.get("tolerance", 3)),
        clean_lines=set(_lines(data.get("clean_lines"), "clean_lines")),
        entries=entries,
    )


# --------------------------------------------------------------------------- находки


@dataclass
class View:
    id: str
    key: str  # ключ ручной разметки: хэш (criterion, первая цитата)
    criterion: str
    title: str
    own: str  # слова агента: title + explanation + suggestion + criterion=<…>
    full_text: str  # own + цитаты
    lines: set[int]
    where: str  # первая ссылка на место, для вывода


def judge_key(finding: dict) -> str:
    quotes = finding.get("quotes") or []
    anchor = quotes[0].get("text", "") if quotes else finding.get("title", "")
    raw = f"{finding.get('criterion', '')}|{norm(anchor)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def view(finding: dict) -> View:
    quotes = finding.get("quotes") or []
    own = norm(
        " ".join(
            [
                finding.get("title", ""),
                finding.get("explanation", ""),
                finding.get("suggestion", ""),
                f"criterion={finding.get('criterion', '')}",
            ]
        )
    )
    quoted = norm(" ".join(q.get("text", "") for q in quotes))
    lines = {int(n) for q in quotes for n in _LINE.findall(q.get("location") or "")}
    where = next((q["location"] for q in quotes if q.get("location")), "")
    return View(
        id=finding.get("id", "?"),
        key=judge_key(finding),
        criterion=finding.get("criterion", ""),
        title=finding.get("title", ""),
        own=own,
        full_text=f"{own} {quoted}",
        lines=lines,
        where=where,
    )


def alt_matches(alt: Alt, v: View) -> bool:
    return all(t.rx.search(v.own if t.own_only else v.full_text) for t in alt)


def anchored(entry: Entry, v: View, tolerance: int) -> bool:
    """Цитата находки стоит рядом с местом дефекта. Нет якорей, нет строк или tolerance < 0 — не проверяем."""
    if tolerance < 0 or not entry.anchors or not v.lines:
        return True
    return any(abs(a - n) <= tolerance for a in entry.anchors for n in v.lines)


# --------------------------------------------------------------------------- оценка


def score_entry(entry: Entry, views: list[View], tolerance: int) -> dict:
    near = [v for v in views if anchored(entry, v, tolerance)]
    full = [v.id for v in near if any(alt_matches(a, v) for a in entry.full)]
    by_alt = [[v.id for v in near if alt_matches(a, v)] for a in entry.partial]
    partial = sorted({fid for ids in by_alt for fid in ids} - set(full))
    covered = sum(1 for ids in by_alt if ids)
    if full:
        status, how = "full", "одной находкой"
    elif entry.combine and covered >= entry.combine:
        status, how = "full", f"частями: {covered} из {len(entry.partial)}"
    elif partial:
        status, how = "partial", ""
    else:
        status, how = "none", ""
    points = {"full": entry.weight, "partial": entry.weight / 2, "none": 0.0}[status]
    return {
        "id": entry.id,
        "weight": entry.weight,
        "status": status,
        "points": points,
        "how": how,
        "full_findings": full,
        "partial_findings": partial,
        "what": entry.what,
    }


def load_judgments(path: Path | None) -> dict[str, float | None]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, float | None] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = value.get("score")
        if value is not None and float(value) not in _VALID_SCORES:
            raise ValueError(f"{path}: {key}: оценка {value} — допустимы 1, 0.5, 0")
        out[str(key)] = None if value is None else float(value)
    return out


def evaluate(
    report_path: Path,
    golden: Golden,
    judgments: dict[str, float | None],
    tolerance: int | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings = report.get("findings") or []
    views = [view(f) for f in findings]
    tol = golden.tolerance if tolerance is None else tolerance
    for e in golden.entries:
        for prefix, w in (weights or {}).items():
            if e.id.startswith(prefix):
                e.weight = w

    rows = [score_entry(e, views, tol) for e in golden.entries]
    matched = {fid for r in rows for fid in r["full_findings"] + r["partial_findings"]}
    got = sum(r["points"] for r in rows)
    total = sum(r["weight"] for r in rows)
    by_weight: dict[str, dict[str, float]] = {}
    for r in rows:
        bucket = by_weight.setdefault(f"{r['weight']:g}", {"points": 0.0, "max": 0.0, "count": 0})
        bucket["points"] += r["points"]
        bucket["max"] += r["weight"]
        bucket["count"] += 1

    def near_clean(v: View) -> bool:
        return any(abs(c - n) <= max(tol, 0) for c in golden.clean_lines for n in v.lines)

    outside = [
        {
            "id": v.id,
            "key": v.key,
            "criterion": v.criterion,
            "title": v.title,
            "where": v.where,
            "on_clean_line": near_clean(v),
        }
        for v in views
        if v.id not in matched
    ]

    graded = []
    for v in views:
        if judgments.get(v.key) is not None:
            graded.append({"id": v.id, "key": v.key, "score": judgments[v.key], "by": "manual"})
        elif v.id in matched:
            graded.append({"id": v.id, "key": v.key, "score": 1.0, "by": "golden"})
        else:
            graded.append({"id": v.id, "key": v.key, "score": None, "by": None, "title": v.title})
    scored = [g for g in graded if g["score"] is not None]
    keys = [v.key for v in views]

    passport = report.get("passport") or {}
    doc = golden.document
    mismatch = bool(doc and passport.get("source_name") and Path(doc).name != passport["source_name"])
    return {
        "report": str(report_path),
        "golden": str(golden.path),
        "document": passport.get("source_name"),
        "document_mismatch": mismatch,
        "provider": passport.get("provider"),
        "model": passport.get("model"),
        "agent_version": passport.get("agent_version"),
        "prompt_version": passport.get("prompt_version"),
        "fatal": bool(report.get("fatal")),
        "tolerance": tol,
        "findings": len(views),
        "recall": {
            "points": got,
            "max": total,
            "value": got / total if total else None,
            "full": sum(r["status"] == "full" for r in rows),
            "partial": sum(r["status"] == "partial" for r in rows),
            "missed": sum(r["status"] == "none" for r in rows),
            "by_weight": by_weight,
        },
        "entries": rows,
        "outside_golden": outside,
        "precision": {
            "points": sum(g["score"] for g in scored),
            "graded": len(scored),
            "value": sum(g["score"] for g in scored) / len(scored) if scored else None,
            "manual": sum(g["by"] == "manual" for g in graded),
            "golden": sum(g["by"] == "golden" for g in graded),
            "ungraded": [g for g in graded if g["score"] is None],
            "key_collisions": sorted({k for k in keys if keys.count(k) > 1}),
        },
        "grades": graded,
    }


# --------------------------------------------------------------------------- вывод


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def print_human(res: dict, judge_path: Path | None) -> None:
    rec, prec = res["recall"], res["precision"]
    print(
        f"Отчёт:  {res['report']}  (находок: {res['findings']}; {res['provider']} {res['model'] or ''})".rstrip()
    )
    print(f"Эталон: {res['golden']}  (записей: {len(res['entries'])}, допуск по строкам ±{res['tolerance']})")
    if res["document_mismatch"]:
        print(f"!  Отчёт по документу «{res['document']}», а эталон — для другого документа")
    if res["fatal"]:
        print("!  Документ не прочитан (fatal): находок нет")
    print()
    mark = {"full": "✓", "partial": "½", "none": "·"}
    for r in res["entries"]:
        hits = r["full_findings"] + [f"{fid}½" for fid in r["partial_findings"]]
        found = ", ".join(hits[:4]) + (" …" if len(hits) > 4 else "") if hits else "—"
        if r["how"].startswith("частями"):
            found += f" ({r['how']})"
        print(
            f"{mark[r['status']]} {r['id']:<5} {r['points']:>4g}/{r['weight']:<4g} {found:<24} {r['what'][:72]}"
        )

    print()
    if rec["max"]:
        print(f"Взвешенная полнота: {rec['points']:g} / {rec['max']:g} = {pct(rec['value'])}")
        groups = ", ".join(
            f"вес {w}: {b['points']:g}/{b['max']:g} ({b['count']} шт.)" for w, b in rec["by_weight"].items()
        )
        print(f"  {groups}")
        print(f"  полностью {rec['full']}, частично {rec['partial']}, не найдено {rec['missed']}")
    else:
        print(f"Эталон пуст: полнота не считается, ожидается 0 находок — найдено {res['findings']}")

    outside = res["outside_golden"]
    print(
        f"\nНаходки вне эталона ({len(outside)} из {res['findings']}) — шум или новый дефект, проверить вручную:"
    )
    for o in outside:
        flag = "  ← чистая строка" if o["on_clean_line"] else ""
        print(f"  {o['id']}  {o['key']}  {o['criterion']:<18} {o['title'][:60]}  [{o['where']}]{flag}")
    if not outside:
        print("  нет")

    src = f"разметка {judge_path}" if judge_path else "без --judge: только совпадения с эталоном"
    if not res["findings"]:
        print(f"\nТочность ({src}): находок нет — оценивать нечего")
        return
    print(
        f"\nТочность ({src}): {prec['points']:g} / {prec['graded']} = {pct(prec['value'])}"
        f"  [вручную {prec['manual']}, по эталону {prec['golden']}, без оценки {len(prec['ungraded'])}]"
    )
    if prec["key_collisions"]:
        print(f"!  Одинаковый ключ у разных находок: {', '.join(prec['key_collisions'])}")
    if prec["ungraded"]:
        target = judge_path or "eval/judgments.yaml"
        print(f"  Не размечены — допишите в {target} оценку 1 / 0.5 / 0:")
        for g in prec["ungraded"]:
            print(f"    {g['key']}:   # {g['id']} {g['title'][:70]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Взвешенная полнота и точность отчёта ravel по эталону.")
    ap.add_argument("report", type=Path, nargs="?", default=Path("reports/eval/requirements.json"))
    ap.add_argument(
        "--golden", type=Path, default=DEFAULT_GOLDEN, help="эталон (по умолчанию eval/golden.yaml)"
    )
    ap.add_argument("--judge", type=Path, help="ручная разметка находок: ключ → 1 / 0.5 / 0")
    ap.add_argument("--json", action="store_true", help="вывести итог в JSON (для сравнения прогонов)")
    ap.add_argument("--tolerance", type=int, help="допуск по строкам (по умолчанию из эталона)")
    ap.add_argument("--no-lines", action="store_true", help="не сверять строки цитат")
    ap.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="ПРЕФИКС=ВЕС",
        help="переопределить вес записей по префиксу id, например K=2 (шкала аудитора со знаменателем 32)",
    )
    args = ap.parse_args(argv)

    try:
        weights = {p: float(w) for p, w in (item.split("=", 1) for item in args.weight)}
        golden = load_golden(args.golden)
        judgments = load_judgments(args.judge)
        tolerance = -1 if args.no_lines else args.tolerance
        res = evaluate(args.report, golden, judgments, tolerance, weights)
        res["report"], res["golden"] = rel(args.report), rel(args.golden)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"score.py: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_human(res, args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
