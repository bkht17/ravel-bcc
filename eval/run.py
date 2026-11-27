"""Прогон оценки одной командой: агент на всех документах eval, затем оценка каждого отчёта.

    uv run python eval/run.py              # без LLM (--provider none)
    uv run python eval/run.py -p gemini    # с LLM; ответы кэшируются в .cache/llm

Отчёты — в reports/eval/<документ>.{md,json,…}, сводка всех оценок — в reports/eval/summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from score import evaluate, load_golden, load_judgments, pct, print_human, rel

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent

# имя → (документ, эталон)
CASES: dict[str, tuple[Path, Path]] = {
    "requirements": (ROOT / "samples" / "requirements.md", EVAL_DIR / "golden.yaml"),
    "library": (EVAL_DIR / "foreign" / "library.md", EVAL_DIR / "foreign" / "library.golden.yaml"),
    "timer": (EVAL_DIR / "foreign" / "timer.md", EVAL_DIR / "foreign" / "timer.golden.yaml"),
    "rooms": (EVAL_DIR / "foreign" / "rooms.md", EVAL_DIR / "foreign" / "rooms.golden.yaml"),
    "converter": (EVAL_DIR / "foreign" / "converter.md", EVAL_DIR / "foreign" / "converter.golden.yaml"),
}
# коды ravel check: 0 — всё проверено, 2 — частично (есть ограничения прогона), 1 — документ не прочитан
OK_CODES = {0, 1, 2}


def run_agent(doc: Path, out_md: Path, provider: str, model: str | None, no_cache: bool) -> int:
    cmd = [sys.executable, "-m", "ravel", "check", str(doc), "-o", str(out_md), "-p", provider]
    if model:
        cmd += ["-m", model]
    if no_cache:
        cmd.append("--no-cache")
    print(f"\n$ ravel check {rel(doc)} -o {rel(out_md)} -p {provider}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Прогнать ravel на эталонных документах и оценить отчёты.")
    ap.add_argument("-p", "--provider", default="none", help="gemini | ollama | none (по умолчанию none)")
    ap.add_argument("-m", "--model", help="имя модели (иначе из .env)")
    ap.add_argument("--no-cache", action="store_true", help="не брать ответы LLM из кэша")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "reports" / "eval", help="каталог отчётов")
    ap.add_argument("--judge", type=Path, default=EVAL_DIR / "judgments.yaml", help="ручная разметка находок")
    ap.add_argument("--only", choices=sorted(CASES), action="append", help="только эти документы")
    ap.add_argument("--score-only", action="store_true", help="не запускать агента, оценить готовые отчёты")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    judge = args.judge if args.judge.exists() else None
    judgments = load_judgments(judge)
    results: dict[str, dict | None] = {}

    for name in args.only or list(CASES):
        doc, golden_path = CASES[name]
        out_md = args.out / f"{name}.md"
        if not args.score_only:
            code = run_agent(doc, out_md, args.provider, args.model, args.no_cache)
            if code == 3:
                print(f"LLM недоступен для провайдера «{args.provider}» — оценка прервана.", file=sys.stderr)
                return 3
            if code not in OK_CODES:
                print(f"ravel завершился с кодом {code} на {rel(doc)} — отчёт не оценён.", file=sys.stderr)
                results[name] = None
                continue
        report = out_md.with_suffix(".json")
        if not report.exists():
            print(f"Нет отчёта {rel(report)} — пропускаю.", file=sys.stderr)
            results[name] = None
            continue
        res = evaluate(report, load_golden(golden_path), judgments)
        res["report"], res["golden"] = rel(report), rel(golden_path)
        print(f"\n{'═' * 30} {name} {'═' * 30}")
        print_human(res, judge and Path(rel(judge)))
        results[name] = res

    print(f"\n{'═' * 30} итог {'═' * 30}")
    print(f"{'документ':<14} {'полнота':<22} {'точность':<20} {'находок':>8} {'вне эталона':>12}")
    for name, res in results.items():
        if res is None:
            print(f"{name:<14} {'не оценён':<22}")
            continue
        rec, prec = res["recall"], res["precision"]
        recall = (
            f"{rec['points']:g}/{rec['max']:g} = {pct(rec['value'])}" if rec["max"] else "— (эталон пуст)"
        )
        precision = f"{prec['points']:g}/{prec['graded']} = {pct(prec['value'])}" if prec["graded"] else "—"
        unj = f" (+{len(prec['ungraded'])}?)" if prec["ungraded"] else ""
        print(
            f"{name:<14} {recall:<22} {precision + unj:<20} {res['findings']:>8} {len(res['outside_golden']):>12}"
        )

    summary = {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provider": args.provider,
        "model": args.model,
        "judge": judge and rel(judge),
        "results": results,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСводка: {rel(summary_path)}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
