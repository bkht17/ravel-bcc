"""Полнота агента против эталона: uv run python eval/score.py reports/report.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

GOLDEN = Path(__file__).with_name("golden.yaml")


def haystack(f: dict) -> str:
    parts = [f["title"], f["explanation"], f.get("suggestion", "")] + [q["text"] for q in f["quotes"]]
    return " ".join(parts).lower().replace("ё", "е")


def main(report_path: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    golden = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    texts = {f["id"]: haystack(f) for f in report["findings"]}
    hit_findings: set[str] = set()
    hits = 0
    for g in golden:
        matched = [
            fid for fid, t in texts.items()
            if any(all(w.lower().replace("ё", "е") in t for w in words) for words in g["any_of"])
        ]
        hits += bool(matched)
        hit_findings.update(matched)
        mark = "✓" if matched else "·"
        print(f"{mark} {g['id']} {g['what'][:80]}" + (f"  ← {', '.join(matched[:3])}" if matched else ""))
    total = len(golden)
    print(f"\nПолнота (recall): {hits}/{total} = {hits / total:.0%}")
    extra = len(texts) - len(hit_findings)
    print(f"Находок вне эталона: {extra} из {len(texts)} (проверить вручную: шум или новый дефект)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/report.json")
