"""Report → Markdown (шаблон) + JSON + questions.yaml + trace.jsonl.

Все цифры в отчёте считает код из Report — LLM тексты отчёта не пишет.
"""

from __future__ import annotations

from collections import Counter
from importlib.resources import files
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from ravel.fsutil import atomic_write
from ravel.models import CRITERION_LABELS, SEVERITY_LABELS, Question, Report, Severity


def _cell(text: str | None, limit: int = 160) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(report: Report) -> str:
    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["cell"] = _cell
    template = env.from_string(files("ravel").joinpath("report.md.j2").read_text(encoding="utf-8"))

    by_req: dict[str, list] = {r.id: [] for r in report.requirements}
    for f in report.findings:
        if f.scope == "requirement":
            for rid in f.requirement_ids:
                by_req.setdefault(rid, []).append(f)
    set_findings = [f for f in report.findings if f.scope == "set"]
    sev = Counter(f.severity for f in report.findings)
    crit_counter = Counter(f.criterion for f in report.findings)
    reqs_with_defects = len({rid for f in report.findings for rid in f.requirement_ids})

    return template.render(
        r=report,
        p=report.passport,
        by_req=by_req,
        set_findings=set_findings,
        critical=[f for f in report.findings if f.severity == Severity.critical],
        sev={s: sev.get(s, 0) for s in Severity},
        by_criterion=sorted(crit_counter.items(), key=lambda kv: -kv[1]),
        reqs_with_defects=reqs_with_defects,
        CL=CRITERION_LABELS,
        SL=SEVERITY_LABELS,
    )


def write_outputs(report: Report, out_md: Path) -> dict[str, Path]:
    stem = out_md.with_suffix("")
    paths = {
        "report": out_md,
        "json": stem.with_suffix(".json"),
        "questions": Path(f"{stem}.questions.yaml"),
        "trace": Path(f"{stem}.trace.jsonl"),
    }
    atomic_write(paths["json"], report.model_dump_json(indent=2))
    atomic_write(paths["questions"], questions_yaml(report.questions))
    atomic_write(paths["trace"], "".join(s.model_dump_json() + "\n" for s in report.trace))
    atomic_write(paths["report"], render_markdown(report))  # последним: md = «прогон завершён»
    return paths


def questions_yaml(questions: list[Question]) -> str:
    header = (
        "# Уточняющие вопросы к автору требований.\n"
        "# Впишите ответы в поле answer и запустите проверку снова с --answers <этот файл>.\n"
    )
    data = [
        {
            "id": q.id,
            "priority": q.priority.value,
            "question": q.question,
            "why": q.why,
            "options": q.options,
            "requirements": q.requirement_ids,
            "findings": q.finding_ids,
            "answer": q.answer or "",
        }
        for q in questions
    ]
    return header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)


def load_answers(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    out = []
    for item in data:
        if not isinstance(item, dict) or not str(item.get("answer") or "").strip():
            continue
        out.append(
            Question(
                id=str(item.get("id", "")),
                question=str(item.get("question", "")),
                why=str(item.get("why", "")),
                answer=str(item["answer"]).strip(),
            )
        )
    return out
