"""Report → Markdown (шаблон) + JSON + questions.yaml + trace.jsonl.

Все цифры в отчёте считает код из Report — LLM тексты отчёта не пишет.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from importlib.resources import files
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from ravel.fsutil import atomic_write_group
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


def output_paths(out_md: Path) -> dict[str, Path]:
    stem = out_md.with_suffix("")
    return {
        "report": out_md,
        "json": stem.with_suffix(".json"),
        "questions": Path(f"{stem}.questions.yaml"),
        "trace": Path(f"{stem}.trace.jsonl"),
    }


_PASSPORT_ANCHOR = "| Версия промптов |"


class ReportRenderError(RuntimeError):
    """Отчёт не удалось собрать; на диск ничего не записано."""


def _with_run_id_row(md: str, run_id: str) -> str:
    """Вставить строку «ID прогона» в паспорт (после «Версия промптов»), не трогая шаблон.

    Если строки паспорта нет — ReportRenderError: иначе run_id молча пропал бы из MD,
    оставшись в JSON/YAML/trace.
    """
    lines = md.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(_PASSPORT_ANCHOR):
            if not line.endswith("\n"):
                lines[i] = line + "\n"
            lines.insert(i + 1, f"| ID прогона | `{run_id}` |\n")
            return "".join(lines)
    raise ReportRenderError(
        "в паспорте отчёта нет строки «Версия промптов» — некуда вставить «ID прогона»; файлы не записаны"
    )


def write_outputs(report: Report, out_md: Path, run_id: str | None = None) -> dict[str, Path]:
    run_id = run_id or str(uuid.uuid4())
    paths = output_paths(out_md)
    # Все четыре содержимых собираем до записи: любая ошибка здесь не трогает целевую папку.
    md = _with_run_id_row(render_markdown(report), run_id)
    data = {"run_id": run_id, **json.loads(report.model_dump_json())}
    trace = json.dumps({"run_id": run_id}) + "\n" + "".join(s.model_dump_json() + "\n" for s in report.trace)
    contents = {
        paths["json"]: json.dumps(data, ensure_ascii=False, indent=2),
        paths["questions"]: questions_yaml(report.questions, run_id),
        paths["trace"]: trace,
        paths["report"]: md,  # последним: md = «прогон завершён»
    }
    atomic_write_group(contents)
    return paths


def questions_yaml(questions: list[Question], run_id: str | None = None) -> str:
    header = (
        (f"# run_id: {run_id}\n" if run_id else "")
        + "# Уточняющие вопросы к автору требований.\n"
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
