"""Report → Markdown (шаблон) + JSON + questions.yaml + trace.jsonl.

Все цифры в отчёте считает код из Report — LLM тексты отчёта не пишет.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from importlib.resources import files
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from ravel.fsutil import atomic_write_group
from ravel.models import CRITERION_LABELS, SEVERITY_LABELS, Question, Report, Severity
from ravel.orchestrator import coverage, llm_mode, noncritical_failures, undecided_reasons

# «<» перед буквой, «/», «!» или «?» markdown-просмотрщик примет за HTML-тег и спрячет текст
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")
_SPACES = re.compile(r"[ \t\r\f\v]*\n[ \t\r\f\v\n]*|[ \t\r\f\v]+")

# Ячейка таблицы нарезки: столько символов формулировки, дальше «…» (полный текст — в разделе 3)
CELL_LIMIT = 160


def _escape(text: str, table: bool = False) -> str:
    """Текст документа как данные: «<tag>» не исчезает; в ячейке таблицы «|» не делит ячейку."""
    if table:
        text = text.replace("|", "\\|")
    return _TAG_START.sub(r"\\<", text)


def _one_line(text: str | None) -> str:
    return _SPACES.sub(" ", text or "").strip()


def _cell(text: str | None, limit: int = CELL_LIMIT) -> str:
    """Ячейка таблицы: одна строка, сокращение до limit символов."""
    text = _one_line(text)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return _escape(text, table=True)


def _truncated(text: str | None, limit: int = CELL_LIMIT) -> bool:
    return len(_one_line(text)) > limit


def _inline(text: str | None) -> str:
    """Полная цитата в строке списка: без сокращения, переносы строк — пробелы."""
    return _escape(_one_line(text))


def _blockquote(text: str | None) -> str:
    """Полный текст цитатой markdown с сохранением переносов строк."""
    lines = _escape((text or "").strip()).split("\n")
    return "\n".join(f"> {ln.rstrip()}  " if ln.strip() else ">" for ln in lines).rstrip()


def render_markdown(report: Report, run_id: str | None = None) -> str:
    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["cell"] = _cell
    env.filters["inline"] = _inline
    env.filters["blockquote"] = _blockquote
    env.tests["truncated"] = _truncated
    template = env.from_string(files("ravel").joinpath("report.md.j2").read_text(encoding="utf-8"))

    by_req: dict[str, list] = {r.id: [] for r in report.requirements}
    for f in report.findings:
        if f.scope == "requirement":
            for rid in f.requirement_ids:
                by_req.setdefault(rid, []).append(f)
    set_findings = [f for f in report.findings if f.scope == "set"]
    # находки уровня набора, затрагивающие требование (ссылка на раздел 4)
    set_by_req: dict[str, list] = {}
    for f in set_findings:
        for rid in f.requirement_ids:
            set_by_req.setdefault(rid, []).append(f)
    sev = Counter(f.severity for f in report.findings)
    crit_counter = Counter(f.criterion for f in report.findings)
    # «Требований с нарушениями» — уникальные продуктовые ID: метаданные и примеры не требования
    product_ids = {r.id for r in report.requirements if r.kind == "requirement"}
    reqs_with_defects = len({rid for f in report.findings for rid in f.requirement_ids if rid in product_ids})
    by_id = {r.id: r for r in report.requirements}

    done, total = coverage(report)
    return template.render(
        r=report,
        llm=llm_mode(report),
        reasons=undecided_reasons(report),
        checked_n=done,
        total_n=total,
        set_check_failed=any(m.startswith("set_checker") for m in report.llm_failures),
        p=report.passport,
        by_req=by_req,
        set_by_req=set_by_req,
        run_id=run_id,
        noncritical=noncritical_failures(report),
        set_findings=set_findings,
        critical=[f for f in report.findings if f.severity == Severity.critical],
        sev={s: sev.get(s, 0) for s in Severity},
        by_criterion=sorted(crit_counter.items(), key=lambda kv: -kv[1]),
        reqs_with_defects=reqs_with_defects,
        by_id=by_id,
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


class ReportRenderError(RuntimeError):
    """Отчёт не удалось собрать; на диск ничего не записано."""


def _check_run_id_row(md: str, run_id: str) -> None:
    """Строка «ID прогона» рисуется шаблоном; если её нет в паспорте — файлы не пишем.

    Иначе run_id молча пропал бы из MD, оставшись в JSON/YAML/trace.
    """
    if f"| ID прогона | `{run_id}` |" not in md:
        raise ReportRenderError(
            "в паспорте отчёта нет строки «ID прогона» (после «Версия промптов»); файлы не записаны"
        )


def write_outputs(report: Report, out_md: Path, run_id: str | None = None) -> dict[str, Path]:
    run_id = run_id or str(uuid.uuid4())
    paths = output_paths(out_md)
    # Все четыре содержимых собираем до записи: любая ошибка здесь не трогает целевую папку.
    md = render_markdown(report, run_id)
    _check_run_id_row(md, run_id)
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
        (f"# run_id: {run_id}\n" if run_id else "") + "# Уточняющие вопросы к автору требований.\n"
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
