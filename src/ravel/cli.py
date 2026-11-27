"""CLI: ravel check <требования> -o <отчёт.md>"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from ravel import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Агент статической проверки требований.")


def _check_output_paths(src: Path, outs: dict[str, Path]) -> str | None:
    """Вернуть текст ошибки, если выходы опасны (совпадают с входом, каталог, дубли), иначе None."""
    src_res = src.resolve()
    seen: dict[Path, Path] = {}
    for out in outs.values():
        res = out.resolve()
        if out.is_dir():
            return f"Выходной путь — каталог: {out}"
        same = res == src_res
        if not same and out.exists() and src.exists():
            try:
                same = os.path.samefile(out, src)
            except OSError:
                same = False
        if same:
            return f"Выходной файл совпадает с входным: {out}"
        if res in seen:
            return f"Выходные файлы совпадают между собой: {seen[res]} и {out}"
        seen[res] = out
    return None


@app.callback()
def main() -> None:
    load_dotenv()


@app.command()
def check(
    input: Annotated[Path, typer.Argument(help="Файл требований: .pdf, .docx, .md, .txt")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Куда записать отчёт (.md)")] = Path(
        "reports/report.md"
    ),
    provider: Annotated[
        str, typer.Option("-p", "--provider", envvar="RAVEL_PROVIDER", help="gemini | ollama | none")
    ] = "gemini",
    model: Annotated[str | None, typer.Option("-m", "--model", help="Имя модели (иначе из .env)")] = None,
    answers: Annotated[
        Path | None, typer.Option("--answers", help="questions.yaml с ответами автора прошлого прогона")
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("-i", "--interactive", help="Задать уточняющие вопросы в терминале и перепроверить"),
    ] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Не использовать кэш ответов LLM")] = False,
) -> None:
    """Проверить документ требований и записать отчёт."""
    from ravel import orchestrator
    from ravel.llm import LLMUnavailable, get_client
    from ravel.render import ReportRenderError, load_answers, output_paths, write_outputs

    # Проверка путей — до get_client, чтения документа и любых обращений к LLM.
    bad = _check_output_paths(input, output_paths(output))
    if bad:
        typer.secho(bad, fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    run_id = str(uuid.uuid4())

    try:
        llm = get_client(provider, model)
        if llm is not None and no_cache:
            llm.cache_dir = None
    except LLMUnavailable as e:
        typer.secho(f"LLM недоступен: {e}", fg=typer.colors.RED, err=True)
        typer.secho("Запустите с --provider none, чтобы получить отчёт только по правилам.", err=True)
        raise typer.Exit(3) from e

    prior = load_answers(answers) if answers else []
    if llm is not None:
        llm.on_event = lambda msg: typer.secho(f"      {msg}", fg=typer.colors.BRIGHT_BLACK, err=True)
    step = lambda name: typer.secho(f"  → {name}", fg=typer.colors.BRIGHT_BLACK, err=True)  # noqa: E731

    typer.secho(f"ravel {__version__}: {input} [{llm.label if llm else 'без LLM'}]", err=True)
    report = orchestrator.run(input, llm, prior, on_step=step)

    if interactive and report.questions and not report.fatal:
        typer.secho(
            f"\nУ агента {len(report.questions)} уточняющих вопросов. Enter — пропустить.\n", err=True
        )
        for q in report.questions:
            typer.secho(f"{q.id} [{q.priority.value}] {q.question}", bold=True, err=True)
            typer.secho(f"   зачем: {q.why}", err=True)
            for k, opt in enumerate(q.options, start=1):
                typer.secho(f"   {k}) {opt}", err=True)
            ans = typer.prompt("   ответ", default="", show_default=False).strip()
            if ans.isdigit() and 1 <= int(ans) <= len(q.options):
                ans = q.options[int(ans) - 1]
            if ans:
                q.answer = ans
        given = [q for q in report.questions if q.answer]
        if given:
            typer.secho(f"\nПерепроверяю с учётом {len(given)} ответов…", err=True)
            report = orchestrator.run(input, llm, prior + given, on_step=step)

    try:
        paths = write_outputs(report, output, run_id)
    except (ReportRenderError, OSError) as e:
        typer.secho(f"Отчёт не записан: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    code = orchestrator.exit_code(report)
    color = {0: typer.colors.GREEN, 2: typer.colors.YELLOW}.get(code, typer.colors.RED)
    n_req = sum(r.kind == "requirement" for r in report.requirements)
    n_meta = sum(r.kind == "metadata" for r in report.requirements)
    n_ex = sum(r.kind == "example" for r in report.requirements)
    typer.secho(
        f"\nТребований: {n_req} (метаданных: {n_meta}, примеров: {n_ex}), нарушений: {len(report.findings)}, "
        f"вопросов: {len(report.questions)}",
        err=True,
    )
    typer.secho(f"Отчёт: {paths['report']}", fg=color, err=True)
    typer.secho(f"Вопросы: {paths['questions']}  JSON: {paths['json']}  Трасса: {paths['trace']}", err=True)
    raise typer.Exit(code)


@app.command()
def version() -> None:
    """Показать версию агента."""
    from ravel.orchestrator import agent_version

    typer.echo(agent_version())


if __name__ == "__main__":
    app()
