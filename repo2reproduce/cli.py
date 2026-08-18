"""Command-line interface: `r2r reproduce <paper_url> <repo_url>`."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from .orchestrator import RunResult, load_pipeline_state, reproduce, resume_run
from .report import render_markdown

app = typer.Typer(
    add_completion=False, help="Repo2Reproduce: paper + code -> grounded reproduction plan."
)
console = Console()


def _slug(url: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", url)
    return s.strip("-")[:60] or "run"


def _emit_result(
    result: RunResult, run_dir: Path, amd: bool = False, execute: bool = False
) -> None:
    """Render the Python-assembled report and surface durable failure artifacts."""
    del amd, execute  # completion was already decided once by the phase runner
    console.print("\n")
    console.rule("[bold]Result")
    if result.cost_usd is not None:
        console.print(f"[dim]Estimated cost: ${result.cost_usd:.4f}[/dim]")

    if result.report is None:
        console.print("[red]No structured report produced.[/red]")
        if result.raw_result_text:
            (run_dir / "raw_result.txt").write_text(result.raw_result_text, "utf-8")
            console.print(f"[yellow]Raw output saved to {run_dir / 'raw_result.txt'}[/yellow]")
        console.print(f"[yellow]Pipeline state: {run_dir / 'pipeline_state.json'}[/yellow]")
        console.print(f"[yellow]Phase artifacts: {run_dir / 'phases'}[/yellow]")
        raise typer.Exit(code=1)

    md = render_markdown(result.report)
    json_name = "draft_report.json" if result.is_error else "report.json"
    md_name = "draft_report.md" if result.is_error else "report.md"
    json_path = run_dir / json_name
    if not json_path.exists():
        json_path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), "utf-8")
    (run_dir / md_name).write_text(md, "utf-8")
    console.print(f"[green]Report written to:[/green] {run_dir / md_name}")
    console.print(f"[green]JSON written to  :[/green] {json_path}\n")
    console.print(md)

    if result.is_error:
        console.print("\n[red]Incomplete run — required phases or outputs are missing.[/red]")
        if result.raw_result_text:
            console.print(f"[red]{result.raw_result_text}[/red]")
        console.print(
            f"[yellow]Completed work is preserved in {run_dir / 'phases'} and "
            f"{run_dir / 'pipeline_state.json'}. The run exits 1.[/yellow]"
        )
        raise typer.Exit(code=1)


@app.command("reproduce")
def reproduce_cmd(
    paper_url: str = typer.Argument(..., help="arXiv/PDF URL of the paper."),
    repo_url: str = typer.Argument(..., help="GitHub URL of the official code."),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Specific experiment to reproduce (e.g. 'Table 2')."
    ),
    run_dir: Optional[Path] = typer.Option(
        None, "--run-dir", help="Where to store artifacts (default: runs/<timestamp>)."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress streamed agent text."),
    stream: bool = typer.Option(False, "--stream", help="Live token streaming of progress."),
    amd: bool = typer.Option(
        False, "--amd", help="Also make the repo run on AMD ROCm GPUs (Track 2)."
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="With --amd: actually build the image and run the "
        "reproduction on the GPU (default: plan + Dockerfile only).",
    ),
):
    """Read the paper + code, choose an experiment, extract config, synthesize commands."""
    load_dotenv()

    if execute and not amd:
        console.print("[yellow]--execute requires --amd; enabling --amd.[/yellow]")
        amd = True

    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{stamp}-{_slug(repo_url)}"

    console.rule("[bold]Repo2Reproduce")
    console.print(f"[cyan]Paper:[/cyan] {paper_url}")
    console.print(f"[cyan]Code :[/cyan] {repo_url}")
    console.print(f"[cyan]Run  :[/cyan] {run_dir}")
    if amd:
        console.print(f"[cyan]Mode :[/cyan] AMD ROCm ({'execute' if execute else 'plan-only'})")
    console.print("")

    def _on_text(text: str):
        if not quiet and text.strip():
            console.print(text, end="")

    result = asyncio.run(
        reproduce(
            paper_url,
            repo_url,
            run_dir,
            target=target,
            on_text=_on_text,
            stream=stream,
            amd=amd,
            execute=execute,
        )
    )
    _emit_result(result, run_dir, amd=amd, execute=execute)


@app.command("metrics")
def metrics_cmd(
    runs_dir: Path = typer.Argument(
        Path("runs"), help="Directory holding run subdirs, each with a report.json."
    ),
    references: Optional[Path] = typer.Option(
        None,
        "--references",
        "-r",
        help="JSON file mapping run-dir name -> reference metric (P_ref), to pin the "
        "benchmark baseline instead of parsing each report's paper_target.",
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out", help="Also write the full metrics breakdown to this JSON file."
    ),
):
    """Compute Exec Rate and Performance Gap (↓) over a set of runs."""
    from .metrics import evaluate_runs

    if not runs_dir.is_dir():
        console.print(f"[red]Runs dir not found: {runs_dir}[/red]")
        raise typer.Exit(code=1)

    refs: dict[str, float] = {}
    if references is not None:
        try:
            refs = {k: float(v) for k, v in json.loads(references.read_text("utf-8")).items()}
        except (OSError, json.JSONDecodeError, ValueError) as e:
            console.print(f"[red]Could not read references file {references}: {e}[/red]")
            raise typer.Exit(code=1)

    summary = evaluate_runs(runs_dir, references=refs)
    if summary.n == 0:
        console.print(f"[yellow]No */report.json found under {runs_dir}.[/yellow]")
        raise typer.Exit(code=1)

    from rich.table import Table

    table = Table(title=f"Reproduction metrics — {runs_dir}")
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("executed", justify="center")
    table.add_column("P_ref", justify="right")
    table.add_column("P_agent", justify="right")
    table.add_column("gap ↓", justify="right")
    table.add_column("note", style="dim")
    for r in summary.runs:
        table.add_row(
            r.name,
            "[green]yes[/green]" if r.executed else "[red]no[/red]",
            "-" if r.ref_value is None else f"{r.ref_value:g}",
            "-" if r.agent_value is None else f"{r.agent_value:g}",
            f"{r.gap:.4f}",
            r.note or "",
        )
    console.print(table)

    console.rule("[bold]Summary")
    console.print(f"[bold]Runs           :[/bold] {summary.n}")
    console.print(f"[bold]Executed       :[/bold] {summary.n_executed}")
    console.print(f"[bold]Exec Rate      :[/bold] {summary.exec_rate * 100:.2f}%")
    console.print(f"[bold]Perf Gap (↓)   :[/bold] {summary.perf_gap * 100:.2f}%  (all runs)")
    console.print(
        f"[bold]Perf Gap (exec):[/bold] {summary.perf_gap_executed_only * 100:.2f}%  "
        f"(executed only)"
    )

    if json_out is not None:
        json_out.write_text(json.dumps(summary.to_dict(), indent=2), "utf-8")
        console.print(f"[green]Metrics JSON written to:[/green] {json_out}")


@app.command("seed-memory")
def seed_memory_cmd():
    """Seed the persistent memory with curated AMD/ROCm porting knowledge (idempotent)."""
    load_dotenv()
    from .amd_knowledge import seed_amd_memory

    res = seed_amd_memory()
    console.print(f"[green]AMD memory seeded:[/green] {res}")


@app.command("resume")
def resume_cmd(
    run_dir: Path = typer.Argument(
        ..., help="A prior run dir containing pipeline_state.json and phase artifacts."
    ),
    instruction: str = typer.Argument(
        ..., help="Additional guidance for failed or unfinished phases."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress streamed agent text."),
    stream: bool = typer.Option(False, "--stream", help="Live token streaming of progress."),
):
    """Resume failed/missing phases from durable artifacts."""
    load_dotenv()
    if load_pipeline_state(run_dir) is None:
        console.print(f"[red]No pipeline_state.json in {run_dir}. Run `r2r reproduce` first.[/red]")
        raise typer.Exit(code=1)

    console.rule("[bold]Repo2Reproduce — resume")
    console.print(f"[cyan]Run :[/cyan] {run_dir}")
    console.print(f"[cyan]Ask :[/cyan] {instruction}\n")

    def _on_text(text: str):
        if not quiet and text.strip():
            console.print(text, end="")

    result = asyncio.run(resume_run(run_dir, instruction, on_text=_on_text, stream=stream))
    _emit_result(result, run_dir)


# Allow both `r2r reproduce ...` and a bare `r2r ...` invocation.
@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
