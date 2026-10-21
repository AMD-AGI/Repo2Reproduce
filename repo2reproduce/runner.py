"""Claude Agent SDK entry point for the staged reproduction pipeline."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, SystemMessage, query
from dotenv import load_dotenv

from repo2reproduce.porting.official_hipify import HipifyTool, ensure_hipify
from repo2reproduce.provider import ensure_runtime_path, provider_env

from . import config
from .schemas import ExecutionResult, FinalReport, TargetContract
from .validators import validate_final_report

PACKAGE_DIR = Path(__file__).resolve().parent
FINALIZATION_TOOLS = ["Read", "Glob", "Grep"]


def load_project_environment() -> None:
    """Load provider credentials from the repository's configured environment file."""

    load_dotenv(PACKAGE_DIR.parent / ".env", override=False)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _slug_from_repo(repo_url: str) -> str:
    name = Path(urlparse(repo_url).path.rstrip("/")).name.removesuffix(".git")
    rendered = "".join(character.lower() if character.isalnum() else "-" for character in name)
    return rendered.strip("-") or "repository"


def _default_run_id(repo_url: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"two-stage-{timestamp}-{_slug_from_repo(repo_url)}"


def _render_prompt(name: str, replacements: dict[str, str]) -> str:
    text = (PACKAGE_DIR / "prompts" / name).read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _append_message(path: Path, stage: str, message: Any) -> None:
    event = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "message": _jsonable(message),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def porting_policy_prompt(workspace: str, *, force_kernel_port: bool) -> str:
    """Render the CUDA-to-ROCm policy appended to environment preparation."""

    requirement = (
        "FORCED KERNEL PORT: The selected result must execute the paper's CUDA kernel "
        "contribution after porting it with `hipify_with_fixes` or `targeted_rewrite`. "
        "Every CUDA kernel the selected entrypoint launches, including preprocessing such "
        "as quantization, must run as a ported HIP, Triton, Composable Kernel, or rocWMMA "
        "kernel listed in `rocm_port.ported_kernels`. A PyTorch fallback for any of those "
        "kernels, an existing ROCm fork, OpenCL implementation, the repository's Triton "
        "path, a CPU reference, or disabling the extension does not satisfy the run. You "
        "may use such implementations only as parity references or the full-precision "
        "baseline. Persist every semantic source change "
        "in the listed patch/edit-script files. A blocked result is valid only after both "
        "`hipify_with_fixes` and `targeted_rewrite` were actually attempted."
        if force_kernel_port
        else "STANDARD PORTING: stop at the first strategy that produces the selected result."
    )
    return _render_prompt(
        "rocm_porting.md",
        {"WORKSPACE": workspace, "PORTING_REQUIREMENT": requirement},
    )


def _install_porting_commands(run_dir: Path) -> HipifyTool:
    """Fetch the pinned official HIPIFY and install ``port-scan`` and ``port-hipify``."""

    tool = ensure_hipify(run_dir / "tools")
    bin_dir = run_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in (("port-scan", "scan_main"), ("port-hipify", "hipify_main")):
        path = bin_dir / name
        path.write_text(
            f"#!{sys.executable}\n"
            "from repo2reproduce.porting_commands import "
            f"{entry}\n"
            f"raise SystemExit({entry}())\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
    return tool


def initialize_run(
    run_dir: Path,
    *,
    run_id: str,
    paper_url: str,
    repo_url: str,
    assigned_gpu: str,
    timeout_seconds: int,
    target: str | None,
    rocm_porting: bool = False,
    force_kernel_port: bool = False,
) -> None:
    """Create the isolated run directory and metadata."""

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")

    workspace = run_dir / "workspace"
    hook_dir = workspace / ".claude" / "hooks"
    for path in (
        workspace,
        hook_dir,
        run_dir / "paper",
        run_dir / "bin",
        run_dir / "logs",
        run_dir / "artifacts",
        run_dir / "cache",
        run_dir / "prompts",
        run_dir / "stage_attempts",
    ):
        path.mkdir(parents=True, exist_ok=True)

    wrapper = run_dir / "bin" / "docker"
    shutil.copy2(PACKAGE_DIR / "templates" / "docker-wrapper", wrapper)
    wrapper.chmod(0o755)

    hook = hook_dir / "enforce_bash.py"
    shutil.copy2(PACKAGE_DIR / "templates" / "enforce_bash.py", hook)
    hook.chmod(0o755)
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": f"python3 {hook}"}],
                }
            ]
        }
    }
    _atomic_json(workspace / ".claude" / "settings.json", settings)

    hipify_tool: dict[str, str] | None = None
    if rocm_porting:
        hipify_tool = dataclasses.asdict(_install_porting_commands(run_dir))
    _atomic_json(
        run_dir / "input.json",
        {
            "paper_url": paper_url,
            "repo_url": repo_url,
            "timeout_seconds": timeout_seconds,
            "target": target,
            "rocm_porting": rocm_porting,
            "force_kernel_port": force_kernel_port,
        },
    )
    _atomic_json(
        run_dir / "metadata.json",
        {
            "schema_version": 2,
            "run_id": run_id,
            "paper_url": paper_url,
            "repo_url": repo_url,
            "workspace": str(workspace),
            "assigned_physical_gpu": assigned_gpu,
            "container_visible_gpu": 0,
            "timeout_seconds": timeout_seconds,
            "target": target,
            "rocm_porting": rocm_porting,
            "force_kernel_port": force_kernel_port,
            "hipify": hipify_tool,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "initialized",
            "stages": {},
        },
    )


def _agent_env(run_dir: Path, assigned_gpu: str) -> dict[str, str]:
    env = {
        "RUN_DIR": str(run_dir),
        "RUN_ID": run_dir.name,
        "ASSIGNED_GPU": str(assigned_gpu),
        "PATH": f"{run_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    env.update(provider_env())
    return env


async def _invoke_agent(
    *,
    prompt: str,
    options: ClaudeAgentOptions,
    log_path: Path,
    stage: str,
) -> tuple[ResultMessage, str | None]:
    result: ResultMessage | None = None
    session_id: str | None = None
    try:
        async for message in query(prompt=prompt, options=options):
            _append_message(log_path, stage, message)
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
            if isinstance(message, ResultMessage):
                result = message
                session_id = message.session_id or session_id
    except Exception:
        if result is None:
            raise
    if result is None:
        raise RuntimeError(f"{stage} agent returned no ResultMessage")
    return result, session_id


def _base_options(
    run_dir: Path,
    *,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> dict[str, Any]:
    return {
        "system_prompt": {"type": "preset", "preset": "claude_code"},
        "cwd": str(run_dir / "workspace"),
        "add_dirs": [str(run_dir)],
        "setting_sources": ["project"],
        "strict_mcp_config": True,
        "permission_mode": "dontAsk",
        "max_turns": max_turns,
        "max_budget_usd": max_budget_usd,
        "model": model,
        "env": _agent_env(run_dir, assigned_gpu),
    }


async def finalize_session(
    run_dir: Path,
    *,
    session_id: str | None,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    execution_result: ExecutionResult,
) -> str:
    """Create a stateless evidence-only report.

    ``session_id`` is accepted for backward API compatibility but deliberately ignored:
    recovery must not depend on the preparation session still being resumable.
    """

    del session_id
    prompt = _render_prompt(
        "finalize.md",
        {
            "TARGET_PATH": str(run_dir / "target.json"),
            "AUDIT_PATH": str(run_dir / "configuration_audit.json"),
            "ENVIRONMENT_PATH": str(run_dir / "environment_manifest.json"),
            "MANIFEST_PATH": str(run_dir / "execution_manifest.json"),
            "RESULT_PATH": str(run_dir / "execution_result.json"),
            "LOG_PATH": execution_result.log_path,
            "ARTIFACTS_PATH": str(run_dir / "artifacts"),
        },
    )
    (run_dir / "finalize_prompt.txt").write_text(prompt, encoding="utf-8")
    options_data = _base_options(
        run_dir,
        assigned_gpu=assigned_gpu,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )
    options_data.update(
        {
            "allowed_tools": FINALIZATION_TOOLS,
            "output_format": {
                "type": "json_schema",
                "schema": FinalReport.model_json_schema(),
            },
        }
    )
    result, _ = await _invoke_agent(
        prompt=prompt,
        options=ClaudeAgentOptions(**options_data),
        log_path=run_dir / "logs" / "agent_messages.jsonl",
        stage="finalize",
    )
    if result.is_error:
        detail = result.result or ", ".join(result.errors or []) or "unknown SDK error"
        raise RuntimeError(f"finalization failed: {detail}")
    if result.subtype != "success" or result.structured_output is None:
        raise RuntimeError("finalization returned no structured report")
    report = FinalReport.model_validate(result.structured_output)
    target = TargetContract.model_validate_json(
        (run_dir / "target.json").read_text(encoding="utf-8")
    )
    report.selected_experiment = target.selected_experiment
    report.execution_status = execution_result.status
    report.metric_name = target.metric_name
    report.paper_target_value = target.metric_value
    report.achieved_value = execution_result.achieved_metric
    validate_final_report(target, execution_result, report)
    _atomic_json(run_dir / "final_report.json", report.model_dump(mode="json"))

    match = "not determined" if report.match is None else ("yes" if report.match else "no")
    markdown = "\n".join(
        [
            f"# Reproduction report: {report.selected_experiment}",
            "",
            f"- **Execution:** {report.execution_status}",
            f"- **Metric:** {report.metric_name}",
            f"- **Paper target:** {report.paper_target_value}",
            f"- **Achieved:** {report.achieved_value or 'not produced'}",
            f"- **Comparable:** {'yes' if report.comparable else 'no'}",
            f"- **Match:** {match}",
            "",
            "## Comparison",
            report.comparison_explanation,
            "",
            "## Assumptions",
            *([f"- {item}" for item in report.assumptions] or ["- None reported."]),
            "",
            "## Gaps",
            *([f"- {item}" for item in report.gaps] or ["- None reported."]),
            "",
        ]
    )
    (run_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    return markdown


def _update_metadata(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    _atomic_json(path, data)


def _preserve_previous_result(run_dir: Path) -> None:
    """Keep the prior blocked result/report before a recovery attempt overwrites them."""

    for relative in ("execution_result.json", "final_report.md", "logs/executor.log"):
        source = run_dir / relative
        if not source.exists():
            continue
        destination = source.with_name(f"{source.stem}.pre-recovery{source.suffix}")
        if not destination.exists():
            shutil.copy2(source, destination)


async def recover_existing_run(args: argparse.Namespace) -> int:
    """Resume from the first incomplete or invalid durable stage."""

    ensure_runtime_path()
    run_dir = args.recover_run.resolve()
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"recovery metadata is missing: {metadata_path}")
    if not (run_dir / "input.json").is_file():
        raise FileNotFoundError(f"recovery input is missing: {run_dir / 'input.json'}")

    _update_metadata(
        run_dir,
        status="recovering",
        recovery_started_at=datetime.now(timezone.utc).isoformat(),
        recovery_error=None,
    )
    from .pipeline import run_staged_pipeline

    try:
        return await run_staged_pipeline(run_dir, args, recovering=True)
    except Exception as exc:
        _update_metadata(
            run_dir,
            status="recovery_failed",
            recovery_finished_at=datetime.now(timezone.utc).isoformat(),
            recovery_error=f"{type(exc).__name__}: {exc}",
        )
        raise


async def run(args: argparse.Namespace) -> int:
    ensure_runtime_path()
    assigned_gpu = args.gpu or "0"
    run_id = args.run_id or _default_run_id(args.repo_url)
    run_dir = args.runs_root.resolve() / run_id
    initialize_run(
        run_dir,
        run_id=run_id,
        paper_url=args.paper_url,
        repo_url=args.repo_url,
        assigned_gpu=assigned_gpu,
        timeout_seconds=args.timeout_seconds,
        target=args.target,
        rocm_porting=args.rocm_porting,
        force_kernel_port=args.force_kernel_port,
    )
    from .pipeline import run_staged_pipeline

    return await run_staged_pipeline(run_dir, args, recovering=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="r2r", description=__doc__)
    parser.add_argument("paper_url", nargs="?")
    parser.add_argument("repo_url", nargs="?")
    parser.add_argument(
        "--recover-run",
        type=Path,
        help="Resume an existing run from its first incomplete durable stage",
    )
    parser.add_argument("--target")
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", type=Path, default=config.runs_root())
    parser.add_argument("--gpu", help="Physical AMD GPU index (defaults to 0 or saved value)")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=config.max_turns())
    parser.add_argument("--max-budget-usd", type=float, default=config.max_budget_usd())
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-evidence-validation",
        action="store_true",
        help=(
            "Skip optional paper/code evidence diagnostics; scientific gaps remain "
            "recorded by the agents"
        ),
    )
    parser.add_argument(
        "--rocm-porting",
        action="store_true",
        help=(
            "Require CUDA-to-ROCm porting attempts before a CUDA-related block and a "
            "numerical parity check for ported code. Installs host port-scan and port-hipify."
        ),
    )
    parser.add_argument(
        "--force-kernel-port",
        action="store_true",
        help=(
            "Require the selected CUDA kernel contribution itself to run via "
            "hipify_with_fixes or targeted_rewrite; requires --rocm-porting"
        ),
    )
    return parser


def main() -> None:
    load_project_environment()
    parser = build_parser()
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.max_turns <= 0:
        raise SystemExit("--max-turns must be positive")
    if args.max_budget_usd <= 0:
        raise SystemExit("--max-budget-usd must be positive")
    if args.force_kernel_port and not args.rocm_porting:
        parser.error("--force-kernel-port requires --rocm-porting")
    if (args.rocm_porting or args.force_kernel_port) and args.recover_run:
        parser.error(
            "porting modes are fixed when a run is created; recovered runs reuse the saved mode"
        )
    if args.recover_run:
        if args.paper_url or args.repo_url:
            parser.error("paper_url and repo_url cannot be used with --recover-run")
        operation = recover_existing_run(args)
    else:
        if not args.paper_url or not args.repo_url:
            parser.error("paper_url and repo_url are required unless --recover-run is used")
        operation = run(args)
    raise SystemExit(asyncio.run(operation))


if __name__ == "__main__":
    main()
