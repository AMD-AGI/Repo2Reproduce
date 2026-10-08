"""Claude Agent SDK orchestration for the two-stage reproduction baseline."""

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
from pydantic import ValidationError

from repo2reproduce.provider import ensure_runtime_path, provider_env, reasoning_model

from .executor import execute_manifest
from .schemas import ExecutionManifest, ExecutionResult

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = Path("/shared_nfs/machita/claude-code-two-stage-runs")
PREPARATION_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch"]
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


def initialize_run(
    run_dir: Path,
    *,
    run_id: str,
    paper_url: str,
    repo_url: str,
    assigned_gpu: str,
    timeout_seconds: int,
    target: str | None,
) -> str:
    """Create the isolated directory layout and preparation prompt."""

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")

    workspace = run_dir / "workspace"
    hook_dir = workspace / ".claude" / "hooks"
    for path in (
        workspace,
        hook_dir,
        run_dir / "bin",
        run_dir / "logs",
        run_dir / "artifacts",
        run_dir / "cache",
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

    _atomic_json(
        run_dir / "input.json",
        {
            "paper_url": paper_url,
            "repo_url": repo_url,
            "timeout_seconds": timeout_seconds,
            "target": target,
        },
    )
    prompt = _render_prompt(
        "prepare.md",
        {
            "PAPER_URL": paper_url,
            "REPO_URL": repo_url,
            "WORKSPACE": str(workspace),
            "TIMEOUT_SECONDS": str(timeout_seconds),
            "TARGET": target or "(choose the cheapest representative numerical result)",
        },
    )
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _atomic_json(
        run_dir / "metadata.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "paper_url": paper_url,
            "repo_url": repo_url,
            "workspace": str(workspace),
            "assigned_physical_gpu": assigned_gpu,
            "container_visible_gpu": 0,
            "timeout_seconds": timeout_seconds,
            "target": target,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "initialized",
        },
    )
    return prompt


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


async def prepare_manifest(
    run_dir: Path,
    *,
    prompt: str,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> tuple[ExecutionManifest, str]:
    """Run preparation and allow one semantic-manifest correction."""

    log_path = run_dir / "logs" / "agent_messages.jsonl"
    session_id: str | None = None
    correction: str | None = None
    validation_error: Exception | None = None

    for attempt in range(2):
        options_data = _base_options(
            run_dir,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
        )
        options_data.update(
            {
                "allowed_tools": PREPARATION_TOOLS,
                "output_format": {
                    "type": "json_schema",
                    "schema": ExecutionManifest.model_json_schema(),
                },
            }
        )
        if session_id:
            options_data["resume"] = session_id

        result, observed_session = await _invoke_agent(
            prompt=correction or prompt,
            options=ClaudeAgentOptions(**options_data),
            log_path=log_path,
            stage=f"prepare-{attempt + 1}",
        )
        session_id = observed_session or session_id
        if not session_id:
            raise RuntimeError("preparation agent returned no session ID")
        if result.is_error:
            detail = result.result or ", ".join(result.errors or []) or "unknown SDK error"
            terminal_reason = result.terminal_reason or result.stop_reason or "unknown"
            raise RuntimeError(
                f"preparation agent failed ({terminal_reason}): {detail}"
            )
        if result.subtype != "success" or result.structured_output is None:
            validation_error = RuntimeError(
                f"preparation ended without structured output: {result.subtype}"
            )
        else:
            try:
                manifest = ExecutionManifest.model_validate(result.structured_output)
                return manifest, session_id
            except ValidationError as exc:
                validation_error = exc

        correction = (
            "Your previous structured execution manifest failed deterministic validation. "
            f"Correct only the manifest; do not run the full experiment. Error: {validation_error}"
        )

    raise RuntimeError(f"agent did not produce a valid execution manifest: {validation_error}")


async def finalize_session(
    run_dir: Path,
    *,
    session_id: str,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    execution_result: ExecutionResult,
) -> str:
    prompt = _render_prompt(
        "finalize.md",
        {
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
            "resume": session_id,
        }
    )
    result, _ = await _invoke_agent(
        prompt=prompt,
        options=ClaudeAgentOptions(**options_data),
        log_path=run_dir / "logs" / "agent_messages.jsonl",
        stage="finalize",
    )
    report = result.result or ""
    (run_dir / "final_report.md").write_text(report, encoding="utf-8")
    return report


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
    """Execute an existing ready manifest and resume its saved Claude session."""

    ensure_runtime_path()
    run_dir = args.recover_run.resolve()
    metadata_path = run_dir / "metadata.json"
    manifest_path = run_dir / "execution_manifest.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"recovery metadata is missing: {metadata_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"recovery manifest is missing: {manifest_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = ExecutionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.status != "ready":
        raise RuntimeError("only a ready execution manifest can be recovered")
    session_id = metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("recovery metadata contains no resumable Claude session ID")

    assigned_gpu = args.gpu or str(metadata.get("assigned_physical_gpu", "0"))
    model = args.model or str(metadata.get("model") or reasoning_model())
    _preserve_previous_result(run_dir)
    _update_metadata(
        run_dir,
        status="recovery_executing",
        recovery_started_at=datetime.now(timezone.utc).isoformat(),
        recovery_error=None,
    )

    try:
        execution_result = await asyncio.to_thread(
            execute_manifest,
            run_dir,
            manifest,
            assigned_gpu=assigned_gpu,
        )
        _update_metadata(run_dir, status=f"recovery_{execution_result.status}")
        await finalize_session(
            run_dir,
            session_id=session_id,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
            execution_result=execution_result,
        )
        _update_metadata(
            run_dir,
            status="completed",
            execution_status=execution_result.status,
            recovered=True,
            recovery_finished_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"Recovered run directory: {run_dir}")
        print(f"Execution status: {execution_result.status}")
        print(f"Final report: {run_dir / 'final_report.md'}")
        return 0 if execution_result.status == "succeeded" else 1
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
    prompt = initialize_run(
        run_dir,
        run_id=run_id,
        paper_url=args.paper_url,
        repo_url=args.repo_url,
        assigned_gpu=assigned_gpu,
        timeout_seconds=args.timeout_seconds,
        target=args.target,
    )
    model = args.model or reasoning_model()

    try:
        manifest, session_id = await prepare_manifest(
            run_dir,
            prompt=prompt,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
        )
        _atomic_json(
            run_dir / "execution_manifest.json",
            manifest.model_dump(mode="json"),
        )
        _update_metadata(
            run_dir,
            status="prepared",
            session_id=session_id,
            model=model,
            manifest_status=manifest.status,
        )
        if manifest.status == "blocked":
            _update_metadata(
                run_dir,
                status="blocked",
                finished_at=datetime.now(timezone.utc).isoformat(),
                blocked_reason=manifest.blocked_reason,
            )
            print(f"Preparation blocked: {manifest.blocked_reason}", file=sys.stderr)
            return 2
        if args.prepare_only:
            print(f"Prepared manifest: {run_dir / 'execution_manifest.json'}")
            return 0

        execution_result = await asyncio.to_thread(
            execute_manifest,
            run_dir,
            manifest,
            assigned_gpu=assigned_gpu,
        )
        _update_metadata(run_dir, status=f"execution_{execution_result.status}")
        await finalize_session(
            run_dir,
            session_id=session_id,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
            execution_result=execution_result,
        )
        _update_metadata(
            run_dir,
            status="completed",
            execution_status=execution_result.status,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"Run directory: {run_dir}")
        print(f"Execution status: {execution_result.status}")
        print(f"Final report: {run_dir / 'final_report.md'}")
        return 0 if execution_result.status == "succeeded" else 1
    except Exception as exc:
        _update_metadata(
            run_dir,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_url", nargs="?")
    parser.add_argument("repo_url", nargs="?")
    parser.add_argument(
        "--recover-run",
        type=Path,
        help="Execute an existing manifest and resume its saved Claude session",
    )
    parser.add_argument("--target")
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--gpu", help="Physical AMD GPU index (defaults to 0 or saved value)")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-budget-usd", type=float, default=100.0)
    parser.add_argument("--prepare-only", action="store_true")
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
    if args.recover_run:
        if args.paper_url or args.repo_url:
            parser.error("paper_url and repo_url cannot be used with --recover-run")
        if args.prepare_only:
            parser.error("--prepare-only cannot be used with --recover-run")
        operation = recover_existing_run(args)
    else:
        if not args.paper_url or not args.repo_url:
            parser.error("paper_url and repo_url are required unless --recover-run is used")
        operation = run(args)
    raise SystemExit(asyncio.run(operation))


if __name__ == "__main__":
    main()
