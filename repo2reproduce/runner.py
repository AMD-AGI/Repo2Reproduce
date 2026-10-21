"""Claude Agent SDK entry point for the staged reproduction pipeline."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from dotenv import load_dotenv
from pydantic import ValidationError

from repo2reproduce.provider import ensure_runtime_path, provider_env

from . import config
from .schemas import (
    ConfigurationAudit,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionResult,
    FinalReport,
    TargetContract,
)
from .validators import StageValidationError, validate_final_report

PACKAGE_DIR = Path(__file__).resolve().parent
SKILLS_DIR = PACKAGE_DIR / "skills"
DEFAULT_RUNS_ROOT = Path("/shared_nfs/machita/claude-code-two-stage-runs")
FINALIZATION_TOOLS = ["Read", "Glob", "Grep"]
CONFIGURATION_AUDIT_SKILL = "configuration-audit"
EXECUTION_REPAIR_SKILL = "execution-repair"
RESULT_COMPARISON_SKILL = "paper-result-comparison"
MATCH_TOLERANCE_PERCENT = 5


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


class SkillUnavailableError(RuntimeError):
    """Raised when Claude Code did not discover a stage's required skill."""


@dataclasses.dataclass
class SkillTrace:
    """Skills discovered by Claude Code and requested by the model in one session."""

    available: set[str] | None = None
    requests: dict[str, str] = dataclasses.field(default_factory=dict)
    failed_requests: set[str] = dataclasses.field(default_factory=set)

    @property
    def invoked(self) -> set[str]:
        return {
            name
            for tool_use_id, name in self.requests.items()
            if tool_use_id not in self.failed_requests
        }

    def observe(self, message: Any) -> None:
        if isinstance(message, SystemMessage) and message.subtype == "init":
            skills = message.data.get("skills")
            if isinstance(skills, list):
                self.available = {str(name) for name in skills}
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name == "Skill":
                    name = block.input.get("skill")
                    if isinstance(name, str):
                        self.requests[block.id] = name.strip().lstrip("/")
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock) and block.is_error:
                    self.failed_requests.add(block.tool_use_id)


def require_skill_available(stage: str, skill: str, trace: SkillTrace) -> None:
    """Fail fast when the CLI session cannot load the stage's skill at all."""

    if trace.available is not None and skill not in trace.available:
        raise SkillUnavailableError(
            f"{stage}: Claude Code did not discover the required `{skill}` skill; "
            "expected it under workspace/.claude/skills/"
        )


def require_skill_invoked(skill: str, trace: SkillTrace) -> None:
    if skill not in trace.invoked:
        raise StageValidationError(
            f"the required `{skill}` skill was not loaded with the Skill tool. Invoke it, "
            "re-check this stage's work against it, and return the corrected output"
        )


def _skill_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def install_skills(run_dir: Path) -> dict[str, str]:
    """Install the packaged stage skills as project skills of the run workspace.

    Claude Code discovers project skills under ``<cwd>/.claude/skills``; every stage runs
    with ``cwd`` set to the workspace and the ``project`` setting source enabled.
    """

    destination_root = run_dir / "workspace" / ".claude" / "skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    installed: dict[str, str] = {}
    for source in sorted(SKILLS_DIR.iterdir()):
        if not (source / "SKILL.md").is_file():
            continue
        destination = destination_root / source.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        installed[source.name] = _skill_digest(destination)
    return installed


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
    skills = install_skills(run_dir)

    _atomic_json(
        run_dir / "input.json",
        {
            "paper_url": paper_url,
            "repo_url": repo_url,
            "timeout_seconds": timeout_seconds,
            "target": target,
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
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "initialized",
            "skills": skills,
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
    skill_trace: SkillTrace | None = None,
) -> tuple[ResultMessage, str | None]:
    result: ResultMessage | None = None
    session_id: str | None = None
    try:
        async for message in query(prompt=prompt, options=options):
            _append_message(log_path, stage, message)
            if skill_trace is not None:
                skill_trace.observe(message)
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


def _run_relative_path(run_dir: Path, value: str | None, fallback: str) -> str:
    """Return a stable run-root-relative evidence path."""

    if not value:
        return fallback
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return fallback


def _render_final_report(report: FinalReport) -> str:
    """Render the concise human-facing view of the structured final report."""

    location = report.result_location
    location_details = [
        ("Table/Figure", location.table_or_figure),
        ("Row", location.row),
        ("Column", location.column),
        ("Cell", location.cell),
    ]
    location_lines = [
        f"- **{label}:** {value}" for label, value in location_details if value
    ]

    baseline = report.baseline
    if baseline.applicable:
        baseline_lines = [
            f"- **Method:** {baseline.method}",
            f"- **Value:** {baseline.value}",
            f"- **Metric direction:** {baseline.metric_direction.replace('_', ' ')}",
            f"- **Applicability:** {baseline.applicability_explanation}",
        ]
        if baseline.paper_evidence is not None:
            baseline_lines.extend(
                [
                    f'- **Paper evidence:** "{baseline.paper_evidence.quote}"',
                    f"- **Evidence location:** {baseline.paper_evidence.locator}",
                ]
            )
    else:
        baseline_lines = [
            "- **Applicable baseline:** None",
            f"- **Metric direction:** {baseline.metric_direction.replace('_', ' ')}",
            f"- **Reason:** {baseline.applicability_explanation}",
        ]

    comparison = report.comparison
    relative_error = (
        "not applicable"
        if comparison.relative_error_percent is None
        else f"{comparison.relative_error_percent:.6g}%"
    )
    assumptions = [
        (
            f"- **[{item.category}] {item.id} — {item.setting}:** "
            f"{item.value_used or 'interpretation only'}; "
            f"paper {item.paper_status.replace('_', ' ')}; "
            f"source {item.source_type.replace('_', ' ')}. {item.rationale}"
        )
        for item in report.assumptions
    ]

    artifacts = report.artifacts
    artifact_items = [
        ("Configuration used", artifacts.configuration_used),
        ("Configuration audit", artifacts.configuration_audit),
        ("Environment manifest", artifacts.environment_manifest),
        ("Dockerfile", artifacts.dockerfile),
        ("Environment build log", artifacts.build_log),
        ("Scientific launch script", artifacts.launch_script),
        ("Execution result", artifacts.execution_result),
        ("Metric artifact", artifacts.metric_artifact),
        ("Executor log", artifacts.executor_log),
        ("Detailed assumptions", artifacts.assumptions),
    ]
    artifact_lines = [
        f"- **{label}:** `{path}`" for label, path in artifact_items if path
    ]
    if artifacts.launch_script_sha256:
        artifact_lines.append(
            f"- **Launch script SHA-256:** `{artifacts.launch_script_sha256}`"
        )

    return "\n".join(
        [
            f"# Reproduction report: {report.selected_experiment}",
            "",
            "## Selected result",
            f"- **Paper location:** {location.description}",
            *location_lines,
            f"- **Metric:** {report.metric_name}",
            "",
            "## Result",
            f"- **Execution:** {report.execution_status}",
            f"- **Paper target:** {report.paper_target_value}",
            f"- **Reproduced value:** {report.achieved_value or 'not produced'}",
            f"- **Relative error:** {relative_error}",
            f"- **Verdict:** {comparison.verdict.replace('_', ' ')}",
            f"- **Calculation:** {comparison.calculation}",
            f"- **Explanation:** {comparison.explanation}",
            "",
            "## Paper baseline",
            *baseline_lines,
            "",
            "## Assumptions and fallbacks",
            *(assumptions or ["- None reported."]),
            "",
            "## Deviations",
            *([f"- {item}" for item in report.deviations] or ["- None reported."]),
            "",
            "## Gaps",
            *([f"- {item}" for item in report.gaps] or ["- None reported."]),
            "",
            "## Evidence files",
            *artifact_lines,
            "",
        ]
    )


async def _request_final_report(
    run_dir: Path,
    *,
    prompt: str,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    skill: str = RESULT_COMPARISON_SKILL,
) -> FinalReport:
    """Request a structured report, allowing one schema-correction attempt."""

    correction: str | None = None
    session_id: str | None = None
    skill_trace = SkillTrace()
    last_error = "finalization returned no structured report"
    for attempt in range(1, 3):
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
                "skills": [skill],
                "output_format": {
                    "type": "json_schema",
                    "schema": FinalReport.model_json_schema(),
                },
            }
        )
        if session_id:
            options_data["resume"] = session_id
        attempt_prompt = prompt
        if correction:
            attempt_prompt = correction if session_id else f"{prompt}\n\n{correction}"
        result, observed_session = await _invoke_agent(
            prompt=attempt_prompt,
            options=ClaudeAgentOptions(**options_data),
            log_path=run_dir / "logs" / "agent_messages.jsonl",
            stage=f"finalize-{attempt}",
            skill_trace=skill_trace,
        )
        require_skill_available("finalization", skill, skill_trace)
        session_id = observed_session or session_id
        if result.is_error:
            last_error = (
                result.result
                or ", ".join(result.errors or [])
                or "unknown SDK validation error"
            )
        elif result.subtype != "success" or result.structured_output is None:
            last_error = "finalization returned no structured report"
        else:
            try:
                report = FinalReport.model_validate(result.structured_output)
                require_skill_invoked(skill, skill_trace)
                return report
            except (ValidationError, StageValidationError) as exc:
                last_error = str(exc)

        correction = (
            "Your previous final report failed the required output schema or stage "
            "contract. Correct only the report structure or fields identified below. "
            "Re-read the persisted evidence when needed; do not launch or repeat the "
            "experiment and do not change any frozen scientific value. Return only the "
            f"corrected FinalReport. Validation error: {last_error}"
        )

    raise RuntimeError(f"finalization failed after one correction retry: {last_error}")


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
            "PAPER_MD": str(run_dir / "paper" / "paper.md"),
            "PAPER_PARSE": str(run_dir / "paper" / "paper_parse.json"),
            "LOG_PATH": execution_result.log_path,
            "ARTIFACTS_PATH": str(run_dir / "artifacts"),
            "REPAIRS_PATH": str(run_dir / "execution_repairs"),
            "SKILL": RESULT_COMPARISON_SKILL,
            "MATCH_TOLERANCE_PERCENT": str(MATCH_TOLERANCE_PERCENT),
        },
    )
    (run_dir / "finalize_prompt.txt").write_text(prompt, encoding="utf-8")
    report = await _request_final_report(
        run_dir,
        prompt=prompt,
        assigned_gpu=assigned_gpu,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )
    target = TargetContract.model_validate_json(
        (run_dir / "target.json").read_text(encoding="utf-8")
    )
    audit = ConfigurationAudit.model_validate_json(
        (run_dir / "configuration_audit.json").read_text(encoding="utf-8")
    )
    environment = EnvironmentManifest.model_validate_json(
        (run_dir / "environment_manifest.json").read_text(encoding="utf-8")
    )
    manifest = ExecutionManifest.model_validate_json(
        (run_dir / "execution_manifest.json").read_text(encoding="utf-8")
    )
    report.selected_experiment = target.selected_experiment
    report.result_location.description = target.result_location
    report.execution_status = execution_result.status
    report.metric_name = target.metric_name
    report.paper_target_value = target.metric_value
    report.achieved_value = execution_result.achieved_metric
    report.artifacts.configuration_used = "execution_manifest.json"
    report.artifacts.configuration_audit = "configuration_audit.json"
    report.artifacts.environment_manifest = "environment_manifest.json"
    report.artifacts.dockerfile = environment.dockerfile
    report.artifacts.build_log = environment.build_log_path
    report.artifacts.launch_script = manifest.launch_script
    report.artifacts.launch_script_sha256 = manifest.launch_script_sha256
    report.artifacts.execution_result = "execution_result.json"
    report.artifacts.metric_artifact = (
        execution_result.metric_source or manifest.metric_artifact
    )
    report.artifacts.executor_log = _run_relative_path(
        run_dir,
        execution_result.log_path,
        "logs/executor.log",
    )
    report.artifacts.assumptions = "assumptions.json"
    validate_final_report(
        run_dir,
        target,
        audit,
        environment,
        manifest,
        execution_result,
        report,
    )
    _atomic_json(run_dir / "final_report.json", report.model_dump(mode="json"))
    _atomic_json(
        run_dir / "assumptions.json",
        {
            "schema_version": 1,
            "assumptions": [
                item.model_dump(mode="json") for item in report.assumptions
            ],
        },
    )
    markdown = _render_final_report(report)
    (run_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    return markdown


def _update_metadata(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    _atomic_json(path, data)


def _preserve_previous_result(run_dir: Path) -> None:
    """Keep the prior blocked result/report before a recovery attempt overwrites them."""

    for relative in (
        "execution_result.json",
        "final_report.json",
        "final_report.md",
        "assumptions.json",
        "logs/executor.log",
    ):
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
        skills=install_skills(run_dir),
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
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--gpu", help="Physical AMD GPU index (defaults to 0 or saved value)")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=config.max_turns())
    parser.add_argument("--max-budget-usd", type=float, default=config.max_budget_usd())
    parser.add_argument(
        "--max-execution-repairs",
        type=int,
        default=2,
        help="Maximum LLM repairs after failed executions (0–2; default: 2)",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-evidence-validation",
        action="store_true",
        help=(
            "Skip optional paper/code evidence diagnostics; scientific gaps remain "
            "recorded by the agents"
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
    if not 0 <= args.max_execution_repairs <= 2:
        raise SystemExit("--max-execution-repairs must be between 0 and 2")
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
