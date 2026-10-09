"""State-machine orchestration for the paper-grounded reproduction workflow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel, ValidationError

from repo2reproduce.provider import reasoning_model
from repo2reproduce.tools import code as code_io
from repo2reproduce.tools import paper as paper_io

from .executor import execute_manifest, validate_manifest_files
from .schemas import (
    AuditStageOutput,
    ConfigurationAudit,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionResult,
    ExecutionSpec,
    PreparationStageOutput,
    TargetContract,
)
from .validators import (
    StageValidationError,
    sha256_file,
    validate_audit,
    validate_preparation,
    validate_target,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
READ_ONLY_TOOLS = ["Read", "Glob", "Grep"]
PREPARATION_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_stage(
    run_dir: Path,
    stage: str,
    status: str,
    *,
    session_id: str | None = None,
    error: str | None = None,
    increment_attempt: bool = False,
) -> None:
    from .runner import _atomic_json

    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    stages = metadata.setdefault("stages", {})
    record = stages.setdefault(stage, {"attempts": 0})
    if increment_attempt:
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["started_at"] = _utc_now()
    record["status"] = status
    record["updated_at"] = _utc_now()
    if session_id:
        record["session_id"] = session_id
    if error:
        record["error"] = error
    else:
        record.pop("error", None)
    metadata["status"] = status if status in {"blocked", "failed"} else stage
    _atomic_json(metadata_path, metadata)


def _stage_succeeded(run_dir: Path, stage: str) -> bool:
    try:
        metadata = _read_json(run_dir / "metadata.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return metadata.get("stages", {}).get(stage, {}).get("status") == "succeeded"


async def _run_structured_stage(
    run_dir: Path,
    *,
    stage: str,
    prompt: str,
    schema: type[ModelT],
    tools: list[str],
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    validator: Callable[[ModelT], None],
) -> tuple[ModelT, str]:
    """Invoke one constrained Claude stage and allow one evidence-based correction."""

    from .runner import _base_options, _invoke_agent

    log_path = run_dir / "logs" / "agent_messages.jsonl"
    session_id: str | None = None
    correction: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, 3):
        _update_stage(
            run_dir,
            stage,
            "running",
            session_id=session_id,
            increment_attempt=True,
        )
        options_data = _base_options(
            run_dir,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
        )
        options_data.update(
            {
                "allowed_tools": tools,
                "output_format": {
                    "type": "json_schema",
                    "schema": schema.model_json_schema(),
                },
            }
        )
        if session_id:
            options_data["resume"] = session_id
        result, observed_session = await _invoke_agent(
            prompt=correction or prompt,
            options=ClaudeAgentOptions(**options_data),
            log_path=log_path,
            stage=f"{stage}-{attempt}",
        )
        session_id = observed_session or session_id
        if result.is_error:
            detail = result.result or ", ".join(result.errors or []) or "unknown SDK error"
            terminal_reason = result.terminal_reason or result.stop_reason or "unknown"
            last_error = RuntimeError(f"{stage} failed ({terminal_reason}): {detail}")
        elif result.subtype != "success" or result.structured_output is None:
            last_error = RuntimeError(f"{stage} ended without structured output")
        else:
            try:
                artifact = schema.model_validate(result.structured_output)
                attempt_path = run_dir / "stage_attempts" / f"{stage}-{attempt}.json"
                from .runner import _atomic_json

                _atomic_json(attempt_path, artifact.model_dump(mode="json"))
                validator(artifact)
                if not session_id:
                    raise RuntimeError(f"{stage} returned no session ID")
                return artifact, session_id
            except (ValidationError, StageValidationError, ValueError, RuntimeError) as exc:
                last_error = exc

        if not session_id:
            break
        correction = (
            "Your previous stage output failed deterministic validation. Correct the "
            "structured artifact and any files created by this stage. Do not skip or "
            "weaken the failed requirement. Validation error: "
            f"{last_error}"
        )

    message = str(last_error or f"{stage} failed without a diagnostic")
    _update_stage(run_dir, stage, "failed", session_id=session_id, error=message)
    raise RuntimeError(message)


async def bootstrap_sources(
    run_dir: Path,
    *,
    paper_url: str,
    repo_url: str,
) -> dict[str, Any]:
    """Deterministically parse the paper and clone the official repository."""

    from .runner import _atomic_json

    bootstrap_path = run_dir / "bootstrap.json"
    required = [
        run_dir / "paper" / "paper.md",
        run_dir / "paper" / "paper_parse.json",
        run_dir / "workspace" / "repo",
    ]
    if bootstrap_path.is_file() and all(path.exists() for path in required):
        _update_stage(run_dir, "bootstrap", "succeeded")
        return _read_json(bootstrap_path)

    _update_stage(run_dir, "bootstrap", "running", increment_attempt=True)

    def _bootstrap() -> dict[str, Any]:
        paper_dir = run_dir / "paper"
        parse_json = paper_dir / "paper_parse.json"
        paper_md = paper_dir / "paper.md"
        if not (parse_json.is_file() and paper_md.is_file()):
            pdf = paper_io.download_pdf(paper_url, paper_dir / "paper.pdf")
            parsed = paper_io.parse_paper(pdf, paper_dir)
            paper_summary = {
                "pdf_path": parsed.pdf_path,
                "paper_md": parsed.merged_markdown_path,
                "paper_parse": parsed.parse_json_path,
                "num_blocks": len(parsed.blocks),
                "num_tables": len(parsed.tables),
            }
        else:
            payload = _read_json(parse_json)
            paper_summary = {
                "pdf_path": str(paper_dir / "paper.pdf"),
                "paper_md": str(paper_md),
                "paper_parse": str(parse_json),
                "num_blocks": payload.get("num_blocks", 0),
                "num_tables": payload.get("num_tables", 0),
            }

        repository = run_dir / "workspace" / "repo"
        clone = code_io.clone_repo(repo_url, repository)
        commit = clone.get("commit")
        if not commit:
            try:
                from git import Repo

                commit = Repo(repository).head.commit.hexsha
            except Exception as exc:
                raise RuntimeError(
                    "cannot determine the exact repository commit for auditability"
                ) from exc

        paper_audit = {
            "schema_version": 1,
            **paper_summary,
            "paper_md_sha256": sha256_file(paper_md),
            "paper_parse_sha256": sha256_file(parse_json),
        }
        _atomic_json(run_dir / "paper_audit.json", paper_audit)
        return {
            "schema_version": 1,
            "paper_url": paper_url,
            "repo_url": repo_url,
            "paper": paper_summary,
            "repository_path": "repo",
            "repository_commit": str(commit),
            "clone": clone,
        }

    try:
        result = await asyncio.to_thread(_bootstrap)
        _atomic_json(bootstrap_path, result)
        _update_stage(run_dir, "bootstrap", "succeeded")
        return result
    except Exception as exc:
        _update_stage(run_dir, "bootstrap", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


async def select_target(
    run_dir: Path,
    *,
    requested_target: str | None,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> TargetContract:
    from .runner import _atomic_json, _render_prompt

    path = run_dir / "target.json"
    if path.is_file() and _stage_succeeded(run_dir, "target_selection"):
        target = TargetContract.model_validate_json(path.read_text(encoding="utf-8"))
        validate_target(run_dir, target)
        return target

    prompt = _render_prompt(
        "select_target.md",
        {
            "PAPER_MD": str(run_dir / "paper" / "paper.md"),
            "PAPER_PARSE": str(run_dir / "paper" / "paper_parse.json"),
            "REPOSITORY": str(run_dir / "workspace" / "repo"),
            "TARGET": requested_target or "(select one central locally reproducible result)",
        },
    )
    (run_dir / "prompts" / "target_selection.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts" / "target_selection.md").write_text(prompt, encoding="utf-8")
    target, session_id = await _run_structured_stage(
        run_dir,
        stage="target_selection",
        prompt=prompt,
        schema=TargetContract,
        tools=READ_ONLY_TOOLS,
        assigned_gpu=assigned_gpu,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        validator=lambda artifact: validate_target(run_dir, artifact),
    )
    _atomic_json(path, target.model_dump(mode="json"))
    _update_stage(run_dir, "target_selection", "succeeded", session_id=session_id)
    return target


async def audit_configuration(
    run_dir: Path,
    *,
    target: TargetContract,
    bootstrap: dict[str, Any],
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> tuple[ConfigurationAudit, ExecutionSpec]:
    from .runner import _atomic_json, _render_prompt

    audit_path = run_dir / "configuration_audit.json"
    spec_path = run_dir / "execution_spec.json"
    repository_commit = str(bootstrap["repository_commit"])
    if (
        audit_path.is_file()
        and spec_path.is_file()
        and _stage_succeeded(run_dir, "configuration_audit")
    ):
        audit = ConfigurationAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
        spec = ExecutionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        validate_audit(
            run_dir,
            target,
            audit,
            spec,
            repository_commit=repository_commit,
        )
        return audit, spec

    target_hash = sha256_file(run_dir / "target.json")
    prompt = _render_prompt(
        "audit_configuration.md",
        {
            "TARGET_PATH": str(run_dir / "target.json"),
            "TARGET_SHA256": target_hash,
            "PAPER_MD": str(run_dir / "paper" / "paper.md"),
            "PAPER_PARSE": str(run_dir / "paper" / "paper_parse.json"),
            "REPOSITORY": str(run_dir / "workspace" / "repo"),
            "REPOSITORY_RELATIVE": str(bootstrap["repository_path"]),
            "REPOSITORY_COMMIT": repository_commit,
        },
    )
    prompt_path = run_dir / "prompts" / "configuration_audit.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    def _validate(output: AuditStageOutput) -> None:
        if output.audit.status != output.execution_spec.status:
            raise StageValidationError("audit and execution spec statuses must agree")
        validate_audit(
            run_dir,
            target,
            output.audit,
            output.execution_spec,
            repository_commit=repository_commit,
        )

    output, session_id = await _run_structured_stage(
        run_dir,
        stage="configuration_audit",
        prompt=prompt,
        schema=AuditStageOutput,
        tools=READ_ONLY_TOOLS,
        assigned_gpu=assigned_gpu,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        validator=_validate,
    )
    _atomic_json(audit_path, output.audit.model_dump(mode="json"))
    output.execution_spec.audit_sha256 = sha256_file(audit_path)
    _atomic_json(spec_path, output.execution_spec.model_dump(mode="json"))
    _update_stage(run_dir, "configuration_audit", "succeeded", session_id=session_id)
    return output.audit, output.execution_spec


async def prepare_environment(
    run_dir: Path,
    *,
    target: TargetContract,
    audit: ConfigurationAudit,
    spec: ExecutionSpec,
    timeout_seconds: int,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> tuple[EnvironmentManifest, ExecutionManifest]:
    from .runner import _atomic_json, _render_prompt

    environment_path = run_dir / "environment_manifest.json"
    manifest_path = run_dir / "execution_manifest.json"
    if (
        environment_path.is_file()
        and manifest_path.is_file()
        and _stage_succeeded(run_dir, "environment_preparation")
    ):
        environment = EnvironmentManifest.model_validate_json(
            environment_path.read_text(encoding="utf-8")
        )
        manifest = ExecutionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        validate_preparation(run_dir, target, audit, spec, environment, manifest)
        validate_manifest_files(run_dir, manifest)
        return environment, manifest

    target_hash = sha256_file(run_dir / "target.json")
    audit_hash = sha256_file(run_dir / "configuration_audit.json")
    spec_hash = sha256_file(run_dir / "execution_spec.json")
    prompt = _render_prompt(
        "prepare_environment.md",
        {
            "TARGET_PATH": str(run_dir / "target.json"),
            "TARGET_SHA256": target_hash,
            "AUDIT_PATH": str(run_dir / "configuration_audit.json"),
            "AUDIT_SHA256": audit_hash,
            "SPEC_PATH": str(run_dir / "execution_spec.json"),
            "SPEC_SHA256": spec_hash,
            "REPOSITORY": str(run_dir / "workspace" / spec.repository_path),
            "WORKSPACE": str(run_dir / "workspace"),
            "TIMEOUT_SECONDS": str(timeout_seconds),
        },
    )
    prompt_path = run_dir / "prompts" / "environment_preparation.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    def _validate(output: PreparationStageOutput) -> None:
        if output.environment.status != output.manifest.status:
            raise StageValidationError("environment and execution manifest statuses must agree")
        _atomic_json(environment_path, output.environment.model_dump(mode="json"))
        output.manifest.environment_sha256 = sha256_file(environment_path)
        _atomic_json(manifest_path, output.manifest.model_dump(mode="json"))
        if output.manifest.status == "ready":
            validate_preparation(
                run_dir,
                target,
                audit,
                spec,
                output.environment,
                output.manifest,
            )
            validate_manifest_files(run_dir, output.manifest)

    output, session_id = await _run_structured_stage(
        run_dir,
        stage="environment_preparation",
        prompt=prompt,
        schema=PreparationStageOutput,
        tools=PREPARATION_TOOLS,
        assigned_gpu=assigned_gpu,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        validator=_validate,
    )
    _atomic_json(environment_path, output.environment.model_dump(mode="json"))
    output.manifest.environment_sha256 = sha256_file(environment_path)
    _atomic_json(manifest_path, output.manifest.model_dump(mode="json"))
    _update_stage(run_dir, "environment_preparation", "succeeded", session_id=session_id)
    return output.environment, output.manifest


def _blocked(run_dir: Path, stage: str, reason: str) -> int:
    from .runner import _update_metadata

    _update_stage(run_dir, stage, "blocked", error=reason)
    _update_metadata(run_dir, status="blocked", blocked_reason=reason, finished_at=_utc_now())
    return 2


async def run_staged_pipeline(run_dir: Path, args: Any, *, recovering: bool = False) -> int:
    """Run or resume the five-stage state machine."""

    from .runner import (
        _atomic_json,
        _preserve_previous_result,
        _update_metadata,
        finalize_session,
    )

    input_data = _read_json(run_dir / "input.json")
    paper_url = str(input_data["paper_url"])
    repo_url = str(input_data["repo_url"])
    requested_target = input_data.get("target")
    timeout_seconds = int(input_data["timeout_seconds"])
    metadata = _read_json(run_dir / "metadata.json")
    assigned_gpu = args.gpu or str(metadata.get("assigned_physical_gpu", "0"))
    model = args.model or str(metadata.get("model") or reasoning_model())
    _update_metadata(run_dir, model=model, assigned_physical_gpu=assigned_gpu)

    try:
        bootstrap = await bootstrap_sources(
            run_dir,
            paper_url=paper_url,
            repo_url=repo_url,
        )
        target = await select_target(
            run_dir,
            requested_target=requested_target,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
        )
        if target.status == "blocked":
            return _blocked(
                run_dir,
                "target_selection",
                target.blocked_reason or "target selection blocked",
            )

        audit, spec = await audit_configuration(
            run_dir,
            target=target,
            bootstrap=bootstrap,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
        )
        if audit.status == "blocked" or spec.status == "blocked":
            return _blocked(
                run_dir,
                "configuration_audit",
                audit.blocked_reason
                or spec.blocked_reason
                or "scientific configuration audit blocked",
            )

        environment, manifest = await prepare_environment(
            run_dir,
            target=target,
            audit=audit,
            spec=spec,
            timeout_seconds=timeout_seconds,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
        )
        if environment.status == "blocked" or manifest.status == "blocked":
            return _blocked(
                run_dir,
                "environment_preparation",
                environment.blocked_reason
                or manifest.blocked_reason
                or "environment preparation blocked",
            )
        if args.prepare_only:
            _update_metadata(run_dir, status="prepared", manifest_status=manifest.status)
            print(f"Prepared manifest: {run_dir / 'execution_manifest.json'}")
            return 0

        result_path = run_dir / "execution_result.json"
        execution_result: ExecutionResult | None = None
        executed_now = False
        if result_path.is_file():
            try:
                existing = ExecutionResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
                if existing.status == "succeeded":
                    execution_result = existing
            except ValidationError:
                execution_result = None

        if execution_result is None:
            executed_now = True
            if recovering and result_path.exists():
                _preserve_previous_result(run_dir)
            _update_stage(run_dir, "execution", "running", increment_attempt=True)
            execution_result = await asyncio.to_thread(
                execute_manifest,
                run_dir,
                manifest,
                assigned_gpu=assigned_gpu,
            )
            execution_result.manifest_sha256 = sha256_file(run_dir / "execution_manifest.json")
            _atomic_json(
                result_path,
                execution_result.model_dump(mode="json"),
            )
            _update_stage(run_dir, "execution", "succeeded")

        report_path = run_dir / "final_report.md"
        if executed_now or not report_path.is_file():
            _update_stage(run_dir, "finalization", "running", increment_attempt=True)
            await finalize_session(
                run_dir,
                session_id=None,
                assigned_gpu=assigned_gpu,
                model=model,
                max_turns=args.max_turns,
                max_budget_usd=args.max_budget_usd,
                execution_result=execution_result,
            )
            _update_stage(run_dir, "finalization", "succeeded")

        _update_metadata(
            run_dir,
            status="completed",
            execution_status=execution_result.status,
            recovered=recovering,
            finished_at=_utc_now(),
        )
        print(f"Run directory: {run_dir}")
        print(f"Execution status: {execution_result.status}")
        print(f"Final report: {report_path}")
        return 0 if execution_result.status == "succeeded" else 1
    except Exception as exc:
        _update_metadata(
            run_dir,
            status="failed",
            finished_at=_utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
