"""State-machine orchestration for the paper-grounded reproduction workflow."""

from __future__ import annotations

import asyncio
import copy
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
    PreparationStageOutput,
    TargetContract,
    preparation_stage_model,
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
    normalizer: Callable[[Any], Any] | None = None,
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
                raw_output = result.structured_output
                if normalizer is not None:
                    raw_output = normalizer(raw_output)
                artifact = schema.model_validate(raw_output)
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
    skip_evidence_validation: bool,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> TargetContract:
    from .runner import _atomic_json, _render_prompt

    path = run_dir / "target.json"
    if path.is_file() and _stage_succeeded(run_dir, "target_selection"):
        target = TargetContract.model_validate_json(path.read_text(encoding="utf-8"))
        validate_target(
            run_dir,
            target,
            verify_evidence=not skip_evidence_validation,
        )
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
        validator=lambda artifact: validate_target(
            run_dir,
            artifact,
            verify_evidence=not skip_evidence_validation,
        ),
    )
    _atomic_json(path, target.model_dump(mode="json"))
    _update_stage(run_dir, "target_selection", "succeeded", session_id=session_id)
    return target


async def audit_configuration(
    run_dir: Path,
    *,
    target: TargetContract,
    bootstrap: dict[str, Any],
    skip_evidence_validation: bool,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
) -> ConfigurationAudit:
    from .runner import _atomic_json, _render_prompt

    audit_path = run_dir / "configuration_audit.json"
    repository_commit = str(bootstrap["repository_commit"])
    if audit_path.is_file() and _stage_succeeded(run_dir, "configuration_audit"):
        audit = ConfigurationAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
        validate_audit(
            run_dir,
            target,
            audit,
            repository_commit=repository_commit,
            verify_evidence=not skip_evidence_validation,
        )
        return audit

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
        validate_audit(
            run_dir,
            target,
            output.audit,
            repository_commit=repository_commit,
            verify_evidence=not skip_evidence_validation,
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
    _update_stage(run_dir, "configuration_audit", "succeeded", session_id=session_id)
    return output.audit


def _normalize_preparation_output(
    raw_output: Any,
    *,
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
) -> Any:
    """Canonicalize LLM path spellings and Python-owned metadata before validation."""

    if not isinstance(raw_output, dict):
        return raw_output
    output = copy.deepcopy(raw_output)
    environment = output.get("environment")
    manifest = output.get("manifest")
    if not isinstance(environment, dict) or not isinstance(manifest, dict):
        return output

    manifest["selected_experiment"] = target.selected_experiment
    manifest["metric_name"] = target.metric_name
    manifest["paper_target_value"] = target.metric_value
    manifest["target_sha256"] = sha256_file(run_dir / "target.json")
    manifest["audit_sha256"] = sha256_file(run_dir / "configuration_audit.json")
    manifest["repository_commit"] = audit.repository_commit
    manifest["environment_sha256"] = None
    return output


def _migrate_legacy_preparation_paths(raw_output: Any, run_dir: Path) -> Any:
    """Convert previously persisted path conventions to run-root-relative paths."""

    if not isinstance(raw_output, dict):
        return raw_output
    output = copy.deepcopy(raw_output)
    environment = output.get("environment")
    manifest = output.get("manifest")
    if not isinstance(environment, dict) or not isinstance(manifest, dict):
        return output

    dockerfile = environment.get("dockerfile")
    if isinstance(dockerfile, str):
        if dockerfile.startswith("/run/workspace/"):
            environment["dockerfile"] = "workspace/" + dockerfile.removeprefix("/run/workspace/")
        elif "\n" in dockerfile:
            candidate = run_dir / "workspace" / "Dockerfile"
            expected_hash = environment.get("dockerfile_sha256")
            if (
                candidate.is_file()
                and isinstance(expected_hash, str)
                and sha256_file(candidate) == expected_hash
            ):
                environment["dockerfile"] = "workspace/Dockerfile"
        elif not dockerfile.startswith("workspace/"):
            environment["dockerfile"] = f"workspace/{dockerfile}"

    def artifact_path(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith("/run/artifacts/"):
            return "artifacts/" + value.removeprefix("/run/artifacts/")
        if not value.startswith("artifacts/"):
            return f"artifacts/{value}"
        return value

    environment["build_log_path"] = artifact_path(environment.get("build_log_path"))

    host_repository = manifest.get("host_repository")
    if isinstance(host_repository, str):
        host_path = Path(host_repository)
        if host_repository.startswith("/run/workspace/"):
            host_repository = "workspace/" + host_repository.removeprefix("/run/workspace/")
        elif host_path.is_absolute():
            try:
                relative = host_path.resolve().relative_to((run_dir / "workspace").resolve())
                host_repository = str(Path("workspace") / relative)
            except ValueError:
                pass
        elif not host_repository.startswith("workspace/"):
            host_repository = f"workspace/{host_repository}"
        manifest["host_repository"] = host_repository
        if host_repository.startswith("workspace/"):
            manifest["container_repository"] = f"/run/{host_repository}"

    launch_script = manifest.get("launch_script")
    if isinstance(launch_script, str):
        if launch_script.startswith("/run/workspace/"):
            launch_script = "workspace/" + launch_script.removeprefix("/run/workspace/")
        elif not launch_script.startswith("workspace/"):
            launch_script = f"workspace/{launch_script}"
        manifest["launch_script"] = launch_script

    manifest["expected_outputs"] = [
        artifact_path(path) for path in manifest.get("expected_outputs", [])
    ]
    manifest["metric_artifact"] = artifact_path(manifest.get("metric_artifact"))
    metric_extractor = manifest.get("metric_extractor")
    if isinstance(metric_extractor, dict):
        metric_extractor["source"] = artifact_path(metric_extractor.get("source"))
    smoke_test = manifest.get("smoke_test")
    if isinstance(smoke_test, dict):
        smoke_test["log_path"] = artifact_path(smoke_test.get("log_path"))
    return output


def replay_preparation_artifacts(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    *,
    rocm_porting: bool = False,
    force_kernel_port: bool = False,
) -> tuple[EnvironmentManifest, ExecutionManifest]:
    """Normalize, persist, and strictly validate an existing Stage 4 response."""

    from .runner import _atomic_json

    environment_path = run_dir / "environment_manifest.json"
    manifest_path = run_dir / "execution_manifest.json"
    persisted = {
        "environment": _read_json(environment_path),
        "manifest": _read_json(manifest_path),
    }
    persisted = _migrate_legacy_preparation_paths(persisted, run_dir)
    normalized = _normalize_preparation_output(
        persisted,
        run_dir=run_dir,
        target=target,
        audit=audit,
    )
    output_model = preparation_stage_model(rocm_porting, force_kernel_port)
    output = output_model.model_validate(normalized)
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
            output.environment,
            output.manifest,
        )
        validate_manifest_files(run_dir, output.manifest)
    return output.environment, output.manifest


async def prepare_environment(
    run_dir: Path,
    *,
    target: TargetContract,
    audit: ConfigurationAudit,
    timeout_seconds: int,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    rocm_porting: bool = False,
    force_kernel_port: bool = False,
) -> tuple[EnvironmentManifest, ExecutionManifest]:
    from .runner import _atomic_json, _render_prompt

    environment_path = run_dir / "environment_manifest.json"
    manifest_path = run_dir / "execution_manifest.json"

    def _normalize(raw_output: Any) -> Any:
        return _normalize_preparation_output(
            raw_output,
            run_dir=run_dir,
            target=target,
            audit=audit,
        )

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
                output.environment,
                output.manifest,
            )
            validate_manifest_files(run_dir, output.manifest)

    if environment_path.is_file() and manifest_path.is_file():
        try:
            environment, manifest = replay_preparation_artifacts(
                run_dir,
                target,
                audit,
                rocm_porting=rocm_porting,
                force_kernel_port=force_kernel_port,
            )
            _update_stage(run_dir, "environment_preparation", "succeeded")
            return environment, manifest
        except (
            json.JSONDecodeError,
            ValidationError,
            StageValidationError,
            ValueError,
            RuntimeError,
        ):
            if _stage_succeeded(run_dir, "environment_preparation"):
                raise

    target_hash = sha256_file(run_dir / "target.json")
    audit_hash = sha256_file(run_dir / "configuration_audit.json")
    prompt = _render_prompt(
        "prepare_environment.md",
        {
            "TARGET_PATH": str(run_dir / "target.json"),
            "TARGET_SHA256": target_hash,
            "AUDIT_PATH": str(run_dir / "configuration_audit.json"),
            "AUDIT_SHA256": audit_hash,
            "REPOSITORY": str(run_dir / "workspace" / audit.repository_path),
            "REPOSITORY_COMMIT": audit.repository_commit,
            "WORKSPACE": str(run_dir / "workspace"),
            "TIMEOUT_SECONDS": str(timeout_seconds),
        },
    )
    if rocm_porting:
        from .runner import porting_policy_prompt

        prompt += "\n" + porting_policy_prompt(
            str(run_dir / "workspace"), force_kernel_port=force_kernel_port
        )
    prompt_path = run_dir / "prompts" / "environment_preparation.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    output, session_id = await _run_structured_stage(
        run_dir,
        stage="environment_preparation",
        prompt=prompt,
        schema=preparation_stage_model(rocm_porting, force_kernel_port),
        tools=PREPARATION_TOOLS,
        assigned_gpu=assigned_gpu,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        validator=_validate,
        normalizer=_normalize,
    )
    _update_stage(run_dir, "environment_preparation", "succeeded", session_id=session_id)
    return output.environment, output.manifest


def _blocked(run_dir: Path, stage: str, reason: str) -> int:
    from .runner import _update_metadata

    _update_stage(run_dir, stage, "blocked", error=reason)
    _update_metadata(run_dir, status="blocked", blocked_reason=reason, finished_at=_utc_now())
    return 2


async def run_staged_pipeline(run_dir: Path, args: Any, *, recovering: bool = False) -> int:
    """Run or resume the six-stage state machine."""

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
    skip_evidence_validation = bool(getattr(args, "skip_evidence_validation", False))
    _update_metadata(
        run_dir,
        model=model,
        assigned_physical_gpu=assigned_gpu,
        skip_evidence_validation=skip_evidence_validation,
    )

    try:
        bootstrap = await bootstrap_sources(
            run_dir,
            paper_url=paper_url,
            repo_url=repo_url,
        )
        target = await select_target(
            run_dir,
            requested_target=requested_target,
            skip_evidence_validation=skip_evidence_validation,
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

        audit = await audit_configuration(
            run_dir,
            target=target,
            bootstrap=bootstrap,
            skip_evidence_validation=skip_evidence_validation,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
        )

        environment, manifest = await prepare_environment(
            run_dir,
            target=target,
            audit=audit,
            timeout_seconds=timeout_seconds,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
            rocm_porting=bool(input_data.get("rocm_porting")),
            force_kernel_port=bool(input_data.get("force_kernel_port")),
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
            preparation_updates: dict[str, Any] = {
                "status": "prepared",
                "manifest_status": manifest.status,
                "error": None,
                "finished_at": _utc_now(),
            }
            if recovering:
                preparation_updates["recovery_finished_at"] = _utc_now()
                preparation_updates["recovery_error"] = None
            _update_metadata(run_dir, **preparation_updates)
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

        completion_updates: dict[str, Any] = {
            "status": "completed",
            "execution_status": execution_result.status,
            "recovered": recovering,
            "finished_at": _utc_now(),
            "error": None,
        }
        if recovering:
            completion_updates["recovery_finished_at"] = _utc_now()
            completion_updates["recovery_error"] = None
        _update_metadata(run_dir, **completion_updates)
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
