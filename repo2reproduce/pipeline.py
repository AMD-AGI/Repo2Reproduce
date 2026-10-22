"""State-machine orchestration for the paper-grounded reproduction workflow."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
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
    ExecutionRepairOutput,
    ExecutionResult,
    PreparationStageOutput,
    TargetContract,
    execution_repair_model,
    preparation_stage_model,
)
from .validators import (
    StageValidationError,
    sha256_file,
    validate_audit,
    validate_execution_repair,
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
    skills: Sequence[str] = (),
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
    if skills:
        record["skills"] = list(skills)
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
    initial_session_id: str | None = None,
    skills: Sequence[str] = (),
) -> tuple[ModelT, str]:
    """Invoke one constrained Claude stage and allow one evidence-based correction.

    ``skills`` names the project skills exposed to this stage; each must be loaded before
    the stage output is accepted. Stages without skills see none.
    """

    from .runner import (
        SkillTrace,
        SkillUnavailableError,
        _base_options,
        _invoke_agent,
        require_skill_available,
        require_skill_invoked,
    )

    log_path = run_dir / "logs" / "agent_messages.jsonl"
    session_id = initial_session_id
    resume_fallback_available = initial_session_id is not None
    correction: str | None = None
    last_error: Exception | None = None
    skill_trace = SkillTrace()

    maximum_attempts = 4 if initial_session_id else 2
    for attempt in range(1, maximum_attempts + 1):
        _update_stage(
            run_dir,
            stage,
            "running",
            session_id=session_id,
            increment_attempt=True,
            skills=skills,
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
                "skills": list(skills),
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
            skill_trace=skill_trace,
        )
        try:
            for skill in skills:
                require_skill_available(stage, skill, skill_trace)
        except SkillUnavailableError as exc:
            _update_stage(run_dir, stage, "failed", session_id=session_id, error=str(exc))
            raise
        session_id = observed_session or session_id
        invocation_failed = False
        if result.is_error:
            detail = result.result or ", ".join(result.errors or []) or "unknown SDK error"
            terminal_reason = result.terminal_reason or result.stop_reason or "unknown"
            last_error = RuntimeError(f"{stage} failed ({terminal_reason}): {detail}")
            invocation_failed = True
        elif result.subtype != "success" or result.structured_output is None:
            last_error = RuntimeError(f"{stage} ended without structured output")
            invocation_failed = True
        else:
            try:
                raw_output = result.structured_output
                if normalizer is not None:
                    raw_output = normalizer(raw_output)
                artifact = schema.model_validate(raw_output)
                attempt_path = run_dir / "stage_attempts" / f"{stage}-{attempt}.json"
                from .runner import _atomic_json

                _atomic_json(attempt_path, artifact.model_dump(mode="json"))
                for skill in skills:
                    require_skill_invoked(skill, skill_trace)
                validator(artifact)
                if not session_id:
                    raise RuntimeError(f"{stage} returned no session ID")
                return artifact, session_id
            except (ValidationError, StageValidationError, ValueError, RuntimeError) as exc:
                last_error = exc

        if resume_fallback_available and invocation_failed:
            # Saved Claude sessions can expire. Retry the same evidence-complete prompt
            # statelessly, then retain the ordinary correction attempt if validation fails.
            session_id = None
            resume_fallback_available = False
            correction = None
            skill_trace = SkillTrace()
            continue
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


def _needs_repository_baseline(bootstrap: dict[str, Any]) -> bool:
    clone = bootstrap.get("clone") or {}
    return bool(clone.get("subpath")) and not clone.get("snapshot_commit")


def _restore_repository_baseline(
    run_dir: Path,
    bootstrap: dict[str, Any],
    repo_url: str,
) -> dict[str, Any]:
    """Give a subtree clone from an older run its own pristine git baseline.

    The baseline is rebuilt from a fresh clone of the same upstream commit, so files the
    run already changed still appear in later repository diffs. When the upstream branch
    has moved, the current tree becomes the baseline and the bootstrap record says so.
    """

    repository = run_dir / "workspace" / str(bootstrap["repository_path"])
    clone = dict(bootstrap["clone"])
    if (repository / ".git").exists():
        raise RuntimeError(f"repository already has unrecorded git metadata: {repository}")
    cache = run_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(cache)) as temporary:
        pristine = Path(temporary) / "repo"
        fresh = code_io.clone_repo(repo_url, pristine)
        if fresh.get("commit") == bootstrap["repository_commit"]:
            shutil.move(str(pristine / ".git"), str(repository / ".git"))
            clone["snapshot_commit"] = fresh["snapshot_commit"]
            clone["snapshot_origin"] = "pristine_upstream"
        else:
            clone["snapshot_commit"] = code_io.snapshot_repository(
                repository,
                f"Workspace snapshot at recovery of {bootstrap['repository_commit']}",
            )
            clone["snapshot_origin"] = "workspace_at_recovery"
    return {**bootstrap, "clone": clone}


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
        bootstrap = _read_json(bootstrap_path)
        if _needs_repository_baseline(bootstrap):
            bootstrap = await asyncio.to_thread(
                _restore_repository_baseline, run_dir, bootstrap, repo_url
            )
            _atomic_json(bootstrap_path, bootstrap)
        _update_stage(run_dir, "bootstrap", "succeeded")
        return bootstrap

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
    from .runner import CONFIGURATION_AUDIT_SKILL, _atomic_json, _render_prompt

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
            "SKILL": CONFIGURATION_AUDIT_SKILL,
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
        skills=[CONFIGURATION_AUDIT_SKILL],
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
    from .runner import ROCM_PORTING_SKILL, _atomic_json, _render_prompt

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
        skills=[ROCM_PORTING_SKILL] if rocm_porting else [],
    )
    _update_stage(run_dir, "environment_preparation", "succeeded", session_id=session_id)
    return output.environment, output.manifest


def _failure_fingerprint(run_dir: Path, result: ExecutionResult) -> str:
    """Return a stable fingerprint for one deterministic execution failure."""

    log_path = Path(result.log_path)
    if not log_path.is_absolute():
        log_path = run_dir / log_path
    log_sha256 = sha256_file(log_path) if log_path.is_file() else None
    payload = {
        "status": result.status,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "error": result.error,
        "missing_outputs": sorted(result.missing_outputs),
        "metric_extraction_error": result.metric_extraction_error,
        "log_sha256": log_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _workspace_hashes(run_dir: Path) -> dict[str, str]:
    """Hash mutable, reasonably sized workspace files for no-op detection."""

    hashes: dict[str, str] = {}
    workspace = run_dir / "workspace"
    if not workspace.is_dir():
        return hashes
    for path in workspace.rglob("*"):
        if not path.is_file() or path.is_symlink() or ".git" in path.relative_to(workspace).parts:
            continue
        # Source, dependency, launch, and generated config files are expected to be small.
        # Avoid hashing downloaded model/data payloads merely to detect a repair no-op.
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        relative = path.relative_to(run_dir).as_posix()
        hashes[relative] = sha256_file(path)
    return hashes


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise StageValidationError(f"cannot capture repository repair evidence: {detail}")
    return completed.stdout


def _repository_baseline(run_dir: Path, audit: ConfigurationAudit) -> str:
    """Return the local commit that holds the pristine upstream repository files.

    Full clones check out the upstream commit itself; subtree clones carry a local
    snapshot commit of the upstream subtree instead.
    """

    bootstrap_path = run_dir / "bootstrap.json"
    if bootstrap_path.is_file():
        snapshot = (_read_json(bootstrap_path).get("clone") or {}).get("snapshot_commit")
        if snapshot:
            return str(snapshot)
    return audit.repository_commit


def _capture_repository_patch(
    run_dir: Path,
    audit: ConfigurationAudit,
    attempt: int,
) -> tuple[str, str, list[str]]:
    """Persist the complete tracked and untracked repository diff."""

    repository = run_dir / "workspace" / audit.repository_path
    toplevel = _git_output(repository, "rev-parse", "--show-toplevel").strip()
    if Path(toplevel).resolve() != repository.resolve():
        raise StageValidationError(
            f"repository is not its own git checkout (git resolved {toplevel})"
        )
    baseline = _repository_baseline(run_dir, audit)
    head = _git_output(repository, "rev-parse", "HEAD").strip()
    if head != baseline:
        raise StageValidationError("repair changed the checked-out repository base commit")

    status = _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all")
    changed_files: list[str] = []
    untracked: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        changed_files.append(value)
        if line.startswith("?? "):
            untracked.append(value)

    tracked_patch = _git_output(repository, "diff", "--binary", baseline, "--", ".")
    patch_parts = [tracked_patch]
    for relative in untracked:
        candidate = repository / relative
        if not candidate.is_file():
            continue
        completed = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            detail = completed.stderr.strip() or "unknown git diff error"
            raise StageValidationError(f"cannot capture untracked repository file: {detail}")
        patch_parts.append(completed.stdout)

    relative_path = f"execution_repairs/repair-{attempt}.patch"
    patch_path = run_dir / relative_path
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("".join(patch_parts), encoding="utf-8")
    return relative_path, sha256_file(patch_path), sorted(set(changed_files))


def _archive_execution_attempt(
    run_dir: Path,
    result: ExecutionResult,
    manifest: ExecutionManifest,
    attempt: int,
) -> Path:
    """Archive failed evidence and remove outputs that could create stale success."""

    destination = run_dir / "artifacts" / "execution-attempts" / f"attempt-{attempt}"
    destination.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "execution_result.json"
    if result_path.is_file():
        shutil.copy2(result_path, destination / "execution_result.json")
    log_path = Path(result.log_path)
    if not log_path.is_absolute():
        log_path = run_dir / log_path
    if log_path.is_file():
        shutil.copy2(log_path, destination / "executor.log")

    full_run = run_dir / "artifacts" / "full-run"
    if full_run.exists():
        # Containers write full-run as root. The host may rename a directory it cannot
        # write only within the same parent, so archive it as a sibling and link to it.
        archived_full_run = full_run.with_name(f"full-run.attempt-{attempt}")
        full_run.rename(archived_full_run)
        (destination / "full-run").symlink_to(Path("..") / ".." / archived_full_run.name)
    for relative in manifest.expected_outputs:
        output = run_dir / relative
        if not output.exists():
            continue
        archived = destination / "expected-outputs" / Path(relative).relative_to("artifacts")
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(archived))
    return destination


def _repair_state(
    environment: EnvironmentManifest,
    manifest: ExecutionManifest,
    workspace_hashes: dict[str, str],
    repository_diff_sha256: str,
) -> str:
    """Hash only mutable delivery/environment state, never smoke-test evidence."""

    payload = {
        "environment": {
            "image_tag": environment.image_tag,
            "dockerfile": environment.dockerfile,
            "dockerfile_sha256": environment.dockerfile_sha256,
            "build_log_path": environment.build_log_path,
            "build_succeeded": environment.build_succeeded,
        },
        "manifest": {
            "image_tag": manifest.image_tag,
            "launch_script": manifest.launch_script,
            "launch_script_sha256": manifest.launch_script_sha256,
            "host_repository": manifest.host_repository,
            "container_repository": manifest.container_repository,
            "timeout_seconds": manifest.timeout_seconds,
            "expected_outputs": manifest.expected_outputs,
            "metric_artifact": manifest.metric_artifact,
            "metric_extractor": (
                manifest.metric_extractor.model_dump(mode="json")
                if manifest.metric_extractor
                else None
            ),
            "scientific_configuration": [
                item.model_dump(mode="json") for item in manifest.scientific_configuration
            ],
        },
        "workspace_hashes": workspace_hashes,
        "repository_diff_sha256": repository_diff_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


async def repair_execution(
    run_dir: Path,
    *,
    target: TargetContract,
    audit: ConfigurationAudit,
    environment: EnvironmentManifest,
    manifest: ExecutionManifest,
    failed_result: ExecutionResult,
    attempt: int,
    max_repairs: int,
    failure_fingerprint: str,
    assigned_gpu: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    rocm_porting: bool = False,
    force_kernel_port: bool = False,
) -> tuple[EnvironmentManifest, ExecutionManifest, str]:
    """Ask the preparation agent for one bounded repair and validate it."""

    from .runner import (
        EXECUTION_REPAIR_SKILL,
        ROCM_PORTING_SKILL,
        _atomic_json,
        _render_prompt,
        porting_policy_prompt,
    )

    target_sha256 = sha256_file(run_dir / "target.json")
    audit_sha256 = sha256_file(run_dir / "configuration_audit.json")
    target_bytes = (run_dir / "target.json").read_bytes()
    audit_bytes = (run_dir / "configuration_audit.json").read_bytes()
    before_workspace = _workspace_hashes(run_dir)
    before_patch_path, before_patch_sha, _ = _capture_repository_patch(run_dir, audit, 0)
    (run_dir / before_patch_path).unlink()
    before_state = _repair_state(environment, manifest, before_workspace, before_patch_sha)
    metadata = _read_json(run_dir / "metadata.json")
    preparation_session = (
        metadata.get("stages", {}).get("environment_preparation", {}).get("session_id")
    )
    prompt = _render_prompt(
        "repair_execution.md",
        {
            "REPAIR_ATTEMPT": str(attempt),
            "MAX_REPAIRS": str(max_repairs),
            "TARGET_PATH": str(run_dir / "target.json"),
            "AUDIT_PATH": str(run_dir / "configuration_audit.json"),
            "ENVIRONMENT_PATH": str(run_dir / "environment_manifest.json"),
            "MANIFEST_PATH": str(run_dir / "execution_manifest.json"),
            "RESULT_PATH": str(run_dir / "execution_result.json"),
            "LOG_PATH": failed_result.log_path,
            "REPOSITORY": str(run_dir / "workspace" / audit.repository_path),
            "REPOSITORY_COMMIT": audit.repository_commit,
            "REPAIRS_PATH": str(run_dir / "execution_repairs"),
            "SKILL": EXECUTION_REPAIR_SKILL,
        },
    )
    skills = [EXECUTION_REPAIR_SKILL]
    if rocm_porting:
        prompt += "\n" + porting_policy_prompt(
            str(run_dir / "workspace"), force_kernel_port=force_kernel_port
        )
        skills.append(ROCM_PORTING_SKILL)
    prompt_path = run_dir / "prompts" / f"execution_repair-{attempt}.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    def _normalize(raw_output: Any) -> Any:
        output = _normalize_preparation_output(
            raw_output,
            run_dir=run_dir,
            target=target,
            audit=audit,
        )
        if not isinstance(output, dict):
            return output
        repair = output.get("repair")
        environment_data = output.get("environment")
        manifest_data = output.get("manifest")
        if not all(isinstance(value, dict) for value in (repair, environment_data, manifest_data)):
            return output

        repair["attempt"] = attempt
        repair["failure_fingerprint"] = failure_fingerprint
        after_workspace = _workspace_hashes(run_dir)
        actual_changes = {
            path for path, digest in after_workspace.items() if before_workspace.get(path) != digest
        }
        actual_changes.update(set(before_workspace) - set(after_workspace))
        reported = {
            change.get("path"): change
            for change in repair.get("changes", [])
            if isinstance(change, dict) and isinstance(change.get("path"), str)
        }
        changes: list[dict[str, Any]] = []
        for path in sorted(actual_changes):
            candidate = run_dir / path
            if not candidate.is_file():
                continue
            item = reported.get(path, {})
            changes.append(
                {
                    "path": path,
                    "kind": item.get(
                        "kind",
                        "source_patch" if path.startswith("workspace/repo/") else "other",
                    ),
                    "summary": item.get("summary") or "File changed during execution repair.",
                    "sha256": sha256_file(candidate),
                }
            )
        repair["changes"] = changes

        dockerfile = environment_data.get("dockerfile")
        if isinstance(dockerfile, str) and (run_dir / dockerfile).is_file():
            environment_data["dockerfile_sha256"] = sha256_file(run_dir / dockerfile)
        launch_script = manifest_data.get("launch_script")
        if isinstance(launch_script, str) and (run_dir / launch_script).is_file():
            manifest_data["launch_script_sha256"] = sha256_file(run_dir / launch_script)

        patch_path, patch_sha, changed_files = _capture_repository_patch(run_dir, audit, attempt)
        repair["repository_patch"] = {
            "base_commit": audit.repository_commit,
            "local_baseline_commit": _repository_baseline(run_dir, audit),
            "diff_path": patch_path,
            "diff_sha256": patch_sha,
            "changed_files": changed_files,
        }
        # Match the exact bytes written by runner._atomic_json.
        environment_bytes = json.dumps(environment_data, indent=2).encode("utf-8")
        manifest_data["environment_sha256"] = hashlib.sha256(environment_bytes).hexdigest()
        return output

    def _validate(output: ExecutionRepairOutput) -> None:
        validate_execution_repair(
            run_dir,
            target,
            audit,
            manifest,
            output,
            attempt=attempt,
            failure_fingerprint=failure_fingerprint,
            target_sha256=target_sha256,
            audit_sha256=audit_sha256,
        )
        after_workspace = _workspace_hashes(run_dir)
        after_state = _repair_state(
            output.environment,
            output.manifest,
            after_workspace,
            output.repair.repository_patch.diff_sha256,
        )
        if after_state == before_state:
            raise StageValidationError("execution repair was a no-op")

        environment_path = run_dir / "environment_manifest.json"
        manifest_path = run_dir / "execution_manifest.json"
        previous_environment = environment_path.read_bytes()
        previous_manifest = manifest_path.read_bytes()
        try:
            _atomic_json(environment_path, output.environment.model_dump(mode="json"))
            output.manifest.environment_sha256 = sha256_file(environment_path)
            _atomic_json(manifest_path, output.manifest.model_dump(mode="json"))
            validate_preparation(
                run_dir,
                target,
                audit,
                output.environment,
                output.manifest,
            )
            validate_manifest_files(run_dir, output.manifest)
        except Exception:
            environment_path.write_bytes(previous_environment)
            manifest_path.write_bytes(previous_manifest)
            raise

    try:
        output, session_id = await _run_structured_stage(
            run_dir,
            stage=f"execution_repair_{attempt}",
            prompt=prompt,
            schema=execution_repair_model(rocm_porting, force_kernel_port),
            tools=PREPARATION_TOOLS,
            assigned_gpu=assigned_gpu,
            model=model,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            validator=_validate,
            normalizer=_normalize,
            initial_session_id=preparation_session,
            skills=skills,
        )
    except Exception:
        (run_dir / "target.json").write_bytes(target_bytes)
        (run_dir / "configuration_audit.json").write_bytes(audit_bytes)
        raise
    repair_path = run_dir / "execution_repairs" / f"repair-{attempt}.json"
    _atomic_json(repair_path, output.repair.model_dump(mode="json"))
    _update_stage(
        run_dir,
        f"execution_repair_{attempt}",
        "succeeded",
        session_id=session_id,
    )
    return output.environment, output.manifest, session_id


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
            max_repairs = int(getattr(args, "max_execution_repairs", 2))
            previous_repairs = list((run_dir / "execution_repairs").glob("repair-*.json"))
            repair_count = len(previous_repairs)
            execution_attempt = int(
                _read_json(run_dir / "metadata.json")
                .get("stages", {})
                .get("execution", {})
                .get("attempts", 0)
            )
            seen_failure_fingerprints: set[str] = set()
            for repair_path in previous_repairs:
                try:
                    fingerprint = _read_json(repair_path).get("failure_fingerprint")
                    if isinstance(fingerprint, str):
                        seen_failure_fingerprints.add(fingerprint)
                except (OSError, json.JSONDecodeError):
                    continue
            while True:
                execution_attempt += 1
                _update_stage(run_dir, "execution", "running", increment_attempt=True)
                execution_result = await asyncio.to_thread(
                    execute_manifest,
                    run_dir,
                    manifest,
                    assigned_gpu=assigned_gpu,
                )
                execution_result.manifest_sha256 = sha256_file(
                    run_dir / "execution_manifest.json"
                )
                _atomic_json(
                    result_path,
                    execution_result.model_dump(mode="json"),
                )
                _update_stage(
                    run_dir,
                    "execution",
                    execution_result.status,
                    error=execution_result.error,
                )

                if execution_result.status != "failed" or repair_count >= max_repairs:
                    break
                fingerprint = _failure_fingerprint(run_dir, execution_result)
                if fingerprint in seen_failure_fingerprints:
                    _update_stage(
                        run_dir,
                        "execution",
                        "failed",
                        error="repeated execution failure fingerprint; repair loop stopped",
                    )
                    break
                seen_failure_fingerprints.add(fingerprint)
                _archive_execution_attempt(
                    run_dir,
                    execution_result,
                    manifest,
                    execution_attempt,
                )
                repair_count += 1
                try:
                    environment, manifest, _ = await repair_execution(
                        run_dir,
                        target=target,
                        audit=audit,
                        environment=environment,
                        manifest=manifest,
                        failed_result=execution_result,
                        attempt=repair_count,
                        max_repairs=max_repairs,
                        failure_fingerprint=fingerprint,
                        assigned_gpu=assigned_gpu,
                        model=model,
                        max_turns=args.max_turns,
                        max_budget_usd=args.max_budget_usd,
                        rocm_porting=bool(input_data.get("rocm_porting")),
                        force_kernel_port=bool(input_data.get("force_kernel_port")),
                    )
                except (RuntimeError, StageValidationError, ValidationError, ValueError) as exc:
                    _update_stage(
                        run_dir,
                        f"execution_repair_{repair_count}",
                        "failed",
                        error=str(exc),
                    )
                    break

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
