"""Deterministic gates between the staged Claude calls."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from repo2reproduce.tools.paper import find_quote

from .schemas import (
    CodeEvidence,
    ConfigurationAudit,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionRepairOutput,
    ExecutionResult,
    FinalReport,
    PaperEvidence,
    ScientificConfigurationItem,
    TargetContract,
)


class StageValidationError(ValueError):
    """Raised when an agent artifact fails a deterministic workflow gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _paper_evidence(target: TargetContract) -> Iterable[PaperEvidence]:
    for requirement in target.requirements:
        yield requirement.evidence
    yield from target.supporting_evidence


def _audit_paper_evidence(audit: ConfigurationAudit) -> Iterable[PaperEvidence]:
    for item in audit.configuration:
        if item.paper_evidence is not None:
            yield item.paper_evidence


def verify_paper_evidence(parse_json: Path, evidence: Iterable[PaperEvidence]) -> list[str]:
    errors: list[str] = []
    for item in evidence:
        result = find_quote(parse_json, item.quote)
        if not result.get("found"):
            errors.append(f"paper quote did not verify: {item.quote[:120]!r}")
            item.verified = False
            continue
        item.verified = True
        locator = result.get("locator")
        if isinstance(locator, str) and locator:
            item.locator = locator
    return errors


def validate_target(
    run_dir: Path,
    target: TargetContract,
    *,
    verify_evidence: bool = True,
) -> None:
    if target.status != "ready":
        return
    parse_json = run_dir / "paper" / "paper_parse.json"
    if not parse_json.is_file():
        raise StageValidationError("paper/paper_parse.json is missing")
    warnings = verify_paper_evidence(parse_json, _paper_evidence(target)) if verify_evidence else []
    categories = {item.category for item in target.requirements}
    required_categories = {"dataset", "metric"}
    missing_categories = sorted(required_categories - categories)
    if missing_categories:
        warnings.append(
            "target requirements do not cover required categories: " + ", ".join(missing_categories)
        )
    for warning in warnings:
        if warning not in target.warnings:
            target.warnings.append(warning)


def _safe_repo_file(repository: Path, relative: str) -> Path:
    candidate_path = Path(relative)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise StageValidationError(f"unsafe code evidence path: {relative}")
    root = repository.resolve()
    candidate = (repository / candidate_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StageValidationError(f"code evidence escapes repository: {relative}") from exc
    if not candidate.is_file():
        raise StageValidationError(f"code evidence file does not exist: {relative}")
    return candidate


def verify_code_evidence(repository: Path, evidence: CodeEvidence) -> list[str]:
    try:
        path = _safe_repo_file(repository, evidence.path)
    except StageValidationError as exc:
        return [str(exc)]
    if evidence.snippet:
        content = path.read_text(encoding="utf-8", errors="replace")
        if _normalized(evidence.snippet) not in _normalized(content):
            return [f"code evidence snippet was not found in {evidence.path}"]
    return []


def _item_code_evidence(item: ScientificConfigurationItem) -> Iterable[CodeEvidence]:
    if item.code_evidence is not None:
        yield item.code_evidence
    if item.consumed_by is not None:
        yield item.consumed_by


def validate_audit(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    *,
    repository_commit: str,
    verify_evidence: bool = True,
) -> None:
    target_path = run_dir / "target.json"
    expected_target_hash = sha256_file(target_path)
    errors: list[str] = []
    if audit.target_sha256 != expected_target_hash:
        errors.append("configuration audit target_sha256 does not match target.json")
    if audit.repository_commit != repository_commit:
        errors.append("configuration audit repository commit does not match the clone")

    warnings: list[str] = []
    if verify_evidence:
        parse_json = run_dir / "paper" / "paper_parse.json"
        warnings.extend(verify_paper_evidence(parse_json, _audit_paper_evidence(audit)))

        repository = run_dir / "workspace" / audit.repository_path
        evidence: list[CodeEvidence] = [
            *audit.entry_points,
            *(step.evidence for step in audit.execution_path),
        ]
        for item in audit.configuration:
            evidence.extend(_item_code_evidence(item))
        for code_ref in evidence:
            warnings.extend(verify_code_evidence(repository, code_ref))

    requirement_ids = {item.id for item in target.requirements}
    mapped_ids = {
        requirement_id for item in audit.configuration for requirement_id in item.requirement_ids
    }
    missing_requirements = sorted(requirement_ids - mapped_ids)
    unknown_requirements = sorted(mapped_ids - requirement_ids)
    if missing_requirements:
        warnings.append(
            "configuration audit omitted target requirements: " + ", ".join(missing_requirements)
        )
    if unknown_requirements:
        warnings.append(
            "configuration audit references unknown requirements: "
            + ", ".join(unknown_requirements)
        )

    if errors:
        raise StageValidationError("; ".join(errors))
    for warning in warnings:
        if warning not in audit.warnings:
            audit.warnings.append(warning)


def validate_preparation(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    environment: EnvironmentManifest,
    manifest: ExecutionManifest,
) -> None:
    if environment.status != "ready" or manifest.status != "ready":
        return
    errors: list[str] = []
    target_hash = sha256_file(run_dir / "target.json")
    audit_hash = sha256_file(run_dir / "configuration_audit.json")
    environment_hash = sha256_file(run_dir / "environment_manifest.json")
    expected_hashes = {
        "target_sha256": target_hash,
        "audit_sha256": audit_hash,
        "environment_sha256": environment_hash,
    }
    for field, expected in expected_hashes.items():
        if getattr(manifest, field) != expected:
            errors.append(f"execution manifest {field} does not match its source artifact")

    if manifest.selected_experiment != target.selected_experiment:
        errors.append("execution manifest selected experiment differs from target")
    if manifest.repository_commit != audit.repository_commit:
        errors.append("execution manifest repository commit differs from configuration audit")
    if manifest.entrypoint not in audit.entry_points:
        errors.append("execution manifest entrypoint is absent from configuration audit")
    if manifest.metric_name != target.metric_name:
        errors.append("execution manifest metric differs from target")
    if manifest.paper_target_value != target.metric_value:
        errors.append("execution manifest paper target value differs from target")
    if manifest.metric_artifact not in manifest.expected_outputs:
        errors.append("execution manifest expected outputs omit the metric artifact")
    if (
        manifest.metric_extractor is None
        or manifest.metric_extractor.source != manifest.metric_artifact
    ):
        errors.append("metric extractor must read the manifest metric artifact")
    if manifest.image_tag != environment.image_tag:
        errors.append("execution manifest image differs from environment manifest")

    manifest_inputs = {item.audit_item_id: item for item in manifest.scientific_configuration}
    if manifest.launch_script:
        launch_path = run_dir / manifest.launch_script
        if launch_path.is_file():
            launch_text = launch_path.read_text(encoding="utf-8", errors="replace")
            for resolved in manifest.scientific_configuration:
                if resolved.rendered and resolved.rendered not in launch_text:
                    errors.append(
                        "launch script does not contain rendered input "
                        f"{resolved.audit_item_id}: {resolved.rendered}"
                    )

    assert environment.dockerfile is not None
    assert environment.dockerfile_sha256 is not None
    assert environment.build_log_path is not None
    dockerfile = run_dir / environment.dockerfile
    if not dockerfile.is_file():
        errors.append("environment Dockerfile is missing")
    elif sha256_file(dockerfile) != environment.dockerfile_sha256:
        errors.append("environment Dockerfile hash does not match")
    build_log = run_dir / environment.build_log_path
    if not build_log.is_file() or build_log.stat().st_size == 0:
        errors.append("environment build log is missing or empty")

    audit_ids = {item.id for item in audit.configuration}
    unknown_inputs = sorted(set(manifest_inputs) - audit_ids)
    if unknown_inputs:
        errors.append(
            "manifest scientific configuration references unknown audit items: "
            + ", ".join(unknown_inputs)
        )
    if errors:
        raise StageValidationError("; ".join(errors))


def validate_execution_repair(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    previous_manifest: ExecutionManifest,
    output: ExecutionRepairOutput,
    *,
    attempt: int,
    failure_fingerprint: str,
    target_sha256: str,
    audit_sha256: str,
) -> None:
    """Reject repairs that alter frozen science or lack concrete repair evidence."""

    repair = output.repair
    errors: list[str] = []
    if repair.attempt != attempt:
        errors.append(f"repair attempt must be {attempt}")
    if repair.failure_fingerprint != failure_fingerprint:
        errors.append("repair failure fingerprint differs from the Python-owned fingerprint")
    if sha256_file(run_dir / "target.json") != target_sha256:
        errors.append("repair modified immutable target.json")
    if sha256_file(run_dir / "configuration_audit.json") != audit_sha256:
        errors.append("repair modified immutable configuration_audit.json")

    environment = output.environment
    manifest = output.manifest
    if environment.status != "ready" or manifest.status != "ready":
        errors.append("execution repair must return ready environment and execution manifests")
    if manifest.selected_experiment != target.selected_experiment:
        errors.append("repair changed the selected experiment")
    if manifest.metric_name != target.metric_name:
        errors.append("repair changed the target metric")
    if manifest.paper_target_value != target.metric_value:
        errors.append("repair changed the paper target value")
    if manifest.repository_commit != audit.repository_commit:
        errors.append("repair changed the pinned repository commit")

    before_values = {
        item.audit_item_id: item.value for item in previous_manifest.scientific_configuration
    }
    after_values = {item.audit_item_id: item.value for item in manifest.scientific_configuration}
    if after_values != before_values:
        errors.append("repair changed immutable scientific configuration values")

    if not repair.changes:
        errors.append("repair made no evidenced file changes")
    for change in repair.changes:
        if change.path in {"target.json", "configuration_audit.json"}:
            errors.append(f"repair cannot change immutable artifact: {change.path}")
            continue
        path = run_dir / change.path
        if not path.is_file():
            errors.append(f"repair evidence file is missing: {change.path}")
        elif not change.sha256 or sha256_file(path) != change.sha256:
            errors.append(f"repair file hash does not match: {change.path}")

    smoke = repair.smoke_test
    smoke_log = run_dir / smoke.log_path
    if smoke.exit_code != 0:
        errors.append("repair smoke test did not succeed")
    if not smoke_log.is_file() or smoke_log.stat().st_size == 0:
        errors.append("repair smoke-test log is missing or empty")

    patch = repair.repository_patch
    patch_path = run_dir / patch.diff_path
    if patch.base_commit != audit.repository_commit:
        errors.append("repository patch is not based on the pinned commit")
    if not patch_path.is_file() or sha256_file(patch_path) != patch.diff_sha256:
        errors.append("repository patch evidence is missing or has the wrong hash")

    if errors:
        raise StageValidationError("; ".join(errors))


def validate_final_report(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    environment: EnvironmentManifest,
    manifest: ExecutionManifest,
    result: ExecutionResult,
    report: FinalReport,
) -> None:
    """Prevent finalization from substituting targets or unsupported result values."""

    errors: list[str] = []
    if report.selected_experiment != target.selected_experiment:
        errors.append("final report selected experiment differs from target")
    if report.metric_name != target.metric_name:
        errors.append("final report metric differs from target")
    if report.paper_target_value != target.metric_value:
        errors.append("final report paper value differs from target")
    if report.result_location.description != target.result_location:
        errors.append("final report paper location differs from target")
    if report.execution_status != result.status:
        errors.append("final report execution status differs from execution result")
    if report.achieved_value != result.achieved_metric:
        errors.append("final report achieved value differs from deterministic extraction")
    if (result.status != "succeeded" or result.achieved_metric is None) and (
        report.comparison.comparable
        or report.comparison.verdict in {"match", "miss"}
    ):
        errors.append("final report claims comparability without a successful extracted metric")

    report_paths = report.artifacts
    expected_paths = {
        "configuration_used": "execution_manifest.json",
        "configuration_audit": "configuration_audit.json",
        "environment_manifest": "environment_manifest.json",
        "dockerfile": environment.dockerfile,
        "build_log": environment.build_log_path,
        "launch_script": manifest.launch_script,
        "execution_result": "execution_result.json",
        "metric_artifact": result.metric_source or manifest.metric_artifact,
        "assumptions": "assumptions.json",
    }
    log_path = Path(result.log_path)
    if log_path.is_absolute():
        try:
            expected_executor_log = log_path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            expected_executor_log = "logs/executor.log"
    else:
        expected_executor_log = log_path.as_posix()
    expected_paths["executor_log"] = expected_executor_log
    for field_name, expected in expected_paths.items():
        if getattr(report_paths, field_name) != expected:
            errors.append(f"final report {field_name} path differs from its manifest")
    if report_paths.launch_script_sha256 != manifest.launch_script_sha256:
        errors.append("final report launch script hash differs from execution manifest")

    audit_ids = {item.id for item in audit.configuration}
    assumption_audit_ids = {
        audit_item_id
        for assumption in report.assumptions
        for audit_item_id in assumption.audit_item_ids
    }
    unknown_assumption_ids = sorted(assumption_audit_ids - audit_ids)
    if unknown_assumption_ids:
        errors.append(
            "final report assumptions reference unknown audit items: "
            + ", ".join(unknown_assumption_ids)
        )
    missing_fallbacks = sorted(
        item.id
        for item in audit.configuration
        if item.finding == "paper_missing" and item.id not in assumption_audit_ids
    )
    if missing_fallbacks:
        errors.append(
            "final report omitted paper-missing assumptions: "
            + ", ".join(missing_fallbacks)
        )
    if errors:
        raise StageValidationError("; ".join(errors))
