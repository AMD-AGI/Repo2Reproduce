"""Deterministic gates between the staged Claude calls."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from repo2reproduce.tools.paper import find_quote

from .schemas import (
    AuditItem,
    CodeEvidence,
    ConfigurationAudit,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionResult,
    ExecutionSpec,
    FinalReport,
    PaperEvidence,
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
    for item in [*audit.code_to_paper, *audit.paper_to_code]:
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


def validate_target(run_dir: Path, target: TargetContract) -> None:
    if target.status != "ready":
        return
    parse_json = run_dir / "paper" / "paper_parse.json"
    if not parse_json.is_file():
        raise StageValidationError("paper/paper_parse.json is missing")
    errors = verify_paper_evidence(parse_json, _paper_evidence(target))
    categories = {item.category for item in target.requirements}
    required_categories = {"dataset", "metric"}
    missing_categories = sorted(required_categories - categories)
    if missing_categories:
        errors.append(
            "target requirements do not cover required categories: " + ", ".join(missing_categories)
        )
    if errors:
        raise StageValidationError("; ".join(errors))


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


def _item_code_evidence(item: AuditItem) -> Iterable[CodeEvidence]:
    if item.code_evidence is not None:
        yield item.code_evidence
    if item.consumed_by is not None:
        yield item.consumed_by


def validate_audit(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    spec: ExecutionSpec,
    *,
    repository_commit: str,
) -> None:
    if audit.status != "ready" or spec.status != "ready":
        return
    target_path = run_dir / "target.json"
    expected_target_hash = sha256_file(target_path)
    errors: list[str] = []
    if audit.target_sha256 != expected_target_hash:
        errors.append("configuration audit target_sha256 does not match target.json")
    if spec.target_sha256 != expected_target_hash:
        errors.append("execution spec target_sha256 does not match target.json")
    if spec.repository_commit != repository_commit:
        errors.append("execution spec repository commit does not match the cloned repository")

    parse_json = run_dir / "paper" / "paper_parse.json"
    errors.extend(verify_paper_evidence(parse_json, _audit_paper_evidence(audit)))

    repository = run_dir / "workspace" / spec.repository_path
    evidence: list[CodeEvidence] = [
        *audit.entry_points,
        spec.entrypoint,
        *(step.evidence for step in audit.execution_path),
    ]
    for item in [*audit.code_to_paper, *audit.paper_to_code]:
        evidence.extend(_item_code_evidence(item))
    for code_ref in evidence:
        errors.extend(verify_code_evidence(repository, code_ref))

    requirement_ids = {item.id for item in target.requirements}
    mapped_ids = {
        item.requirement_id for item in audit.paper_to_code if item.requirement_id is not None
    }
    missing_requirements = sorted(requirement_ids - mapped_ids)
    unknown_requirements = sorted(mapped_ids - requirement_ids)
    if missing_requirements:
        errors.append(
            "paper-to-code audit omitted target requirements: " + ", ".join(missing_requirements)
        )
    if unknown_requirements:
        errors.append(
            "paper-to-code audit references unknown requirements: "
            + ", ".join(unknown_requirements)
        )
    if any(item.direction != "code_to_paper" for item in audit.code_to_paper):
        errors.append("code_to_paper contains an item with the wrong direction")
    if any(item.direction != "paper_to_code" for item in audit.paper_to_code):
        errors.append("paper_to_code contains an item with the wrong direction")

    audit_items = {item.id: item for item in [*audit.code_to_paper, *audit.paper_to_code]}
    resolved_ids = {item.audit_item_id for item in spec.resolved_inputs}
    expected_resolved_ids = {
        item.id for item in audit.code_to_paper if item.resolved_value is not None
    }
    omitted_inputs = sorted(expected_resolved_ids - resolved_ids)
    if omitted_inputs:
        errors.append("execution spec omitted resolved code inputs: " + ", ".join(omitted_inputs))
    for resolved in spec.resolved_inputs:
        item = audit_items.get(resolved.audit_item_id)
        if item is None:
            errors.append(f"resolved input references unknown audit item: {resolved.audit_item_id}")
            continue
        if item.resolved_value is None:
            errors.append(
                f"resolved input references audit item without a resolved value: {item.id}"
            )
        elif resolved.value != item.resolved_value:
            errors.append(f"resolved input value disagrees with audit item: {item.id}")

    if errors:
        raise StageValidationError("; ".join(errors))


def validate_preparation(
    run_dir: Path,
    target: TargetContract,
    audit: ConfigurationAudit,
    spec: ExecutionSpec,
    environment: EnvironmentManifest,
    manifest: ExecutionManifest,
) -> None:
    if environment.status != "ready" or manifest.status != "ready":
        return
    errors: list[str] = []
    target_hash = sha256_file(run_dir / "target.json")
    audit_hash = sha256_file(run_dir / "configuration_audit.json")
    spec_hash = sha256_file(run_dir / "execution_spec.json")
    environment_hash = sha256_file(run_dir / "environment_manifest.json")
    expected_hashes = {
        "target_sha256": target_hash,
        "audit_sha256": audit_hash,
        "execution_spec_sha256": spec_hash,
        "environment_sha256": environment_hash,
    }
    for field, expected in expected_hashes.items():
        if getattr(manifest, field) != expected:
            errors.append(f"execution manifest {field} does not match its source artifact")

    if manifest.selected_experiment != target.selected_experiment:
        errors.append("execution manifest selected experiment differs from target")
    if manifest.repository_commit != spec.repository_commit:
        errors.append("execution manifest repository commit differs from execution spec")
    if manifest.entrypoint != spec.entrypoint:
        errors.append("execution manifest entrypoint differs from execution spec")
    if manifest.metric_name != target.metric_name:
        errors.append("execution manifest metric differs from target")
    if manifest.paper_target_value != target.metric_value:
        errors.append("execution manifest paper target value differs from target")
    if manifest.metric_artifact != spec.result_artifact:
        errors.append("execution manifest metric artifact differs from execution spec")
    if spec.result_artifact not in manifest.expected_outputs:
        errors.append("execution manifest expected outputs omit the metric artifact")
    if (
        manifest.metric_extractor is None
        or manifest.metric_extractor.source != spec.result_artifact
    ):
        errors.append("metric extractor must read the execution specification result artifact")
    if manifest.image_tag != environment.image_tag:
        errors.append("execution manifest image differs from environment manifest")

    manifest_inputs = {
        item.audit_item_id: item.model_dump(mode="json")
        for item in manifest.scientific_configuration
    }
    spec_inputs = {
        item.audit_item_id: item.model_dump(mode="json") for item in spec.resolved_inputs
    }
    if manifest_inputs != spec_inputs:
        errors.append("execution manifest scientific configuration differs from execution spec")
    if manifest.launch_script:
        launch_path = run_dir / "workspace" / manifest.launch_script
        if launch_path.is_file():
            launch_text = launch_path.read_text(encoding="utf-8", errors="replace")
            for resolved in spec.resolved_inputs:
                if resolved.rendered and resolved.rendered not in launch_text:
                    errors.append(
                        "launch script does not contain rendered input "
                        f"{resolved.audit_item_id}: {resolved.rendered}"
                    )

    assert environment.dockerfile is not None
    assert environment.dockerfile_sha256 is not None
    assert environment.build_log_path is not None
    dockerfile = run_dir / "workspace" / environment.dockerfile
    if not dockerfile.is_file():
        errors.append("environment Dockerfile is missing")
    elif sha256_file(dockerfile) != environment.dockerfile_sha256:
        errors.append("environment Dockerfile hash does not match")
    build_log = run_dir / environment.build_log_path
    if not build_log.is_file() or build_log.stat().st_size == 0:
        errors.append("environment build log is missing or empty")

    audit_ids = {item.id for item in [*audit.code_to_paper, *audit.paper_to_code]}
    unknown_inputs = sorted(set(manifest_inputs) - audit_ids)
    if unknown_inputs:
        errors.append(
            "manifest scientific configuration references unknown audit items: "
            + ", ".join(unknown_inputs)
        )
    if errors:
        raise StageValidationError("; ".join(errors))


def validate_final_report(
    target: TargetContract,
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
    if report.execution_status != result.status:
        errors.append("final report execution status differs from execution result")
    if report.achieved_value != result.achieved_metric:
        errors.append("final report achieved value differs from deterministic extraction")
    if (result.status != "succeeded" or result.achieved_metric is None) and (
        report.comparable or report.match is not None
    ):
        errors.append("final report claims comparability without a successful extracted metric")
    if errors:
        raise StageValidationError("; ".join(errors))
