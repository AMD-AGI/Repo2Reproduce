"""Structured contracts for the staged Claude reproduction pipeline."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ScientificCategory = Literal[
    "dataset",
    "preprocessing",
    "model",
    "checkpoint",
    "optimization",
    "training",
    "inference",
    "simulation",
    "evaluation",
    "metric",
    "aggregation",
    "seed",
    "sample_count",
    "threshold",
    "other",
]


def _relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    return value


class PaperEvidence(BaseModel):
    """Verbatim paper evidence for a scientific setting or target."""

    quote: str = Field(..., min_length=8)
    locator: str
    verified: bool = False


class CodeEvidence(BaseModel):
    """A real repository location supporting an audit claim."""

    path: str
    locator: str | None = None
    snippet: str | None = None


class ScientificRequirement(BaseModel):
    """One paper-stated requirement for the selected result."""

    id: str = Field(..., min_length=1)
    category: ScientificCategory
    name: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    critical: bool = True
    evidence: PaperEvidence


class TargetContract(BaseModel):
    """Frozen paper-grounded definition of the single result to reproduce."""

    schema_version: Literal[2] = 2
    status: Literal["ready", "blocked"]
    selected_experiment: str
    contribution: str
    result_location: str
    metric_name: str
    metric_value: str
    metric_units: str | None = None
    metric_definition: str
    aggregation: str
    selection_reason: str
    requirements: list[ScientificRequirement] = Field(default_factory=list)
    supporting_evidence: list[PaperEvidence] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> TargetContract:
        if self.status == "blocked":
            if not self.blocked_reason or not self.blocked_reason.strip():
                raise ValueError("blocked_reason is required when target selection is blocked")
            return self
        required_text = {
            "selected_experiment": self.selected_experiment,
            "contribution": self.contribution,
            "result_location": self.result_location,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_definition": self.metric_definition,
            "aggregation": self.aggregation,
            "selection_reason": self.selection_reason,
        }
        missing = [name for name, value in required_text.items() if not value.strip()]
        if missing:
            raise ValueError(f"ready target is missing: {', '.join(missing)}")
        if not self.requirements:
            raise ValueError("ready target requires at least one scientific requirement")
        ids = [item.id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("target scientific requirement ids must be unique")
        return self


AuditStatus = Literal[
    "match",
    "conflict_resolved",
    "paper_only",
    "code_only_assumption",
    "hardcoded",
    "missing",
    "paper_detail_not_implemented",
]


class AuditItem(BaseModel):
    """One direction of the bidirectional paper/code configuration audit."""

    id: str = Field(..., min_length=1)
    direction: Literal["code_to_paper", "paper_to_code"]
    requirement_id: str | None = None
    category: ScientificCategory
    name: str = Field(..., min_length=1)
    critical: bool = True
    status: AuditStatus
    code_value: str | None = None
    paper_value: str | None = None
    resolved_value: str | None = None
    paper_evidence: PaperEvidence | None = None
    code_evidence: CodeEvidence | None = None
    consumed_by: CodeEvidence | None = None
    resolution: str

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> AuditItem:
        if self.status in {"match", "conflict_resolved"}:
            if self.paper_evidence is None or self.code_evidence is None:
                raise ValueError(f"{self.status} requires paper and code evidence")
            if self.paper_value is None or self.code_value is None or self.resolved_value is None:
                raise ValueError(f"{self.status} requires paper, code, and resolved values")
        if self.status == "conflict_resolved" and self.resolved_value != self.paper_value:
            raise ValueError("conflicts must resolve to the paper value")
        if self.status == "code_only_assumption":
            if self.code_evidence is None or self.code_value is None:
                raise ValueError("code_only_assumption requires code evidence and code value")
            if self.resolved_value is None:
                raise ValueError("code_only_assumption requires an explicit resolved value")
        if self.status in {
            "paper_only",
            "paper_detail_not_implemented",
        } and (self.paper_evidence is None or self.paper_value is None):
            raise ValueError(f"{self.status} requires paper evidence and paper value")
        if self.status == "hardcoded" and (self.code_evidence is None or self.code_value is None):
            raise ValueError("hardcoded requires code evidence and code value")
        if self.direction == "paper_to_code" and not self.requirement_id:
            raise ValueError("paper_to_code items require requirement_id")
        return self


class ExecutionPathStep(BaseModel):
    order: int = Field(..., ge=1)
    role: str
    evidence: CodeEvidence
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class ConfigurationAudit(BaseModel):
    """Closed bidirectional inventory for the selected experiment."""

    schema_version: Literal[2] = 2
    status: Literal["ready", "blocked"]
    target_sha256: str
    experiment_kind: str
    entry_points: list[CodeEvidence] = Field(default_factory=list)
    execution_path: list[ExecutionPathStep] = Field(default_factory=list)
    code_to_paper: list[AuditItem] = Field(default_factory=list)
    paper_to_code: list[AuditItem] = Field(default_factory=list)
    code_to_paper_complete: bool = False
    paper_to_code_complete: bool = False
    critical_gaps: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_audit_state(self) -> ConfigurationAudit:
        if not _SHA256.fullmatch(self.target_sha256):
            raise ValueError("target_sha256 must be a lowercase SHA-256 digest")
        all_items = [*self.code_to_paper, *self.paper_to_code]
        ids = [item.id for item in all_items]
        if len(ids) != len(set(ids)):
            raise ValueError("configuration audit item ids must be unique")
        if self.status == "blocked":
            if not self.blocked_reason or not self.blocked_reason.strip():
                raise ValueError("blocked_reason is required when audit is blocked")
            return self
        if not self.entry_points or not self.execution_path:
            raise ValueError("ready audit requires entry points and an execution path")
        if not self.code_to_paper_complete or not self.paper_to_code_complete:
            raise ValueError("ready audit requires both audit directions to be complete")
        unresolved = [
            item.id
            for item in all_items
            if item.critical
            and item.status
            in {"paper_only", "hardcoded", "missing", "paper_detail_not_implemented"}
        ]
        if unresolved or self.critical_gaps:
            detail = unresolved + self.critical_gaps
            raise ValueError(f"ready audit contains critical unresolved items: {detail}")
        return self


class ResolvedInput(BaseModel):
    """A resolved scientific input that must reach the final execution."""

    audit_item_id: str
    name: str
    value: str
    delivery: Literal["cli", "config", "environment", "data", "artifact", "hardcoded", "other"]
    rendered: str | None = Field(
        None,
        description="Exact launch-script fragment when this value is delivered by the command.",
    )

    @model_validator(mode="after")
    def validate_delivery(self) -> ResolvedInput:
        if self.delivery == "cli" and not self.rendered:
            raise ValueError("CLI-delivered inputs require an exact rendered fragment")
        return self


class ExecutionSpec(BaseModel):
    """Scientific execution specification produced by the audit stage."""

    schema_version: Literal[2] = 2
    status: Literal["ready", "blocked"]
    target_sha256: str
    audit_sha256: str | None = None
    repository_path: str
    repository_commit: str
    experiment_kind: str
    entrypoint: CodeEvidence
    command_outline: str
    resolved_inputs: list[ResolvedInput] = Field(default_factory=list)
    result_artifact: str
    metric_extraction: str
    approved_scientific_patches: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> ExecutionSpec:
        if not _SHA256.fullmatch(self.target_sha256):
            raise ValueError("target_sha256 must be a lowercase SHA-256 digest")
        if self.audit_sha256 is not None and not _SHA256.fullmatch(self.audit_sha256):
            raise ValueError("audit_sha256 must be a lowercase SHA-256 digest")
        self.repository_path = _relative_path(self.repository_path, "repository_path")
        self.result_artifact = _relative_path(self.result_artifact, "result_artifact")
        if self.status == "blocked":
            if not self.blocked_reason or not self.blocked_reason.strip():
                raise ValueError("blocked_reason is required when execution spec is blocked")
            return self
        if not self.command_outline.strip() or not self.resolved_inputs:
            raise ValueError("ready execution spec requires a command outline and resolved inputs")
        ids = [item.audit_item_id for item in self.resolved_inputs]
        if len(ids) != len(set(ids)):
            raise ValueError("resolved input audit_item_id values must be unique")
        return self


class AuditStageOutput(BaseModel):
    """Single structured output from code tracing and reconciliation."""

    audit: ConfigurationAudit
    execution_spec: ExecutionSpec


class SmokeTestResult(BaseModel):
    """Evidence that the prepared image and command shape passed a bounded pilot."""

    exit_code: int
    log_path: str
    description: str

    @model_validator(mode="after")
    def validate_log_path(self) -> SmokeTestResult:
        self.log_path = _relative_path(self.log_path, "smoke_test.log_path")
        return self


class EnvironmentManifest(BaseModel):
    """Durable evidence for the environment build that precedes full execution."""

    schema_version: Literal[2] = 2
    status: Literal["ready", "blocked"]
    image_tag: str | None = None
    dockerfile: str | None = None
    dockerfile_sha256: str | None = None
    build_log_path: str | None = None
    build_succeeded: bool = False
    notes: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_environment(self) -> EnvironmentManifest:
        if self.status == "blocked":
            if not self.blocked_reason or not self.blocked_reason.strip():
                raise ValueError("blocked_reason is required when environment is blocked")
            return self
        required = {
            "image_tag": self.image_tag,
            "dockerfile": self.dockerfile,
            "dockerfile_sha256": self.dockerfile_sha256,
            "build_log_path": self.build_log_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"ready environment is missing: {', '.join(missing)}")
        assert self.dockerfile is not None
        assert self.dockerfile_sha256 is not None
        assert self.build_log_path is not None
        self.dockerfile = _relative_path(self.dockerfile, "dockerfile")
        self.build_log_path = _relative_path(self.build_log_path, "build_log_path")
        if not _SHA256.fullmatch(self.dockerfile_sha256):
            raise ValueError("dockerfile_sha256 must be a lowercase SHA-256 digest")
        if not self.build_succeeded:
            raise ValueError("ready environment requires a successful build")
        return self


class MetricExtractor(BaseModel):
    """Machine-readable rule for extracting the reproduced value."""

    source: str = Field(
        ...,
        description="Path relative to the run artifacts directory.",
    )
    format: Literal["json", "regex"]
    selector: str = Field(
        ...,
        min_length=1,
        description="Dot-separated JSON path or regex with a named/first capture group.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> MetricExtractor:
        self.source = _relative_path(self.source, "metric_extractor.source")
        return self


class ExecutionManifest(BaseModel):
    """Validated handoff produced by the preparation agent."""

    schema_version: Literal[1, 2] = 1
    status: Literal["ready", "blocked"]
    selected_experiment: str

    image_tag: str | None = None
    launch_script: str | None = None
    launch_script_sha256: str | None = None
    host_repository: str | None = None
    container_repository: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0, le=604800)
    expected_outputs: list[str] = Field(default_factory=list)
    smoke_test: SmokeTestResult | None = None
    blocked_reason: str | None = None
    target_sha256: str | None = None
    audit_sha256: str | None = None
    execution_spec_sha256: str | None = None
    environment_sha256: str | None = None
    repository_commit: str | None = None
    entrypoint: CodeEvidence | None = None
    scientific_configuration: list[ResolvedInput] = Field(default_factory=list)
    metric_name: str | None = None
    metric_artifact: str | None = None
    paper_target_value: str | None = None
    metric_extractor: MetricExtractor | None = None

    @model_validator(mode="after")
    def validate_state(self) -> ExecutionManifest:
        if not self.selected_experiment.strip():
            raise ValueError("selected_experiment cannot be empty")

        if self.status == "blocked":
            if not self.blocked_reason or not self.blocked_reason.strip():
                raise ValueError("blocked_reason is required when status is blocked")
            return self

        required = {
            "image_tag": self.image_tag,
            "launch_script": self.launch_script,
            "launch_script_sha256": self.launch_script_sha256,
            "host_repository": self.host_repository,
            "container_repository": self.container_repository,
            "timeout_seconds": self.timeout_seconds,
            "smoke_test": self.smoke_test,
        }
        missing = [name for name, value in required.items() if value is None or value == ""]
        if missing:
            raise ValueError(f"ready manifest is missing: {', '.join(missing)}")

        assert self.launch_script is not None
        assert self.host_repository is not None
        assert self.container_repository is not None
        assert self.launch_script_sha256 is not None
        assert self.smoke_test is not None

        self.launch_script = _relative_path(self.launch_script, "launch_script")
        self.host_repository = _relative_path(self.host_repository, "host_repository")
        expected_container_repository = str(PurePosixPath("/run/workspace") / self.host_repository)
        if self.container_repository != expected_container_repository:
            raise ValueError(
                "container_repository must map exactly to host_repository below /run/workspace"
            )
        if not _SHA256.fullmatch(self.launch_script_sha256):
            raise ValueError("launch_script_sha256 must be a lowercase SHA-256 digest")
        if self.smoke_test.exit_code != 0:
            raise ValueError("ready manifest requires a successful smoke test")
        if not self.expected_outputs:
            raise ValueError("ready manifest requires at least one expected output")

        for output in self.expected_outputs:
            _relative_path(output, "expected_outputs item")
        if self.schema_version == 2:
            v2_required = {
                "target_sha256": self.target_sha256,
                "audit_sha256": self.audit_sha256,
                "execution_spec_sha256": self.execution_spec_sha256,
                "repository_commit": self.repository_commit,
                "entrypoint": self.entrypoint,
                "metric_name": self.metric_name,
                "metric_artifact": self.metric_artifact,
                "paper_target_value": self.paper_target_value,
                "metric_extractor": self.metric_extractor,
            }
            missing_v2 = [name for name, value in v2_required.items() if not value]
            if missing_v2:
                raise ValueError(f"version 2 manifest is missing: {', '.join(missing_v2)}")
            for name in (
                "target_sha256",
                "audit_sha256",
                "execution_spec_sha256",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            if not self.scientific_configuration:
                raise ValueError("version 2 manifest requires scientific_configuration")
            assert self.metric_artifact is not None
            self.metric_artifact = _relative_path(self.metric_artifact, "metric_artifact")
        return self


class PreparationStageOutput(BaseModel):
    """Single structured output from environment preparation and smoke testing."""

    environment: EnvironmentManifest
    manifest: ExecutionManifest


class ExecutionResult(BaseModel):
    """Durable result emitted by the Python executor."""

    schema_version: Literal[1] = 1
    status: Literal["succeeded", "failed", "timed_out", "blocked"]
    command: list[str]
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    log_path: str
    expected_outputs: list[str]
    missing_outputs: list[str]
    error: str | None = None
    manifest_sha256: str | None = None
    achieved_metric: str | None = None
    metric_source: str | None = None
    metric_extraction_error: str | None = None
