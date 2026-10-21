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
    if (
        path.is_absolute()
        or ".." in path.parts
        or value in {"", "."}
        or any(character in value for character in ("\n", "\r", "\0"))
    ):
        raise ValueError(f"{field_name} must be a non-empty relative path")
    return value


def _run_root_path(value: str, field_name: str, prefix: str) -> str:
    value = _relative_path(value, field_name)
    if PurePosixPath(value).parts[0] != prefix:
        raise ValueError(f"{field_name} must be relative to the run root and start with {prefix}/")
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
    warnings: list[str] = Field(default_factory=list)
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


class ExecutionPathStep(BaseModel):
    order: int = Field(..., ge=1)
    role: str
    evidence: CodeEvidence
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


ScientificFinding = Literal[
    "match",
    "paper_preferred",
    "paper_missing",
    "paper_partially_implemented",
    "paper_not_implemented",
]


class ScientificConfigurationItem(BaseModel):
    """One canonical scientific input shared by both audit directions."""

    id: str = Field(..., min_length=1)
    category: ScientificCategory
    name: str = Field(..., min_length=1)
    requirement_ids: list[str] = Field(default_factory=list)
    paper_value: str | None = None
    paper_evidence: PaperEvidence | None = None
    code_default: str | None = None
    code_experiment_value: str | None = None
    code_evidence: CodeEvidence | None = None
    consumed_by: CodeEvidence | None = None
    chosen_value: str | None = None
    finding: ScientificFinding
    notes: str = ""

    @model_validator(mode="after")
    def validate_finding(self) -> ScientificConfigurationItem:
        if self.paper_value is not None and self.paper_evidence is None:
            raise ValueError("paper values require verbatim paper evidence")
        if self.finding in {"match", "paper_preferred"} and (
            self.paper_value is None or self.code_evidence is None or self.chosen_value is None
        ):
            raise ValueError(
                f"{self.finding} requires paper value, code evidence, and chosen value"
            )
        if self.finding == "paper_missing":
            if self.paper_value is not None or self.code_evidence is None:
                raise ValueError("paper_missing requires code evidence and no paper value")
            if self.chosen_value is None:
                raise ValueError("paper_missing requires an explicit code fallback")
        if self.finding == "paper_partially_implemented" and (
            self.paper_value is None or self.code_evidence is None
        ):
            raise ValueError("paper_partially_implemented requires paper value and code evidence")
        if self.finding == "paper_not_implemented" and (
            self.paper_value is None or not self.requirement_ids
        ):
            raise ValueError("paper_not_implemented requires a paper value and target requirement")
        return self


class ConfigurationAudit(BaseModel):
    """Canonical scientific configuration built from both audit directions."""

    schema_version: Literal[3] = 3
    target_sha256: str
    repository_path: str
    repository_commit: str
    experiment_kind: str
    entry_points: list[CodeEvidence] = Field(default_factory=list)
    execution_path: list[ExecutionPathStep] = Field(default_factory=list)
    configuration: list[ScientificConfigurationItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_audit_state(self) -> ConfigurationAudit:
        if not _SHA256.fullmatch(self.target_sha256):
            raise ValueError("target_sha256 must be a lowercase SHA-256 digest")
        self.repository_path = _relative_path(self.repository_path, "repository_path")
        if not self.repository_commit.strip():
            raise ValueError("repository_commit cannot be empty")
        if not self.entry_points or not self.execution_path:
            raise ValueError("configuration audit requires entry points and an execution path")
        if not self.configuration:
            raise ValueError("configuration audit requires scientific configuration items")
        ids = [item.id for item in self.configuration]
        if len(ids) != len(set(ids)):
            raise ValueError("configuration audit item ids must be unique")
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


class AuditStageOutput(BaseModel):
    """Single structured output from code tracing and reconciliation."""

    audit: ConfigurationAudit


class SmokeTestResult(BaseModel):
    """Evidence that the prepared image and command shape passed a bounded pilot."""

    exit_code: int
    log_path: str = Field(
        ...,
        description="Run-root-relative path starting with artifacts/.",
    )
    description: str

    @model_validator(mode="after")
    def validate_log_path(self) -> SmokeTestResult:
        self.log_path = _run_root_path(self.log_path, "smoke_test.log_path", "artifacts")
        return self


class EnvironmentManifest(BaseModel):
    """Durable evidence for the environment build that precedes full execution."""

    schema_version: Literal[2] = 2
    status: Literal["ready", "blocked"]
    image_tag: str | None = None
    dockerfile: str | None = Field(
        default=None,
        description=(
            "Run-root-relative path starting with workspace/. This field is a path, "
            "never Dockerfile contents."
        ),
    )
    dockerfile_sha256: str | None = None
    build_log_path: str | None = Field(
        default=None,
        description="Run-root-relative path starting with artifacts/.",
    )
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
        self.dockerfile = _run_root_path(self.dockerfile, "dockerfile", "workspace")
        self.build_log_path = _run_root_path(self.build_log_path, "build_log_path", "artifacts")
        if not _SHA256.fullmatch(self.dockerfile_sha256):
            raise ValueError("dockerfile_sha256 must be a lowercase SHA-256 digest")
        if not self.build_succeeded:
            raise ValueError("ready environment requires a successful build")
        return self


class MetricExtractor(BaseModel):
    """Machine-readable rule for extracting the reproduced value."""

    source: str = Field(
        ...,
        description="Run-root-relative metric path starting with artifacts/.",
    )
    format: Literal["json", "regex"]
    selector: str = Field(
        ...,
        min_length=1,
        description="Dot-separated JSON path or regex with a named/first capture group.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> MetricExtractor:
        self.source = _run_root_path(self.source, "metric_extractor.source", "artifacts")
        return self


class ExecutionManifest(BaseModel):
    """Validated handoff produced by the preparation agent."""

    schema_version: Literal[1, 3] = 3
    status: Literal["ready", "blocked"]
    selected_experiment: str

    image_tag: str | None = None
    launch_script: str | None = Field(
        default=None,
        description="Run-root-relative path starting with workspace/.",
    )
    launch_script_sha256: str | None = None
    host_repository: str | None = Field(
        default=None,
        description="Run-root-relative repository path starting with workspace/.",
    )
    container_repository: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0, le=604800)
    expected_outputs: list[str] = Field(
        default_factory=list,
        description="Run-root-relative output paths, each starting with artifacts/.",
    )
    smoke_test: SmokeTestResult | None = None
    blocked_reason: str | None = None
    target_sha256: str | None = None
    audit_sha256: str | None = None
    environment_sha256: str | None = None
    repository_commit: str | None = None
    entrypoint: CodeEvidence | None = None
    scientific_configuration: list[ResolvedInput] = Field(default_factory=list)
    metric_name: str | None = None
    metric_artifact: str | None = Field(
        default=None,
        description="Run-root-relative metric path starting with artifacts/.",
    )
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

        self.launch_script = _run_root_path(self.launch_script, "launch_script", "workspace")
        self.host_repository = _run_root_path(self.host_repository, "host_repository", "workspace")
        expected_container_repository = str(PurePosixPath("/run") / self.host_repository)
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
            _run_root_path(output, "expected_outputs item", "artifacts")
        if self.schema_version == 3:
            versioned_required = {
                "target_sha256": self.target_sha256,
                "audit_sha256": self.audit_sha256,
                "repository_commit": self.repository_commit,
                "entrypoint": self.entrypoint,
                "metric_name": self.metric_name,
                "metric_artifact": self.metric_artifact,
                "paper_target_value": self.paper_target_value,
                "metric_extractor": self.metric_extractor,
            }
            missing_versioned = [name for name, value in versioned_required.items() if not value]
            if missing_versioned:
                raise ValueError(
                    f"version {self.schema_version} manifest is missing: "
                    f"{', '.join(missing_versioned)}"
                )
            hash_names = ["target_sha256", "audit_sha256"]
            for name in hash_names:
                value = getattr(self, name)
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            if not self.scientific_configuration:
                raise ValueError(
                    f"version {self.schema_version} manifest requires scientific_configuration"
                )
            assert self.metric_artifact is not None
            self.metric_artifact = _run_root_path(
                self.metric_artifact, "metric_artifact", "artifacts"
            )
        return self


class PreparationStageOutput(BaseModel):
    """Single structured output from environment preparation and smoke testing."""

    environment: EnvironmentManifest
    manifest: ExecutionManifest


RepairChangeKind = Literal[
    "environment",
    "dependency",
    "launch_script",
    "generated_config",
    "source_patch",
    "metric_extractor",
    "other",
]


class RepairChange(BaseModel):
    """One file-level change made to repair a failed execution."""

    path: str
    kind: RepairChangeKind
    summary: str = Field(..., min_length=1)
    sha256: str | None = None

    @model_validator(mode="after")
    def validate_change(self) -> RepairChange:
        self.path = _relative_path(self.path, "repair change path")
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise ValueError("repair change sha256 must be a lowercase SHA-256 digest")
        return self


class RepositoryPatchEvidence(BaseModel):
    """Complete repository diff against the pinned base commit after a repair."""

    base_commit: str
    diff_path: str
    diff_sha256: str
    changed_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patch(self) -> RepositoryPatchEvidence:
        if not self.base_commit.strip():
            raise ValueError("repository patch base_commit cannot be empty")
        self.diff_path = _relative_path(self.diff_path, "repository diff path")
        if not _SHA256.fullmatch(self.diff_sha256):
            raise ValueError("repository diff_sha256 must be a lowercase SHA-256 digest")
        self.changed_files = [
            _relative_path(path, "repository changed file") for path in self.changed_files
        ]
        return self


class ExecutionRepairRecord(BaseModel):
    """Auditable diagnosis and evidence for one execution repair."""

    schema_version: Literal[1] = 1
    attempt: int = Field(..., ge=1, le=2)
    diagnosis: str = Field(..., min_length=1)
    failure_fingerprint: str
    changes: list[RepairChange] = Field(..., min_length=1)
    repository_patch: RepositoryPatchEvidence
    smoke_test: SmokeTestResult

    @model_validator(mode="after")
    def validate_record(self) -> ExecutionRepairRecord:
        if not _SHA256.fullmatch(self.failure_fingerprint):
            raise ValueError("failure_fingerprint must be a lowercase SHA-256 digest")
        if self.smoke_test.exit_code != 0:
            raise ValueError("execution repair requires a successful smoke test")
        return self


class ExecutionRepairOutput(BaseModel):
    """Structured output from repairing one failed deterministic execution."""

    repair: ExecutionRepairRecord
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


class ReportResultLocation(BaseModel):
    """Paper location of the selected result, with optional structured coordinates."""

    description: str
    table_or_figure: str | None = None
    row: str | None = None
    column: str | None = None
    cell: str | None = None


class BaselineComparison(BaseModel):
    """Applicable paper baseline used in the reproduction verdict."""

    applicable: bool
    method: str | None = None
    value: str | None = None
    metric_direction: Literal[
        "higher_is_better",
        "lower_is_better",
        "not_applicable",
        "unknown",
    ]
    paper_evidence: PaperEvidence | None = None
    applicability_explanation: str

    @model_validator(mode="after")
    def validate_baseline(self) -> BaselineComparison:
        if self.applicable:
            missing = [
                name
                for name, value in {
                    "method": self.method,
                    "value": self.value,
                    "paper_evidence": self.paper_evidence,
                }.items()
                if value is None or value == ""
            ]
            if missing:
                raise ValueError(
                    "applicable baseline is missing: " + ", ".join(missing)
                )
            if self.metric_direction not in {"higher_is_better", "lower_is_better"}:
                raise ValueError(
                    "applicable baseline requires a known higher/lower metric direction"
                )
        return self


ReportVerdict = Literal["match", "miss", "run_incomplete", "needs_review"]


class MetricComparison(BaseModel):
    """LLM-authored interpretation of the frozen paper and reproduced values."""

    comparable: bool
    calculation: str
    relative_error_percent: float | None = Field(default=None, ge=0)
    baseline_advantage_preserved: bool | None = None
    verdict: ReportVerdict
    explanation: str

    @model_validator(mode="after")
    def validate_verdict(self) -> MetricComparison:
        if self.verdict in {"match", "miss"} and not self.comparable:
            raise ValueError("match/miss verdict requires comparable values")
        return self


AssumptionCategory = Literal["config", "environment", "evaluation"]
AssumptionSource = Literal[
    "code_default",
    "experiment_config",
    "readme",
    "environment",
    "agent_inference",
    "other",
]


class AssumptionRecord(BaseModel):
    """One explicit fallback or interpretation used by the reproduction."""

    id: str = Field(..., min_length=1)
    category: AssumptionCategory
    setting: str = Field(..., min_length=1)
    value_used: str | None = None
    paper_status: Literal["not_specified", "ambiguous", "not_applicable"]
    source_type: AssumptionSource
    source_path: str | None = None
    source_locator: str | None = None
    audit_item_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)
    impact: str = Field(..., min_length=1)


class ReportArtifactReferences(BaseModel):
    """Run-root-relative paths to authoritative reproduction evidence."""

    configuration_used: str = "execution_manifest.json"
    configuration_audit: str = "configuration_audit.json"
    environment_manifest: str = "environment_manifest.json"
    dockerfile: str | None = None
    build_log: str | None = None
    launch_script: str | None = None
    launch_script_sha256: str | None = None
    execution_result: str = "execution_result.json"
    metric_artifact: str | None = None
    executor_log: str = "logs/executor.log"
    assumptions: str = "assumptions.json"


class FinalReport(BaseModel):
    """Evidence-only final conclusion produced from frozen artifacts."""

    schema_version: Literal[3] = 3
    selected_experiment: str
    result_location: ReportResultLocation
    execution_status: Literal["succeeded", "failed", "timed_out", "blocked"]
    metric_name: str
    paper_target_value: str
    achieved_value: str | None = None
    baseline: BaselineComparison
    comparison: MetricComparison
    artifacts: ReportArtifactReferences = Field(default_factory=ReportArtifactReferences)
    assumptions: list[AssumptionRecord] = Field(default_factory=list)
    deviations: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conclusion(self) -> FinalReport:
        if self.execution_status != "succeeded" and self.comparison.verdict in {
            "match",
            "miss",
        }:
            raise ValueError("an unsuccessful execution cannot have a match/miss verdict")
        if self.execution_status != "succeeded" and self.comparison.comparable:
            raise ValueError("an unsuccessful execution cannot be declared comparable")
        ids = [item.id for item in self.assumptions]
        if len(ids) != len(set(ids)):
            raise ValueError("assumption ids must be unique")
        return self
