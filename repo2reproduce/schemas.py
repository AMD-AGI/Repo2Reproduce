"""Pydantic models for the three grounded outputs and the intermediate maps.

These types define the contract between agents. The orchestrator also exposes the
final report shape as a JSON schema (see ``final_report_json_schema``) so the SDK can
enforce structured output.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Shared evidence primitives
# --------------------------------------------------------------------------- #
class PaperQuote(BaseModel):
    """A verbatim span of paper text with a locator, used for grounding."""

    text: str = Field(..., description="Verbatim text exactly as it appears in the paper.")
    locator: str = Field(..., description="Where it is, e.g. 'p.5, §4.1' or 'p.6, Table 2'.")
    verified: bool = Field(False, description="True once checked by verify_quote_in_paper.")


class CodeRef(BaseModel):
    """A pointer to a real location in the cloned repository."""

    path: str = Field(..., description="Repo-relative file path, e.g. 'train.py'.")
    locator: Optional[str] = Field(None, description="Line/flag/key, e.g. 'L42' or '--lr'.")
    snippet: Optional[str] = Field(None, description="Short verbatim snippet from the file.")


# --------------------------------------------------------------------------- #
# Persisted intermediate worker outputs
# --------------------------------------------------------------------------- #
class PaperMap(BaseModel):
    """Flexible but structured contract for the paper-analysis phase."""

    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    experiments: list[dict[str, Any]] = Field(default_factory=list)
    hyperparameters: list[dict[str, Any]] = Field(default_factory=list)
    datasets: Any = Field(default_factory=list)
    models: Any = Field(default_factory=list)
    hardware: Any = Field(default_factory=list)


class CodeMap(BaseModel):
    """Repository map consumed by the selection/configuration phases."""

    model_config = ConfigDict(extra="allow")

    entry_points: list[dict[str, Any]] = Field(default_factory=list)
    cli_flags: list[dict[str, Any]] = Field(default_factory=list)
    configs: list[dict[str, Any]] = Field(default_factory=list)
    datasets: Any = None
    environment: Any = None
    readme_commands: list[Any] = Field(default_factory=list)


class BaseImageSelection(BaseModel):
    model_config = ConfigDict(extra="allow")

    base_image: str
    rationale: str
    python: Optional[str] = None
    risks: list[str] = Field(default_factory=list)


class EnvironmentPort(BaseModel):
    model_config = ConfigDict(extra="allow")

    requirements: list[str] = Field(default_factory=list)
    dropped_pins: list[dict[str, Any]] = Field(default_factory=list)
    extra_install_cmds: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class KernelMigration(BaseModel):
    model_config = ConfigDict(extra="allow")

    has_custom_kernel: bool = False
    sources: list[str] = Field(default_factory=list)
    build_script: Optional[str] = None
    patch_file: Optional[str] = None
    note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Output 1: experiment selection (paper-grounded)
# --------------------------------------------------------------------------- #
class ExperimentSelection(BaseModel):
    selected_experiment: str = Field(..., description="Human-readable id of the chosen result.")
    target_result: str = Field(..., description="The exact metric/number to reproduce.")
    justification_quotes: list[PaperQuote] = Field(
        default_factory=list,
        description="Exact paper text + the table/figure that defines the chosen experiment.",
    )
    why_this_one: str = Field(
        ..., description="Why this experiment (paper clarity x code support)."
    )
    alternatives_considered: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Output 2: full configuration (paper + code reconciled)
# --------------------------------------------------------------------------- #
class Hyperparameter(BaseModel):
    name: str
    paper_value: Optional[str] = None
    code_default: Optional[str] = None
    chosen_value: Optional[str] = None
    status: Literal["match", "conflict", "paper_only", "code_only", "gap"] = "gap"
    paper_evidence: Optional[PaperQuote] = None
    code_evidence: Optional[CodeRef] = None


class ConfigField(BaseModel):
    value: Optional[str] = None
    paper_evidence: Optional[PaperQuote] = None
    code_evidence: Optional[CodeRef] = None
    note: Optional[str] = None


class Configuration(BaseModel):
    model: ConfigField = Field(default_factory=ConfigField)
    dataset: ConfigField = Field(default_factory=ConfigField)
    data_prep: ConfigField = Field(default_factory=ConfigField)
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    seed: ConfigField = Field(default_factory=ConfigField)
    hardware: ConfigField = Field(default_factory=ConfigField)
    environment: ConfigField = Field(default_factory=ConfigField, description="deps / versions")
    other: list[ConfigField] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Output 3: replication commands (code-grounded)
# --------------------------------------------------------------------------- #
class Command(BaseModel):
    step: Literal["setup", "data", "preprocess", "train", "evaluate", "other"]
    cmd: str
    source: Optional[CodeRef] = Field(None, description="Where this command/flags came from.")
    flags_verified: bool = Field(False, description="All flags exist per extract_argparse.")
    expected: Optional[str] = Field(None, description="Expected result, e.g. '~94.3% top-1'.")
    notes: Optional[str] = None


class ReplicationPlan(BaseModel):
    commands: list[Command] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ReconciliationResult(BaseModel):
    """Updated configuration/commands after the AMD port."""

    model_config = ConfigDict(extra="allow")

    updated_output2: Configuration | Literal["unchanged"]
    updated_output3: ReplicationPlan
    changes: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    issues_found: list[str] = Field(default_factory=list)
    corrections_applied: list[str] = Field(default_factory=list)
    gaps_and_assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    confidence_rationale: Optional[str] = None


# --------------------------------------------------------------------------- #
# Output 4: AMD/ROCm environment (Track 2)
# --------------------------------------------------------------------------- #
class AmdPatch(BaseModel):
    path: str = Field(..., description="Repo-relative file the patch targets.")
    reason: str = Field(..., description="Why this change is needed for ROCm.")
    diff_or_note: Optional[str] = Field(None, description="The change (diff or summary).")


class TorchIntegrity(BaseModel):
    torch: Optional[str] = None
    torchvision: Optional[str] = None
    torchaudio: Optional[str] = None
    hip: Optional[str] = None
    cuda_available: Optional[bool] = None
    device: Optional[str] = None
    intact: bool = Field(False, description="True if ROCm stack unchanged + GPU visible.")


class AmdEnvironment(BaseModel):
    base_image: Optional[str] = None
    base_image_rationale: Optional[str] = None
    dockerfile_path: Optional[str] = None
    patches: list[AmdPatch] = Field(default_factory=list)
    has_custom_kernel: bool = False
    kernel_migration_note: Optional[str] = None
    torch_baseline: Optional[TorchIntegrity] = None
    torch_after_build: Optional[TorchIntegrity] = None
    build_status: Literal["verified", "build_failed", "integrity_failed", "not_attempted"] = (
        "not_attempted"
    )
    smoke_test_cmd: Optional[str] = None
    smoke_test_passed: Optional[bool] = None
    image_tag: Optional[str] = None
    amd_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Output 5: reproduction result (Track D/E)
# --------------------------------------------------------------------------- #
class ReproductionResult(BaseModel):
    executed: bool = False
    command_run: Optional[str] = None
    achieved_metric: Optional[str] = None
    paper_target: Optional[str] = None
    match: Optional[bool] = None
    delta: Optional[str] = None
    notes: Optional[str] = None


# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #
class FinalReport(BaseModel):
    paper_url: str
    repo_url: str
    paper_title: Optional[str] = None
    output1_experiment_selection: ExperimentSelection
    output2_configuration: Configuration
    output3_replication_plan: ReplicationPlan
    output4_amd_environment: Optional[AmdEnvironment] = None
    output5_reproduction_result: Optional[ReproductionResult] = None
    gaps_and_assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    confidence_rationale: Optional[str] = None


def final_report_json_schema() -> dict:
    """JSON schema for ClaudeAgentOptions.output_format structured outputs."""
    return {"type": "json_schema", "schema": FinalReport.model_json_schema()}
