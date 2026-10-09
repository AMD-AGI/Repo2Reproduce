from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from claude_baseline_two_stage import pipeline, runner
from claude_baseline_two_stage.executor import _extract_metric
from claude_baseline_two_stage.schemas import (
    AuditItem,
    CodeEvidence,
    ConfigurationAudit,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionPathStep,
    ExecutionResult,
    ExecutionSpec,
    FinalReport,
    PaperEvidence,
    ResolvedInput,
    ScientificRequirement,
    TargetContract,
)
from claude_baseline_two_stage.validators import (
    StageValidationError,
    sha256_file,
    validate_audit,
    validate_final_report,
    validate_preparation,
    validate_target,
)

DATASET_QUOTE = "We evaluate every configuration on the DemoSet test split."
METRIC_QUOTE = "The table reports median accuracy across four evaluations."


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _paper(run_dir: Path) -> None:
    paper_dir = run_dir / "paper"
    paper_dir.mkdir(parents=True)
    markdown = paper_dir / "paper.md"
    markdown.write_text(f"{DATASET_QUOTE}\n\n{METRIC_QUOTE}\n", encoding="utf-8")
    _write_json(
        paper_dir / "paper_parse.json",
        {
            "markdown_path": str(markdown),
            "page_separator": "<!-- R2R_PAGE -->",
            "blocks": [
                {"type": "paragraph", "page": 1, "text": DATASET_QUOTE},
                {"type": "paragraph", "page": 2, "text": METRIC_QUOTE},
            ],
        },
    )


def _evidence(quote: str) -> PaperEvidence:
    return PaperEvidence(quote=quote, locator="unverified")


def _target(run_dir: Path) -> TargetContract:
    target = TargetContract(
        status="ready",
        selected_experiment="Table 1 / DemoSet / proposed method",
        contribution="The proposed method improves DemoSet accuracy.",
        result_location="Table 1, proposed method row, DemoSet column",
        metric_name="accuracy",
        metric_value="0.90",
        metric_definition="fraction of correct predictions",
        aggregation="median across four evaluations",
        selection_reason="central result with an official implementation",
        requirements=[
            ScientificRequirement(
                id="dataset",
                category="dataset",
                name="evaluation dataset",
                value="DemoSet test split",
                evidence=_evidence(DATASET_QUOTE),
            ),
            ScientificRequirement(
                id="metric",
                category="metric",
                name="reported metric",
                value="median accuracy across four evaluations",
                evidence=_evidence(METRIC_QUOTE),
            ),
        ],
    )
    validate_target(run_dir, target)
    _write_json(run_dir / "target.json", target.model_dump(mode="json"))
    return target


def _code_ref(snippet: str) -> CodeEvidence:
    return CodeEvidence(path="run.py", locator="run", snippet=snippet)


def _audit_and_spec(
    run_dir: Path,
    target: TargetContract,
    *,
    omit_metric_requirement: bool = False,
) -> tuple[ConfigurationAudit, ExecutionSpec]:
    target_hash = sha256_file(run_dir / "target.json")
    dataset_match = AuditItem(
        id="code-dataset",
        direction="code_to_paper",
        category="dataset",
        name="dataset",
        status="match",
        code_value="DemoSet test split",
        paper_value="DemoSet test split",
        resolved_value="DemoSet test split",
        paper_evidence=_evidence(DATASET_QUOTE),
        code_evidence=_code_ref('DATASET = "DemoSet"'),
        consumed_by=_code_ref("def run"),
        resolution="paper and code agree",
    )
    metric_match = AuditItem(
        id="code-metric",
        direction="code_to_paper",
        category="metric",
        name="metric",
        status="match",
        code_value="median accuracy",
        paper_value="median accuracy",
        resolved_value="median accuracy",
        paper_evidence=_evidence(METRIC_QUOTE),
        code_evidence=_code_ref("median_accuracy"),
        consumed_by=_code_ref("def run"),
        resolution="paper and code agree",
    )
    paper_to_code = [
        AuditItem(
            id="paper-dataset",
            direction="paper_to_code",
            requirement_id="dataset",
            category="dataset",
            name="evaluation dataset",
            status="match",
            code_value="DemoSet test split",
            paper_value="DemoSet test split",
            resolved_value="DemoSet test split",
            paper_evidence=_evidence(DATASET_QUOTE),
            code_evidence=_code_ref('DATASET = "DemoSet"'),
            consumed_by=_code_ref("def run"),
            resolution="mapped to the executed loader",
        )
    ]
    if not omit_metric_requirement:
        paper_to_code.append(
            AuditItem(
                id="paper-metric",
                direction="paper_to_code",
                requirement_id="metric",
                category="metric",
                name="reported metric",
                status="match",
                code_value="median accuracy",
                paper_value="median accuracy",
                resolved_value="median accuracy",
                paper_evidence=_evidence(METRIC_QUOTE),
                code_evidence=_code_ref("median_accuracy"),
                consumed_by=_code_ref("def run"),
                resolution="mapped to final aggregation",
            )
        )
    audit = ConfigurationAudit(
        status="ready",
        target_sha256=target_hash,
        experiment_kind="evaluation",
        entry_points=[_code_ref("def run")],
        execution_path=[
            ExecutionPathStep(
                order=1,
                role="load, evaluate, and aggregate",
                evidence=_code_ref("def run"),
                inputs=["dataset"],
                outputs=["median accuracy"],
            )
        ],
        code_to_paper=[dataset_match, metric_match],
        paper_to_code=paper_to_code,
        code_to_paper_complete=True,
        paper_to_code_complete=True,
    )
    spec = ExecutionSpec(
        status="ready",
        target_sha256=target_hash,
        repository_path="repo",
        repository_commit="abc123",
        experiment_kind="evaluation",
        entrypoint=_code_ref("def run"),
        command_outline="python run.py --dataset DemoSet",
        resolved_inputs=[
            ResolvedInput(
                audit_item_id="code-dataset",
                name="dataset",
                value="DemoSet test split",
                delivery="cli",
                rendered="--dataset DemoSet",
            ),
            ResolvedInput(
                audit_item_id="code-metric",
                name="metric",
                value="median accuracy",
                delivery="hardcoded",
            ),
        ],
        result_artifact="full-run/metrics.json",
        metric_extraction="Read median_accuracy from metrics.json",
    )
    return audit, spec


def _repository(run_dir: Path) -> None:
    repository = run_dir / "workspace" / "repo"
    repository.mkdir(parents=True)
    (repository / "run.py").write_text(
        'DATASET = "DemoSet"\n'
        "def median_accuracy(values):\n"
        "    return sorted(values)[len(values) // 2]\n"
        "def run():\n"
        "    return median_accuracy([0.9])\n",
        encoding="utf-8",
    )


def test_target_quotes_are_verified_and_located(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)

    assert all(requirement.evidence.verified for requirement in target.requirements)
    assert target.requirements[0].evidence.locator == "p.1, paragraph"


def test_bidirectional_audit_requires_every_target_requirement(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit, spec = _audit_and_spec(tmp_path, target, omit_metric_requirement=True)

    with pytest.raises(StageValidationError, match="omitted target requirements: metric"):
        validate_audit(
            tmp_path,
            target,
            audit,
            spec,
            repository_commit="abc123",
        )


def test_bidirectional_audit_accepts_closed_mapping(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit, spec = _audit_and_spec(tmp_path, target)

    validate_audit(
        tmp_path,
        target,
        audit,
        spec,
        repository_commit="abc123",
    )


def test_execution_spec_must_include_every_resolved_code_input(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit, spec = _audit_and_spec(tmp_path, target)
    spec.resolved_inputs = spec.resolved_inputs[:1]

    with pytest.raises(
        StageValidationError,
        match="omitted resolved code inputs: code-metric",
    ):
        validate_audit(
            tmp_path,
            target,
            audit,
            spec,
            repository_commit="abc123",
        )


def test_conflict_cannot_resolve_to_code_default() -> None:
    with pytest.raises(ValidationError, match="resolve to the paper value"):
        AuditItem(
            id="learning-rate",
            direction="code_to_paper",
            category="optimization",
            name="learning rate",
            status="conflict_resolved",
            code_value="0.002",
            paper_value="0.01",
            resolved_value="0.002",
            paper_evidence=_evidence("The learning rate is set to 0.01 for all runs."),
            code_evidence=CodeEvidence(path="train.py", snippet="lr=0.002"),
            resolution="incorrectly retained code default",
        )


def test_preparation_must_match_frozen_scientific_artifacts(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit, spec = _audit_and_spec(tmp_path, target)
    _write_json(tmp_path / "configuration_audit.json", audit.model_dump(mode="json"))
    spec.audit_sha256 = sha256_file(tmp_path / "configuration_audit.json")
    _write_json(tmp_path / "execution_spec.json", spec.model_dump(mode="json"))

    execution = tmp_path / "workspace" / "execution"
    execution.mkdir()
    launch = execution / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "REPO_DIR=/run/workspace/repo\n"
        'cd "${REPO_DIR}"\n'
        "mkdir -p /run/artifacts/full-run\n"
        "python run.py --dataset DemoSet | "
        "tee /run/artifacts/full-run/metrics.json\n",
        encoding="utf-8",
    )
    dockerfile = execution / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    build_log = tmp_path / "logs" / "build.log"
    build_log.parent.mkdir()
    build_log.write_text("build passed\n", encoding="utf-8")
    smoke_log = tmp_path / "artifacts" / "smoke" / "smoke.log"
    smoke_log.parent.mkdir(parents=True)
    smoke_log.write_text("smoke passed\n", encoding="utf-8")
    wrapper = tmp_path / "bin" / "docker"
    wrapper.parent.mkdir()
    shutil.copy2(
        Path(__file__).parents[1] / "claude_baseline_two_stage" / "templates" / "docker-wrapper",
        wrapper,
    )
    wrapper.chmod(0o755)

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="execution/Dockerfile",
        dockerfile_sha256=hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        build_log_path="logs/build.log",
        build_succeeded=True,
    )
    _write_json(tmp_path / "environment_manifest.json", environment.model_dump(mode="json"))
    manifest = ExecutionManifest(
        schema_version=2,
        status="ready",
        selected_experiment=target.selected_experiment,
        image_tag="demo:test",
        launch_script="execution/launch.sh",
        launch_script_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),
        host_repository="repo",
        container_repository="/run/workspace/repo",
        timeout_seconds=60,
        expected_outputs=["full-run/metrics.json"],
        smoke_test={
            "exit_code": 0,
            "log_path": "smoke/smoke.log",
            "description": "bounded evaluation",
        },
        target_sha256=sha256_file(tmp_path / "target.json"),
        audit_sha256=sha256_file(tmp_path / "configuration_audit.json"),
        execution_spec_sha256=sha256_file(tmp_path / "execution_spec.json"),
        environment_sha256=sha256_file(tmp_path / "environment_manifest.json"),
        repository_commit="abc123",
        entrypoint=_code_ref("def run"),
        scientific_configuration=spec.resolved_inputs,
        metric_name=target.metric_name,
        metric_artifact=spec.result_artifact,
        paper_target_value=target.metric_value,
        metric_extractor={
            "source": "full-run/metrics.json",
            "format": "json",
            "selector": "median_accuracy",
        },
    )

    validate_preparation(tmp_path, target, audit, spec, environment, manifest)

    metric_output = tmp_path / "artifacts" / "full-run" / "metrics.json"
    metric_output.parent.mkdir(parents=True)
    metric_output.write_text('{"median_accuracy": 0.91}\n', encoding="utf-8")
    assert _extract_metric(tmp_path, manifest) == ("0.91", "full-run/metrics.json")

    manifest.metric_name = "r-squared"
    with pytest.raises(StageValidationError, match="metric differs from target"):
        validate_preparation(tmp_path, target, audit, spec, environment, manifest)


def test_recovery_rerun_always_regenerates_stale_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _paper(run_dir)
    _repository(run_dir)
    target = _target(run_dir)
    audit, spec = _audit_and_spec(run_dir, target)
    _write_json(
        run_dir / "input.json",
        {
            "paper_url": "https://example.test/paper.pdf",
            "repo_url": "https://github.com/example/repo",
            "timeout_seconds": 60,
            "target": None,
        },
    )
    _write_json(
        run_dir / "metadata.json",
        {
            "status": "failed",
            "assigned_physical_gpu": "0",
            "model": "test-model",
            "stages": {},
        },
    )
    (run_dir / "bin").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "execution_manifest.json").write_text("{}\n", encoding="utf-8")
    previous = ExecutionResult(
        status="failed",
        command=["old"],
        exit_code=1,
        timed_out=False,
        duration_seconds=1,
        log_path=str(run_dir / "logs" / "executor.log"),
        expected_outputs=["full-run/metrics.json"],
        missing_outputs=["full-run/metrics.json"],
        error="old failure",
    )
    _write_json(run_dir / "execution_result.json", previous.model_dump(mode="json"))
    (run_dir / "final_report.md").write_text("stale report\n", encoding="utf-8")

    async def fake_bootstrap(*args: object, **kwargs: object) -> dict:
        return {"repository_commit": "abc123", "repository_path": "repo"}

    async def fake_target(*args: object, **kwargs: object) -> TargetContract:
        return target

    async def fake_audit(
        *args: object, **kwargs: object
    ) -> tuple[ConfigurationAudit, ExecutionSpec]:
        return audit, spec

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="execution/Dockerfile",
        dockerfile_sha256="0" * 64,
        build_log_path="logs/build.log",
        build_succeeded=True,
    )
    manifest = ExecutionManifest(
        status="ready",
        selected_experiment=target.selected_experiment,
        image_tag="demo:test",
        launch_script="execution/launch.sh",
        launch_script_sha256="0" * 64,
        host_repository="repo",
        container_repository="/run/workspace/repo",
        timeout_seconds=60,
        expected_outputs=["full-run/metrics.json"],
        smoke_test={
            "exit_code": 0,
            "log_path": "smoke/smoke.log",
            "description": "test",
        },
    )

    async def fake_prepare(
        *args: object, **kwargs: object
    ) -> tuple[EnvironmentManifest, ExecutionManifest]:
        return environment, manifest

    fresh_result = ExecutionResult(
        status="succeeded",
        command=["new"],
        exit_code=0,
        timed_out=False,
        duration_seconds=2,
        log_path=str(run_dir / "logs" / "executor.log"),
        expected_outputs=["full-run/metrics.json"],
        missing_outputs=[],
        achieved_metric="0.91",
    )

    async def fake_finalize(*args: object, **kwargs: object) -> str:
        (run_dir / "final_report.md").write_text("fresh report\n", encoding="utf-8")
        return "fresh report"

    monkeypatch.setattr(pipeline, "bootstrap_sources", fake_bootstrap)
    monkeypatch.setattr(pipeline, "select_target", fake_target)
    monkeypatch.setattr(pipeline, "audit_configuration", fake_audit)
    monkeypatch.setattr(pipeline, "prepare_environment", fake_prepare)
    monkeypatch.setattr(pipeline, "execute_manifest", lambda *args, **kwargs: fresh_result)
    monkeypatch.setattr(runner, "finalize_session", fake_finalize)
    args = argparse.Namespace(
        gpu=None,
        model=None,
        max_turns=10,
        max_budget_usd=1.0,
        prepare_only=False,
    )

    exit_code = asyncio.run(pipeline.run_staged_pipeline(run_dir, args, recovering=True))

    assert exit_code == 0
    assert (run_dir / "final_report.md").read_text(encoding="utf-8") == "fresh report\n"
    assert (run_dir / "final_report.pre-recovery.md").read_text(
        encoding="utf-8"
    ) == "stale report\n"


def test_final_report_cannot_substitute_achieved_value(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)
    result = ExecutionResult(
        status="succeeded",
        command=["python", "run.py"],
        exit_code=0,
        timed_out=False,
        duration_seconds=1,
        log_path="executor.log",
        expected_outputs=["full-run/metrics.json"],
        missing_outputs=[],
        achieved_metric="0.91",
        metric_source="full-run/metrics.json",
    )
    report = FinalReport(
        selected_experiment=target.selected_experiment,
        execution_status="succeeded",
        metric_name=target.metric_name,
        paper_target_value=target.metric_value,
        achieved_value="0.99",
        comparable=True,
        match=True,
        comparison_explanation="Incorrect substituted value.",
    )

    with pytest.raises(
        StageValidationError,
        match="achieved value differs from deterministic extraction",
    ):
        validate_final_report(target, result, report)
