from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from repo2reproduce import pipeline, runner
from repo2reproduce.executor import _extract_metric
from repo2reproduce.schemas import (
    CodeEvidence,
    ConfigurationAudit,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionPathStep,
    ExecutionRepairOutput,
    ExecutionResult,
    FinalReport,
    PaperEvidence,
    PreparationStageOutput,
    ResolvedInput,
    ScientificConfigurationItem,
    ScientificRequirement,
    SmokeTestResult,
    TargetContract,
)
from repo2reproduce.tools import code as code_io
from repo2reproduce.validators import (
    StageValidationError,
    sha256_file,
    validate_audit,
    validate_execution_repair,
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


def _audit(
    run_dir: Path,
    target: TargetContract,
    *,
    omit_metric_requirement: bool = False,
) -> ConfigurationAudit:
    target_hash = sha256_file(run_dir / "target.json")
    dataset_match = ScientificConfigurationItem(
        id="dataset",
        category="dataset",
        name="dataset",
        paper_value="DemoSet test split",
        paper_evidence=_evidence(DATASET_QUOTE),
        code_experiment_value="DemoSet test split",
        code_evidence=_code_ref('DATASET = "DemoSet"'),
        consumed_by=_code_ref("def run"),
        chosen_value="DemoSet test split",
        requirement_ids=["dataset"],
        finding="match",
        notes="paper and code agree",
    )
    metric_match = ScientificConfigurationItem(
        id="metric",
        category="metric",
        name="metric",
        paper_value="median accuracy",
        paper_evidence=_evidence(METRIC_QUOTE),
        code_experiment_value="median accuracy",
        code_evidence=_code_ref("median_accuracy"),
        consumed_by=_code_ref("def run"),
        chosen_value="median accuracy",
        requirement_ids=[] if omit_metric_requirement else ["metric"],
        finding="match",
        notes="mapped to final aggregation",
    )
    audit = ConfigurationAudit(
        target_sha256=target_hash,
        repository_path="repo",
        repository_commit="abc123",
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
        configuration=[dataset_match, metric_match],
    )
    return audit


def _resolved_inputs() -> list[ResolvedInput]:
    return [
        ResolvedInput(
            audit_item_id="dataset",
            name="dataset",
            value="DemoSet test split",
            delivery="cli",
            rendered="--dataset DemoSet",
        ),
        ResolvedInput(
            audit_item_id="metric",
            name="metric",
            value="median accuracy",
            delivery="hardcoded",
        ),
    ]


def _final_report(
    target: TargetContract,
    result: ExecutionResult,
    *,
    achieved_value: str | None = None,
    assumptions: list[dict] | None = None,
) -> FinalReport:
    return FinalReport.model_validate(
        {
            "selected_experiment": target.selected_experiment,
            "result_location": {
                "description": target.result_location,
                "table_or_figure": "Table 1",
                "row": "proposed method",
                "column": "DemoSet",
                "cell": "accuracy",
            },
            "execution_status": result.status,
            "metric_name": target.metric_name,
            "paper_target_value": target.metric_value,
            "achieved_value": (
                result.achieved_metric if achieved_value is None else achieved_value
            ),
            "baseline": {
                "applicable": False,
                "metric_direction": "not_applicable",
                "applicability_explanation": "No comparable baseline is reported.",
            },
            "comparison": {
                "comparable": True,
                "calculation": "abs(0.91 - 0.90) / 0.90 * 100",
                "relative_error_percent": 1.111111,
                "baseline_advantage_preserved": None,
                "verdict": "match",
                "explanation": "The completed result is within five percent.",
            },
            "artifacts": {
                "metric_artifact": result.metric_source,
            },
            "assumptions": assumptions or [],
        }
    )


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


def test_target_can_skip_quote_checks_for_command_preview(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)
    target.requirements[0].evidence.quote = "unverified diagnostic quote"

    validate_target(tmp_path, target, verify_evidence=False)


def test_audit_records_missing_target_requirement_as_warning(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target, omit_metric_requirement=True)

    validate_audit(
        tmp_path,
        target,
        audit,
        repository_commit="abc123",
    )

    assert "configuration audit omitted target requirements: metric" in audit.warnings


def test_canonical_audit_accepts_closed_mapping(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)

    validate_audit(
        tmp_path,
        target,
        audit,
        repository_commit="abc123",
    )


def test_audit_can_skip_evidence_checks_for_command_preview(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    assert audit.configuration[0].paper_evidence is not None
    audit.configuration[0].paper_evidence.quote = "unverified diagnostic quote"
    audit.entry_points[0].path = "missing.py"

    validate_audit(
        tmp_path,
        target,
        audit,
        repository_commit="abc123",
        verify_evidence=False,
    )


def test_conflict_values_are_not_compared_as_exact_strings() -> None:
    item = ScientificConfigurationItem(
        id="learning-rate",
        category="optimization",
        name="learning rate",
        finding="paper_preferred",
        code_default="Adam with configurable learning rate",
        paper_value="Adam, lr=0.01",
        chosen_value="0.01",
        paper_evidence=_evidence("The learning rate is set to 0.01 for all runs."),
        code_evidence=CodeEvidence(path="train.py", snippet="lr=args.learning_rate"),
        notes="resolved to the paper learning rate",
    )

    assert item.chosen_value == "0.01"


def test_preparation_must_match_frozen_scientific_artifacts(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    _write_json(tmp_path / "configuration_audit.json", audit.model_dump(mode="json"))

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
    build_log = tmp_path / "artifacts" / "logs" / "build.log"
    build_log.parent.mkdir(parents=True)
    build_log.write_text("build passed\n", encoding="utf-8")
    smoke_log = tmp_path / "artifacts" / "smoke" / "smoke.log"
    smoke_log.parent.mkdir(parents=True)
    smoke_log.write_text("smoke passed\n", encoding="utf-8")
    wrapper = tmp_path / "bin" / "docker"
    wrapper.parent.mkdir()
    shutil.copy2(
        Path(__file__).parents[1] / "repo2reproduce" / "templates" / "docker-wrapper",
        wrapper,
    )
    wrapper.chmod(0o755)

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="workspace/execution/Dockerfile",
        dockerfile_sha256=hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        build_log_path="artifacts/logs/build.log",
        build_succeeded=True,
    )
    _write_json(tmp_path / "environment_manifest.json", environment.model_dump(mode="json"))
    manifest = ExecutionManifest(
        schema_version=3,
        status="ready",
        selected_experiment=target.selected_experiment,
        image_tag="demo:test",
        launch_script="workspace/execution/launch.sh",
        launch_script_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),
        host_repository="workspace/repo",
        container_repository="/run/workspace/repo",
        timeout_seconds=60,
        expected_outputs=["artifacts/full-run/metrics.json"],
        smoke_test={
            "exit_code": 0,
            "log_path": "artifacts/smoke/smoke.log",
            "description": "bounded evaluation",
        },
        target_sha256=sha256_file(tmp_path / "target.json"),
        audit_sha256=sha256_file(tmp_path / "configuration_audit.json"),
        environment_sha256=sha256_file(tmp_path / "environment_manifest.json"),
        repository_commit="abc123",
        entrypoint=_code_ref("def run"),
        scientific_configuration=_resolved_inputs(),
        metric_name=target.metric_name,
        metric_artifact="artifacts/full-run/metrics.json",
        paper_target_value=target.metric_value,
        metric_extractor={
            "source": "artifacts/full-run/metrics.json",
            "format": "json",
            "selector": "median_accuracy",
        },
    )

    deferred_payload = manifest.model_dump(mode="json")
    deferred_payload["environment_sha256"] = None
    deferred_manifest = ExecutionManifest.model_validate(deferred_payload)
    assert deferred_manifest.environment_sha256 is None

    captured_payload = {
        "environment": environment.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"),
    }
    captured_payload["environment"]["dockerfile"] = "/run/workspace/execution/Dockerfile"
    captured_payload["environment"]["build_log_path"] = "/run/artifacts/logs/build.log"
    captured_manifest = captured_payload["manifest"]
    captured_manifest["host_repository"] = str(tmp_path / "workspace" / "repo")
    captured_manifest["launch_script"] = "/run/workspace/execution/launch.sh"
    captured_manifest["expected_outputs"] = ["/run/artifacts/full-run/metrics.json"]
    captured_manifest["metric_artifact"] = "/run/artifacts/full-run/metrics.json"
    captured_manifest["metric_extractor"]["source"] = "/run/artifacts/full-run/metrics.json"
    captured_manifest["smoke_test"]["log_path"] = "/run/artifacts/smoke/smoke.log"
    captured_manifest["paper_target_value"] = "0.90 ± 0.00"

    migrated = pipeline._migrate_legacy_preparation_paths(captured_payload, tmp_path)
    normalized = pipeline._normalize_preparation_output(
        migrated,
        run_dir=tmp_path,
        target=target,
        audit=audit,
    )
    replayed = PreparationStageOutput.model_validate(normalized)
    assert replayed.environment.build_log_path == "artifacts/logs/build.log"
    assert replayed.manifest.host_repository == "workspace/repo"
    assert replayed.manifest.launch_script == "workspace/execution/launch.sh"
    assert replayed.manifest.expected_outputs == ["artifacts/full-run/metrics.json"]
    assert replayed.manifest.metric_artifact == "artifacts/full-run/metrics.json"
    assert replayed.manifest.paper_target_value == target.metric_value

    validate_preparation(tmp_path, target, audit, environment, manifest)

    metric_output = tmp_path / "artifacts" / "full-run" / "metrics.json"
    metric_output.parent.mkdir(parents=True)
    metric_output.write_text('{"median_accuracy": 0.91}\n', encoding="utf-8")
    assert _extract_metric(tmp_path, manifest) == (
        "0.91",
        "artifacts/full-run/metrics.json",
    )

    manifest.scientific_configuration = manifest.scientific_configuration[:1]
    validate_preparation(tmp_path, target, audit, environment, manifest)
    manifest.scientific_configuration = _resolved_inputs()

    manifest.metric_name = "r-squared"
    with pytest.raises(StageValidationError, match="metric differs from target"):
        validate_preparation(tmp_path, target, audit, environment, manifest)


def test_recovery_rerun_always_regenerates_stale_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _paper(run_dir)
    _repository(run_dir)
    target = _target(run_dir)
    audit = _audit(run_dir, target)
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

    async def fake_audit(*args: object, **kwargs: object) -> ConfigurationAudit:
        return audit

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="workspace/execution/Dockerfile",
        dockerfile_sha256="0" * 64,
        build_log_path="artifacts/logs/build.log",
        build_succeeded=True,
    )
    manifest = ExecutionManifest(
        schema_version=1,
        status="ready",
        selected_experiment=target.selected_experiment,
        image_tag="demo:test",
        launch_script="workspace/execution/launch.sh",
        launch_script_sha256="0" * 64,
        host_repository="workspace/repo",
        container_repository="/run/workspace/repo",
        timeout_seconds=60,
        expected_outputs=["artifacts/full-run/metrics.json"],
        smoke_test={
            "exit_code": 0,
            "log_path": "artifacts/smoke/smoke.log",
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


def _exercise_execution_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    results: list[ExecutionResult],
    *,
    max_repairs: int = 2,
    repair_error: Exception | None = None,
) -> tuple[int, int, int, ExecutionResult]:
    run_dir = tmp_path / "loop-run"
    _paper(run_dir)
    _repository(run_dir)
    target = _target(run_dir)
    audit = _audit(run_dir, target)
    _write_json(run_dir / "configuration_audit.json", audit.model_dump(mode="json"))
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
            "status": "initialized",
            "assigned_physical_gpu": "0",
            "model": "test-model",
            "stages": {},
        },
    )
    (run_dir / "logs").mkdir()
    (run_dir / "artifacts").mkdir()
    _write_json(run_dir / "execution_manifest.json", {"fixture": True})

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="workspace/Dockerfile",
        dockerfile_sha256="0" * 64,
        build_log_path="artifacts/build.log",
        build_succeeded=True,
    )
    manifest = ExecutionManifest(
        schema_version=1,
        status="ready",
        selected_experiment=target.selected_experiment,
        image_tag="demo:test",
        launch_script="workspace/launch.sh",
        launch_script_sha256="0" * 64,
        host_repository="workspace/repo",
        container_repository="/run/workspace/repo",
        timeout_seconds=60,
        expected_outputs=["artifacts/full-run/metrics.json"],
        smoke_test={
            "exit_code": 0,
            "log_path": "artifacts/smoke.log",
            "description": "fixture",
        },
    )
    repair_calls = 0
    execution_calls = 0
    final_result: ExecutionResult | None = None

    async def fake_bootstrap(*args: object, **kwargs: object) -> dict:
        return {"repository_commit": "abc123", "repository_path": "repo"}

    async def fake_target(*args: object, **kwargs: object) -> TargetContract:
        return target

    async def fake_audit(*args: object, **kwargs: object) -> ConfigurationAudit:
        return audit

    async def fake_prepare(
        *args: object, **kwargs: object
    ) -> tuple[EnvironmentManifest, ExecutionManifest]:
        return environment, manifest

    def fake_execute(*args: object, **kwargs: object) -> ExecutionResult:
        nonlocal execution_calls
        result = results[execution_calls]
        execution_calls += 1
        log = run_dir / "logs" / "executor.log"
        log.write_text((result.error or result.status) + "\n", encoding="utf-8")
        if result.status == "failed" and execution_calls == 1:
            stale = run_dir / "artifacts" / "full-run" / "metrics.json"
            stale.parent.mkdir(parents=True)
            stale.write_text('{"stale": true}\n', encoding="utf-8")
        return result

    async def fake_repair(
        *args: object, **kwargs: object
    ) -> tuple[EnvironmentManifest, ExecutionManifest, str]:
        nonlocal repair_calls
        repair_calls += 1
        if repair_error is not None:
            raise repair_error
        return environment, manifest, "repair-session"

    async def fake_finalize(*args: object, **kwargs: object) -> str:
        nonlocal final_result
        final_result = kwargs["execution_result"]
        (run_dir / "final_report.md").write_text("report\n", encoding="utf-8")
        return "report"

    monkeypatch.setattr(pipeline, "bootstrap_sources", fake_bootstrap)
    monkeypatch.setattr(pipeline, "select_target", fake_target)
    monkeypatch.setattr(pipeline, "audit_configuration", fake_audit)
    monkeypatch.setattr(pipeline, "prepare_environment", fake_prepare)
    monkeypatch.setattr(pipeline, "execute_manifest", fake_execute)
    monkeypatch.setattr(pipeline, "repair_execution", fake_repair)
    monkeypatch.setattr(runner, "finalize_session", fake_finalize)
    args = argparse.Namespace(
        gpu=None,
        model=None,
        max_turns=10,
        max_budget_usd=1.0,
        max_execution_repairs=max_repairs,
        prepare_only=False,
    )
    exit_code = asyncio.run(pipeline.run_staged_pipeline(run_dir, args))
    assert final_result is not None
    return exit_code, repair_calls, execution_calls, final_result


def _result(status: str, error: str | None = None) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        command=["run"],
        exit_code=0 if status == "succeeded" else 1,
        timed_out=status == "timed_out",
        duration_seconds=1,
        log_path="logs/executor.log",
        expected_outputs=["artifacts/full-run/metrics.json"],
        missing_outputs=[] if status == "succeeded" else ["artifacts/full-run/metrics.json"],
        error=error,
        achieved_metric="0.91" if status == "succeeded" else None,
    )


def test_execution_loop_immediate_success_has_no_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, repairs, executions, final = _exercise_execution_loop(
        tmp_path, monkeypatch, [_result("succeeded")]
    )
    assert (exit_code, repairs, executions, final.status) == (0, 0, 1, "succeeded")


def test_structured_stage_falls_back_when_saved_session_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(tmp_path / "metadata.json", {"status": "running", "stages": {}})
    observed_resumes: list[str | None] = []
    responses = [
        (
            SimpleNamespace(
                is_error=True,
                result="session unavailable",
                errors=[],
                terminal_reason="error",
                stop_reason=None,
                subtype="error",
                structured_output=None,
            ),
            None,
        ),
        (
            SimpleNamespace(
                is_error=False,
                result=None,
                errors=[],
                terminal_reason=None,
                stop_reason=None,
                subtype="success",
                structured_output={
                    "exit_code": 0,
                    "log_path": "artifacts/repair-smoke.log",
                    "description": "passed",
                },
            ),
            "new-session",
        ),
    ]

    def fake_options(*args: object, **kwargs: object) -> dict:
        return {}

    async def fake_invoke(*, options: object, **kwargs: object) -> tuple[object, str | None]:
        observed_resumes.append(getattr(options, "resume", None))
        return responses.pop(0)

    monkeypatch.setattr(runner, "_base_options", fake_options)
    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    artifact, session_id = asyncio.run(
        pipeline._run_structured_stage(
            tmp_path,
            stage="repair",
            prompt="repair",
            schema=SmokeTestResult,
            tools=[],
            assigned_gpu="0",
            model="test",
            max_turns=1,
            max_budget_usd=1,
            validator=lambda output: None,
            initial_session_id="expired-session",
        )
    )
    assert artifact.exit_code == 0
    assert session_id == "new-session"
    assert observed_resumes == ["expired-session", None]


def _smoke_success(session: str) -> tuple[object, str]:
    return (
        SimpleNamespace(
            is_error=False,
            result=None,
            errors=[],
            terminal_reason=None,
            stop_reason=None,
            subtype="success",
            structured_output={
                "exit_code": 0,
                "log_path": "artifacts/smoke.log",
                "description": "passed",
            },
        ),
        session,
    )


def _observe_skill_session(
    trace: runner.SkillTrace, *, available: list[str], invoked: str | None
) -> None:
    from claude_agent_sdk import AssistantMessage, SystemMessage, ToolUseBlock

    trace.observe(SystemMessage(subtype="init", data={"skills": available}))
    if invoked:
        trace.observe(
            AssistantMessage(
                content=[ToolUseBlock(id="toolu_1", name="Skill", input={"skill": invoked})],
                model="test",
            )
        )


def _run_smoke_stage(tmp_path: Path, skill: str | None) -> tuple[SmokeTestResult, str]:
    return asyncio.run(
        pipeline._run_structured_stage(
            tmp_path,
            stage="configuration_audit",
            prompt="audit",
            schema=SmokeTestResult,
            tools=["Read"],
            assigned_gpu="0",
            model="test",
            max_turns=1,
            max_budget_usd=1,
            validator=lambda output: None,
            skills=[skill] if skill else [],
        )
    )


def test_structured_stage_requires_its_skill_before_accepting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(tmp_path / "metadata.json", {"status": "running", "stages": {}})
    observed: list[dict[str, object]] = []
    loads = [None, runner.CONFIGURATION_AUDIT_SKILL]

    async def fake_invoke(
        *, prompt: str, options: object, skill_trace: runner.SkillTrace, **kwargs: object
    ) -> tuple[object, str | None]:
        observed.append({"prompt": prompt, "skills": options.skills, "tools": options.allowed_tools})
        _observe_skill_session(
            skill_trace,
            available=[runner.CONFIGURATION_AUDIT_SKILL, "debug"],
            invoked=loads.pop(0),
        )
        return _smoke_success("audit-session")

    monkeypatch.setattr(runner, "_base_options", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    artifact, _ = _run_smoke_stage(tmp_path, runner.CONFIGURATION_AUDIT_SKILL)

    assert artifact.exit_code == 0
    assert [item["skills"] for item in observed] == [[runner.CONFIGURATION_AUDIT_SKILL]] * 2
    assert all(item["tools"] == ["Read"] for item in observed)
    assert "was not loaded with the Skill tool" in str(observed[1]["prompt"])
    stage = json.loads((tmp_path / "metadata.json").read_text())["stages"]["configuration_audit"]
    assert stage["skills"] == [runner.CONFIGURATION_AUDIT_SKILL]
    assert stage["attempts"] == 2


def test_structured_stage_without_skill_hides_all_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(tmp_path / "metadata.json", {"status": "running", "stages": {}})
    observed_skills: list[object] = []

    async def fake_invoke(*, options: object, **kwargs: object) -> tuple[object, str | None]:
        observed_skills.append(options.skills)
        return _smoke_success("prepare-session")

    monkeypatch.setattr(runner, "_base_options", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    _run_smoke_stage(tmp_path, None)

    assert observed_skills == [[]]


def test_structured_stage_fails_fast_when_skill_is_not_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(tmp_path / "metadata.json", {"status": "running", "stages": {}})
    calls = 0

    async def fake_invoke(
        *, skill_trace: runner.SkillTrace, **kwargs: object
    ) -> tuple[object, str | None]:
        nonlocal calls
        calls += 1
        _observe_skill_session(skill_trace, available=["debug"], invoked=None)
        return _smoke_success("audit-session")

    monkeypatch.setattr(runner, "_base_options", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    with pytest.raises(runner.SkillUnavailableError, match="did not discover"):
        _run_smoke_stage(tmp_path, runner.CONFIGURATION_AUDIT_SKILL)

    assert calls == 1
    stage = json.loads((tmp_path / "metadata.json").read_text())["stages"]["configuration_audit"]
    assert stage["status"] == "failed"


def _subtree_run(tmp_path: Path, *, snapshot: bool) -> tuple[Path, ConfigurationAudit]:
    parent_checkout = tmp_path / "checkout"
    parent_checkout.mkdir()
    (parent_checkout / "README.md").write_text("orchestrator checkout\n", encoding="utf-8")
    code_io.snapshot_repository(parent_checkout, "parent checkout")
    run_dir = parent_checkout / "runs" / "run-1"
    _paper(run_dir)
    _repository(run_dir)
    target = _target(run_dir)
    audit = _audit(run_dir, target)
    clone = {"subpath": "project", "commit": audit.repository_commit}
    if snapshot:
        clone["snapshot_commit"] = code_io.snapshot_repository(
            run_dir / "workspace" / "repo", "pristine subtree"
        )
    _write_json(
        run_dir / "bootstrap.json",
        {"repository_path": "repo", "repository_commit": audit.repository_commit, "clone": clone},
    )
    return run_dir, audit


def test_subtree_patch_is_diffed_against_its_local_snapshot(tmp_path: Path) -> None:
    run_dir, audit = _subtree_run(tmp_path, snapshot=True)
    repository = run_dir / "workspace" / "repo"
    (repository / "run.py").write_text('DATASET = "DemoSet"\n', encoding="utf-8")
    (repository / "compat.py").write_text("SHIM = True\n", encoding="utf-8")

    patch_path, patch_sha, changed = pipeline._capture_repository_patch(run_dir, audit, 1)

    patch = (run_dir / patch_path).read_text(encoding="utf-8")
    assert changed == ["compat.py", "run.py"]
    assert "-def median_accuracy(values):" in patch
    assert "+SHIM = True" in patch
    assert "README.md" not in patch
    assert patch_sha == sha256_file(run_dir / patch_path)
    snapshot = json.loads((run_dir / "bootstrap.json").read_text())["clone"]["snapshot_commit"]
    assert pipeline._repository_baseline(run_dir, audit) == snapshot


def test_patch_capture_refuses_a_parent_checkout(tmp_path: Path) -> None:
    run_dir, audit = _subtree_run(tmp_path, snapshot=False)
    with pytest.raises(StageValidationError, match="not its own git checkout"):
        pipeline._capture_repository_patch(run_dir, audit, 1)


@pytest.mark.parametrize(
    ("upstream_commit", "origin"),
    [("abc123", "pristine_upstream"), ("moved", "workspace_at_recovery")],
)
def test_legacy_subtree_run_gets_a_repository_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream_commit: str,
    origin: str,
) -> None:
    run_dir, audit = _subtree_run(tmp_path, snapshot=False)
    repository = run_dir / "workspace" / "repo"
    pristine_source = (repository / "run.py").read_text(encoding="utf-8")
    (repository / "run.py").write_text("PREPARED = True\n", encoding="utf-8")

    def fake_clone(repo_url: str, dest: Path) -> dict:
        dest.mkdir(parents=True)
        (dest / "run.py").write_text(pristine_source, encoding="utf-8")
        return {
            "commit": upstream_commit,
            "snapshot_commit": code_io.snapshot_repository(dest, "pristine subtree"),
        }

    monkeypatch.setattr(code_io, "clone_repo", fake_clone)
    bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
    assert pipeline._needs_repository_baseline(bootstrap)
    restored = pipeline._restore_repository_baseline(run_dir, bootstrap, "https://example")
    _write_json(run_dir / "bootstrap.json", restored)

    assert restored["clone"]["snapshot_origin"] == origin
    assert not pipeline._needs_repository_baseline(restored)
    _, _, changed = pipeline._capture_repository_patch(run_dir, audit, 1)
    assert changed == (["run.py"] if origin == "pristine_upstream" else [])


@pytest.mark.parametrize(
    ("prompt_name", "skill"),
    [
        ("audit_configuration.md", runner.CONFIGURATION_AUDIT_SKILL),
        ("repair_execution.md", runner.EXECUTION_REPAIR_SKILL),
        ("finalize.md", runner.RESULT_COMPARISON_SKILL),
    ],
)
def test_skill_backed_prompts_name_their_skill(prompt_name: str, skill: str) -> None:
    prompt = runner._render_prompt(prompt_name, {"SKILL": skill})
    assert f"invoke the `{skill}` skill with the Skill tool" in prompt


def test_execution_loop_repairs_failure_then_succeeds_and_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, repairs, executions, final = _exercise_execution_loop(
        tmp_path,
        monkeypatch,
        [_result("failed", "first failure"), _result("succeeded")],
    )
    assert (exit_code, repairs, executions, final.status) == (0, 1, 2, "succeeded")
    archived = tmp_path / "loop-run" / "artifacts" / "execution-attempts" / "attempt-1"
    assert (archived / "execution_result.json").is_file()
    assert (archived / "executor.log").is_file()
    assert (archived / "full-run" / "metrics.json").is_file()
    assert (archived / "full-run").resolve() == (
        tmp_path / "loop-run" / "artifacts" / "full-run.attempt-1"
    ).resolve()
    assert not (tmp_path / "loop-run" / "artifacts" / "full-run").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_archive_handles_container_owned_full_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    full_run = run_dir / "artifacts" / "full-run"
    (full_run / "train").mkdir(parents=True)
    (full_run / "train" / "model.pt").write_text("weights", encoding="utf-8")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "executor.log").write_text("failed\n", encoding="utf-8")
    for directory in (full_run / "train", full_run):
        directory.chmod(0o555)
    result = _result("failed", "launch command exited with code 1")
    manifest = SimpleNamespace(expected_outputs=["artifacts/full-run/metrics.json"])

    try:
        archived = pipeline._archive_execution_attempt(run_dir, result, manifest, 1)
        assert (archived / "full-run" / "train" / "model.pt").read_text() == "weights"
        assert not full_run.exists()
    finally:
        for name in ("full-run", "full-run.attempt-1"):
            directory = run_dir / "artifacts" / name
            if directory.exists():
                directory.chmod(0o755)
                (directory / "train").chmod(0o755)


def test_execution_loop_stops_after_two_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, repairs, executions, final = _exercise_execution_loop(
        tmp_path,
        monkeypatch,
        [
            _result("failed", "failure one"),
            _result("failed", "failure two"),
            _result("failed", "failure three"),
        ],
    )
    assert (exit_code, repairs, executions, final.status) == (1, 2, 3, "failed")


@pytest.mark.parametrize("status", ["timed_out", "blocked"])
def test_execution_loop_does_not_repair_timeout_or_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    exit_code, repairs, executions, final = _exercise_execution_loop(
        tmp_path, monkeypatch, [_result(status, status)]
    )
    assert (exit_code, repairs, executions, final.status) == (1, 0, 1, status)


def test_execution_loop_stops_on_repeated_failure_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = _result("failed", "same failure")
    exit_code, repairs, executions, final = _exercise_execution_loop(
        tmp_path, monkeypatch, [failure, failure]
    )
    assert (exit_code, repairs, executions, final.status) == (1, 1, 2, "failed")


def test_execution_loop_stops_when_repair_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, repairs, executions, final = _exercise_execution_loop(
        tmp_path,
        monkeypatch,
        [_result("failed", "repairable")],
        repair_error=StageValidationError("execution repair was a no-op"),
    )
    assert (exit_code, repairs, executions, final.status) == (1, 1, 1, "failed")


def test_execution_repair_rejects_changed_scientific_value(tmp_path: Path) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    _write_json(tmp_path / "configuration_audit.json", audit.model_dump(mode="json"))
    changed_file = tmp_path / "workspace" / "launch.sh"
    changed_file.write_text("repaired\n", encoding="utf-8")
    smoke_log = tmp_path / "artifacts" / "execution-repairs" / "repair-1" / "smoke.log"
    smoke_log.parent.mkdir(parents=True)
    smoke_log.write_text("passed\n", encoding="utf-8")
    patch_path = tmp_path / "execution_repairs" / "repair-1.patch"
    patch_path.parent.mkdir()
    patch_path.write_text("", encoding="utf-8")

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="workspace/Dockerfile",
        dockerfile_sha256="0" * 64,
        build_log_path="artifacts/build.log",
        build_succeeded=True,
    )
    previous = ExecutionManifest(
        schema_version=1,
        status="ready",
        selected_experiment=target.selected_experiment,
        image_tag="demo:test",
        launch_script="workspace/launch.sh",
        launch_script_sha256=sha256_file(changed_file),
        host_repository="workspace/repo",
        container_repository="/run/workspace/repo",
        timeout_seconds=60,
        expected_outputs=["artifacts/full-run/metrics.json"],
        smoke_test={
            "exit_code": 0,
            "log_path": "artifacts/execution-repairs/repair-1/smoke.log",
            "description": "passed",
        },
        repository_commit=audit.repository_commit,
        scientific_configuration=_resolved_inputs(),
        metric_name=target.metric_name,
        paper_target_value=target.metric_value,
    )
    changed = previous.model_copy(deep=True)
    changed.scientific_configuration[0].value = "DifferentSet"
    fingerprint = "1" * 64
    output = ExecutionRepairOutput.model_validate(
        {
            "repair": {
                "attempt": 1,
                "diagnosis": "launch failed",
                "failure_fingerprint": fingerprint,
                "changes": [
                    {
                        "path": "workspace/launch.sh",
                        "kind": "launch_script",
                        "summary": "repaired launch",
                        "sha256": sha256_file(changed_file),
                    }
                ],
                "repository_patch": {
                    "base_commit": audit.repository_commit,
                    "diff_path": "execution_repairs/repair-1.patch",
                    "diff_sha256": sha256_file(patch_path),
                    "changed_files": [],
                },
                "smoke_test": changed.smoke_test.model_dump(mode="json"),
            },
            "environment": environment.model_dump(mode="json"),
            "manifest": changed.model_dump(mode="json"),
        }
    )

    with pytest.raises(StageValidationError, match="scientific configuration values"):
        validate_execution_repair(
            tmp_path,
            target,
            audit,
            previous,
            output,
            attempt=1,
            failure_fingerprint=fingerprint,
            target_sha256=sha256_file(tmp_path / "target.json"),
            audit_sha256=sha256_file(tmp_path / "configuration_audit.json"),
        )


def test_final_report_cannot_substitute_achieved_value(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    environment = EnvironmentManifest(status="blocked", blocked_reason="test fixture")
    manifest = ExecutionManifest(
        status="blocked",
        selected_experiment=target.selected_experiment,
        blocked_reason="test fixture",
    )
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
    report = _final_report(target, result, achieved_value="0.99")

    with pytest.raises(
        StageValidationError,
        match="achieved value differs from deterministic extraction",
    ):
        validate_final_report(
            tmp_path,
            target,
            audit,
            environment,
            manifest,
            result,
            report,
        )


def test_final_report_must_cover_every_paper_missing_fallback(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    audit.configuration.append(
        ScientificConfigurationItem(
            id="batch-size",
            category="training",
            name="batch size",
            code_default="64",
            code_evidence=_code_ref("def run"),
            consumed_by=_code_ref("def run"),
            chosen_value="64",
            finding="paper_missing",
            notes="paper does not specify the batch size",
        )
    )
    environment = EnvironmentManifest(status="blocked", blocked_reason="test fixture")
    manifest = ExecutionManifest(
        status="blocked",
        selected_experiment=target.selected_experiment,
        blocked_reason="test fixture",
    )
    result = ExecutionResult(
        status="succeeded",
        command=["python", "run.py"],
        exit_code=0,
        timed_out=False,
        duration_seconds=1,
        log_path="logs/executor.log",
        expected_outputs=["artifacts/full-run/metrics.json"],
        missing_outputs=[],
        achieved_metric="0.91",
        metric_source="artifacts/full-run/metrics.json",
    )
    report = _final_report(target, result)

    with pytest.raises(
        StageValidationError,
        match="omitted paper-missing assumptions: batch-size",
    ):
        validate_final_report(
            tmp_path,
            target,
            audit,
            environment,
            manifest,
            result,
            report,
        )

    report = _final_report(
        target,
        result,
        assumptions=[
            {
                "id": "A1",
                "category": "config",
                "setting": "batch size",
                "value_used": "64",
                "paper_status": "not_specified",
                "source_type": "code_default",
                "source_path": "run.py",
                "audit_item_ids": ["batch-size"],
                "rationale": "The official entrypoint consumes its default.",
                "impact": "May affect optimization.",
            }
        ],
    )
    validate_final_report(
        tmp_path,
        target,
        audit,
        environment,
        manifest,
        result,
        report,
    )
