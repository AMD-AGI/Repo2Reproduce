from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    SystemMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from pydantic import ValidationError

from repo2reproduce import runner
from repo2reproduce.executor import execute_manifest
from repo2reproduce.runner import initialize_run
from repo2reproduce.schemas import ExecutionManifest, FinalReport

PACKAGED_SKILLS = {
    runner.CONFIGURATION_AUDIT_SKILL,
    runner.EXECUTION_REPAIR_SKILL,
    runner.RESULT_COMPARISON_SKILL,
}


def _skill_messages(skill: str, *, tool_use_id: str = "toolu_skill") -> list[object]:
    return [
        SystemMessage(subtype="init", data={"session_id": "s", "skills": sorted(PACKAGED_SKILLS)}),
        AssistantMessage(
            content=[ToolUseBlock(id=tool_use_id, name="Skill", input={"skill": skill})],
            model="test",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id=tool_use_id, content="Launching")]),
    ]


def _manifest(script: Path, **updates: object) -> ExecutionManifest:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "ready",
        "selected_experiment": "tiny evaluation",
        "image_tag": "example/reproduction:test",
        "launch_script": "workspace/execution/launch.sh",
        "launch_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "host_repository": "workspace/repo",
        "container_repository": "/run/workspace/repo",
        "timeout_seconds": 10,
        "expected_outputs": ["artifacts/full-run/metrics.json"],
        "smoke_test": {
            "exit_code": 0,
            "log_path": "artifacts/smoke/smoke.log",
            "description": "one-batch pilot",
        },
    }
    payload.update(updates)
    return ExecutionManifest.model_validate(payload)


def _prepared_run(tmp_path: Path, *, create_output: bool = True) -> tuple[Path, Path]:
    run_dir = tmp_path / "run-1"
    script = run_dir / "workspace" / "execution" / "launch.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "REPO_DIR=/run/workspace/repo\n"
        'cd "${REPO_DIR}"\n'
        "mkdir -p /run/artifacts/full-run\n"
        "python train.py \\\n"
        "  --epochs 1 \\\n"
        "  --batch-size 2 | tee /run/artifacts/full-run/stdout.log\n",
        encoding="utf-8",
    )
    (run_dir / "workspace" / "repo").mkdir()
    smoke = run_dir / "artifacts" / "smoke" / "smoke.log"
    smoke.parent.mkdir(parents=True)
    smoke.write_text("smoke passed\n", encoding="utf-8")

    wrapper = run_dir / "bin" / "docker"
    wrapper.parent.mkdir()
    output_statement = (
        'mkdir -p "$RUN_DIR/artifacts/full-run"\n'
        'printf \'{"score": 1}\\n\' > "$RUN_DIR/artifacts/full-run/metrics.json"\n'
        if create_output
        else ""
    )
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$@" > "$RUN_DIR/logs/fake-docker-argv.txt"\n'
        f"{output_statement}",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return run_dir, script


def test_ready_manifest_requires_successful_smoke_test(tmp_path: Path) -> None:
    _, script = _prepared_run(tmp_path)
    with pytest.raises(ValidationError, match="successful smoke test"):
        _manifest(
            script,
            smoke_test={
                "exit_code": 1,
                "log_path": "artifacts/smoke/smoke.log",
                "description": "failed",
            },
        )


def test_blocked_manifest_requires_reason() -> None:
    manifest = ExecutionManifest.model_validate(
        {
            "status": "blocked",
            "selected_experiment": "tiny evaluation",
            "blocked_reason": "dataset is unavailable",
        }
    )
    assert manifest.status == "blocked"

    with pytest.raises(ValidationError, match="blocked_reason"):
        ExecutionManifest.model_validate(
            {"status": "blocked", "selected_experiment": "tiny evaluation"}
        )


def test_manifest_rejects_unsafe_relative_paths(tmp_path: Path) -> None:
    _, script = _prepared_run(tmp_path)
    with pytest.raises(ValidationError, match="relative path"):
        _manifest(script, launch_script="../launch.sh")


def test_initialize_run_installs_isolation_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "initialized"
    initialize_run(
        run_dir,
        run_id="initialized",
        paper_url="https://arxiv.org/abs/0000.00000",
        repo_url="https://github.com/example/project",
        assigned_gpu="2",
        timeout_seconds=120,
        target=None,
    )

    assert (run_dir / "bin" / "docker").stat().st_mode & 0o111
    settings = json.loads(
        (run_dir / "workspace" / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    hook_command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert str(run_dir / "workspace" / ".claude" / "hooks" / "enforce_bash.py") in hook_command

    skills_root = run_dir / "workspace" / ".claude" / "skills"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["skills"]) == PACKAGED_SKILLS
    for name in PACKAGED_SKILLS:
        installed = skills_root / name / "SKILL.md"
        packaged = runner.SKILLS_DIR / name / "SKILL.md"
        assert installed.read_bytes() == packaged.read_bytes()
        assert len(metadata["skills"][name]) == 64


def test_install_skills_replaces_stale_skill_files(tmp_path: Path) -> None:
    stale = tmp_path / "workspace" / ".claude" / "skills" / runner.EXECUTION_REPAIR_SKILL
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text("outdated", encoding="utf-8")
    (stale / "removed-reference.md").write_text("outdated", encoding="utf-8")

    first = runner.install_skills(tmp_path)
    second = runner.install_skills(tmp_path)

    assert first == second
    assert not (stale / "removed-reference.md").exists()
    assert (stale / "SKILL.md").read_bytes() == (
        runner.SKILLS_DIR / runner.EXECUTION_REPAIR_SKILL / "SKILL.md"
    ).read_bytes()


@pytest.mark.parametrize("name", sorted(PACKAGED_SKILLS))
def test_packaged_skill_frontmatter_and_references(name: str) -> None:
    text = (runner.SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    fields = dict(
        line.split(": ", 1) for line in frontmatter.strip().splitlines() if ": " in line
    )
    assert fields["name"] == name
    assert 0 < len(fields["description"]) <= 1024
    for reference in re.findall(r"\]\(([^)]+\.md)\)", body):
        assert (runner.SKILLS_DIR / name / reference).is_file()


def test_skill_trace_counts_only_successful_skill_requests() -> None:
    trace = runner.SkillTrace()
    for message in _skill_messages(runner.EXECUTION_REPAIR_SKILL):
        trace.observe(message)
    trace.observe(
        AssistantMessage(
            content=[ToolUseBlock(id="denied", name="Skill", input={"skill": "debug"})],
            model="test",
        )
    )
    trace.observe(
        UserMessage(content=[ToolResultBlock(tool_use_id="denied", is_error=True)])
    )

    assert trace.available == PACKAGED_SKILLS
    assert trace.invoked == {runner.EXECUTION_REPAIR_SKILL}
    runner.require_skill_invoked(runner.EXECUTION_REPAIR_SKILL, trace)
    with pytest.raises(runner.StageValidationError, match="debug"):
        runner.require_skill_invoked("debug", trace)
    with pytest.raises(runner.SkillUnavailableError, match="debug"):
        runner.require_skill_available("finalization", "debug", trace)


def test_finalize_prompt_names_skill_and_tolerance() -> None:
    prompt = runner._render_prompt(
        "finalize.md",
        {
            "SKILL": runner.RESULT_COMPARISON_SKILL,
            "MATCH_TOLERANCE_PERCENT": str(runner.MATCH_TOLERANCE_PERCENT),
        },
    )
    assert f"invoke the `{runner.RESULT_COMPARISON_SKILL}` skill" in prompt
    assert "at most 5 percent" in prompt


def test_project_environment_loads_repository_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    def fake_load(path: Path, *, override: bool) -> None:
        observed.update(path=path, override=override)

    monkeypatch.setattr(runner, "load_dotenv", fake_load)
    runner.load_project_environment()

    assert observed == {"path": runner.PACKAGE_DIR.parent / ".env", "override": False}


def test_execution_repair_cli_defaults_to_two_and_allows_zero() -> None:
    parser = runner.build_parser()
    default = parser.parse_args(["paper.pdf", "https://github.com/example/repo"])
    disabled = parser.parse_args(
        [
            "paper.pdf",
            "https://github.com/example/repo",
            "--max-execution-repairs",
            "0",
        ]
    )
    assert default.max_execution_repairs == 2
    assert disabled.max_execution_repairs == 0


def test_runs_root_defaults_inside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("R2R_RUNS_ROOT", raising=False)
    default = runner.build_parser().parse_args([]).runs_root
    assert default == runner.PACKAGE_DIR.parent / "runs" / "multi-step"

    monkeypatch.setenv("R2R_RUNS_ROOT", str(tmp_path / "runs"))
    assert runner.build_parser().parse_args([]).runs_root == tmp_path / "runs"


def test_executor_waits_and_records_success(tmp_path: Path) -> None:
    run_dir, script = _prepared_run(tmp_path)
    manifest = _manifest(script)

    result = execute_manifest(run_dir, manifest, assigned_gpu="3")

    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.missing_outputs == []
    argv = (run_dir / "logs" / "fake-docker-argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv == [
        "run",
        "-w",
        "/run/workspace/repo",
        "example/reproduction:test",
        "bash",
        "/run/workspace/execution/launch.sh",
    ]


def test_executor_blocks_changed_launch_script(tmp_path: Path) -> None:
    run_dir, script = _prepared_run(tmp_path)
    manifest = _manifest(script)
    script.write_text("#!/bin/bash\nexit 99\n", encoding="utf-8")

    result = execute_manifest(run_dir, manifest, assigned_gpu="0")

    assert result.status == "blocked"
    assert "hash does not match" in (result.error or "")
    assert not (run_dir / "logs" / "fake-docker-argv.txt").exists()


def _write_launch_script(script: Path, *lines: str) -> None:
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        + "".join(f"{line}\n" for line in lines)
        + "mkdir -p /run/artifacts/full-run\n"
        "python train.py | tee /run/artifacts/full-run/stdout.log\n",
        encoding="utf-8",
    )


def test_executor_accepts_a_repository_path_composed_from_variables(tmp_path: Path) -> None:
    run_dir, script = _prepared_run(tmp_path)
    _write_launch_script(
        script,
        "RUN_WS=/run/workspace",
        'REPO="$RUN_WS/repo"',
        'cd "$REPO"',
    )

    result = execute_manifest(run_dir, _manifest(script), assigned_gpu="3")

    assert result.status == "succeeded"


def test_executor_blocks_a_launch_script_that_runs_from_a_sibling_directory(
    tmp_path: Path,
) -> None:
    run_dir, script = _prepared_run(tmp_path)
    (run_dir / "workspace" / "repo-hip").mkdir()
    _write_launch_script(
        script,
        "RUN_WS=/run/workspace",
        'REPO_HIP="$RUN_WS/repo-hip"',
        'cd "$REPO_HIP"',
    )

    result = execute_manifest(run_dir, _manifest(script), assigned_gpu="3")

    assert result.status == "blocked"
    assert "exact container_repository" in (result.error or "")
    assert not (run_dir / "logs" / "fake-docker-argv.txt").exists()


def test_executor_fails_when_expected_output_is_missing(tmp_path: Path) -> None:
    run_dir, script = _prepared_run(tmp_path, create_output=False)
    manifest = _manifest(script)

    result = execute_manifest(run_dir, manifest, assigned_gpu="0")

    assert result.status == "failed"
    assert result.exit_code == 0
    assert result.missing_outputs == ["artifacts/full-run/metrics.json"]


def test_recovery_delegates_to_staged_pipeline_without_saved_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "assigned_physical_gpu": "3",
                "model": "saved-model",
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "paper_url": "https://example.test/paper.pdf",
                "repo_url": "https://github.com/example/repo",
                "timeout_seconds": 120,
                "target": None,
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, Any] = {}

    async def fake_pipeline(_run_dir: Path, args: Any, *, recovering: bool) -> int:
        observed.update(run_dir=_run_dir, args=args, recovering=recovering)
        return 0

    monkeypatch.setattr("repo2reproduce.pipeline.run_staged_pipeline", fake_pipeline)
    args = argparse.Namespace(
        recover_run=run_dir,
        gpu=None,
        model=None,
        max_turns=10,
        max_budget_usd=2.0,
    )

    exit_code = asyncio.run(runner.recover_existing_run(args))

    assert exit_code == 0
    assert observed["run_dir"] == run_dir
    assert observed["recovering"] is True
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "recovering"
    assert "session_id" not in metadata


def test_final_report_renderer_links_authoritative_evidence_files() -> None:
    report = FinalReport.model_validate(
        {
            "selected_experiment": "Table 1 / DemoSet / proposed method",
            "result_location": {
                "description": "Table 1, proposed method row, DemoSet column",
                "table_or_figure": "Table 1",
                "row": "proposed method",
                "column": "DemoSet",
                "cell": "accuracy",
            },
            "execution_status": "succeeded",
            "metric_name": "accuracy",
            "paper_target_value": "0.90",
            "achieved_value": "0.91",
            "baseline": {
                "applicable": True,
                "method": "Prior method",
                "value": "0.80",
                "metric_direction": "higher_is_better",
                "paper_evidence": {
                    "quote": "The prior method obtains an accuracy of 0.80.",
                    "locator": "p.4, Table 1",
                },
                "applicability_explanation": "Same dataset and evaluation protocol.",
            },
            "comparison": {
                "comparable": True,
                "calculation": "abs(0.91 - 0.90) / 0.90 * 100",
                "relative_error_percent": 1.111111,
                "baseline_advantage_preserved": True,
                "verdict": "match",
                "explanation": "Within five percent and above the baseline.",
            },
            "artifacts": {
                "dockerfile": "workspace/execution/Dockerfile",
                "build_log": "artifacts/env/build.log",
                "launch_script": "workspace/execution/launch.sh",
                "launch_script_sha256": "0" * 64,
                "metric_artifact": "artifacts/full-run/metric.json",
            },
            "assumptions": [
                {
                    "id": "A1",
                    "category": "config",
                    "setting": "batch size",
                    "value_used": "64",
                    "paper_status": "not_specified",
                    "source_type": "code_default",
                    "source_path": "config.py",
                    "audit_item_ids": ["batch-size"],
                    "rationale": "The official entrypoint uses this default.",
                    "impact": "May affect optimization.",
                }
            ],
        }
    )

    markdown = runner._render_final_report(report)

    assert "**Paper location:** Table 1, proposed method row, DemoSet column" in markdown
    assert "**Verdict:** match" in markdown
    assert "**Configuration used:** `execution_manifest.json`" in markdown
    assert "**Scientific launch script:** `workspace/execution/launch.sh`" in markdown
    assert "**[config] A1 — batch size:** 64" in markdown


def test_final_report_schema_failure_gets_one_correction_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "logs").mkdir()
    observed: list[dict[str, object]] = []
    valid_report = {
        "selected_experiment": "interval experiment",
        "result_location": {"description": "Figure 2c"},
        "execution_status": "succeeded",
        "metric_name": "certified interval",
        "paper_target_value": "(0.25, 0.30]",
        "achieved_value": "(0.25, 0.30]",
        "baseline": {
            "applicable": False,
            "metric_direction": "higher_is_better",
            "applicability_explanation": "No claim-aligned baseline applies.",
        },
        "comparison": {
            "comparable": True,
            "calculation": "The reproduced interval equals the paper interval.",
            "relative_error_percent": None,
            "baseline_advantage_preserved": None,
            "verdict": "match",
            "explanation": "The intervals are identical.",
        },
    }
    responses = [
        (
            SimpleNamespace(
                is_error=True,
                result="baseline field failed schema validation",
                errors=[],
                subtype="error",
                structured_output=None,
            ),
            "finalize-session",
        ),
        (
            SimpleNamespace(
                is_error=False,
                result=None,
                errors=[],
                subtype="success",
                structured_output=valid_report,
            ),
            "finalize-session",
        ),
    ]

    def fake_options(*args: object, **kwargs: object) -> dict:
        return {}

    async def fake_invoke(
        *,
        prompt: str,
        options: object,
        stage: str,
        skill_trace: runner.SkillTrace,
        **kwargs: object,
    ) -> tuple[object, str | None]:
        observed.append(
            {
                "prompt": prompt,
                "resume": getattr(options, "resume", None),
                "stage": stage,
                "skills": getattr(options, "skills", None),
            }
        )
        for message in _skill_messages(runner.RESULT_COMPARISON_SKILL):
            skill_trace.observe(message)
        return responses.pop(0)

    monkeypatch.setattr(runner, "_base_options", fake_options)
    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    report = asyncio.run(
        runner._request_final_report(
            tmp_path,
            prompt="Read the frozen evidence and report.",
            assigned_gpu="0",
            model="test",
            max_turns=5,
            max_budget_usd=1,
        )
    )

    assert report.comparison.verdict == "match"
    assert [item["stage"] for item in observed] == ["finalize-1", "finalize-2"]
    assert observed[0]["skills"] == [runner.RESULT_COMPARISON_SKILL]
    assert observed[0]["resume"] is None
    assert observed[1]["resume"] == "finalize-session"
    assert "failed the required output schema" in str(observed[1]["prompt"])
    assert "do not launch or repeat the experiment" in str(observed[1]["prompt"])


@pytest.mark.parametrize(
    ("verdict", "execution_status", "comparable"),
    [
        ("match", "succeeded", True),
        ("miss", "succeeded", True),
        ("run_incomplete", "failed", False),
        ("needs_review", "succeeded", False),
    ],
)
def test_final_report_supports_every_verdict(
    verdict: str,
    execution_status: str,
    comparable: bool,
) -> None:
    report = FinalReport.model_validate(
        {
            "selected_experiment": "interval experiment",
            "result_location": {"description": "Table 2, interval row"},
            "execution_status": execution_status,
            "metric_name": "certified interval",
            "paper_target_value": "(0.15, 0.20]",
            "achieved_value": "(0.15, 0.20]" if execution_status == "succeeded" else None,
            "baseline": {
                "applicable": False,
                "metric_direction": "not_applicable",
                "applicability_explanation": "The paper reports no comparable baseline.",
            },
            "comparison": {
                "comparable": comparable,
                "calculation": "Interval membership/equality; relative error is not meaningful.",
                "relative_error_percent": None,
                "verdict": verdict,
                "explanation": "Structured verdict fixture.",
            },
        }
    )

    assert report.comparison.verdict == verdict


def test_non_applicable_baseline_can_retain_metric_direction() -> None:
    report = FinalReport.model_validate(
        {
            "selected_experiment": "FLOPs measurement",
            "result_location": {"description": "Table 2, FLOPs row"},
            "execution_status": "failed",
            "metric_name": "Inference FLOPs",
            "paper_target_value": "8.83e6",
            "achieved_value": None,
            "baseline": {
                "applicable": False,
                "metric_direction": "lower_is_better",
                "applicability_explanation": "No claim-aligned baseline applies.",
            },
            "comparison": {
                "comparable": False,
                "calculation": "No metric was produced.",
                "relative_error_percent": None,
                "verdict": "run_incomplete",
                "explanation": "Execution failed before metric extraction.",
            },
        }
    )

    assert report.baseline.metric_direction == "lower_is_better"
