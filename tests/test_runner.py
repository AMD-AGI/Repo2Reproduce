from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from repo2reproduce import runner
from repo2reproduce.executor import execute_manifest
from repo2reproduce.runner import initialize_run
from repo2reproduce.schemas import ExecutionManifest


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


def test_project_environment_loads_repository_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    def fake_load(path: Path, *, override: bool) -> None:
        observed.update(path=path, override=override)

    monkeypatch.setattr(runner, "load_dotenv", fake_load)
    runner.load_project_environment()

    assert observed == {"path": runner.PACKAGE_DIR.parent / ".env", "override": False}


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
