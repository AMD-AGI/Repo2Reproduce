"""Porting rules on the multi-step agent, from run setup through execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from repo2reproduce import pipeline, runner
from repo2reproduce.executor import ManifestExecutionError, execute_manifest, validate_porting_files
from repo2reproduce.pipeline import replay_preparation_artifacts
from repo2reproduce.porting.official_hipify import (
    HIPIFY_COMMIT,
    HIPIFY_REF,
    HIPIFY_REPOSITORY,
    HipifyTool,
)
from repo2reproduce.runner import (
    ROCM_PORTING_SKILL,
    SKILLS_DIR,
    initialize_run,
    porting_policy_prompt,
)
from repo2reproduce.schemas import (
    PORTING_LADDER,
    EnvironmentManifest,
    ExecutionManifest,
    ExecutionRepairOutput,
    ForcedKernelPortingManifest,
    ForcedPortingPreparationStageOutput,
    RocmPortingManifest,
    execution_repair_model,
    preparation_stage_model,
)
from repo2reproduce.validators import StageValidationError, sha256_file, validate_execution_repair
from tests.test_pipeline import _audit, _code_ref, _paper, _repository, _resolved_inputs, _target


def _attempt(strategy: str, outcome: str) -> dict[str, str]:
    return {
        "strategy": strategy,
        "outcome": outcome,
        "evidence": f"{strategy}: checked upstream, the repository, and the build log",
    }


def _script(run_dir: Path) -> Path:
    script = run_dir / "workspace" / "execution" / "launch.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "REPO_DIR=/run/workspace/repo\n"
        'cd "${REPO_DIR}"\n'
        "mkdir -p /run/artifacts/full-run\n"
        "python run.py --dataset DemoSet | tee /run/artifacts/full-run/metrics.json\n",
        encoding="utf-8",
    )
    return script


def _base_manifest(script: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ready",
        "selected_experiment": "kernel benchmark",
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


def _ported_ready_payload(script: Path, **parity_updates: Any) -> dict[str, Any]:
    return {
        **_base_manifest(script),
        "rocm_port": {
            "uses_cuda_specific_code": True,
            "summary": "hipified the custom kernels and widened warp-shuffle masks",
            "attempts": [
                _attempt("existing_rocm_port", "not_applicable"),
                _attempt("portable_repo_path", "not_applicable"),
                _attempt("hipify_with_fixes", "succeeded"),
            ],
            "patches": ["docker/rocm_port.py"],
            "ported_kernels": [
                {
                    "cuda_kernel": "tp_forward_kernel",
                    "cuda_source": "repo/csrc/tp.cu",
                    "implementation": "hip",
                    "port_files": ["repo-hip/csrc/tp.cu"],
                }
            ],
            "parity_check": {
                "reference": "e3nn TensorProduct",
                "inputs": "float32, 1024 random edges, seed 0",
                "error_metric": "max absolute error",
                "observed_error": 2e-6,
                "tolerance": 1e-4,
                "log_path": "parity/parity.log",
                **parity_updates,
            },
        },
    }


def _blocked_porting_payload(reason: str, **port: Any) -> dict[str, Any]:
    return {
        "status": "blocked",
        "selected_experiment": "kernel benchmark",
        "blocked_reason": reason,
        "rocm_port": {
            "uses_cuda_specific_code": True,
            "summary": "custom CUDA attention kernels",
            **port,
        },
    }


def _write_port_sources(run_dir: Path) -> None:
    workspace = run_dir / "workspace"
    for relative, kernel in (
        ("repo/csrc/tp.cu", "__global__ void tp_forward_kernel(float* out) {}\n"),
        ("repo-hip/csrc/tp.cu", "__global__ void tp_forward_kernel(float* out) {}\n"),
        ("docker/rocm_port.py", "print('patched')\n"),
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(kernel, encoding="utf-8")


def _fake_docker(run_dir: Path) -> None:
    wrapper = run_dir / "bin" / "docker"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$@" > "$RUN_DIR/logs/fake-docker-argv.txt"\n'
        'mkdir -p "$RUN_DIR/artifacts/full-run"\n'
        "printf '%s\\n' '{\"score\": 1}' > \"$RUN_DIR/artifacts/full-run/metrics.json\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    smoke = run_dir / "artifacts" / "smoke" / "smoke.log"
    smoke.parent.mkdir(parents=True, exist_ok=True)
    smoke.write_text("smoke passed\n", encoding="utf-8")


def test_porting_blocked_manifest_requires_every_strategy() -> None:
    partial = _blocked_porting_payload(
        "the kernels only build with nvcc",
        attempts=[_attempt("existing_rocm_port", "failed")],
    )
    with pytest.raises(ValidationError, match="missing: portable_repo_path"):
        RocmPortingManifest.model_validate(partial)

    complete = _blocked_porting_payload(
        "the kernels only build with nvcc",
        attempts=[_attempt(strategy, "failed") for strategy in PORTING_LADDER],
    )
    assert RocmPortingManifest.model_validate(complete).status == "blocked"


def test_porting_ladder_applies_when_blocked_reason_mentions_cuda() -> None:
    cuda_reason = _blocked_porting_payload(
        "setup.py raises: Cannot find CUDA_HOME", uses_cuda_specific_code=False
    )
    with pytest.raises(ValidationError, match="must record every porting strategy"):
        RocmPortingManifest.model_validate(cuda_reason)

    data_reason = _blocked_porting_payload(
        "the dataset requires a signed license", uses_cuda_specific_code=False
    )
    assert RocmPortingManifest.model_validate(data_reason).status == "blocked"


def test_kernel_ports_reject_pytorch_and_require_parity(tmp_path: Path) -> None:
    script = _script(tmp_path)
    assert RocmPortingManifest.model_validate(_ported_ready_payload(script)).status == "ready"

    with pytest.raises(ValidationError, match="parity check failed"):
        RocmPortingManifest.model_validate(_ported_ready_payload(script, observed_error=3e-3))

    unlisted = _ported_ready_payload(script)
    unlisted["rocm_port"]["ported_kernels"] = []
    with pytest.raises(ValidationError, match="a PyTorch fallback is not a port"):
        ForcedKernelPortingManifest.model_validate(unlisted)

    pytorch = _ported_ready_payload(script)
    pytorch["rocm_port"]["ported_kernels"][0]["implementation"] = "pytorch"
    with pytest.raises(ValidationError, match="implementation"):
        RocmPortingManifest.model_validate(pytorch)

    fallback = _ported_ready_payload(script)
    fallback["rocm_port"]["attempts"] = [
        _attempt("existing_rocm_port", "failed"),
        _attempt("portable_repo_path", "succeeded"),
    ]
    with pytest.raises(ValidationError, match="hipify_with_fixes or targeted_rewrite"):
        ForcedKernelPortingManifest.model_validate(fallback)


def test_forced_port_files_gate_execution(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    script = _script(run_dir)
    _fake_docker(run_dir)
    manifest = ForcedKernelPortingManifest.model_validate(_ported_ready_payload(script))
    _write_port_sources(run_dir)
    parity_log = run_dir / "artifacts" / "parity" / "parity.log"
    parity_log.parent.mkdir(parents=True)
    parity_log.write_text("max error 2e-6\n", encoding="utf-8")

    with pytest.raises(ManifestExecutionError, match="port-scan.json"):
        validate_porting_files(run_dir, manifest)
    blocked = execute_manifest(run_dir, manifest, assigned_gpu="0")
    assert blocked.status == "blocked"
    assert not (run_dir / "logs" / "fake-docker-argv.txt").exists()

    (run_dir / "artifacts" / "port-scan.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "artifacts" / "port-hipify.json").write_text("{}\n", encoding="utf-8")
    validate_porting_files(run_dir, manifest)
    result = execute_manifest(run_dir, manifest, assigned_gpu="0")
    assert result.status == "succeeded"
    assert result.exit_code == 0


def test_initialize_run_installs_porting_tools_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[Path] = []

    def fake_ensure_hipify(cache_dir: Path) -> HipifyTool:
        fetched.append(cache_dir)
        return HipifyTool(
            repository=HIPIFY_REPOSITORY,
            ref=HIPIFY_REF,
            commit=HIPIFY_COMMIT,
            script=str(cache_dir / f"HIPIFY-{HIPIFY_REF}" / "bin" / "hipify-perl"),
        )

    monkeypatch.setattr(runner, "ensure_hipify", fake_ensure_hipify)
    common: dict[str, Any] = {
        "paper_url": "https://arxiv.org/abs/0000.00000",
        "repo_url": "https://github.com/example/project",
        "assigned_gpu": "0",
        "timeout_seconds": 60,
        "target": None,
    }
    initialize_run(tmp_path / "baseline", run_id="baseline", **common)
    initialize_run(
        tmp_path / "porting",
        run_id="porting",
        rocm_porting=True,
        force_kernel_port=True,
        **common,
    )

    baseline = json.loads((tmp_path / "baseline" / "metadata.json").read_text(encoding="utf-8"))
    porting = json.loads((tmp_path / "porting" / "input.json").read_text(encoding="utf-8"))
    assert baseline["rocm_porting"] is False
    assert not (tmp_path / "baseline" / "bin" / "port-scan").exists()
    assert porting["rocm_porting"] is True and porting["force_kernel_port"] is True
    assert fetched == [tmp_path / "porting" / "tools"]
    script = (tmp_path / "porting" / "bin" / "port-hipify").read_text(encoding="utf-8")
    assert "repo2reproduce.porting_commands" in script
    assert (tmp_path / "porting" / "bin" / "port-scan").stat().st_mode & 0o111

    policy = porting_policy_prompt(str(tmp_path / "porting" / "workspace"), force_kernel_port=True)
    assert "FORCED KERNEL PORT" in policy
    assert "port-scan CLONE" in policy
    assert f"Invoke the `{ROCM_PORTING_SKILL}` skill" in policy
    skill = (SKILLS_DIR / ROCM_PORTING_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "Never call `torch.utils.hipify`" in skill
    assert "No PyTorch fallback" in skill
    plain = porting_policy_prompt(str(tmp_path / "workspace"), force_kernel_port=False)
    assert "STANDARD PORTING" in plain
    assert "FORCED KERNEL PORT" not in plain


def test_preparation_schema_requires_rocm_port_only_in_porting_mode() -> None:
    assert "rocm_port" not in ExecutionManifest.model_json_schema()["properties"]
    porting_schema = preparation_stage_model(True).model_json_schema()
    assert "rocm_port" in json.dumps(porting_schema)
    forced_schema = preparation_stage_model(True, True).model_json_schema()
    assert forced_schema["$defs"]["ForcedKernelPortingManifest"] or forced_schema


def test_replay_and_execute_a_forced_port(
    tmp_path: Path,
) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    (tmp_path / "configuration_audit.json").write_text(
        audit.model_dump_json(indent=2), encoding="utf-8"
    )
    script = _script(tmp_path)
    _write_port_sources(tmp_path)
    _fake_docker(tmp_path)
    dockerfile = tmp_path / "workspace" / "execution" / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    build_log = tmp_path / "artifacts" / "logs" / "build.log"
    build_log.parent.mkdir(parents=True)
    build_log.write_text("build passed\n", encoding="utf-8")
    parity = tmp_path / "artifacts" / "parity" / "parity.log"
    parity.parent.mkdir(parents=True)
    parity.write_text("max error 2e-6\n", encoding="utf-8")
    (tmp_path / "artifacts" / "port-scan.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifacts" / "port-hipify.json").write_text("{}\n", encoding="utf-8")

    environment = EnvironmentManifest(
        status="ready",
        image_tag="demo:test",
        dockerfile="workspace/execution/Dockerfile",
        dockerfile_sha256=hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        build_log_path="artifacts/logs/build.log",
        build_succeeded=True,
    )
    (tmp_path / "environment_manifest.json").write_text(
        environment.model_dump_json(indent=2), encoding="utf-8"
    )
    from repo2reproduce.validators import sha256_file

    manifest = ForcedKernelPortingManifest.model_validate(
        {
            **_ported_ready_payload(script),
            "schema_version": 3,
            "selected_experiment": target.selected_experiment,
            "image_tag": "demo:test",
            "target_sha256": sha256_file(tmp_path / "target.json"),
            "audit_sha256": sha256_file(tmp_path / "configuration_audit.json"),
            "environment_sha256": sha256_file(tmp_path / "environment_manifest.json"),
            "repository_commit": "abc123",
            "entrypoint": _code_ref("def run").model_dump(mode="json"),
            "scientific_configuration": [
                item.model_dump(mode="json") for item in _resolved_inputs()
            ],
            "metric_name": target.metric_name,
            "metric_artifact": "artifacts/full-run/metrics.json",
            "paper_target_value": target.metric_value,
            "metric_extractor": {
                "source": "artifacts/full-run/metrics.json",
                "format": "json",
                "selector": "score",
            },
        }
    )
    (tmp_path / "execution_manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    replayed_environment, replayed_manifest = replay_preparation_artifacts(
        tmp_path,
        target,
        audit,
        rocm_porting=True,
        force_kernel_port=True,
    )
    assert isinstance(replayed_manifest, ForcedKernelPortingManifest)
    assert replayed_environment.status == "ready"
    assert replayed_manifest.rocm_port.ported_kernels[0].cuda_kernel == "tp_forward_kernel"

    result = execute_manifest(tmp_path, replayed_manifest, assigned_gpu="1")
    assert result.status == "succeeded"
    assert result.achieved_metric == "1"
    argv = (tmp_path / "logs" / "fake-docker-argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv[0] == "run"
    assert "/run/workspace/execution/launch.sh" in argv


def test_cli_rejects_forced_port_without_porting_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["r2r", "--force-kernel-port", "https://arxiv.org/abs/1", "https://github.com/a/b"],
    )
    with pytest.raises(SystemExit):
        runner.main()


def test_preparation_output_model_is_the_forced_variant() -> None:
    assert preparation_stage_model(True, True) is ForcedPortingPreparationStageOutput


def test_run_root_spellings_of_porting_paths_are_rebased(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    (tmp_path / "configuration_audit.json").write_text(audit.model_dump_json(), encoding="utf-8")
    payload = _ported_ready_payload(_script(tmp_path), log_path="artifacts/parity/parity.log")
    port = payload["rocm_port"]
    port["patches"] = ["workspace/docker/rocm_port.py"]
    port["ported_kernels"][0]["cuda_source"] = "workspace/repo/csrc/tp.cu"
    port["ported_kernels"][0]["port_files"] = ["workspace/repo-hip/csrc/tp.cu"]
    raw = {"environment": {"status": "ready"}, "manifest": payload}

    normalized = pipeline._normalize_preparation_output(
        raw, run_dir=tmp_path, target=target, audit=audit
    )["manifest"]["rocm_port"]

    assert normalized["patches"] == ["docker/rocm_port.py"]
    assert normalized["ported_kernels"][0]["cuda_source"] == "repo/csrc/tp.cu"
    assert normalized["ported_kernels"][0]["port_files"] == ["repo-hip/csrc/tp.cu"]
    assert normalized["parity_check"]["log_path"] == "parity/parity.log"


def test_repair_output_keeps_the_porting_manifest() -> None:
    assert execution_repair_model(False) is ExecutionRepairOutput
    for rocm_porting, force, manifest_model in (
        (True, False, RocmPortingManifest),
        (True, True, ForcedKernelPortingManifest),
    ):
        model = execution_repair_model(rocm_porting, force)
        assert model.model_fields["manifest"].annotation is manifest_model
        assert "rocm_port" in json.dumps(model.model_json_schema())


class _StopStage(Exception):
    pass


def _capture_stage_skills(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    observed: list[list[str]] = []

    async def fake_stage(*args: object, skills: list[str], **kwargs: object) -> None:
        observed.append(list(skills))
        raise _StopStage

    monkeypatch.setattr(pipeline, "_run_structured_stage", fake_stage)
    return observed


@pytest.mark.parametrize(
    ("rocm_porting", "expected"),
    [(False, []), (True, [ROCM_PORTING_SKILL])],
)
def test_preparation_enables_the_porting_skill_only_in_porting_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rocm_porting: bool,
    expected: list[str],
) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    (tmp_path / "configuration_audit.json").write_text(audit.model_dump_json(), encoding="utf-8")
    observed = _capture_stage_skills(monkeypatch)

    with pytest.raises(_StopStage):
        asyncio.run(
            pipeline.prepare_environment(
                tmp_path,
                target=target,
                audit=audit,
                timeout_seconds=60,
                assigned_gpu="0",
                model="test",
                max_turns=1,
                max_budget_usd=1,
                rocm_porting=rocm_porting,
            )
        )

    assert observed == [expected]
    prompt = (tmp_path / "prompts" / "environment_preparation.md").read_text(encoding="utf-8")
    assert (f"`{ROCM_PORTING_SKILL}` skill" in prompt) is rocm_porting


def test_porting_repair_loads_both_skills_and_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _paper(tmp_path)
    _repository(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    (tmp_path / "configuration_audit.json").write_text(audit.model_dump_json(), encoding="utf-8")
    (tmp_path / "metadata.json").write_text('{"stages": {}}', encoding="utf-8")
    manifest = RocmPortingManifest.model_validate(_ported_ready_payload(_script(tmp_path)))
    environment = EnvironmentManifest(status="blocked", blocked_reason="not built in this test")

    def fake_patch(run_dir: Path, audit: object, attempt: int) -> tuple[str, str, list[str]]:
        patch = run_dir / "execution_repairs" / f"repair-{attempt}.patch"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text("", encoding="utf-8")
        return patch.relative_to(run_dir).as_posix(), "0" * 64, []

    monkeypatch.setattr(pipeline, "_capture_repository_patch", fake_patch)
    observed = _capture_stage_skills(monkeypatch)

    with pytest.raises(_StopStage):
        asyncio.run(
            pipeline.repair_execution(
                tmp_path,
                target=target,
                audit=audit,
                environment=environment,
                manifest=manifest,
                failed_result=SimpleNamespace(log_path="logs/executor.log"),
                attempt=1,
                max_repairs=2,
                failure_fingerprint="0" * 64,
                assigned_gpu="0",
                model="test",
                max_turns=1,
                max_budget_usd=1,
                rocm_porting=True,
                force_kernel_port=True,
            )
        )

    assert observed == [[runner.EXECUTION_REPAIR_SKILL, ROCM_PORTING_SKILL]]
    prompt = (tmp_path / "prompts" / "execution_repair-1.md").read_text(encoding="utf-8")
    assert "FORCED KERNEL PORT" in prompt
    assert f"`{ROCM_PORTING_SKILL}` skill" in prompt


def test_repair_cannot_drop_a_ported_kernel(tmp_path: Path) -> None:
    _paper(tmp_path)
    target = _target(tmp_path)
    audit = _audit(tmp_path, target)
    (tmp_path / "configuration_audit.json").write_text(audit.model_dump_json(), encoding="utf-8")
    script = _script(tmp_path)
    previous = RocmPortingManifest.model_validate(_ported_ready_payload(script))
    repaired_payload = _ported_ready_payload(script)
    repaired_payload["rocm_port"]["ported_kernels"][0]["cuda_kernel"] = "other_kernel"
    repaired = RocmPortingManifest.model_validate(repaired_payload)
    output = SimpleNamespace(
        repair=SimpleNamespace(
            attempt=1,
            failure_fingerprint="0" * 64,
            changes=[],
            smoke_test=SimpleNamespace(exit_code=0, log_path="artifacts/smoke/smoke.log"),
            repository_patch=SimpleNamespace(
                base_commit=audit.repository_commit, diff_path="missing.patch", diff_sha256=""
            ),
        ),
        environment=SimpleNamespace(status="ready"),
    )

    common = {
        "attempt": 1,
        "failure_fingerprint": "0" * 64,
        "target_sha256": sha256_file(tmp_path / "target.json"),
        "audit_sha256": sha256_file(tmp_path / "configuration_audit.json"),
    }
    output.manifest = repaired
    with pytest.raises(StageValidationError, match=r"removed ported kernels: tp_forward_kernel"):
        validate_execution_repair(tmp_path, target, audit, previous, output, **common)
    output.manifest = ExecutionManifest.model_validate(_base_manifest(script))
    with pytest.raises(StageValidationError, match="dropped the rocm_port record"):
        validate_execution_repair(tmp_path, target, audit, previous, output, **common)
