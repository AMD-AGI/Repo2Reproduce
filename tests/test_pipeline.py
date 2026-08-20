from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path

from repo2reproduce import orchestrator
from repo2reproduce.hooks import build_hooks


def _outputs(*, build_status: str = "not_attempted", executor_ok: bool = True):
    plan = {
        "commands": [
            {
                "step": "evaluate",
                "cmd": "python eval.py --model demo",
                "flags_verified": True,
            }
        ],
        "assumptions": [],
    }
    return {
        "paper-analyst": {
            "title": "Demo Paper",
            "experiments": [],
            "hyperparameters": [],
        },
        "code-cartographer": {
            "entry_points": [],
            "cli_flags": [],
            "configs": [],
            "readme_commands": [],
        },
        "base-image-selector": {
            "base_image": "rocm/pytorch:test",
            "rationale": "test fixture",
            "risks": [],
        },
        "experiment-selector": {
            "selected_experiment": "Table 1 demo",
            "target_result": "1.0 accuracy",
            "justification_quotes": [],
            "why_this_one": "test fixture",
            "alternatives_considered": [],
        },
        "env-porter": {
            "requirements": [],
            "dropped_pins": [],
            "extra_install_cmds": [],
            "notes": [],
        },
        "kernel-migrator": {
            "has_custom_kernel": True,
            "sources": ["kernel.cu"],
            "note": "fixture",
        },
        "config-extractor": {},
        "command-synthesizer": plan,
        "build-verifier": {
            "base_image": "rocm/pytorch:test",
            "build_status": build_status,
            "has_custom_kernel": False,
            "amd_notes": [],
        },
        "reconciler": {
            "updated_output2": "unchanged",
            "updated_output3": plan,
            "changes": [],
        },
        "executor": {
            "executed": executor_ok,
            "command_run": "python eval.py --model demo" if executor_ok else None,
            "achieved_metric": "1.0 accuracy" if executor_ok else None,
            "paper_target": "1.0 accuracy",
            "match": True if executor_ok else None,
            "delta": "0" if executor_ok else None,
            "notes": "fixture",
        },
        "verifier": {
            "issues_found": [],
            "corrections_applied": [],
            "gaps_and_assumptions": [],
            "confidence": "high",
            "confidence_rationale": "fixture",
        },
    }


def _install_fakes(monkeypatch, outputs, failures=None, calls=None):
    failures = failures or set()
    calls = calls if calls is not None else []

    async def fake_bootstrap(phase: str, run_dir: Path, value: str):
        del value
        calls.append(phase)
        if phase == "ingest-paper":
            paper = run_dir / "paper"
            paper.mkdir(parents=True, exist_ok=True)
            (paper / "paper.md").write_text("# fixture", encoding="utf-8")
            (paper / "paper_parse.json").write_text("{}", encoding="utf-8")
            return {"markdown_path": str(paper / "paper.md")}
        repo = run_dir / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        return {"repo_path": str(repo)}

    async def fake_worker(**kwargs):
        phase = kwargs["phase"]
        calls.append(phase)
        if phase in failures:
            raise RuntimeError(f"fixture failure: {phase}")
        return orchestrator.WorkerOutcome(
            output=outputs[phase],
            raw_text=None,
            cost_usd=0.01,
            is_error=False,
            session_id=f"session-{phase}",
        )

    monkeypatch.setattr(orchestrator, "_bootstrap_operation", fake_bootstrap)
    monkeypatch.setattr(orchestrator, "_run_worker_session", fake_worker)
    monkeypatch.setattr(orchestrator.config, "max_budget_usd", lambda: 100.0)
    monkeypatch.setattr(orchestrator.config, "max_turns", lambda: 1000)
    return calls


def _run(tmp_path, *, amd=False, execute=False):
    return asyncio.run(
        orchestrator._run_pipeline(
            paper_url="https://example.test/paper",
            repo_url="https://example.test/repo",
            run_dir=tmp_path,
            target="Table 1",
            amd=amd,
            execute=execute,
            on_text=None,
            stream=False,
            enable_reviewers=False,
        )
    )


def test_execute_mode_rejects_not_attempted_build_output():
    error = orchestrator._semantic_phase_error(
        "build-verifier",
        {"build_status": "not_attempted"},
        execute=True,
    )
    assert error
    assert "execute mode" in error
    assert (
        orchestrator._semantic_phase_error(
            "build-verifier",
            {"build_status": "verified"},
            execute=True,
        )
        is None
    )
    assert (
        orchestrator._semantic_phase_error(
            "build-verifier",
            {"build_status": "not_attempted"},
            execute=False,
        )
        is None
    )


def test_track1_success_persists_every_phase(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, _outputs())

    result = _run(tmp_path)

    assert not result.is_error
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "pipeline_state.json").exists()
    assert orchestrator.PipelineStore(tmp_path).succeeded("paper-analyst")
    assert orchestrator.PipelineStore(tmp_path).succeeded("verifier")
    assert calls.index("experiment-selector") > calls.index("paper-analyst")
    assert calls.index("config-extractor") > calls.index("experiment-selector")
    assert calls.index("command-synthesizer") > calls.index("config-extractor")


def test_parallel_failure_does_not_cancel_independent_worker(monkeypatch, tmp_path):
    calls = _install_fakes(
        monkeypatch,
        _outputs(),
        failures={"paper-analyst"},
    )

    result = _run(tmp_path)
    store = orchestrator.PipelineStore(tmp_path)

    assert result.is_error
    assert store.load("paper-analyst").status == "failed"
    assert store.load("code-cartographer").status == "succeeded"
    assert store.load("experiment-selector").status == "skipped"
    assert "code-cartographer" in calls
    assert (tmp_path / "draft_report.json").exists()
    assert not (tmp_path / "report.json").exists()


def test_no_custom_kernel_is_explicit_optional_skip(monkeypatch, tmp_path):
    outputs = _outputs(build_status="not_attempted")
    _install_fakes(monkeypatch, outputs)
    monkeypatch.setattr(
        orchestrator.amd_tools,
        "detect_cuda_kernels",
        lambda repo: {"has_custom_kernel": False, "cuda_sources": []},
    )

    result = _run(tmp_path, amd=True, execute=False)
    record = orchestrator.PipelineStore(tmp_path).load("kernel-migrator")

    assert not result.is_error
    assert record.status == "skipped"
    assert record.skip_kind == "not_applicable"
    assert record.output["has_custom_kernel"] is False
    assert result.report["output4_amd_environment"]["build_status"] == "not_attempted"


def test_execute_build_failure_is_honest_complete_result(monkeypatch, tmp_path):
    outputs = _outputs(build_status="build_failed", executor_ok=False)
    calls = _install_fakes(monkeypatch, outputs)
    monkeypatch.setattr(
        orchestrator.amd_tools,
        "detect_cuda_kernels",
        lambda repo: {"has_custom_kernel": False, "cuda_sources": []},
    )

    result = _run(tmp_path, amd=True, execute=True)
    executor = orchestrator.PipelineStore(tmp_path).load("executor")

    assert not result.is_error
    assert executor.status == "skipped"
    assert executor.output["executed"] is False
    assert "executor" not in calls
    assert result.report["output4_amd_environment"]["build_status"] == "build_failed"


def test_resume_reuses_success_and_retries_failed_phase(monkeypatch, tmp_path):
    outputs = _outputs()
    first_calls = _install_fakes(
        monkeypatch,
        outputs,
        failures={"paper-analyst"},
    )
    first = _run(tmp_path)
    assert first.is_error
    assert "code-cartographer" in first_calls

    second_calls = []
    _install_fakes(monkeypatch, outputs, calls=second_calls)
    result = asyncio.run(
        orchestrator.resume_run(
            tmp_path,
            instruction="Retry the failed paper analysis.",
        )
    )

    assert not result.is_error
    assert "paper-analyst" in second_calls
    assert "code-cartographer" not in second_calls
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"


def test_runtime_uses_fixed_sdk_and_non_orchestration_hooks():
    sdk_version = tuple(int(part) for part in version("claude-agent-sdk").split(".")[:3])
    hooks = build_hooks(execute=True, amd=True)

    assert sdk_version >= (0, 2, 136)
    assert set(hooks) == {"PreToolUse", "SubagentStart", "SubagentStop"}
    assert len(hooks["PreToolUse"]) == 2  # Bash safety + observability logger
