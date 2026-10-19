from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from repo2reproduce import porting_commands
from repo2reproduce.runner import PACKAGE_DIR


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "workspace" / "repo").mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    return run_dir


def test_hook_allows_only_a_single_porting_command(tmp_path: Path) -> None:
    hook = PACKAGE_DIR / "templates" / "enforce_bash.py"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def decide(command: str) -> str:
        proc = subprocess.run(
            [sys.executable, str(hook)],
            input=json.dumps({"tool_input": {"command": command}}),
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "RUN_DIR": str(run_dir)},
        )
        payload = json.loads(proc.stdout)
        return payload["hookSpecificOutput"]["permissionDecision"]

    assert decide(f"port-scan {run_dir}/workspace/repo --min-severity review --json") == "allow"
    assert (
        decide(
            f"port-hipify {run_dir}/workspace/repo --output {run_dir}/workspace/repo-hip "
            "--overwrite --timeout 900"
        )
        == "allow"
    )
    assert decide("port-scan /tmp/repo; rm -rf /") == "deny"
    assert decide("port-scan $(pwd)") == "deny"
    assert decide("cd /tmp && port-scan repo") == "deny"
    assert decide("python -m repo2reproduce.porting.profiler") == "deny"


def test_scan_stays_inside_the_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _run_dir(tmp_path)
    source = run_dir / "workspace" / "repo" / "kernel.cu"
    source.write_text(
        "__global__ void k(int v) { __shfl_sync(0xffffffff, v, 0); }\n", encoding="utf-8"
    )
    monkey_env = pytest.MonkeyPatch()
    monkey_env.setenv("RUN_DIR", str(run_dir))
    try:
        assert porting_commands.scan_main([str(source.parent)]) == 0
        with pytest.raises(SystemExit, match="must stay inside"):
            porting_commands.scan_main([str(tmp_path)])
    finally:
        monkey_env.undo()

    report = json.loads((run_dir / "artifacts" / "port-scan.json").read_text(encoding="utf-8"))
    assert report["counts"]["high"] == 1
    assert "sync_mask_32bit" in capsys.readouterr().out
    assert "sync_mask_32bit" in (run_dir / "artifacts" / "port-scan.txt").read_text(
        encoding="utf-8"
    )


def test_hipify_translates_into_the_workspace_with_the_run_tool_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_dir(tmp_path)
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    seen: dict[str, object] = {}

    def fake_hipify(root: Path, output: Path, **kwargs: object) -> object:
        seen["root"] = root
        seen["output"] = output
        seen.update(kwargs)

        class Report:
            error = None

            def to_text(self) -> str:
                return "translated"

            def to_dict(self) -> dict[str, str]:
                return {"status": "translated"}

        return Report()

    monkeypatch.setattr(porting_commands, "hipify_repository", fake_hipify)
    clone = run_dir / "workspace" / "repo"
    assert porting_commands.hipify_main([str(clone)]) == 0
    workspace = (run_dir / "workspace").resolve()
    assert seen["root"] == clone.resolve()
    assert seen["output"] == workspace / "repo-hip"
    assert seen["cache_dir"] == run_dir.resolve() / "tools"
    assert seen["overwrite"] is False
    assert (run_dir / "artifacts" / "port-hipify.txt").read_text(encoding="utf-8") == (
        "translated\n"
    )

    custom = run_dir / "workspace" / "ported"
    assert porting_commands.hipify_main([str(clone), "--output", str(custom), "--overwrite"]) == 0
    assert seen["output"] == workspace / "ported" and seen["overwrite"] is True
    with pytest.raises(SystemExit, match="--output must stay inside"):
        porting_commands.hipify_main([str(clone), "--output", str(tmp_path / "outside")])
    with pytest.raises(SystemExit, match="not the whole workspace"):
        porting_commands.hipify_main([str(run_dir / "workspace")])
