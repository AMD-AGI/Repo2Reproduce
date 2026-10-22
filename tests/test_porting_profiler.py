from __future__ import annotations

import subprocess
from typing import Any

import pytest

from repo2reproduce.porting import profiler
from repo2reproduce.porting.profiler import KernelStat, Mount, RunCapture

# Header captured from rocprofv3 1.1.0 (ROCm 7.2) run_kernel_stats.csv.
HEADER = '"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"'


def _stats_csv(*rows: tuple[str, int, int, int, int]) -> str:
    lines = [HEADER]
    for name, calls, total, low, high in rows:
        lines.append(f'"{name}",{calls},{total},{total / calls:.6f},50.0,{low},{high},1.0')
    return "\n".join(lines)


def test_parse_kernel_stats_merges_processes() -> None:
    text = (
        _stats_csv(("void gemm<float, 4>(float*)", 2, 300, 100, 200), ("relu", 1, 50, 50, 50))
        + "\n"
        + _stats_csv(("void gemm<float, 4>(float*)", 1, 120, 120, 120))
    )

    stats = {stat.name: stat for stat in profiler.parse_kernel_stats(text)}

    gemm = stats["void gemm<float, 4>(float*)"]
    assert (gemm.calls, gemm.total_ns, gemm.average_ns) == (3, 420.0, 140.0)
    assert (gemm.min_ns, gemm.max_ns) == (100.0, 200.0)
    assert stats["relu"].calls == 1


def _capture(index: int, exit_code: int, gemm_total: float) -> RunCapture:
    return RunCapture(
        index,
        exit_code,
        (
            KernelStat("gemm", 10, gemm_total, gemm_total / 10),
            KernelStat("__amd_rocclr_fillBufferAligned", 1, 5.0, 5.0),
        ),
    )


def test_summarize_drops_warmup_failures_and_runtime_kernels() -> None:
    captures = [
        _capture(1, 0, 9000.0),
        _capture(2, 0, 100.0),
        _capture(3, 0, 130.0),
        _capture(4, 0, 110.0),
        _capture(5, 1, 50.0),
    ]

    kernels, median_total, cv_total = profiler.summarize(captures, warmup=1)

    assert [kernel.name for kernel in kernels] == ["gemm"]
    gemm = kernels[0]
    assert (gemm.runs_present, gemm.median_total_ns, gemm.median_average_ns) == (3, 110.0, 11.0)
    assert (gemm.min_total_ns, gemm.max_total_ns) == (100.0, 130.0)
    assert median_total == 110.0
    assert gemm.cv == pytest.approx(cv_total)
    assert 0.1 < gemm.cv < 0.15


def test_script_and_docker_command_follow_rocprofv3_constraints() -> None:
    script = profiler.build_profile_script(
        ["python", "bench.py", "--size", "4096 x"],
        runs=5,
        warmup=2,
        workdir="/work",
        kernel_regex="gemm|attn",
        run_timeout=60,
    )

    assert "cd /work" in script
    assert "for i in $(seq 1 7); do" in script
    assert (
        "timeout -s KILL 60 rocprofv3 --kernel-trace --stats -f csv --kernel-include-regex "
        "'gemm|attn' -d \"$out/$i\" -o \"run_%pid%\" -- python bench.py --size '4096 x'"
    ) in script

    argv = profiler.build_docker_argv(
        "rocm/pytorch:test",
        script,
        workdir="/work",
        name="r2r-rocprof-test",
        gpu="7",
        mounts=[Mount("/data/repo", "/work")],
    )
    assert argv[:8] == [
        "docker",
        "run",
        "--rm",
        "--name",
        "r2r-rocprof-test",
        "--ipc=host",
        "-w",
        "/work",
    ]
    assert "--device=/dev/kfd" in argv and "--device=/dev/dri" in argv
    assert argv[argv.index("-e") + 1] == "ROCR_VISIBLE_DEVICES=7"
    assert "type=bind,source=/data/repo,target=/work,readonly" in argv
    assert argv[-4:] == ["rocm/pytorch:test", "bash", "-lc", script]


def test_profile_rejects_root_working_directory() -> None:
    with pytest.raises(ValueError, match="other than '/'"):
        profiler.profile_in_docker("rocm/pytorch:test", ["python", "x.py"], workdir="/")


def _container_output(totals: list[int], failing: int) -> str:
    lines: list[str] = []
    for index, total in enumerate(totals, 1):
        code = 1 if index == failing else 0
        lines.append(f"@@r2r-rocprof begin {index} {code}")
        if code == 0:
            lines.append(_stats_csv(("gemm", 4, total, total // 4, total // 4)))
        lines.append(f"@@r2r-rocprof end {index}")
        if code:
            lines += [
                f"@@r2r-rocprof log {index}",
                "RuntimeError: boom",
                f"@@r2r-rocprof endlog {index}",
            ]
    return "\n".join(lines)


def test_profile_in_docker_summarises_measured_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            argv, 0, stdout=_container_output([5_000_000, 2_000_000, 2_400_000, 1], failing=4)
        )

    monkeypatch.setattr(profiler.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(profiler.subprocess, "run", fake_run)

    report = profiler.profile_in_docker(
        "rocm/pytorch:test", ["python", "bench.py"], runs=3, warmup=1, run_timeout=100
    )

    assert observed["timeout"] == 100 * 4 + 300
    assert report.exit_codes == [0, 0, 0, 1]
    assert report.measured_ok == 2
    assert report.median_total_ns == 2_200_000
    assert len(report.errors) == 1 and "measured run 4 exited with 1" in report.errors[0]
    assert "RuntimeError: boom" in report.errors[0]
    text = report.to_text()
    assert "Measured runs that succeeded: 2/3" in text
    assert "Median GPU kernel time per run: 2.2 ms" in text


def test_short_kernel_name_drops_templates_and_parameters() -> None:
    name = (
        "void at::native::vectorized_elementwise_kernel<4, at::native::Fn<float>, "
        "std::array<char*, 2ul> >(int, at::native::Fn<float>)"
    )

    assert profiler.short_kernel_name(name) == "at::native::vectorized_elementwise_kernel<...>"
    assert profiler.short_kernel_name("Cijk_Ailk_Bljk_S_B", width=10) == "Cijk_Ai..."
    anonymous = "void at::native::(anonymous namespace)::distribution_kernel<float, 4>(long, float)"
    assert (
        profiler.short_kernel_name(anonymous) == "at::native::{anonymous}::distribution_kernel<...>"
    )


def test_mount_specs() -> None:
    assert Mount.parse("/data:/work") == Mount("/data", "/work", True)
    assert Mount.parse("/data:/work:rw").read_only is False
    with pytest.raises(ValueError):
        Mount.parse("/data")
