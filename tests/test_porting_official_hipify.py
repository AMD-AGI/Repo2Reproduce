from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from repo2reproduce.porting import official_hipify as hipify

FIXTURES = Path(hipify.__file__).parent / "fixtures"
TOOL = hipify.HipifyTool(
    repository=hipify.HIPIFY_REPOSITORY,
    ref=hipify.HIPIFY_REF,
    commit=hipify.HIPIFY_COMMIT,
    script="/opt/HIPIFY/bin/hipify-perl",
)

# Captured from hipify-perl -print-stats of HIPIFY rocm-7.2.4.
HIPIFY_STDERR = """\
  probe.cu:1: warning: unsupported device function "__brkpt": __global__ void k(int v){ __brkpt(); }
  probe.cu:1: warning: unsupported device data type identifier: nv_bfloat162
  warning: probe.cu:#2 : void h(){ cudaFooBarUnknown(); }
  fa.cu:49: warning: deprecated CUDA identifier: __shfl_down since CUDA 9.0
  fa.cu:50: warning: removed CUDA identifier: cudaBindTexture since CUDA 12.0
  fa.cu:51: warning: experimental ROCm HIP identifier: cudaGraphFoo since 7.0

[HIPIFY] info: file 'probe.cu' statistics:
  CONVERTED refs count: 0
  TOTAL lines of code: 2
  WARNINGS: 3
[HIPIFY] info: CONVERTED refs by names:

[HIPIFY] info: file 'va.cu' statistics:
  CONVERTED refs count: 16
  TOTAL lines of code: 77
  WARNINGS: 0
[HIPIFY] info: CONVERTED refs by names:
  cudaFree => hipFree: 3
  cuda_runtime.h => hip/hip_runtime.h: 1
"""


def test_parse_hipify_output_reads_stats_and_warnings() -> None:
    files, warnings = hipify.parse_hipify_output(HIPIFY_STDERR)

    assert [stats.path for stats in files] == ["probe.cu", "va.cu"]
    assert files[1].stats == {
        "CONVERTED refs count": 16,
        "TOTAL lines of code": 77,
        "WARNINGS": 0,
    }
    assert files[1].converted["cuda_runtime.h => hip/hip_runtime.h"] == 1
    assert [(w.path, w.line, w.kind, w.identifiers) for w in warnings] == [
        ("probe.cu", 1, "unsupported", ("__brkpt",)),
        ("probe.cu", 1, "unsupported", ("nv_bfloat162",)),
        ("probe.cu", 2, "unknown", ("cudaFooBarUnknown",)),
        ("fa.cu", 49, "deprecated", ("__shfl_down",)),
        ("fa.cu", 50, "deprecated", ("cudaBindTexture",)),
        ("fa.cu", 51, "experimental", ("cudaGraphFoo",)),
    ]


def _repository(root: Path) -> None:
    (root / "kernels").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "build").mkdir()
    (root / "kernels" / "a.cu").write_text(
        "#include <cuda_runtime.h>\n__global__ void k() {}\nvoid launch() { cudaMalloc(0, 4); }\n",
        encoding="utf-8",
    )
    (root / "kernels" / "UPPER.CU").write_text("__global__ void u() {}\n", encoding="utf-8")
    (root / "src" / "host.cpp").write_text(
        "#include <ATen/cuda/CUDAContext.h>\n"
        "// at::cuda::getCurrentCUDAStream() in a comment is ignored\n"
        "void f() { auto s = at::cuda::getCurrentCUDAStream(); cudaMalloc(0, 4); }\n",
        encoding="utf-8",
    )
    (root / "src" / "cpu.cpp").write_text(
        "float g(float x) { return sqrtf(x); }\n", encoding="utf-8"
    )
    (root / "build" / "gen.cu").write_text("__global__ void g() {}\n", encoding="utf-8")
    (root / "setup.py").write_text(
        "from torch.utils.cpp_extension import CUDAExtension\n", encoding="utf-8"
    )


def test_selects_cuda_sources_like_findcode(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "kernels" / "link.cu").symlink_to(tmp_path / "kernels" / "a.cu")

    assert hipify.cuda_source_files(tmp_path) == [
        "kernels/UPPER.CU",
        "kernels/a.cu",
        "src/host.cpp",
    ]
    assert hipify.uses_pytorch_extension(tmp_path)


def test_find_leftovers_skips_comments() -> None:
    text = (
        "/* __CUDA_ARCH__ in a block comment */\n"
        "#include <ATen/cuda/HIPContext.h>\n"
        "#if __CUDA_ARCH__ >= 800 // at::cuda::ignored\n"
        "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    )

    leftovers = hipify.find_leftovers("k.cu", text)

    assert [(item.line, item.identifiers) for item in leftovers] == [
        (2, ("ATen/cuda/HIPContext.h",)),
        (3, ("__CUDA_ARCH__",)),
        (4, ("C10_CUDA_KERNEL_LAUNCH_CHECK",)),
    ]


def _fake_hipify_perl(calls: list[dict[str, Any]], fail: set[str] = frozenset()) -> Any:
    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "cwd": kwargs["cwd"]})
        source = argv[-1]
        if source in fail:
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="error: boom")
        output = Path(argv[3].removeprefix("-o="))
        text = (Path(kwargs["cwd"]) / source).read_text(encoding="utf-8")
        translated, count = re.subn(r"\bcuda(Malloc|_runtime\.h)", r"hip\1", text)
        translated = translated.replace("CUDAContext", "HIPContext")
        output.write_text(translated, encoding="utf-8")
        stderr = (
            f"\n[HIPIFY] info: file '{source}' statistics:\n"
            f"  CONVERTED refs count: {count}\n"
            f"  TOTAL lines of code: {text.count(chr(10))}\n"
            "  WARNINGS: 0\n"
            if count
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=stderr)

    return fake_run


def test_translate_tree_translates_a_copy_and_reports_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _repository(root)
    original = (root / "src" / "host.cpp").read_text(encoding="utf-8")
    output = tmp_path / "repo-hip"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(hipify.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        hipify.subprocess, "run", _fake_hipify_perl(calls, fail={"kernels/UPPER.CU"})
    )

    report = hipify.translate_tree(root, output, tool=TOOL)

    assert report.error is None
    assert [call["argv"][-1] for call in calls] == [
        "kernels/UPPER.CU",
        "kernels/a.cu",
        "src/host.cpp",
    ]
    first = calls[1]["argv"]
    assert first[:3] == ["perl", TOOL.script, "-print-stats"]
    assert first[3] == f"-o={output / 'kernels' / '.a.cu.hipify-tmp'}"
    assert first[4] == "--"
    assert calls[1]["cwd"] == root.resolve()
    assert (root / "src" / "host.cpp").read_text(encoding="utf-8") == original
    assert "hipMalloc" in (output / "kernels" / "a.cu").read_text(encoding="utf-8")
    assert (output / "setup.py").is_file() and (output / "src" / "cpu.cpp").is_file()
    assert not list(output.rglob("*.hipify-tmp"))
    assert report.failures == {"kernels/UPPER.CU": "exit code 2: error: boom"}
    assert [(s.path, s.changed) for s in report.files] == [
        ("kernels/a.cu", True),
        ("src/host.cpp", True),
    ]
    assert report.total("CONVERTED refs count") == 3
    assert report.leftover_identifiers() == {
        "ATen/cuda/HIPContext.h": 1,
        "at::cuda::getCurrentCUDAStream": 1,
    }
    data = report.to_dict()
    assert data["changed_files"] == ["kernels/a.cu", "src/host.cpp"]
    assert data["tool"]["commit"] == hipify.HIPIFY_COMMIT
    text = report.to_text()
    assert f"{hipify.HIPIFY_REF} @ {hipify.HIPIFY_COMMIT[:12]}" in text
    assert "without torch.utils.hipify" in text
    assert "at::cuda::getCurrentCUDAStream x1 (src/host.cpp:3)" in text
    assert "kernels/UPPER.CU: exit code 2" in text

    rerun = hipify.translate_tree(root, output, tool=TOOL)
    assert rerun.error is not None and "already exists" in rerun.error
    assert hipify.translate_tree(root, output, tool=TOOL, overwrite=True).error is None


def test_translate_tree_rejects_outputs_overlapping_the_repository(tmp_path: Path) -> None:
    _repository(tmp_path / "repo")

    inside = hipify.translate_tree(tmp_path / "repo", tmp_path / "repo" / "hip", tool=TOOL)
    around = hipify.translate_tree(tmp_path / "repo", tmp_path, tool=TOOL)
    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = hipify.translate_tree(empty, tmp_path / "empty-hip", tool=TOOL)

    assert inside.error is not None and "outside the repository" in inside.error
    assert around.error is not None and "outside the repository" in around.error
    assert nothing.error == "no CUDA sources found"
    assert not (tmp_path / "empty-hip").exists()


def test_ensure_hipify_clones_the_pinned_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    head = {"commit": hipify.HIPIFY_COMMIT}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "clone" in argv:
            staging = Path(argv[-1])
            (staging / "bin").mkdir(parents=True)
            (staging / "bin" / "hipify-perl").write_text("#!/usr/bin/perl\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="")
        return subprocess.CompletedProcess(argv, 0, stdout=head["commit"] + "\n")

    monkeypatch.setattr(hipify.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(hipify.subprocess, "run", fake_run)

    tool = hipify.ensure_hipify(tmp_path)

    clone = calls[0]
    assert clone[clone.index("--branch") + 1] == "rocm-7.2.4"
    assert "https://github.com/ROCm/HIPIFY" in clone
    assert tool.script == str(tmp_path / "HIPIFY-rocm-7.2.4" / "bin" / "hipify-perl")
    assert tool.commit == hipify.HIPIFY_COMMIT
    hipify.ensure_hipify(tmp_path)
    assert sum("clone" in argv for argv in calls) == 1

    head["commit"] = "0" * 40
    with pytest.raises(hipify.HipifyToolError, match="expected rocm-7.2.4"):
        hipify.ensure_hipify(tmp_path)


@pytest.mark.skipif(
    not os.environ.get("R2R_TEST_HIPIFY_PERL"),
    reason="set R2R_TEST_HIPIFY_PERL to a rocm-7.2.4 bin/hipify-perl to run the real tool",
)
def test_real_hipify_perl_translates_fixtures(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("vector_add.cu", "flash_attention_simplified.cu"):
        (root / name).write_bytes((FIXTURES / name).read_bytes())
    (root / "ext.cu").write_text(
        "#include <ATen/cuda/CUDAContext.h>\n"
        "__global__ void k(float* x) {}\n"
        "void f() { k<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(nullptr); }\n",
        encoding="utf-8",
    )
    tool = hipify.HipifyTool(
        repository=hipify.HIPIFY_REPOSITORY,
        ref=hipify.HIPIFY_REF,
        commit=hipify.HIPIFY_COMMIT,
        script=os.environ["R2R_TEST_HIPIFY_PERL"],
    )

    report = hipify.translate_tree(root, tmp_path / "repo-hip", tool=tool)

    assert report.error is None and not report.failures
    translated = (tmp_path / "repo-hip" / "vector_add.cu").read_text(encoding="utf-8")
    assert "hipMalloc" in translated and "cudaMalloc" not in translated
    assert report.identifiers("deprecated") == {"__shfl_down": 2, "__shfl": 1}
    assert report.leftover_identifiers() == {
        "ATen/cuda/HIPContext.h": 1,
        "at::cuda::getCurrentCUDAStream": 1,
    }
