from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo2reproduce.porting import scanner
from repo2reproduce.porting.scanner import scan_path, scan_source

FIXTURES = Path(scanner.__file__).parent / "fixtures"


def _hits(findings: list[scanner.Finding]) -> set[tuple[str, int]]:
    return {(finding.rule, finding.line) for finding in findings}


def _severity(findings: list[scanner.Finding], rule: str) -> set[str]:
    return {finding.severity for finding in findings if finding.rule == rule}


def test_rocmport_fixtures_only_need_review() -> None:
    report = scan_path(FIXTURES)

    assert report.files_scanned == 5
    assert report.counts() == {"blocker": 0, "high": 0, "review": 11, "info": 0}
    by_file: dict[str, set[tuple[str, int]]] = {}
    for finding in report.findings:
        by_file.setdefault(finding.path, set()).add((finding.rule, finding.line))
    assert by_file["reduction.cu"] == {("warp_sync_tail_32", 50), ("warp_sync_tail_32", 59)}
    assert by_file["flash_attention_simplified.cu"] == {
        ("shfl_offset_no_width", 49),
        ("shfl_broadcast_no_width", 50),
        ("shfl_offset_no_width", 59),
    }
    assert ("lane_math_32", 13) in by_file["vector_add.cu"]
    assert {rule for rule, _ in by_file["matrix_multiply.cu"]} == {"lane_math_32"}
    assert ("lane_math_32", 56) not in by_file["matrix_multiply.cu"]


def test_comments_and_plain_loops_are_ignored() -> None:
    source = (
        "// v += __shfl_xor_sync(0xffffffff, v, 16);\n"
        '/* asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane)); */\n'
        "__global__ void k(float* x) {\n"
        "  for (int i = 0; i < 32; ++i) x[i] += 1.f;\n"
        "}\n"
    )

    assert scan_source(source, "k.cu") == []


def test_warp_sync_masks_follow_hip_rules() -> None:
    source = (
        "#define FULL_MASK 0xffffffff\n"
        "__device__ float f(float v, unsigned long long m) {\n"
        "  v += __shfl_xor_sync(0xffffffff, v, 16);\n"
        "  v += __shfl_xor_sync(0xffffffffffffffffULL, v, 8);\n"
        "  v += __shfl_down_sync(m, v, 4);\n"
        "  int any = __any_sync(FULL_MASK, v > 0);\n"
        "  __syncwarp();\n"
        "  return v;\n"
        "}\n"
    )
    findings = scan_source(source, "k.cu")

    assert _hits(findings) == {
        ("sync_mask_32bit", 3),
        ("sync_mask", 4),
        ("sync_mask", 5),
        ("sync_mask_32bit", 6),
    }
    assert _severity(findings, "sync_mask_32bit") == {"high"}
    assert "__hip_check_mask" in findings[0].hint


def test_ballot_masks_must_stay_64_bit() -> None:
    source = (
        "__device__ int f(int p) {\n"
        "  unsigned int lanes = __ballot_sync(0xffffffffffffffffULL, p);\n"
        "  unsigned long long ok = __ballot(p);\n"
        "  int n = __popc(__ballot(p));\n"
        "  if (__activemask() == 0xffffffff) return n;\n"
        "  return lanes + (int)ok;\n"
        "}\n"
    )
    findings = scan_source(source, "k.cu")

    assert {line for rule, line in _hits(findings) if rule == "ballot_truncation"} == {2, 4, 5}


def test_cuda_only_preprocessor_branches_are_skipped() -> None:
    source = (
        "#if defined(__HIP_PLATFORM_AMD__)\n"
        "#define SHFL(v, o, w) __shfl_xor((v), (o), (w))\n"
        "#else\n"
        "#define SHFL(v, o, w) __shfl_xor_sync(0xFFFFFFFFu, (v), (o), (w))\n"
        "#endif\n"
        "#ifndef USE_ROCM\n"
        '__device__ unsigned lane() { unsigned l; asm volatile("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }\n'
        "#endif\n"
        "#if defined(__HIP_PLATFORM_AMD__) && FAST_PATH\n"
        "#else\n"
        '__device__ void g() { asm volatile("bar.sync 0;"); }\n'
        "#endif\n"
    )

    assert _hits(scan_source(source, "k.cu")) == {("inline_asm", 11)}


def test_inline_ptx_and_hopper_features_are_blockers() -> None:
    source = (
        "__device__ void mma(unsigned* d, const unsigned* a) {\n"
        '  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0}, {%1};"\n'
        '               : "=r"(d[0]) : "r"(a[0]));\n'
        '  asm volatile("wgmma.fence.sync.aligned;");\n'
        "}\n"
    )
    findings = scan_source(source, "k.cu")

    assert _hits(findings) == {("inline_asm", 2), ("inline_asm", 4), ("hopper_feature", 4)}
    assert "`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`" in findings[0].message


def test_shared_memory_is_checked_against_64_kib() -> None:
    source = (
        "#define TILE 128\n"
        "constexpr int kSmem = 96 * 1024;\n"
        "__global__ void k() {\n"
        "  __shared__ float big[TILE][TILE + 4];\n"
        "  __shared__ __half ok[64][64];\n"
        "}\n"
        "void launch(int bytes) {\n"
        "  cudaFuncSetAttribute(k<float, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);\n"
        "  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);\n"
        "}\n"
    )
    findings = scan_source(source, "k.cu")

    assert _hits(findings) == {
        ("lds_over_64k", 4),
        ("dynamic_lds_opt_in", 8),
        ("dynamic_lds_opt_in", 9),
    }
    big = next(finding for finding in findings if finding.rule == "lds_over_64k")
    assert "66.0 KiB" in big.message
    severities = {finding.line: finding.severity for finding in findings}
    assert severities[8] == "high" and severities[9] == "review"


def test_fp8_libraries_and_cutlass() -> None:
    source = (
        "#include <cuda_fp8.h>\n"
        "#include <cub/cub.cuh>\n"
        "#include <cutlass/gemm/device/gemm.h>\n"
        "__global__ void k(__nv_fp8_e4m3* x) { cub::WarpReduce<float> r; cute::Tensor t; }\n"
    )
    findings = scan_source(source, "k.cu")

    assert _hits(findings) == {
        ("fp8_format", 1),
        ("fp8_format", 4),
        ("cuda_library", 2),
        ("cutlass", 3),
    }
    assert "FNUZ E4M3 tops out at 240" in next(f for f in findings if f.rule == "fp8_format").hint


def test_setup_py_flags_and_cuda_requirements() -> None:
    source = (
        '"""Builds the kernels with -gencode flags."""\n'
        "from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME\n"
        "if CUDA_HOME is None:\n"
        '    raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available.")\n'
        'NVCC = ["-O3", "--use_fast_math", "-gencode", f"arch=compute_{a},code=sm_{a}"]\n'
        'DEBUG = ["-lineinfo"]\n'
    )
    findings = scan_source(source, "setup.py")

    assert _hits(findings) == {("cuda_required", 4), ("nvcc_flags", 5), ("nvcc_flags", 6)}
    severities = {finding.line: finding.severity for finding in findings}
    assert severities == {4: "high", 5: "high", 6: "info"}

    guarded = scan_source(
        "import torch\n"
        "if torch.version.hip is None:\n"
        '    NVCC = ["-gencode", "arch=compute_80,code=sm_80"]\n',
        "setup.py",
    )
    assert [(f.rule, f.severity) for f in guarded] == [("nvcc_flags", "review")]
    assert "line 2" in guarded[0].message


def test_cmake_cuda_language_and_flags() -> None:
    source = (
        "# set(CMAKE_CUDA_FLAGS --expt-extended-lambda)\n"
        "project(demo LANGUAGES CXX CUDA)\n"
        'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")\n'
        'message(FATAL_ERROR "CUDA toolkit not found")\n'
    )
    findings = scan_source(source, "CMakeLists.txt")

    assert _hits(findings) == {
        ("cmake_cuda_language", 2),
        ("nvcc_flags", 3),
        ("cuda_required", 4),
    }


def test_python_dispatch_packages_and_fp8() -> None:
    source = (
        "import torch\n"
        "import flash_attn\n"
        "import pycuda.autoinit\n"
        "major, minor = torch.cuda.get_device_capability()\n"
        "x = torch.empty(4).to(torch.float8_e4m3fn)\n"
        "y = torch.empty(4).to(torch.float8_e4m3fnuz)\n"
    )
    findings = scan_source(source, "model.py")

    assert _hits(findings) == {
        ("python_cuda_package", 2),
        ("python_cuda_package", 3),
        ("capability_dispatch", 4),
        ("fp8_format", 5),
    }
    severities = {finding.line: finding.severity for finding in findings}
    assert severities[2] == "info" and severities[3] == "high"


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_repository_scan_finds_eager_extension_imports_and_triton(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "setup.py",
        "from torch.utils.cpp_extension import CUDAExtension\n"
        'ext = CUDAExtension(\n    name="pkg._fused", sources=["csrc/fused.cu"])\n',
    )
    _write(tmp_path, "csrc/fused.cu", "__global__ void k(float* x) { x[0] = 1.f; }\n")
    _write(tmp_path, "pkg/__init__.py", "from .core import run\n")
    _write(
        tmp_path,
        "pkg/core.py",
        "try:\n    from . import _kernels\nexcept ImportError:\n    _kernels = None\n"
        "from .quant import quantize\n",
    )
    _write(tmp_path, "pkg/quant.py", "from . import _fused\n")
    _write(tmp_path, "pkg/util.py", "from ._C import ops\n")
    _write(tmp_path, "pkg/_version.py", "__version__ = '1'\n")
    _write(tmp_path, "pkg/meta.py", "from ._version import __version__\n")
    _write(
        tmp_path,
        "pkg/triton_ops.py",
        "import triton\n\n@triton.jit\ndef a(x):\n    pass\n\n@triton.jit\ndef b(x):\n    pass\n",
    )

    report = scan_path(tmp_path)
    eager = {
        (finding.path, finding.line, finding.severity)
        for finding in report.findings
        if finding.rule == "eager_extension_import"
    }
    assert eager == {("pkg/quant.py", 1, "high"), ("pkg/util.py", 1, "review")}
    triton = [finding for finding in report.findings if finding.rule == "triton_kernels"]
    assert [(f.path, f.line, f.message) for f in triton] == [
        ("pkg/triton_ops.py", 3, "2 Triton kernel(s).")
    ]


def test_generated_hip_translations_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "csrc/k.cu", "__global__ void k(float* x) { x[0] = 1.f; }\n")
    _write(
        tmp_path,
        "csrc/k.hip",
        "__global__ void k(float v) { v += __shfl_xor_sync(0xffffffff, v, 1); }\n",
    )

    report = scan_path(tmp_path)

    assert report.files_scanned == 1
    assert report.findings == []


def test_cli_prints_json_and_filters_by_severity(capsys: pytest.CaptureFixture[str]) -> None:
    assert scanner.main([str(FIXTURES), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"]["review"] == 11
    assert {"path", "line", "rule", "severity", "message", "hint"} <= set(data["findings"][0])

    assert scanner.main([str(FIXTURES), "--min-severity", "high"]) == 0
    assert "0 findings" in capsys.readouterr().out
