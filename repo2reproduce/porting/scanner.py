"""Report-only scanner for CUDA code that needs attention when porting to ROCm.

The rules target AMD Instinct gfx942 (MI300X/MI325X) under ROCm 7.2: 64-lane wavefronts,
64 KiB of LDS per workgroup, FNUZ FP8, and HIP's stricter warp-sync masks. Each finding says
what to check and how to confirm it. The scanner never rewrites code, because the same
pattern is correct in one kernel and wrong in another; ports are confirmed with a parity
test, not by this report.

Adapted from ROCmPort-AI ``backend/tools/static_analyzer.py``
(https://github.com/tazwaryayyyy/ROCmPort-AI, commit 943d16c), Copyright 2026 Tazwar Ahnaf
Enan, licensed under the Apache License 2.0 (see ``LICENSE-ROCmPort-AI`` in this package).
Modified for Repo2Reproduce: rules, severities and hints rewritten for ROCm 7.2 semantics,
comment-aware matching, repository-wide scanning with file and line locations, and new
build-script, Python-import, FP8 and shared-memory checks.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SEVERITIES = ("blocker", "high", "review", "info")
LDS_BYTES_PER_WORKGROUP = 64 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024

CUDA_SOURCE_SUFFIXES = frozenset(
    {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}
)
BUILD_FILE_NAMES = frozenset(
    {"setup.py", "build.py", "CMakeLists.txt", "Makefile", "makefile", "meson.build"}
)
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
)

_CATEGORIES = {
    "inline_asm": "inline_asm",
    "hopper_feature": "nvidia_only",
    "wmma_api": "nvidia_only",
    "cutlass": "library",
    "cuda_library": "library",
    "python_cuda_package": "library",
    "sync_mask_32bit": "warp_width",
    "sync_mask": "warp_width",
    "ballot_truncation": "warp_width",
    "shfl_broadcast_no_width": "warp_width",
    "shfl_offset_no_width": "warp_width",
    "warp_size_32": "warp_width",
    "lane_math_32": "warp_width",
    "warp_sync_tail_32": "warp_width",
    "mixed_warp_width": "warp_width",
    "lds_over_64k": "shared_memory",
    "dynamic_lds_opt_in": "shared_memory",
    "fp8_format": "numerics",
    "nvcc_flags": "build",
    "cmake_cuda_language": "build",
    "cuda_required": "build",
    "eager_extension_import": "python_import",
    "capability_dispatch": "dispatch",
    "triton_kernels": "portable_path",
}

_HINTS = {
    "inline_asm": (
        "Replace the PTX with a HIP intrinsic (for example `%laneid` -> `__lane_id()`) or an "
        "AMD builtin. Tensor-core and async-copy PTX (`mma`, `ldmatrix`, `cp.async`, `wgmma`) "
        "needs a rewrite in Triton, rocWMMA or Composable Kernel. Prefer a portable path in "
        "the repository if one exists, and confirm any rewrite with a parity test."
    ),
    "hopper_feature": (
        "gfx942 has no equivalent of Hopper-only features (wgmma, TMA, thread-block "
        "clusters). Use a portable path in the repository or redesign the kernel in Triton "
        "or Composable Kernel, and report the substitution."
    ),
    "wmma_api": (
        "hipify does not convert `nvcuda::wmma`. rocWMMA offers a similar fragment API "
        "(`rocwmma::fragment`, `load_matrix_sync`, `mma_sync`) with its own supported shapes "
        "on gfx942; check the fragment sizes and data types the kernel uses."
    ),
    "cutlass": (
        "CUTLASS/CuTe has no ROCm port. Use the repository's non-CUTLASS path if there is "
        "one; otherwise reimplement the needed GEMM or attention with hipBLASLt, Composable "
        "Kernel or Triton."
    ),
    "cuda_library": (
        "hipify maps the API to the ROCm library named in the finding, but coverage and "
        "performance are not identical; check every call the selected experiment depends on."
    ),
    "python_cuda_package": (
        "The package ships CUDA code. Use the ROCm build named in the finding when there is "
        "one; otherwise find the call sites the selected experiment needs and replace or "
        "avoid them."
    ),
    "sync_mask_32bit": (
        "HIP's `*_sync` warp functions accept only a 64-bit mask (a compile-time "
        "static_assert), and at run time the mask must equal exactly the active lanes "
        "(`__hip_check_mask`). Widening to `0xffffffffffffffffULL` is only safe when all 64 "
        "lanes of every wavefront are active; with blocks such as dim3(32, odd) it fails at "
        "run time. For code that works in 32-lane groups, use the non-sync form with an "
        "explicit width (`__shfl_xor(v, offset, 32)`) or pass `__activemask()`, then run a "
        "parity test."
    ),
    "sync_mask": (
        "Make sure the mask is a 64-bit value equal to the lanes that are actually active; "
        "HIP checks this at run time. A full 64-bit mask fails when a block's thread count "
        "leaves a wavefront partly empty."
    ),
    "ballot_truncation": (
        "`__ballot`, `__ballot_sync` and `__activemask` return 64-bit masks on gfx942. Keep "
        "them in `unsigned long long` or `uint64_t`, use `__popcll`, `__ffsll` and "
        "`__clzll`, and compare against the mask of active lanes instead of `0xffffffff`."
    ),
    "shfl_broadcast_no_width": (
        "Without a width, `__shfl(v, src)` reads lane `src` of the whole 64-lane wavefront, "
        "so threads 32-63 receive the value from the first 32-lane group. If the kernel "
        "works in 32-lane groups and blocks have more than 32 threads, pass width=32."
    ),
    "shfl_offset_no_width": (
        "Without a width the shuffle spans all 64 lanes. Reductions with offsets up to 16 "
        "and lane = threadIdx.x % 32 still stay inside each 32-lane half; check that results "
        "are only read from lanes whose sources belong to the same group, or pass width=32 to "
        "make the grouping explicit."
    ),
    "warp_size_32": (
        'Usually fine when every use means "a 32-lane group". It is wrong where it stands '
        "for the hardware warp: ballot or active-mask widths, `__syncwarp`, comparisons with "
        "`warpSize`, or full-wavefront shuffles. Don't change it to 64 without changing the "
        "code that depends on it; confirm with a parity test."
    ),
    "lane_math_32": (
        "Consistent 32-lane grouping still works on 64-lane wavefronts (each wavefront holds "
        "two groups) as long as shuffles use width 32 or offsets below 32. It breaks when "
        "mixed with full-wavefront operations, so don't switch to `% 64` in isolation."
    ),
    "warp_sync_tail_32": (
        "The classic warp-synchronous tail (volatile shared memory, +32 down to +1) stays "
        "correct on 64-lane wavefronts because the 32 active threads share one wavefront, so "
        "it does not need a `< 64` stage. Check that the shared memory is `volatile` or that "
        "the steps are separated by `__syncwarp()`."
    ),
    "mixed_warp_width": (
        "`warpSize` is 64 on gfx942, while the 32-lane arithmetic in this file assumes 32. "
        "Check every place the two meet: array sizes, lane ids, shuffle widths and masks."
    ),
    "lds_over_64k": (
        "Reduce the tile size or split the buffer. NVIDIA GPUs allow up to 227 KiB per block "
        "(H100), so tiles tuned for them often do not fit in gfx942's 64 KiB."
    ),
    "dynamic_lds_opt_in": (
        "gfx942 allows at most 64 KiB of LDS per workgroup, so launches asking for more fail. "
        "hipify turns this into `hipFuncSetAttribute(const void* func, ...)`, and templated "
        "kernels need an explicit `(const void*)` cast."
    ),
    "fp8_format": (
        "gfx942's FP8 instructions use the FNUZ formats (`__hip_fp8_e4m3_fnuz`, "
        "`torch.float8_e4m3fnuz`), and hipify maps `__nv_fp8_e4m3` to them. FNUZ E4M3 tops "
        "out at 240 instead of 448 and encodes NaN differently, so scales computed for OCP "
        "FP8 can overflow; compare the numerics against the reference."
    ),
    "nvcc_flags": (
        "Pass these flags only when building for CUDA, for example under "
        "`if torch.version.hip is None:` or `if not IS_HIP_EXTENSION:`."
    ),
    "cmake_cuda_language": (
        "On ROCm the project needs the HIP language (`enable_language(HIP)` or "
        "`LANGUAGES HIP`), with the `.cu` sources set to `LANGUAGE HIP` and "
        "`CMAKE_HIP_ARCHITECTURES=gfx942`, or hipified sources."
    ),
    "cuda_required": (
        "On ROCm, PyTorch's `CUDAExtension` builds with hipcc and uses ROCM_HOME, so a hard "
        "CUDA check stops a build that could work. Relax the check when `torch.version.hip` "
        "is set, or skip the extension and use a portable path."
    ),
    "eager_extension_import": (
        "Guard the import with `try`/`except ImportError` or import the portable modules "
        "directly, and record the change as a patch."
    ),
    "capability_dispatch": (
        "On gfx942 `torch.cuda.get_device_capability()` returns (9, 4), so checks written "
        "for NVIDIA compute capabilities treat it like Hopper (sm_90). Check which path this "
        "selects on ROCm, and branch on `torch.version.hip` first."
    ),
    "triton_kernels": (
        "Triton compiles for gfx942, so this may be a portable path for the selected "
        "experiment; check its numerics against the reference."
    ),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    category: str
    severity: str
    message: str
    hint: str
    snippet: str = ""


@dataclass
class ScanReport:
    root: str
    files_scanned: int
    findings: list[Finding] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts = {severity: 0 for severity in SEVERITIES}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def filtered(self, min_severity: str) -> ScanReport:
        limit = SEVERITIES.index(min_severity)
        kept = [f for f in self.findings if SEVERITIES.index(f.severity) <= limit]
        return ScanReport(root=self.root, files_scanned=self.files_scanned, findings=kept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "counts": self.counts(),
            "findings": [asdict(finding) for finding in self.findings],
        }

    def to_text(self, max_findings: int = 200) -> str:
        counts = self.counts()
        summary = ", ".join(f"{severity}: {counts[severity]}" for severity in SEVERITIES)
        lines = [
            (
                f"CUDA-to-ROCm scan of {self.root}: {self.files_scanned} files, "
                f"{len(self.findings)} findings ({summary})"
            )
        ]
        if not self.findings:
            lines.append("No CUDA-specific hazards found.")
            return "\n".join(lines)
        lines.append("")
        for finding in self.findings[:max_findings]:
            lines.append(
                f"[{finding.severity}] {finding.rule} {finding.path}:{finding.line}: "
                f"{finding.message}"
            )
            if finding.snippet:
                lines.append(f"    {finding.snippet}")
        if len(self.findings) > max_findings:
            lines.append(f"... {len(self.findings) - max_findings} more findings not shown")
        lines += ["", "What to check, by rule:"]
        for rule in dict.fromkeys(finding.rule for finding in self.findings):
            lines.append(f"- {rule}: {_HINTS[rule]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Source helpers
# --------------------------------------------------------------------------- #
_C_TOKEN = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


def _strip_c_comments(text: str) -> str:
    """Blank out C/C++ comments while keeping string literals and line numbers."""

    def blank(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("/"):
            return re.sub(r"[^\n]", " ", token)
        return token

    return _C_TOKEN.sub(blank, text)


_PP_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_HIP_MACRO = (
    r"(?:__HIP_PLATFORM_AMD__|__HIP_PLATFORM_HCC__|__HIPCC__|__HIP__|USE_ROCM"
    r"|__HIP_DEVICE_COMPILE__)"
)
_HIP_NEGATED = re.compile(r"!\s*(?:defined\s*\(?\s*)?" + _HIP_MACRO + r"\b")
_HIP_TESTED = re.compile(r"\b" + _HIP_MACRO + r"\b")


def _classify_condition(keyword: str, condition: str) -> tuple[bool, bool]:
    """Return (branch is CUDA-only, #else is CUDA-only) for a HIP-detecting conditional."""

    if not _HIP_TESTED.search(condition):
        return False, False
    if keyword == "ifdef":
        return False, True
    if keyword == "ifndef":
        return True, False
    if _HIP_NEGATED.search(condition):
        return (False, False) if "||" in condition else (True, False)
    return (False, False) if "&&" in condition else (False, True)


def _mask_cuda_only_branches(code: str) -> str:
    """Blank preprocessor branches that only compile when the code is not built for HIP."""

    lines = code.split("\n")
    # Each frame records whether the current branch is CUDA-only and what the #else means.
    stack: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        directive = _PP_DIRECTIVE.match(line)
        if directive:
            keyword, condition = directive.groups()
            if keyword in {"if", "ifdef", "ifndef"}:
                cuda_only, else_cuda_only = _classify_condition(keyword, condition)
                stack.append({"cuda_only": cuda_only, "else_cuda_only": else_cuda_only})
            elif keyword == "elif" and stack:
                # After a positive HIP test, ROCm never reaches an #elif; otherwise it may.
                stack[-1]["cuda_only"] = stack[-1]["else_cuda_only"]
            elif keyword == "else" and stack:
                stack[-1]["cuda_only"] = stack[-1]["else_cuda_only"]
            elif keyword == "endif" and stack:
                stack.pop()
            continue
        if any(frame["cuda_only"] for frame in stack):
            lines[index] = " " * len(line)
    return "\n".join(lines)


class _Lines:
    def __init__(self, text: str) -> None:
        self.text = text
        self._starts = [0] + [match.end() for match in re.finditer("\n", text)]

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self._starts, offset)

    def offset_of(self, line: int) -> int:
        return self._starts[line - 1]

    def snippet(self, line: int) -> str:
        start = self._starts[line - 1]
        end = self.text.find("\n", start)
        content = self.text[start : end if end != -1 else len(self.text)].strip()
        return content if len(content) <= 160 else content[:157] + "..."


class _Collector:
    def __init__(self, path: str, lines: _Lines) -> None:
        self.path = path
        self.lines = lines
        self.findings: list[Finding] = []
        self._seen: set[tuple[str, int | str]] = set()

    def add(
        self, offset: int, rule: str, severity: str, message: str, *, once_per_file: bool = False
    ) -> None:
        line = self.lines.line_of(offset)
        key = (rule, message) if once_per_file else (rule, line)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(
            Finding(
                path=self.path,
                line=line,
                rule=rule,
                category=_CATEGORIES[rule],
                severity=severity,
                message=message,
                hint=_HINTS[rule],
                snippet=self.lines.snippet(line),
            )
        )

    def add_line(self, line: int, rule: str, severity: str, message: str) -> None:
        self.add(self.lines.offset_of(line), rule, severity, message)

    def has(self, *rules: str) -> bool:
        return any(finding.rule in rules for finding in self.findings)


def _call_args(text: str, open_index: int, limit: int = 4000) -> list[str] | None:
    """Split the argument list of the call whose ``(`` is at ``open_index``."""

    depth = 0
    args: list[str] = []
    start = open_index + 1
    index = open_index
    end = min(len(text), open_index + limit)
    while index < end:
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                last = text[start:index].strip()
                if last or args:
                    args.append(last)
                return args
        elif char == "," and depth == 1:
            args.append(text[start:index].strip())
            start = index + 1
        elif char in "\"'":
            close = text.find(char, index + 1)
            if close == -1:
                return None
            index = close
        index += 1
    return None


_TYPE_BYTES = {
    "char": 1,
    "signed char": 1,
    "unsigned char": 1,
    "bool": 1,
    "int8_t": 1,
    "uint8_t": 1,
    "short": 2,
    "unsigned short": 2,
    "int16_t": 2,
    "uint16_t": 2,
    "half": 2,
    "__half": 2,
    "__nv_bfloat16": 2,
    "nv_bfloat16": 2,
    "__hip_bfloat16": 2,
    "at::Half": 2,
    "at::BFloat16": 2,
    "int": 4,
    "unsigned": 4,
    "unsigned int": 4,
    "float": 4,
    "int32_t": 4,
    "uint32_t": 4,
    "half2": 4,
    "__half2": 4,
    "__nv_bfloat162": 4,
    "char4": 4,
    "uchar4": 4,
    "float3": 12,
    "int3": 12,
    "double": 8,
    "long": 8,
    "unsigned long": 8,
    "long long": 8,
    "unsigned long long": 8,
    "int64_t": 8,
    "uint64_t": 8,
    "size_t": 8,
    "float2": 8,
    "int2": 8,
    "uint2": 8,
    "float4": 16,
    "int4": 16,
    "uint4": 16,
    "double2": 16,
}
_QUALIFIERS = re.compile(r"\b(?:const|volatile|static|__restrict__|signed(?=\s+(?:int|long)))\s*")
_SIZEOF = re.compile(r"\bsizeof\s*\(\s*([A-Za-z_][\w:]*(?:\s+[A-Za-z_]\w*)*)\s*\)")
_INT_SUFFIX = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)(?:[uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)\b")
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_CONST_INT = re.compile(
    r"\b(?:static\s+)?(?:constexpr|const)\s+(?:static\s+)?"
    r"(?:unsigned\s+int|unsigned|int|long|size_t|int32_t|uint32_t|int64_t|uint64_t|auto)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([^;{}]+);"
)


def _normalize_type(name: str) -> str:
    return " ".join(_QUALIFIERS.sub("", name).split())


def _eval_int(expr: str, names: dict[str, int]) -> int | None:
    expr = _SIZEOF.sub(
        lambda m: str(_TYPE_BYTES.get(_normalize_type(m.group(1)), "__unknown__")), expr
    )
    expr = _INT_SUFFIX.sub(r"\1", expr).strip()
    if not expr:
        return None
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None
    return _eval_node(node, names)


def _eval_node(node: ast.AST, names: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_node(node.operand, names)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _eval_node(node.left, names), _eval_node(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)) and right:
            return left // right
        if isinstance(node.op, ast.LShift) and 0 <= right < 64:
            return left << right
        if isinstance(node.op, ast.RShift) and 0 <= right < 64:
            return left >> right
    return None


def _macro_texts(code: str) -> dict[str, str]:
    texts = {name: value for name, value in _DEFINE.findall(code)}
    texts.update({name: value.strip() for name, value in _CONST_INT.findall(code)})
    return texts


def _macro_ints(texts: dict[str, str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for name, text in texts.items():
        value = _eval_int(text, values)
        if value is not None:
            values[name] = value
    return values


# --------------------------------------------------------------------------- #
# CUDA / C++ rules
# --------------------------------------------------------------------------- #
_ASM = re.compile(r"\b(?:asm|__asm__)\s*(?:volatile\s*|__volatile__\s*)?\(")
_ASM_MNEMONIC = re.compile(r"\"\s*([a-z][a-z0-9_.]*)")
_HOPPER = re.compile(
    r"\bwgmma\b|\bcp\.async\.bulk\b|\bcuTensorMapEncode\w*|\bCUtensorMap\b|__cluster_dims__"
    r"|\bthis_cluster\s*\(|\bsetmaxnreg\b"
)
_WMMA = re.compile(
    r"\bnvcuda::wmma\b|\bwmma::(?:fragment|load_matrix_sync|store_matrix_sync|mma_sync"
    r"|fill_fragment)\b|#\s*include\s*<mma\.h>"
)
_CUTLASS = re.compile(r"#\s*include\s*[<\"](?:cutlass|cute)/|\b(?:cutlass|cute)::")
_LIBRARIES = (
    ("CUB", re.compile(r"#\s*include\s*[<\"]cub/|\bcub::"), "hipCUB (`hipcub::`)"),
    ("Thrust", re.compile(r"#\s*include\s*[<\"]thrust/|\bthrust::"), "rocThrust"),
    ("cuBLASLt", re.compile(r"\bcublasLt\w*"), "hipBLASLt"),
    ("cuBLAS", re.compile(r"\bcublas(?!Lt)\w*"), "hipBLAS / rocBLAS"),
    ("cuDNN", re.compile(r"\bcudnn\w*"), "MIOpen"),
    ("NCCL", re.compile(r"\bnccl[A-Z]\w*|#\s*include\s*[<\"]nccl"), "RCCL"),
    ("cuRAND", re.compile(r"\bcurand\w*"), "hipRAND / rocRAND"),
    ("cuSPARSE", re.compile(r"\bcusparse\w*"), "hipSPARSE"),
    ("cuFFT", re.compile(r"\bcufft\w*"), "hipFFT"),
    ("cuSOLVER", re.compile(r"\bcusolver\w*"), "hipSOLVER"),
    ("NVTX", re.compile(r"\bnvtx[A-Z]\w*|#\s*include\s*[<\"]nvtx"), "roctx"),
)
_SYNC_CALL = re.compile(
    r"\b(__(?:shfl(?:_up|_down|_xor)?|ballot|any|all|match_any|match_all"
    r"|reduce_(?:add|min|max|and|or|xor))_sync|__syncwarp)\s*\("
)
_HEX_MASK = re.compile(r"^0[xX]([0-9a-fA-F]+)([uUlL]*)$")
_BALLOT = r"__(?:ballot(?:_sync)?|activemask)"
_BALLOT_32BIT_DECL = re.compile(
    r"\b(?:unsigned\s+int|unsigned|uint32_t|int32_t|int|uint)\s+[A-Za-z_]\w*\s*=\s*"
    r"(?:\(\s*\w+\s*\)\s*)?" + _BALLOT + r"\s*\("
)
_BALLOT_32BIT_OP = re.compile(r"\b__(?:popc|ffs|clz|brev)\s*\(\s*" + _BALLOT + r"\b")
_BALLOT_32BIT_CMP = re.compile(_BALLOT + r"\s*\([^;{}]*\)\s*[!=]=\s*0[xX][fF]{8}[uU]?\b")
_SHFL = re.compile(r"\b(__shfl(?:_up|_down|_xor)?)\s*\(")
_WARP_SIZE_32 = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+\w*warp_?size\w*[ \t]+\(?[ \t]*32[ \t]*\)?[ \t]*$"
    r"|\b(?:constexpr|const)\s+(?:static\s+)?(?:int|unsigned(?:\s+int)?|size_t|int32_t"
    r"|uint32_t|auto)\s+\w*warp_?size\w*\s*=\s*32\b",
    re.IGNORECASE | re.MULTILINE,
)
_THREAD = r"(?:threadIdx\.[xyz]|\btid\b|\bthread_?[iI]d\w*|\blocal_?[iI]d\w*)"
_LANE_MATH_32 = re.compile(
    _THREAD + r"\s*%\s*32\b|" + _THREAD + r"\s*/\s*32\b|"
    r"(?:" + _THREAD + r"|\blane\w*)\s*&\s*(?:31|0[xX]1[fF])\b|" + _THREAD + r"\s*>>\s*5\b"
)
_TAIL_IF_32 = re.compile(r"\bif\s*\(\s*(?:threadIdx\.x|tid|lane\w*|thread_?[iI]d\w*)\s*<\s*32\s*\)")
_LOOP_DOWN_TO_32 = re.compile(r"\bfor\s*\([^;)]*;\s*[A-Za-z_]\w*\s*>\s*32\s*;")
_WARPSIZE_TOKEN = re.compile(r"\bwarpSize\b")
_SHARED_ARRAY = re.compile(
    r"__shared__\s+(?:(?:__align__|alignas)\s*\(\s*\d+\s*\)\s*)?"
    r"(?P<type>(?:(?:const|volatile|unsigned|signed)\s+)*[A-Za-z_][\w:]*"
    r"(?:\s+(?:int|long|char|short))*)\s+"
    r"(?:(?:__align__|alignas)\s*\(\s*\d+\s*\)\s*)?"
    r"(?P<name>[A-Za-z_]\w*)\s*(?P<dims>(?:\[[^\]\n]*\]\s*)+)[;=]"
)
_FUNC_SET_ATTRIBUTE = re.compile(r"\b(?:cuda|hip)FuncSetAttribute\s*\(")
_FP8_CUDA = re.compile(r"\b__nv_fp8(?:x\d)?_e(?:4m3|5m2)\b|#\s*include\s*[<\"]cuda_fp8\.h[>\"]")


def _is_32bit_mask(mask: str) -> bool:
    mask = mask.strip()
    match = _HEX_MASK.match(mask)
    if match:
        digits, suffix = match.groups()
        return len(digits.lstrip("0") or "0") <= 8 and "ll" not in suffix.lower()
    return mask in {"~0", "~0u", "~0U", "-1", "(unsigned)-1", "(unsigned int)-1"}


def _scan_c_source(text: str, path: str) -> list[Finding]:
    code = _mask_cuda_only_branches(_strip_c_comments(text))
    out = _Collector(path, _Lines(text))
    macro_texts = _macro_texts(code)
    macros = _macro_ints(macro_texts)

    for match in _ASM.finditer(code):
        mnemonic = _ASM_MNEMONIC.search(code, match.end(), match.end() + 300)
        detail = f" (`{mnemonic.group(1)[:64]}`)" if mnemonic else ""
        out.add(
            match.start(),
            "inline_asm",
            "blocker",
            f"Inline assembly{detail}; in CUDA sources this is NVIDIA PTX, which hipcc "
            "cannot compile and hipify does not translate.",
        )
    for match in _HOPPER.finditer(code):
        out.add(
            match.start(), "hopper_feature", "blocker", f"Hopper-only feature `{match.group(0)}`."
        )
    for match in _WMMA.finditer(code):
        out.add(
            match.start(), "wmma_api", "high", "NVIDIA WMMA tensor-core API.", once_per_file=True
        )
    for match in _CUTLASS.finditer(code):
        out.add(match.start(), "cutlass", "blocker", "CUTLASS/CuTe dependency.", once_per_file=True)
    for library, pattern, target in _LIBRARIES:
        match = pattern.search(code)
        if match:
            out.add(
                match.start(),
                "cuda_library",
                "info",
                f"Uses {library}; on ROCm this maps to {target}.",
                once_per_file=True,
            )

    for match in _SYNC_CALL.finditer(code):
        args = _call_args(code, match.end() - 1)
        if not args:
            continue
        name, mask = match.group(1), args[0]
        resolved = macro_texts.get(mask, mask)
        if _is_32bit_mask(resolved):
            out.add(
                match.start(),
                "sync_mask_32bit",
                "high",
                f"`{name}` with the 32-bit mask `{mask}`; HIP requires a 64-bit mask equal "
                "to the active lanes.",
            )
        else:
            out.add(match.start(), "sync_mask", "review", f"`{name}` with mask `{mask}`.")

    for pattern, detail in (
        (_BALLOT_32BIT_DECL, "stored in a 32-bit variable"),
        (_BALLOT_32BIT_OP, "passed to a 32-bit bit operation"),
        (_BALLOT_32BIT_CMP, "compared with a 32-bit full mask"),
    ):
        for match in pattern.finditer(code):
            out.add(
                match.start(),
                "ballot_truncation",
                "high",
                f"64-bit ballot or active-lane mask {detail}.",
            )

    for match in _SHFL.finditer(code):
        args = _call_args(code, match.end() - 1)
        if args is None or len(args) != 2:
            continue
        name = match.group(1)
        if name == "__shfl":
            out.add(
                match.start(),
                "shfl_broadcast_no_width",
                "review",
                f"`__shfl({args[0]}, {args[1]})` broadcast without a width argument.",
            )
        else:
            out.add(
                match.start(),
                "shfl_offset_no_width",
                "review",
                f"`{name}({args[0]}, {args[1]})` without a width argument.",
            )

    for match in _WARP_SIZE_32.finditer(code):
        out.add(
            match.start(),
            "warp_size_32",
            "review",
            "Warp size hard-coded as 32; the gfx942 wavefront has 64 lanes.",
        )
    for match in _LANE_MATH_32.finditer(code):
        out.add(match.start(), "lane_math_32", "review", "Lane or warp index computed with 32.")
    for match in _TAIL_IF_32.finditer(code):
        out.add(
            match.start(),
            "warp_sync_tail_32",
            "review",
            "Warp-synchronous reduction tail guarded by `< 32`.",
        )
    for match in _LOOP_DOWN_TO_32.finditer(code):
        out.add(
            match.start(),
            "warp_sync_tail_32",
            "review",
            "Shared-memory reduction loop that stops at 32 before a warp-synchronous tail.",
        )
    warp_size = _WARPSIZE_TOKEN.search(code)
    if warp_size and out.has("lane_math_32", "warp_size_32"):
        out.add(
            warp_size.start(),
            "mixed_warp_width",
            "review",
            "File uses both `warpSize` (64 on gfx942) and 32-lane arithmetic.",
        )

    for match in _SHARED_ARRAY.finditer(code):
        element = _TYPE_BYTES.get(_normalize_type(match.group("type")))
        dims = [_eval_int(dim, macros) for dim in re.findall(r"\[([^\]]*)\]", match.group("dims"))]
        if element is None or not dims or any(dim is None for dim in dims):
            continue
        size = element
        for dim in dims:
            size *= dim  # type: ignore[operator]
        if size > LDS_BYTES_PER_WORKGROUP:
            out.add(
                match.start(),
                "lds_over_64k",
                "high",
                f"Static shared memory `{match.group('name')}` is {size / 1024:.1f} KiB, above "
                "the 64 KiB of LDS per workgroup on gfx942.",
            )
    for match in _FUNC_SET_ATTRIBUTE.finditer(code):
        args = _call_args(code, match.end() - 1)
        if not args or not any("MaxDynamicSharedMemorySize" in arg for arg in args):
            continue
        requested = _eval_int(args[-1], macros)
        if requested is not None and requested > LDS_BYTES_PER_WORKGROUP:
            out.add(
                match.start(),
                "dynamic_lds_opt_in",
                "high",
                f"Requests {requested} bytes ({requested / 1024:.0f} KiB) of dynamic shared "
                "memory; gfx942 allows 64 KiB per workgroup.",
            )
        else:
            out.add(
                match.start(),
                "dynamic_lds_opt_in",
                "review",
                "Opt-in to large dynamic shared memory.",
            )
    for match in _FP8_CUDA.finditer(code):
        out.add(match.start(), "fp8_format", "review", f"OCP FP8 type `{match.group(0)}`.")
    return out.findings


# --------------------------------------------------------------------------- #
# Build-file rules
# --------------------------------------------------------------------------- #
# hipcc (ROCm 7.2) fails with "unknown argument" on these.
_NVCC_REJECTED = re.compile(
    r"(?<![\w-])(?:-gencode|--generate-code|-arch=sm_\w+|--gpu-architecture"
    r"|arch=compute_(?:\d+|\{)|code=sm_(?:\d+|\{)|--ptxas-options|-Xptxas|--use_fast_math"
    r"|--generate-line-info|--expt-relaxed-constexpr|--expt-extended-lambda|-maxrregcount"
    r"|--maxrregcount|--resource-usage)"
)
# hipcc accepts these with an "unused argument" warning and they have no effect.
_NVCC_IGNORED = re.compile(r"(?<![\w-])(?:-lineinfo|-use_fast_math)\b")
_ROCM_BRANCH = re.compile(
    r"torch\.version\.hip|getattr\(\s*torch\.version\s*,\s*[\"']hip[\"']|IS_HIP_EXTENSION"
    r"|ROCM_HOME|USE_ROCM|__HIP_PLATFORM_AMD__|\bis_rocm\b|\bis_hip\b"
)
_CMAKE_CUDA = re.compile(
    r"\benable_language\s*\(\s*CUDA\b|\bLANGUAGES\b[^)\n]*\bCUDA\b|\bCMAKE_CUDA_ARCHITECTURES\b"
    r"|\bfind_package\s*\(\s*CUDA(?:Toolkit)?\b"
)
_CMAKE_FATAL_CUDA = re.compile(r"message\s*\(\s*FATAL_ERROR[^)]*\bCUDA", re.IGNORECASE)
_CUDA_REQUIREMENT = re.compile(
    r"\bCUDA_HOME\b|\bnvcc\b|\bcuda\b[^\n]{0,80}?\b(?:required|must|not found|not available"
    r"|unavailable|needed)\b|\brequires? cuda\b",
    re.IGNORECASE,
)


def _code_lines(text: str) -> list[str]:
    return ["" if line.lstrip().startswith("#") else line for line in text.splitlines()]


def _python_code_lines(text: str, tree: ast.Module | None) -> list[str]:
    """Source lines with comment lines and docstrings blanked."""

    lines = _code_lines(text)
    if tree is None:
        return lines
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            for number in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                lines[number - 1] = ""
    return lines


def _parse_python(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _scan_build(
    text: str, path: str, *, python: bool, tree: ast.Module | None = None
) -> list[Finding]:
    out = _Collector(path, _Lines(text))
    lines = _python_code_lines(text, tree) if python else _code_lines(text)
    rocm = next((number for number, line in enumerate(lines, 1) if _ROCM_BRANCH.search(line)), None)
    for number, line in enumerate(lines, 1):
        rejected = list(dict.fromkeys(m.group(0) for m in _NVCC_REJECTED.finditer(line)))
        ignored = list(dict.fromkeys(m.group(0) for m in _NVCC_IGNORED.finditer(line)))
        if rejected:
            listed = ", ".join(f"`{flag}`" for flag in rejected + ignored)
            if rocm is None:
                out.add_line(
                    number,
                    "nvcc_flags",
                    "high",
                    f'nvcc-only compiler flag(s) {listed}; hipcc fails with "unknown argument".',
                )
            else:
                out.add_line(
                    number,
                    "nvcc_flags",
                    "review",
                    f"nvcc-only compiler flag(s) {listed}. The file already checks for ROCm "
                    f"(line {rocm}); confirm these flags are only used on the CUDA side.",
                )
        elif ignored:
            listed = ", ".join(f"`{flag}`" for flag in ignored)
            out.add_line(
                number,
                "nvcc_flags",
                "info",
                f"nvcc-only compiler flag(s) {listed}; hipcc only warns that they are "
                "unused, so they have no effect on ROCm.",
            )
        if not python and _CMAKE_CUDA.search(line):
            out.add_line(
                number,
                "cmake_cuda_language",
                "review",
                "CMake builds this project with the CUDA language.",
            )
        if not python and _CMAKE_FATAL_CUDA.search(line):
            out.add_line(number, "cuda_required", "high", "Build aborts when CUDA is not found.")
    if python and tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                message = " ".join(_string_constants(node.exc))
            elif isinstance(node, ast.Assert) and node.msg is not None:
                message = " ".join(_string_constants(node.msg))
            else:
                continue
            if _CUDA_REQUIREMENT.search(message):
                out.add_line(
                    node.lineno,
                    "cuda_required",
                    "high",
                    f'Build aborts when CUDA is not found: "{message.strip()[:100]}".',
                )
    return out.findings


def _string_constants(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


# --------------------------------------------------------------------------- #
# Python rules
# --------------------------------------------------------------------------- #
_TRITON_JIT = re.compile(r"^\s*@triton\.jit\b")
_CAPABILITY = re.compile(
    r"\bget_device_capability\s*\(|\bget_device_properties\s*\([^)]*\)\.major\b"
)
_FP8_TORCH = re.compile(r"\btorch\.float8_e4m3fn\b|\btorch\.float8_e5m2\b")
_EXTENSION_NAME = re.compile(r"\bCUDAExtension\s*\(\s*(?:name\s*=\s*)?[\"']([\w.]+)[\"']")
_LOAD_NAME = re.compile(r"\bload(?:_inline)?\s*\([^)]*?\bname\s*=\s*[\"'](\w+)[\"']", re.DOTALL)
_PYTHON_PACKAGES = {
    "flash_attn": ("info", "ROCm/flash-attention (Composable Kernel and Triton backends)"),
    "xformers": ("info", "ROCm/xformers"),
    "apex": ("info", "ROCm/apex"),
    "bitsandbytes": ("info", "a ROCm-enabled bitsandbytes release"),
    "pycuda": ("high", None),
    "tensorrt": ("high", None),
    "cuda.bindings": ("high", None),
    "cuda.cudart": ("high", None),
    "cuda.cuda": ("high", None),
    "cuda.nvrtc": ("high", None),
    "nvidia.dali": ("high", None),
    "pynvml": ("review", "amdsmi for device queries"),
}


@dataclass
class _RepoContext:
    extension_modules: frozenset[str] = frozenset()
    has_cuda_sources: bool = False


def _scan_python(
    text: str,
    path: str,
    file: Path | None,
    context: _RepoContext,
    tree: ast.Module | None,
) -> list[Finding]:
    out = _Collector(path, _Lines(text))
    lines = _python_code_lines(text, tree)
    triton = [number for number, line in enumerate(lines, 1) if _TRITON_JIT.search(line)]
    if triton:
        out.add_line(triton[0], "triton_kernels", "info", f"{len(triton)} Triton kernel(s).")
    for number, line in enumerate(lines, 1):
        if _CAPABILITY.search(line):
            out.add_line(
                number,
                "capability_dispatch",
                "review",
                "Branches on the CUDA compute capability.",
            )
        match = _FP8_TORCH.search(line)
        if match:
            out.add_line(number, "fp8_format", "review", f"OCP FP8 dtype `{match.group(0)}`.")
    if tree is None:
        return out.findings
    _check_python_packages(tree, out)
    if file is not None:
        _check_eager_imports(tree, path, file, context, out)
    return out.findings


def _check_python_packages(tree: ast.Module, out: _Collector) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        for module in modules:
            for package, (severity, target) in _PYTHON_PACKAGES.items():
                if module != package and not module.startswith(package + "."):
                    continue
                where = f"; look for {target}" if target else "; it has no ROCm build"
                out.add_line(
                    node.lineno,
                    "python_cuda_package",
                    severity,
                    f"Imports `{package}`, which ships CUDA code{where}.",
                )


def _check_eager_imports(
    tree: ast.Module, path: str, file: Path, context: _RepoContext, out: _Collector
) -> None:
    package = list(Path(path).parts[:-1])
    for node in tree.body:
        # Each candidate is (dotted module name, sibling name to look for on disk or None).
        candidates: list[tuple[str, str | None]]
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.level - 1 > len(package):
                continue
            base = package[: len(package) - (node.level - 1)]
            directory = file.parents[node.level - 1]
            if node.module:
                module_parts = node.module.split(".")
                leaf = module_parts[0] if len(module_parts) == 1 else None
                candidates = [(".".join(base + module_parts), leaf)]
            else:
                candidates = [(".".join(base + [alias.name]), alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            directory = None
            candidates = [(node.module, None)]
        elif isinstance(node, ast.Import):
            directory = None
            candidates = [(alias.name, None) for alias in node.names]
        else:
            continue
        for module, leaf in candidates:
            declared = any(
                module == name or module.endswith("." + name) for name in context.extension_modules
            )
            undeclared_private = (
                not declared
                and context.has_cuda_sources
                and directory is not None
                and leaf is not None
                and leaf.startswith("_")
                and not leaf.startswith("__")
                and not (directory / f"{leaf}.py").exists()
                and not (directory / leaf).is_dir()
            )
            if declared or undeclared_private:
                out.add_line(
                    node.lineno,
                    "eager_extension_import",
                    "high" if declared else "review",
                    f"Imports the compiled extension `{module}` when the module loads. Where "
                    "it isn't built on ROCm, importing this module fails, and so does every "
                    "module that imports it without a guard, including portable code.",
                )
                break


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def iter_repo_files(root: Path, skip_dirs: Iterable[str] = ()) -> Iterator[Path]:
    """Yield the repository's files in a stable order, skipping VCS, cache and build dirs."""

    skipped = SKIP_DIRS | frozenset(skip_dirs)
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name not in skipped and not name.endswith(".egg-info")
        )
        for name in sorted(filenames):
            yield Path(current) / name


def _is_generated(path: Path) -> bool:
    if path.suffix == ".prehip":
        return True
    stem = path.stem.removesuffix("_hip")
    if path.suffix == ".hip" or path.stem.endswith("_hip"):
        originals = (".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp")
        return any((path.parent / f"{stem}{suffix}").exists() for suffix in originals)
    return False


def _is_build_file(path: Path, text: str) -> bool:
    if path.name in BUILD_FILE_NAMES or path.suffix in {".cmake", ".mk"}:
        return True
    return path.suffix == ".py" and bool(
        re.search(r"cpp_extension|CUDAExtension|\bnvcc\b|extra_cuda_cflags", text)
    )


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_source(text: str, path: str = "<source>") -> list[Finding]:
    """Scan one file's contents; the suffix of ``path`` selects the rules."""

    suffix = Path(path).suffix
    tree = _parse_python(text) if suffix == ".py" else None
    findings: list[Finding] = []
    if suffix in CUDA_SOURCE_SUFFIXES:
        findings += _scan_c_source(text, path)
    if suffix == ".py":
        findings += _scan_python(text, path, None, _RepoContext(), tree)
    if _is_build_file(Path(path), text):
        findings += _scan_build(text, path, python=suffix == ".py", tree=tree)
    return _sorted(findings)


def scan_path(path: Path | str, *, skip_dirs: Iterable[str] = ()) -> ScanReport:
    """Scan a file or a repository directory."""

    root = Path(path).resolve()
    if root.is_file():
        base, candidates = root.parent, [root]
    else:
        base, candidates = root, [p for p in iter_repo_files(root, skip_dirs)]
    texts: dict[Path, str] = {}
    for candidate in candidates:
        relevant = candidate.suffix in CUDA_SOURCE_SUFFIXES or candidate.suffix == ".py"
        relevant = relevant or candidate.name in BUILD_FILE_NAMES
        relevant = relevant or candidate.suffix in {".cmake", ".mk"}
        if not relevant or _is_generated(candidate):
            continue
        text = _read(candidate)
        if text is not None:
            texts[candidate] = text

    extension_modules: set[str] = set()
    for candidate, text in texts.items():
        if _is_build_file(candidate, text):
            extension_modules.update(_EXTENSION_NAME.findall(text))
            extension_modules.update(_LOAD_NAME.findall(text))
    context = _RepoContext(
        extension_modules=frozenset(extension_modules),
        has_cuda_sources=any(candidate.suffix in {".cu", ".cuh"} for candidate in texts),
    )

    findings: list[Finding] = []
    for candidate, text in texts.items():
        relative = candidate.relative_to(base).as_posix()
        tree = _parse_python(text) if candidate.suffix == ".py" else None
        if candidate.suffix in CUDA_SOURCE_SUFFIXES:
            findings += _scan_c_source(text, relative)
        if candidate.suffix == ".py":
            findings += _scan_python(text, relative, candidate, context, tree)
        if _is_build_file(candidate, text):
            findings += _scan_build(text, relative, python=candidate.suffix == ".py", tree=tree)
    return ScanReport(root=str(root), files_scanned=len(texts), findings=_sorted(findings))


def _sorted(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITIES.index(f.severity), f.path, f.line, f.rule))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="Repository directory or single source file")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("--min-severity", choices=SEVERITIES, default="info")
    parser.add_argument("--max-findings", type=int, default=200)
    args = parser.parse_args(argv)
    report = scan_path(args.path).filtered(args.min_severity)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_text(max_findings=args.max_findings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
