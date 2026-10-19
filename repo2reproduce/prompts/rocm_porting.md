CUDA-to-ROCm porting policy:

This benchmark runs on AMD Instinct GPUs with ROCm. When the selected experiment depends on
CUDA-specific code (custom CUDA kernels, inline PTX, or CUDA-only libraries), porting that code
to ROCm is part of preparing the environment and is in scope. Keep the paper's algorithm,
numerical formats, and hyperparameters unchanged.

{{PORTING_REQUIREMENT}}

Before returning `status: "blocked"` for a CUDA or NVIDIA reason, try these strategies in order:

1. `existing_rocm_port`: an official ROCm fork, upstream ROCm support, or a ROCm build of the
   CUDA dependency.
2. `portable_repo_path`: a Triton or other GPU-kernel implementation of the same computation
   inside the official repository, including paths that are not the default and paths that only
   fail because a package eagerly imports a CUDA extension. A PyTorch implementation of the
   kernel's math (eager tensor operations, SDPA, or `torch.compile` of them) is a fallback, not a
   portable path, and does not count.
3. `hipify_with_fixes`: translate the CUDA sources with AMD's official HIPIFY by running
   `port-hipify` (below), then rewrite by hand what HIPIFY could not translate: the identifiers it
   reports, PyTorch C++ CUDA API leftovers, 64-lane wavefronts and 64-bit lane masks,
   shared-memory limits, NVIDIA-only compiler flags, and API overload differences. Build the
   translated copy with hipcc.
4. `targeted_rewrite`: when the translated kernel cannot be fixed, reimplement only the kernels
   the selected experiment calls, in HIP, Triton, rocWMMA, or Composable Kernel, with the same
   numerics as the original. Start from the HIPIFY-translated copy so the host code stays
   translated by HIPIFY.

No PyTorch fallback: every CUDA kernel the selected entrypoint launches must run on ROCm as a
ported GPU kernel, in HIP, Triton, Composable Kernel, or rocWMMA. This includes preprocessing and
postprocessing kernels such as quantization, mean subtraction, scaling, packing, layout
transforms, and fused epilogues, even when they look cheap. Do not reimplement any of them with
PyTorch tensor operations; if a HIPIFY-translated helper kernel fails, fix it or rewrite it as a
kernel. PyTorch may only allocate inputs, call the ported entrypoint, and implement the parity
reference and the full-precision baseline.

Do not hipify with PyTorch: never call `torch.utils.hipify` or `hipify_python`, and do not rely
on `torch.utils.cpp_extension.CUDAExtension` to translate sources during the ROCm build, because
that runs `torch.utils.hipify` on them. For PyTorch extensions, compile the HIPIFY output with
`CppExtension` and `BuildExtension`, which compile `.cu` and `.hip` sources with hipcc, adding the
include directories, libraries, and macros that `CUDAExtension` would have added. Alternatively,
build it with CMake, Make, or hipcc. The build log must not contain `torch.utils.hipify` output
(lines that map a source to a `_hip` copy and end in `[ok]` or `[skipped, ...]`).

In standard mode, stop at the first strategy that gives a working path to a representative
result. In forced-kernel mode, `existing_rocm_port` and `portable_repo_path` are investigation and
reference steps only: success requires `hipify_with_fixes` or `targeted_rewrite`. Keep every source
change in a patch file or idempotent edit script under the host workspace, applied by the
Dockerfile or launch script.

Numerical parity check: when the selected path runs ported or substituted code, the smoke test
must also compare that code with a trusted reference on identical random inputs, such as the
repository's reference implementation, an established library (for example e3nn or
`torch.nn.functional.scaled_dot_product_attention`), or the paper's algorithm written in plain
PyTorch. A PyTorch reference is only compared against; it never replaces a ported kernel in the
smoke test or the launch. Use the selected experiment's dtypes and shapes, choose a tolerance
that fits those dtypes, and save the complete comparison log below /run/artifacts. A comparison
that exceeds the tolerance means preparation is not finished.

Record the porting work in the manifest's `rocm_port` object:

- `uses_cuda_specific_code`: whether the official code path of the selected experiment depends
  on CUDA-specific code.
- `summary`: what was ported or substituted, and why the result still reflects the paper.
- `attempts`: one entry per strategy tried, with `strategy`, `outcome` (`succeeded`, `failed`, or
  `not_applicable`), and concrete `evidence` such as URLs, file paths, commands, and errors.
- `patches`: patch files or edit scripts, relative to the host workspace.
- `ported_kernels`: required when `hipify_with_fixes` or `targeted_rewrite` succeeded. One entry
  per CUDA kernel the selected entrypoint launches, preprocessing kernels included: `cuda_kernel`
  (the kernel or launcher name as written in the CUDA source), `cuda_source` (its file in the
  clone, relative to the host workspace), `implementation` (`hip`, `triton`,
  `composable_kernel`, or `rocwmma`), and `port_files` (files relative to the host workspace that
  define the replacement kernel). Before execution the runner checks that `cuda_source` mentions
  `cuda_kernel` and that a port file defines a kernel in the declared language (`__global__`,
  `@triton.jit`, `ck::`, or `rocwmma::`).
- `parity_check`: `reference`, `inputs`, `error_metric` (lower is better), `observed_error`,
  `tolerance`, and `log_path` relative to /run/artifacts.

A blocked manifest must record all four strategies when `uses_cuda_specific_code` is true or the
blocked reason concerns CUDA or NVIDIA. A ready manifest whose selected path uses CUDA-specific
code must record the strategy that succeeded and a passing `parity_check`.

After the repository is cloned, run these host commands before choosing a strategy. Each bash
call must be exactly one command: no `cd`, pipes, redirects, or shell substitutions. `CLONE` is
the host path of the clone, under {{WORKSPACE}}.

```
port-scan CLONE
port-hipify CLONE
```

`port-scan` reports CUDA constructs that need a ROCm decision and defaults to high severity;
add `--min-severity review` to include 32-lane patterns. `port-hipify` copies the clone to
`CLONE-hip` (choose another workspace path with `--output PATH`) and translates every CUDA source
in the copy with `hipify-perl` from https://github.com/ROCm/HIPIFY, pinned to a release tag. The
clone itself is never modified. Its report lists the files HIPIFY changed and the identifiers
HIPIFY could not translate, and it lists CUDA references HIPIFY leaves without a warning:
PyTorch's C++ CUDA API (`ATen/cuda/...` headers, `at::cuda::`, `c10::cuda::`, `C10_CUDA_*`) and
`__CUDA_ARCH__`. Rewrite each of these yourself, for example `ATen/cuda/CUDAContext.h` to
`ATen/hip/HIPContext.h`, `at::cuda::getCurrentCUDAStream()` to
`at::hip::getCurrentHIPStreamMasqueradingAsCUDA()`, and `__CUDA_ARCH__` checks to
`__HIP_DEVICE_COMPILE__` or gfx-architecture macros. Copies of both reports are saved as
`port-scan.txt` and `port-hipify.txt` below the run's `artifacts/` directory. A blocker or high
finding needs a decision recorded in the attempt evidence.

`CLONE-hip` is generated output. Rerunning `port-hipify` with `--overwrite` regenerates it and
discards edits made inside it, so keep your rewrites in patch files or edit scripts applied on
top of the HIPIFY output. Point the Dockerfile or launch script at the translated copy, below
/run/workspace, and apply the patches there. The translated copy is then the repository the
manifest describes: set `host_repository` to it (for example `workspace/repo-hip`), set
`container_repository` to its exact container path (`/run/workspace/repo-hip`), and have the
launch script `cd` to that same directory. Naming the untranslated clone in the manifest while
the launch script runs from the translated copy is rejected before execution.

For a forced kernel port, run both commands and use compile/runtime feedback to fix semantics,
not just spelling: inline PTX or NVIDIA intrinsics, wave64 masks and subgroup widths, shared-memory
limits, FP8 formats, compiler flags, library APIs, and architecture dispatch. The parity smoke test
and final launch must call the ported kernel entrypoint, and every kernel that entrypoint
launches must be a ported GPU kernel listed in `ported_kernels`. Merely running its
CPU/reference, OpenCL, or repository Triton implementation, or replacing any of its kernels with
PyTorch, is not a successful forced port.
