# CUDA-to-ROCm porting in the two-stage agent

Many papers ship their contribution as a custom CUDA kernel: a quantized attention, a fused GEMV, a KV-cache codec. The benchmark hosts here are AMD Instinct GPUs (gfx942, MI300X/MI325X) with no CUDA or nvcc. To reproduce such a paper honestly, the agent has to run that kernel on ROCm, not quietly replace it with something slower or different.

The multi-step agent in `repo2reproduce/` has a porting mode that makes this explicit. This document explains how it differs from a run without that mode, how each piece works, and what it produced on real repositories.

## Baseline versus porting mode

Both modes share the same two-stage loop. In stage 1, a Claude agent reads the paper and code, builds a Docker image, runs a bounded smoke test, writes `execution/launch.sh`, and returns an execution manifest. In stage 2, a deterministic Python executor runs that launch script in a fresh container, and a short report-only session writes `final_report.md`.

The baseline (`feature/claude-two-stage-baseline`) gives the agent no CUDA tooling and no porting rules. When a repository depends on CUDA kernels, the agent can do anything that makes the smoke test pass: reimplement the kernel in PyTorch, switch to a Triton path already in the repo, disable the extension, or give up. The runner never checks that the paper's kernel ran.

Porting mode is the baseline plus three commits on `feature/no-pytorch-fallback`:

- `ba3d2e4` adds the porting mode, the CUDA scanner, and the HIPIFY report.
- `6d66dfc` switches translation to AMD's official HIPIFY and forbids PyTorch's `torch.utils.hipify`.
- `df28a2b` rejects PyTorch fallbacks: every CUDA kernel on the selected path must be listed as a ported GPU kernel.

| | Baseline | `--rocm-porting` | `--rocm-porting --force-kernel-port` |
|---|---|---|---|
| Porting policy | none | `rocm-kernel-porting` skill plus the thin `prompts/rocm_porting.md` | same, with the forced requirement |
| Host tools | none | `port-scan`, `port-hipify` | same |
| Translation | whatever the build does, often `CUDAExtension` auto-hipify | official `hipify-perl` into `<clone>-hip`, then hand rewrites | same |
| Acceptable outcome | smoke test exits 0 | the first strategy on the ladder that yields the result | `hipify_with_fixes` or `targeted_rewrite` must succeed |
| PyTorch | may replace the kernel | parity reference and full-precision baseline only | same |
| Numerical parity | not required | required for ported code, checked against a tolerance | required, and must call the ported kernel |
| Blocked on CUDA | any reason accepted | all four strategies recorded with evidence | both kernel strategies actually attempted |

## Running it

From the `Repo2Reproduce` checkout on a compute node:

```bash
mkdir -p logs/two-stage-launches

nohup setsid .venv/bin/python -m repo2reproduce \
  https://arxiv.org/abs/2505.18231 \
  https://github.com/DHdroid/NSNQuant \
  --run-id nsnquant \
  --gpu 5 \
  --timeout-seconds 5400 \
  --rocm-porting \
  --force-kernel-port \
  --target "Reproduce Section 4.5 / Figure 4 memory and throughput ..." \
  > logs/two-stage-launches/nsnquant.log 2>&1 < /dev/null &
```

`--force-kernel-port` requires `--rocm-porting`. The mode is fixed when a run is created and saved in `input.json` and `metadata.json`; `--recover-run` reuses it and rejects the flags. Creating a porting run needs `git`, `perl`, and GitHub access on the host, because the runner clones HIPIFY. See `HOWTOUSE.md` for the other options.

## How porting mode works

Porting mode adds a policy to environment preparation, two host tools, and validation in the runner and executor. The goal is to port the CUDA-specific code the selected experiment needs while keeping the paper's algorithm, numerical formats, and hyperparameters unchanged.

The policy is split in two. The reusable reasoning (the strategy ladder, the no-fallback and no-PyTorch-hipify rules, HIPIFY leftovers, gfx942 fixes, the build recipe, and the parity method) lives in the `rocm-kernel-porting` skill under `repo2reproduce/skills/`, with a gfx942 reference file. The thin `repo2reproduce/prompts/rocm_porting.md` supplies only the run's mode, host-tool rules, workspace contract, and the `rocm_port` fields. Environment preparation enables and must load the skill; execution repair in a porting run loads it together with `execution-repair`, keeps the porting manifest model, and is rejected if it drops `rocm_port` or any ported kernel.

**Baseline:** prepare an image and a smoke test. A passing smoke test becomes a ready manifest; otherwise the run is blocked. Stage 2 runs `launch.sh` and writes `final_report.md`.

**Porting mode:** pin HIPIFY, clone, `port-scan`, `port-hipify`, then rewrite leftovers into patch files. Try an existing ROCm port, then an in-repo GPU kernel. A PyTorch path does not count. If neither works, HIPIFY with hand fixes, then a targeted rewrite. Forced mode skips ahead to those last two. Build without PyTorch's hipify, check parity, record `rocm_port`, and pass the on-disk checks. Then the same stage 2.

### 1. The runner prepares the porting tools

When `--rocm-porting` is set, `initialize_run` in `repo2reproduce/runner.py`:

- records the mode in `input.json` and `metadata.json`; environment preparation then appends the porting policy, with the standard or forced requirement filled in;
- clones [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) at `rocm-7.2.4` into `<run>/tools/HIPIFY-rocm-7.2.4`, verifies commit `fd26301`, and records both in `metadata.json`;
- installs `port-scan` and `port-hipify` in `<run>/bin`, on the agent's PATH.

To use another HIPIFY release, change `HIPIFY_REF` and `HIPIFY_COMMIT` in `repo2reproduce/porting/official_hipify.py`. The profiler in `repo2reproduce/porting/profiler.py` is not installed for the agent.

### 2. Clone the repository and run both tools

After cloning the official repository, the agent runs two host commands before choosing a strategy. `CLONE` is the clone's host path below the run workspace. Each command must be a single shell command: no `cd`, pipes, redirects, or shell substitutions.

```bash
port-scan CLONE
port-hipify CLONE
```

`port-scan` (`repo2reproduce/porting/scanner.py`) reports CUDA constructs that need a ROCm decision. Its default threshold is `high`; `--min-severity review` also includes 32-lane patterns. Findings include inline PTX, 32-bit warp masks and ballots, code that assumes 32 lanes, shared memory beyond gfx942's 64 KiB, nvcc-only build flags, compute-capability dispatch, and Python modules that eagerly import a CUDA extension. Every blocker or high finding needs a decision in the attempt evidence.

`port-hipify` (`repo2reproduce/porting/official_hipify.py`) copies the clone to `CLONE-hip`, or to another workspace path given with `--output PATH`, and translates every CUDA source in the copy with pinned `hipify-perl`. The clone itself is never modified. The report lists files HIPIFY changed and identifiers it could not translate. It also lists CUDA references that HIPIFY leaves without a warning: PyTorch's C++ CUDA API (`ATen/cuda/...`, `at::cuda::`, `c10::cuda::`, `C10_CUDA_*`) and `__CUDA_ARCH__`.

Both reports are copied to `artifacts/port-scan.{txt,json}` and `artifacts/port-hipify.{txt,json}`.

### 3. Rewrite what HIPIFY leaves behind

The agent rewrites each HIPIFY leftover by hand. Examples are `ATen/cuda/CUDAContext.h` to `ATen/hip/HIPContext.h`, `at::cuda::getCurrentCUDAStream()` to `at::hip::getCurrentHIPStreamMasqueradingAsCUDA()`, and `__CUDA_ARCH__` checks to `__HIP_DEVICE_COMPILE__` or gfx-architecture macros.

`CLONE-hip` is generated output. Rerunning `port-hipify --overwrite` regenerates it and discards edits made inside it. Every rewrite must therefore be stored as a patch file or idempotent edit script under the host workspace. The Dockerfile or launch script points at the translated copy below `/run/workspace` and applies those changes there.

For a forced port, the agent uses compile and runtime feedback to fix semantics, not only names. Typical issues are inline PTX and NVIDIA intrinsics, wave64 masks and subgroup widths, shared-memory limits, FP8 formats, compiler flags, library APIs, and architecture dispatch.

### 4. Try the porting strategies in order

Before returning `blocked` for a CUDA or NVIDIA reason, the agent tries these strategies in order:

1. `existing_rocm_port`: an official ROCm fork, upstream ROCm support, or a ROCm build of the CUDA dependency.
2. `portable_repo_path`: a Triton or other GPU-kernel implementation of the same computation inside the official repository. This includes non-default paths and paths that fail only because a package eagerly imports a CUDA extension. A PyTorch implementation of the kernel's math is not a portable path, even if the repository already contains it. This includes eager tensor operations, SDPA, and `torch.compile` of them.
3. `hipify_with_fixes`: use the `port-hipify` copy, fix what HIPIFY could not translate, and build it with hipcc. Fixes include identifiers HIPIFY reports, PyTorch C++ CUDA API leftovers, 64-lane wavefronts and 64-bit lane masks, shared-memory limits, NVIDIA-only compiler flags, and API overload differences.
4. `targeted_rewrite`: if the translated kernel cannot be fixed, reimplement only the kernels the selected experiment calls, in HIP, Triton, rocWMMA, or Composable Kernel, with the original numerics. Start from the HIPIFY copy so the host code remains translated by HIPIFY.

In standard mode, the agent stops at the first strategy that gives a working path to a representative result. In forced mode, `existing_rocm_port` and `portable_repo_path` are only investigation and reference steps. Success requires `hipify_with_fixes` or `targeted_rewrite`.

### 5. Port every launched kernel

There is no PyTorch fallback. Every CUDA kernel launched by the selected entrypoint must run on ROCm as a HIP, Triton, Composable Kernel, or rocWMMA kernel. This includes preprocessing and postprocessing kernels such as quantization, mean subtraction, scaling, packing, layout transforms, and fused epilogues.

If a translated helper kernel fails, the agent fixes it or rewrites it as a kernel. PyTorch may only allocate inputs, call the ported entrypoint, implement the parity reference, and run the full-precision baseline.

For a forced port, the parity smoke test and final launch must call the ported kernel entrypoint. Running the CPU or reference implementation, OpenCL, or the repository's Triton implementation does not count. Replacing any launched kernel with PyTorch also does not count.

### 6. Build without PyTorch's hipify

The agent must not call `torch.utils.hipify` or `hipify_python`. It must not rely on `torch.utils.cpp_extension.CUDAExtension` either, because that runs `torch.utils.hipify` during a ROCm build.

For PyTorch extensions, compile the HIPIFY output with `CppExtension` and `BuildExtension`. They compile `.cu` and `.hip` sources with hipcc; add the include directories, libraries, and macros `CUDAExtension` would normally add. CMake, Make, or direct hipcc builds are also allowed. The build log must not contain `torch.utils.hipify` output: lines that map a source to a `_hip` copy and end in `[ok]` or `[skipped, ...]`.

### 7. Run a numerical parity check

When the selected path runs ported or substituted code, the smoke test compares it with a trusted reference on identical random inputs. The reference can be the repository's reference implementation, an established library such as e3nn or `torch.nn.functional.scaled_dot_product_attention`, or the paper's algorithm in plain PyTorch.

The reference is only for comparison; it never replaces a ported kernel in the smoke test or launch. The check uses the experiment's dtypes and shapes, a tolerance appropriate for those dtypes, and a complete log below `/run/artifacts`. If the error exceeds the tolerance, preparation is not finished.

### 8. Record the work in `rocm_port`

The execution manifest includes a `rocm_port` object:

- `uses_cuda_specific_code`: whether the selected official code path depends on CUDA-specific code.
- `summary`: what was ported or substituted, and why the result still reflects the paper.
- `attempts`: one entry per strategy tried, with `strategy`, `outcome` (`succeeded`, `failed`, or `not_applicable`), and concrete evidence such as URLs, file paths, commands, and errors.
- `patches`: patch files or edit scripts relative to the host workspace.
- `ported_kernels`: required when `hipify_with_fixes` or `targeted_rewrite` succeeds. Each entry gives `cuda_kernel`, `cuda_source`, `implementation`, and `port_files`, including preprocessing kernels.
- `parity_check`: `reference`, `inputs`, `error_metric`, `observed_error`, `tolerance`, and `log_path` relative to `/run/artifacts`.

A blocked manifest must record all four strategies when `uses_cuda_specific_code` is true or the blocked reason concerns CUDA or NVIDIA. A ready manifest with CUDA-specific code must record the strategy that succeeded and a passing `parity_check`.

### 9. Validate the manifest and evidence before execution

`repo2reproduce/schemas.py` validates the manifest. Standard porting runs use `RocmPortingManifest`; forced runs use `ForcedKernelPortingManifest`, which also requires a successful `hipify_with_fixes` or `targeted_rewrite`, at least one persisted patch, and real attempts at both kernel strategies before blocking. Environment preparation returns that manifest inside `PortingPreparationStageOutput` or `ForcedPortingPreparationStageOutput`.

Before Docker starts, `repo2reproduce/executor.py` checks that listed patches and the parity log exist. In forced mode, it also requires `artifacts/port-scan.json` and `artifacts/port-hipify.json`. For each ported kernel, it checks that `cuda_source` mentions `cuda_kernel` and that a port file defines a kernel in the declared language: `__global__`, `@triton.jit`, `ck::`, or `rocwmma::`.

If any check fails, the run stops before execution instead of producing a misleading number.

### What a porting run leaves behind

| Path | Contents |
|---|---|
| `metadata.json` | mode flags and the pinned HIPIFY commit |
| `artifacts/port-scan.{txt,json}` | scanner findings |
| `artifacts/port-hipify.{txt,json}` | HIPIFY translation report |
| `workspace/<repo>-hip/` | HIPIFY-translated copy |
| `workspace/porting/` or `workspace/execution/` | patch files, edit scripts, Dockerfile, smoke and parity scripts |
| `artifacts/smoke/` | smoke and parity logs |
| `execution_manifest.json` | the validated manifest, including `rocm_port` |
| `artifacts/full-run/` | results from the full experiment |
| `final_report.md` | paper value, achieved value, match, and limitations |

## Use cases

All of these were measured on an AMD Instinct MI325X (gfx942, ROCm 7.2, PyTorch 2.9.1).

### NSNQuant: HIPIFY plus hand fixes is enough

[NSNQuant](https://arxiv.org/abs/2505.18231) stores the KV cache as packed vector-quantized codes, and every decode step calls custom kernels in `src/csrc` and `src/csrc/dq_kernels`: codebook matching, fused quantized dot-product, and quantized weighted sum. There is no ROCm fork and no Triton path for that decode.

`port-hipify` translated 39 sources (1569 references, 18 warnings, all `__shfl_down`). `hipify_with_fixes` succeeded with three semantic fixes, kept in `workspace/porting/apply_rocm_patches.py`:

- **Wave64 reductions.** The weighted-sum kernels and `4bit_rtn.cu` reduce with `__shfl_down_sync(0xffffffff, ...)` inside 32-wide groups. On a 64-lane wavefront, two logical groups share one wavefront, so a full-width shuffle mixes them and fails HIP's mask check. The fix is `__shfl_down(v, delta, 32)`, with `half2` bit-cast through an `int`.
- **`math_constants.h`** does not exist on ROCm; `CUDART_INF_F` becomes `__builtin_huge_valf()`.
- **Stacked `#pragma unroll 4` and `#pragma unroll`** on one loop is a hipcc error; the redundant one is dropped.

The ported VQ kernel picked a different code than a PyTorch nearest-codebook search for 0.0488% of 8192 fp16 vectors (tolerance 2%). On LLaMA2-7B, Section 4.5 / Figure 4:

| | FP16 | NSNQuant-2b |
|---|---|---|
| Best throughput, 80 GB cap | 1147 tok/s at batch 64 | 5114 tok/s at batch 768 |
| Largest batch, 80 GB cap | 256 (OOM at 384) | 768, no OOM |
| Batch 256 uncapped | 336 tok/s, 75.8 GB | 4405 tok/s, 28.1 GB |

That is **4.46×** best throughput and a 3× larger batch, against the paper's roughly 3× and 4× on an A100-80GB.

| | Baseline | Our agent |
|---|---|---|
| Porting tools | none | `port-scan` and official `port-hipify` |
| What can satisfy the decode path | a PyTorch substitute, including `vq_torch` already in the repository | the `nsn_tools` CUDA kernels, ported with `hipify_with_fixes` |
| Translation | no required translation | 39 CUDA sources, 1569 references converted |
| gfx942 fixes | none required | width-32 shuffles, `math_constants.h` replaced, conflicting unroll pragma removed |
| Parity | none required | 0.0488% of VQ codes differ from a PyTorch nearest-codebook search (tolerance 2%) |
| Section 4.5 on MI325X | no requirement to run the paper kernels | 4.46× best throughput under an 80 GB cap; FP16 OOMs at batch 384, NSNQuant-2b runs at 768 |

That reproduction predates two policy changes. It built `nsn_tools` with `python setup.py install`, so PyTorch's `CUDAExtension` hipify ran on top of the patch, and it replaced the auxiliary `fast-hadamard-transform` extension with a PyTorch shim as it was not part of the repository's main codebase.


### SageAttention: HIPIFY fails, Triton rewrite succeeds

[SageAttention](https://arxiv.org/abs/2410.02367) and [SageAttention2](https://arxiv.org/abs/2411.10958) share the sm80 kernel `qk_int8_sv_f16_accum_f32_attn`, which uses inline PTX (`mma.sync`, `ldmatrix`, `cp.async`, `ex2.approx`) and `<cuda_pipeline_primitives.h>`. `port-scan` found about 160 PTX blockers, and hipcc stopped on the missing header, so `hipify_with_fixes` was recorded as failed.

`targeted_rewrite` reimplemented the attention kernel in Triton. For SageAttention the Q per-warp and K per-block INT8 quantization kernels were rewritten in Triton as well, rather than in PyTorch. Three kernels are listed in `ported_kernels`.

- Accuracy, 4096 tokens: cosine 0.999921 and relative L1 0.0125 against SDPA, consistent with the paper's near-lossless claim. Against a same-algorithm PyTorch reference: cosine 1.000000.
- Speed, Table 7 Llama2 shape: 106 TOPS for the INT8 kernel against 331 TOPS for SDPA's AOTriton flash attention, so **0.32×** against the paper's 1.77× on an RTX 4090. Replacing the INT8 QK matmul with FP16 made the same kernel 1.67× slower, which confirms the INT8 path ran. The gap is the Triton rewrite versus a hand-tuned PTX kernel, on different hardware.

### SpargeAttn: same pattern as SageAttention

[SpargeAttn](https://arxiv.org/abs/2502.18137) failed HIPIFY on `cuda_pipeline_primitives.h` behind 135 inline-PTX blockers. `qk_int_sv_f16_block_sparse_attn_kernel` was rewritten in Triton and matched SDPA at cosine 0.9999. At 8K tokens, prediction took 0.385 ms and full attention 2.473 ms, a 15.55% overhead against the paper's 3.78%. The ratio is inflated because the ported attention is faster than the paper's while prediction is launch-overhead bound.

### SageAttention in standard mode

On the baseline agent, SageAttention2 does not run. The repository is CUDA-only: `setup.py` requires `CUDA_HOME` and nvcc, the smoke test fails while building `sageattention._qattn_sm89`, and the agent returns `blocked`. It does not attempt HIPIFY and it does not port the kernels.

## Recurring fixes on gfx942

These came up repeatedly across runs.

- **Wavefront width.** NVIDIA warps are 32 lanes; gfx942 wavefronts are 64. Replace `__shfl_*_sync(0xffffffff, ...)` with width-confined `__shfl_*(v, offset, 32)` when kernels use 32-lane groups. Lane masks and ballots need 64-bit types.
- **Inline PTX.** HIPIFY cannot translate it. `mma.sync`, `ldmatrix`, `cp.async`, and `<cuda_pipeline_primitives.h>` usually mean `targeted_rewrite`, most cheaply in Triton.
- **PyTorch C++ API.** Map `ATen/cuda`, `c10/cuda`, `at::cuda::` and `C10_CUDA_*` to their `hip` / `MasqueradingAsCUDA` equivalents.
- **Builds.** Use `CppExtension` with `BuildExtension`, link `c10_hip`, `torch_hip` and `amdhip64`, add `/opt/rocm/include` and `-D__HIP_PLATFORM_AMD__=1`, and set `PYTORCH_ROCM_ARCH=gfx942` (the Docker build has no GPU to detect).
- **Missing headers and types.** `math_constants.h`, `cuda_device_runtime_api.h`, and some `atomicAdd` overloads for `__half2` and `bfloat16` need small shims.
- **Shared memory.** gfx942 allows 64 KiB of LDS per workgroup.
- **Imports.** Import `torch` before a ROCm extension so `libc10_hip.so` is loaded; this broke the NSNQuant reproduction that also ported the Hadamard kernel.

## Limitations

- Ports run on MI325X, so absolute latency and throughput differ from the paper's NVIDIA hardware. Reports compare ratios and trends, and flag the hardware difference.
- A Triton `targeted_rewrite` is numerically faithful but usually slower than the hand-tuned PTX kernel it replaces, as SageAttention and SpargeAttn show.
- The executor checks that port files define a kernel and that parity passed. It cannot prove that the launch actually reaches the ported kernel; the prompt requires that, and final reports cite call counts or control experiments as evidence.
