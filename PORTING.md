# CUDA-to-ROCm porting

Many papers ship their contribution as a CUDA kernel. The benchmark hosts are AMD Instinct
GPUs (gfx942, MI300X/MI325X) with no CUDA, so porting mode makes the agent run the paper's
kernel on ROCm instead of quietly replacing it with something slower or different.

## Modes

| | Default | `--rocm-porting` | `--rocm-porting --force-kernel-port` |
|---|---|---|---|
| Policy | none | `rocm-kernel-porting` skill | same, plus the forced requirement |
| Host tools | none | `port-scan`, `port-hipify` | same |
| Success | smoke test passes | first strategy that yields the result | `hipify_with_fixes` or `targeted_rewrite` |
| PyTorch | may replace the kernel | parity reference and baseline only | same |
| Parity check | not required | required for ported code | required, on the ported kernel |

With the scripts, `--porting rocm` and `--porting force` add these flags:

```bash
scripts/run_one.sh nsnquant --gpu 0 --porting force
```

`--force-kernel-port` requires `--rocm-porting`. The mode is fixed when a run is created;
`--recover` reuses it. A porting run needs `git`, `perl`, and GitHub access on the host to
clone HIPIFY.

## How it works

1. **Tools.** The runner clones the official [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY)
   at `rocm-7.2.4` into `<run>/tools/` and puts `port-scan` and `port-hipify` on the agent's
   PATH. Change the pin in `repo2reproduce/porting/official_hipify.py`.
2. **Scan and translate.** `port-scan CLONE` reports constructs that need a ROCm decision:
   inline PTX, 32-lane masks, shared memory over 64 KiB, nvcc-only flags.
   `port-hipify CLONE` translates a copy at `CLONE-hip` with `hipify-perl` and lists what
   it could not translate. The clone itself is never modified.
3. **Fix leftovers.** The agent rewrites what HIPIFY leaves behind, such as `ATen/cuda`,
   `at::cuda::`, and `__CUDA_ARCH__`. Edits are stored as patch files, because rerunning
   `port-hipify` regenerates `CLONE-hip`.
4. **Strategies, in order.** `existing_rocm_port`, then `portable_repo_path` (a Triton or
   other GPU kernel already in the repository), then `hipify_with_fixes`, then
   `targeted_rewrite` of only the kernels the experiment calls. Forced mode must succeed
   with one of the last two.
5. **No PyTorch fallback.** Every CUDA kernel the experiment launches must run as a HIP,
   Triton, Composable Kernel, or rocWMMA kernel. Builds must not use `torch.utils.hipify`
   or `CUDAExtension`; use `CppExtension`, CMake, or hipcc.
6. **Parity.** The smoke test compares the ported kernel with a trusted reference on the
   experiment's dtypes and shapes, within a tolerance.
7. **Validation.** The manifest records the work in `rocm_port`: attempts, patches,
   `ported_kernels`, and `parity_check`. Before execution, the executor checks that patches
   and the parity log exist, that each port file defines a kernel, and that a
   `hipify_with_fixes` kernel keeps at least half of HIPIFY's output. Any failure stops the
   run before it produces a number.

## Run outputs

| Path | Contents |
|---|---|
| `artifacts/port-scan.{txt,json}` | Scanner findings. |
| `artifacts/port-hipify.{txt,json}` | HIPIFY translation report. |
| `artifacts/hipify-provenance/` | HIPIFY-to-port diffs and overlap. |
| `workspace/<repo>-hip/` | HIPIFY-translated copy. |
| `execution_manifest.json` | The validated manifest, including `rocm_port`. |

## Results on MI325X

| Paper | Strategy | Result |
|---|---|---|
| [NSNQuant](https://arxiv.org/abs/2505.18231) | `hipify_with_fixes` | 4.46× throughput over FP16 on LLaMA2-7B (paper: about 3×). |
| [SageAttention](https://arxiv.org/abs/2410.02367) | `targeted_rewrite` in Triton | Cosine 0.99992 against SDPA; 0.32× SDPA speed (paper: 1.77× on an RTX 4090). |
| [SpargeAttn](https://arxiv.org/abs/2502.18137) | `targeted_rewrite` in Triton | Cosine 0.9999 against SDPA; 15.6% prediction overhead (paper: 3.8%). |

Without porting mode, SageAttention is blocked: its build requires nvcc.

## Common gfx942 fixes

- **Wavefront width.** Wavefronts are 64 lanes. Replace `__shfl_*_sync(0xffffffff, ...)`
  with `__shfl_*(v, offset, 32)` for 32-lane groups; use 64-bit masks and ballots.
- **Inline PTX.** HIPIFY cannot translate `mma.sync`, `ldmatrix`, or `cp.async`; these
  usually need a Triton `targeted_rewrite`.
- **Builds.** Link `c10_hip`, `torch_hip`, and `amdhip64`, add `-D__HIP_PLATFORM_AMD__=1`,
  and set `PYTORCH_ROCM_ARCH=gfx942`.
- **Headers.** `math_constants.h` and some `atomicAdd` overloads need small shims.
- **Imports.** Import `torch` before a ROCm extension.

## Limitations

- Absolute speed differs from the paper's NVIDIA hardware; reports compare ratios.
- A Triton rewrite is faithful but usually slower than the hand-tuned PTX it replaces.
- The executor cannot prove the launch reaches the ported kernel; reports cite call counts
  or control experiments as evidence.
