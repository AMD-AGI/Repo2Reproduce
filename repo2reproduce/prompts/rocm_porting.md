CUDA-to-ROCm porting policy:

This benchmark runs on AMD Instinct GPUs (gfx942) with ROCm. When the selected experiment
depends on CUDA-specific code (custom CUDA kernels, inline PTX, or CUDA-only libraries),
porting that code to ROCm is part of this stage. Invoke the `{{SKILL}}` skill with the Skill
tool and follow its ground rules, strategy ladder, HIPIFY fixes, build rules, and parity
method. Keep the paper's algorithm, numerical formats, and hyperparameters unchanged.

{{PORTING_REQUIREMENT}}

Host tools for this run. After the repository is cloned, run these host commands before
choosing a strategy. Each bash call must be exactly one command: no `cd`, pipes, redirects, or
shell substitutions. `CLONE` is the host path of the clone, under {{WORKSPACE}}.

```
port-scan CLONE
port-hipify CLONE
```

`port-scan` defaults to high severity; add `--min-severity review` to include 32-lane
patterns. `port-hipify` copies the clone to `CLONE-hip` (choose another workspace path with
`--output PATH`) and translates every CUDA source in the copy with the pinned official
`hipify-perl`; the clone itself is never modified. Rerunning it with `--overwrite`
regenerates the copy and discards edits made inside it. Copies of both reports are saved as
`port-scan.txt` and `port-hipify.txt` below the run's `artifacts/` directory.

Workspace contract:

- Keep every source change in a patch file or idempotent edit script under the host
  workspace, applied by the Dockerfile or launch script on top of the translated copy.
- The translated copy is the repository the manifest describes: set `host_repository` to it
  (for example `workspace/repo-hip`), set `container_repository` to its exact container path
  (`/run/workspace/repo-hip`), and have the launch script `cd` to that same directory.
  Naming the untranslated clone in the manifest while the launch script runs from the
  translated copy is rejected before execution.
- Save the complete parity-comparison log below /run/artifacts.

Record the porting work in the manifest's `rocm_port` object:

- `uses_cuda_specific_code`: whether the official code path of the selected experiment
  depends on CUDA-specific code.
- `summary`: what was ported or substituted, and why the result still reflects the paper.
- `attempts`: one entry per strategy tried, with `strategy` (`existing_rocm_port`,
  `portable_repo_path`, `hipify_with_fixes`, or `targeted_rewrite`), `outcome`
  (`succeeded`, `failed`, or `not_applicable`), and concrete `evidence`.
- `patches`: patch files or edit scripts, relative to the host workspace.
- `ported_kernels`: required when `hipify_with_fixes` or `targeted_rewrite` succeeded. One
  entry per CUDA kernel the selected entrypoint launches: `cuda_kernel` (the kernel or
  launcher name as written in the CUDA source), `cuda_source` (its file in the clone,
  relative to the host workspace), `implementation` (`hip`, `triton`, `composable_kernel`,
  or `rocwmma`), and `port_files` (files relative to the host workspace that define the
  replacement kernel). Before execution the runner checks that `cuda_source` mentions
  `cuda_kernel` and that a port file defines a kernel in the declared language
  (`__global__`, `@triton.jit`, `ck::`, or `rocwmma::`).
- `parity_check`: `reference`, `inputs`, `error_metric` (lower is better),
  `observed_error`, `tolerance`, and `log_path` relative to /run/artifacts.

A blocked manifest must record all four strategies when `uses_cuda_specific_code` is true or
the blocked reason concerns CUDA or NVIDIA. A ready manifest whose selected path uses
CUDA-specific code must record the strategy that succeeded and a passing `parity_check`.
