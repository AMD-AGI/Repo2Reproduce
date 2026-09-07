# Repo2Reproduce — Reproduction Issues Log

A living log of issues discovered while running the Repo2Reproduce agent (`r2r reproduce ... --amd --execute`)
on real papers + repos. Add a new dated entry per run. Every entry should be **grounded**:
cite the run dir, the exact file/line in the codebase, and the observed behavior.

**Environment (host):** AMD Instinct MI350X (gfx950), single GPU, ~252 GiB VRAM
(`270,566,162,432` B reported by `rocm-smi`), Docker 27.5.1, ROCm driver present
(`/dev/kfd`, `/dev/dri`). LLM routing: AMD internal gateway (`https://llm-api.amd.com/Unified`),
reasoning `Claude-Opus-4.8`, mapping `Claude-Sonnet-5`.

---

## Entry template (copy for each new run)

```
### <DATE> — <Paper short name> (<arXiv id>) — <repo>
- Run dir: runs/<timestamp>-<repo>/
- Mode: --amd --execute | --amd (plan-only)
- Selected experiment (agent's choice): <...>
- Target result (paper): <metric + locator>
- Model / size: <...>
- Expected memory: <...>
- Expected full-parity duration: <...>
- Outcome: <matched | proof-of-life / not comparable | build_failed | ...>
- Issue(s) found: <short title>
- Root cause (grounded: file:line): <...>
- Fix idea (not yet applied): <...>
```

---

## Issue Index (recurring, cross-run)

| ID | Issue | Grounding | Status |
|----|-------|-----------|--------|
| R2R-001 | `docker_run` has a hard, non-overridable 3600s timeout → executor shrinks long training jobs to a "proof-of-life" subset instead of reproducing the exact run | `tools/amd.py:143`, `tools/mcp_server.py:333-348`, `prompts/executor.md:12-15` | **Fixed 2026-07-26** (see Fixes Log) |

---

## Fixes Log (base issue → change implemented)

### R2R-001 — un-raisable `docker_run` timeout forces workload shrink
**Base issue (observed & grounded):** the `docker_run` MCP tool schema exposes only
`tag`/`command`/`workdir` (`tools/mcp_server.py:337-345`) and calls the helper without a
timeout (`:348`), so it is pinned to the hardcoded `timeout=3600` in `tools/amd.py:143`
(enforced by `subprocess.run(..., timeout=...)`, `:167`; empirically fires — a `sleep 30`
with `timeout=5` raised `TimeoutExpired` at ~5s). Because `prompts/executor.md:12-15` +
`amd_knowledge.py:72` tell the agent to run train+eval in ONE synchronous call and "reduce
epochs/samples so it fits one call", any >1h full-parity job would be killed, so the agent
pre-emptively shrinks it and reports "proof-of-life / not comparable". (Nuance: in the
GraLoRA run the timeout never actually fired — the shrink was prompt-driven; the 3600s cap
is the ceiling that both justifies the instruction and would kill a real full run.)

**Changes implemented (2026-07-26):**
1. **Expose `timeout` on the `docker_run` MCP tool** — added to the schema and passed
   through to the helper (`tools/mcp_server.py`).
2. **Env-configurable, raised defaults** — `tools/amd.py` `docker_run` now resolves its
   timeout from `R2R_DOCKER_RUN_TIMEOUT` (default **21600s / 6h**, up from 3600s); `docker_build`
   from `R2R_DOCKER_BUILD_TIMEOUT` (default **7200s / 2h**).
3. **Relaxed the "reduce" policy** — `prompts/executor.md` + `amd_knowledge.py` + the
   persisted memory entry now instruct the agent to reproduce the paper's FULL configuration
   with a large timeout, and to reduce workload only as a documented LAST RESORT.
4. **Stream `docker_run` output to a persisted log file** — instead of only returning the
   last 8000 chars, full stdout/stderr is streamed to
   `runs/<run>/amd_run/docker_logs/docker_run_<ts>.log` (observable live via `tail -f`); the
   tool returns a tail + `log_file` path, and now catches `TimeoutExpired` (returns
   `timed_out: true` instead of crashing the tool call).

---

## 2026-07-26 — GraLoRA (arXiv:2505.20355, NeurIPS 2025 Spotlight) — github.com/SqueezeBits/GraLoRA

- **Run dir:** `runs/20260725-235956-https-github.com-SqueezeBits-GraLoRA/`
- **Mode:** `--amd --execute` (budget cap removed, `R2R_MAX_TURNS=1000`, `HIP_VISIBLE_DEVICES=0`)
- **Outcome:** ROCm image built + verified; real GraLoRA train→eval exercised end-to-end on the MI350X,
  but reported as **"proof-of-life, NOT comparable"** (confidence: medium). Full-parity 79.8% not reproduced.

### 1. Experiment the agent chose
- **Commonsense Reasoning — Qwen2.5-1.5B, rank 64, GraLoRA** (paper Table 3, p.9).
- **Target result:** GraLoRA average accuracy **79.8%** over 8 commonsense tasks
  (BoolQ 67.2, PIQA 84.2, SIQA 75.9, HellaSwag 85.7, WinoGrande 73.8, ARC-c 77.5, ARC-e 89.9, OBQA 84.4);
  +1.1% avg over LoRA (78.7%).
- **Why this one (agent rationale):** Table 3 gives exact per-task + avg numbers; §4.1 pins the pipeline
  (LLM-Adapters, rank 64); Table 9 pins hyperparameters (rank 64, LR 2e-4, batch 192, 2 epochs, LionW).
  Qwen2.5-1.5B is the smallest fully-scripted model. The paper's GLUE/RoBERTa and SDXL image-gen
  experiments have **no code** in the repo, so they were correctly ruled out as not reproducible.

### 2. Model & size
- **Model:** `Qwen/Qwen2.5-1.5B` (pretrained base, not instruct).
- **Size:** ~1.5B parameters. GraLoRA (r=64, k=4) trainable params **73,859,072 (4.566%)**; adapter ≈ 295 MB.

### 3. Expected memory (single MI350X, bf16)
| Component | Footprint |
|---|---|
| Base weights (1.5B, frozen, bf16) | ~3 GB |
| Adapter grads (73.9M) | ~0.15 GB |
| Optimizer state — LionW (1 momentum tensor) | ~0.15–0.3 GB |
| Activations (cutoff_len=256, grad checkpointing) | a few GB |
| **Total** | **well under ~15 GB** |

→ Fits ~17× over on the 252 GiB card. **Memory is NOT a constraint.** The paper's "2×H100" is uniform
infra for its whole 1.5B–70B sweep, not a requirement for the 1.5B case; deepspeed ZeRO-2 (multi-GPU
memory sharding) is unnecessary on one large GPU, and batch 192 is reproducible via gradient accumulation
(`micro_batch × accum = 192`) on a single GPU.

### 4. Expected full-parity duration (single MI350X)
- Optimizer steps ≈ 170k samples × 2 epochs ÷ batch 192 ≈ **~1,771 steps**.
- Training: order of **~1 hour** (rough; depends on step time).
- Eval: **8 tasks, generative**, ~3,270 examples each (~26k generations) → **substantial additional time**,
  plausibly comparable-to-longer than training.
- **Combined full-parity run clearly exceeds 1 hour.**
- (Contrast — what was actually executed: ~200 train samples, 20 steps (~0.8 epoch), batch 8, boolq only on
  30 examples; train loss 3.11→0.40, train runtime **466 s**; achieved boolq 83.33% (25/30) — a wide-CI
  point estimate, explicitly not the 79.8% target.)

### 5. Why the "proof-of-life / not comparable" outcome occurred — EXACT, GROUNDED root cause

**This is issue R2R-001.** The executor is forced to run the whole reproduction inside a **single
synchronous `docker_run` call whose timeout is hardcoded to 3600 s and cannot be raised by the agent.**

- The MCP `docker_run` tool exposes only `tag`/`command`/`workdir` and calls the helper **without passing a
  timeout**, so it falls back to the default:

  ```347:348:repo2reproduce/tools/mcp_server.py
  async def docker_run(args: dict[str, Any]) -> dict[str, Any]:
      res = amd_tools.docker_run(args["tag"], args["command"], workdir=args.get("workdir"))
  ```

  ```139:167:repo2reproduce/tools/amd.py
  def docker_run(
      tag: str,
      command: str,
      gpus: bool = True,
      timeout: int = 3600,
      ...
      proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
  ```

  If the container runs longer than 3600 s, `subprocess.run` raises `TimeoutExpired` and the call fails.

- The executor prompt then mandates a single foreground call and explicit workload reduction:

  ```12:15:repo2reproduce/prompts/executor.md
  Instead, run the training-and-evaluation SYNCHRONOUSLY in a SINGLE docker_run call, in the
  foreground, and pass a large `timeout` (seconds) sized to the job. ... Reduce the
  workload first (fewer epochs/samples/runs) so it fits comfortably in one call, and say so.
  ```
  (Note: the prompt says "pass a large timeout", but the tool provides **no way to pass one** — the
  instruction is unachievable, so the only remaining lever is shrinking the job.)

- Reinforced by seeded memory:

  ```72:72:repo2reproduce/amd_knowledge.py
      "reduce epochs/samples so it fits one call.",
  ```

**Consequence:** a full-parity GraLoRA run (>1 h) would be killed at 3600 s, so the agent deliberately
shrinks the experiment to fit and honestly flags the result as non-comparable. This is a **tooling/timeout
limitation, not a GPU/memory limitation.**

**Why iTransformer did NOT hit this:** its full 10-epoch protocol finished in ~50 s (well under 3600 s), so
no reduction was needed and it matched the paper exactly (0.16% gap).

### 6. Other issues observed this run (worked around by the agent, worth tracking)
- **Two failed Docker builds:** a stale `sympy==1.12` transitive pin silently downgraded ROCm torch to a
  CUDA `cu121` wheel; surfaced/fixed by locking the torch stack via `PIP_CONSTRAINT`.
- **deepspeed dropped** (no py3.12 dist on ROCm) → single-GPU plain-python launch; **flash-attn absent** →
  eager attention.
- **First training attempt crashed** because `transformers` auto-enabled `wandb` → fixed with
  `WANDB_DISABLED=true`.
- **Data gap:** eval `test.json` per-task files are not shipped and have no repo download script → sourced
  from `AGI-Edgerunners/LLM-Adapters`.

### 7. Fix ideas (NOT yet applied — for later)
- Expose a `timeout` parameter on the `docker_run` MCP tool (schema + pass-through in `mcp_server.py`), and
  raise the `amd.py` default well beyond 3600 s (or make it unbounded/configurable via env, e.g.
  `R2R_DOCKER_RUN_TIMEOUT`).
- Relax the `executor.md` + `amd_knowledge.py` "reduce epochs/samples" instruction so the agent attempts
  full parity when the job fits the (raised) timeout, only reducing as a documented last resort.
- Consider streaming/log-to-file for long `docker_run`s instead of the current `out[-8000:]` tail truncation
  (`tools/amd.py:169`) so long runs remain observable.

---

## 2026-08-31 — Top-15 ICML 2025 batch late failures — `p2p` and `topobenchmark`

- **Batch dir:** `runs/batch-top-15-20260831-003754/`
- **Mode:** `--amd --execute`, parallel runner on physical GPUs `0,1,2,3`
- **Final batch outcome:** 15 jobs completed; 7 runner-level succeeded, 8 runner-level failed.

### 1. `p2p` — mvandenhi/p2p — execution reached but no paper metric

- **Run dir:** `runs/batch-top-15-20260831-003754/runs/10-p2p/`
- **Assigned GPU:** physical GPU `0` (`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`)
- **Selected experiment:** P2P on CIFAR-10.
- **Target result:** CIFAR-10 P2P accuracy `94.45 ± 0.29` from Table 1, recorded in
  `runs/batch-top-15-20260831-003754/runs/10-p2p/phases/executor.json:8`.
- **Outcome:** failed / not reproduced. The executor phase itself completed, but reported
  `executed=false`, `match=false`, and no clean achieved metric.
- **Issue(s) found:** full protocol infeasible under the current harness; DataLoader shared-memory
  limitation; evaluation crash.
- **Root cause (grounded):**
  - Full protocol was not run; only a one-epoch pilot was run:
    `runs/batch-top-15-20260831-003754/runs/10-p2p/phases/executor.json:5-7`.
  - The pilot produced no comparable paper metric because test-set evaluation aborted with
    `corrupted size vs. prev_size` / glibc heap corruption:
    `runs/batch-top-15-20260831-003754/runs/10-p2p/phases/executor.json:7`.
  - The executor measured about `787s` per epoch with `workers=0`; the full `10 seeds × 100 epochs`
    protocol is estimated at about `218h` of training alone:
    `runs/batch-top-15-20260831-003754/runs/10-p2p/phases/executor.json:11`.
  - The runner/harness could not honor the planned `--ipc=host` / shm settings for `docker_run`,
    so DataLoader workers crashed and the run fell back to the slow `workers=0` path:
    `runs/batch-top-15-20260831-003754/runs/10-p2p/phases/verifier.json:8-9`.
- **Fix idea (not yet applied):** give execution containers controllable `--ipc=host` or shm-size
  support, persist the CIFAR dataset volume across attempts, and investigate the `fast-slic` /
  NumPy 2 / Python 3.12 glibc crash before attempting a full multi-seed run.

### 2. `topobenchmark` — geometric-intelligence/topobenchmark — AMD build verification failed

- **Run dir:** `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/`
- **Assigned GPU:** physical GPU `3`; this batch was launched before the runner GPU-env fix, so the
  parent process inherited `ROCR_VISIBLE_DEVICES=3` and `HIP_VISIBLE_DEVICES=3`. The build-verifier
  later tested commands that explicitly set `HIP_VISIBLE_DEVICES=0` inside the container after the
  `ROCR_VISIBLE_DEVICES=3` physical mask.
- **Selected experiment:** TopoTune/CWN MUTAG Table 2 protocol.
- **Target result:** Table 2 CWN/MUTAG values, including `84.26 ± 8.19` ensemble and
  `81.70 ± 5.34` one-Hasse, recorded in
  `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/phases/verifier.json:6`.
- **Outcome:** build failed; no AMD/ROCm build succeeded and no training/evaluation was executed.
- **Issue(s) found:** ROCm source build of PyG compiled dependencies did not converge within the
  build-verifier budget.
- **Root cause (grounded):**
  - Build-verifier status is `failed` with `phase budget exhausted before a valid output was produced`:
    `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/phases/build-verifier.json:2-6`.
  - The final attempted Docker run failed building installable wheels for `torch-scatter` and
    `torch-sparse`:
    `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/amd_run/docker_logs/docker_run_20260831-055822.log:781-788`.
  - The concrete HIP compile error in `torch-sparse` was a ROCm warp-sync mask assertion
    (`The mask must be a 64-bit integer`) while compiling for `gfx950`:
    `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/amd_run/docker_logs/docker_run_20260831-055822.log:672-690`.
  - A narrower prior probe had shown `torch-scatter`, `torch-sparse`, and `torch-cluster` can build and
    import when constrained for `gfx950`:
    `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/amd_run/docker_logs/docker_run_20260831-054839.log:1-11`.
- **Fix idea (not yet applied):** rerun only if the Dockerfile pins `PYTORCH_ROCM_ARCH=gfx950`,
  caps `MAX_JOBS`, and uses the canonical dependency recipe captured in
  `runs/batch-top-15-20260831-003754/runs/15-topobenchmark/amd_run/Dockerfile`; otherwise treat this
  repo as high-risk for ROCm source-build failures.
