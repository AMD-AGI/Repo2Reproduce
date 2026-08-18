# GraLoRA on AMD (MI350X) — Agent Run Debrief

## Run details
- **Paper:** GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning (NeurIPS 2025 Spotlight) — https://arxiv.org/abs/2505.20355
- **Code:** https://github.com/SqueezeBits/GraLoRA
- **Run directory:** `runs/20260726-015638-https-github.com-SqueezeBits-GraLoRA/`
- **Mode:** `--amd --execute` (single AMD gfx950 / MI350X, ROCm 7.2.4), no budget cap, 12h run timeout
- **Experiment:** Table 3 — Commonsense Reasoning, Qwen2.5-1.5B, rank 64, `gralora_k=4`, LionW, 2 epochs, global batch 192
- **LLM routing:** AMD internal gateway — reasoning `Claude-Opus-4.8`, mapping `Claude-Sonnet-5`

## Final result
- **Achieved:** **78.98%** average over 8 commonsense tasks vs paper **79.8%** (delta **−0.82 pt**).
- GraLoRA (78.98%) still beats the paper's LoRA baseline (78.7%) — **core claim reproduced at full parity**.
- Full 170,420 examples, 2 epochs, exact hyperparameters, all 8 full eval sets. Train ~1h48m; full pipeline ~2h23m.
- Autonomous after launch. Estimated cost ~$43.

---

## ✅ Wins from the LLM (Claude) reasoning inside the executor subagent
*These emerged from the model's agency — not hard-coded in r2r.*
- **Self-diagnosed the failure** — read the NaN + `train_loss 5.86e33`, chose to `cat` the Lion optimizer + training code to rule out a code bug, and attributed it to the environment (`TunableOp`).
- **Autonomously recovered** — flipped env vars (`TunableOp=0`) and **relaunched a fresh clean run**, with no human input.
- **Fixed catastrophic eval speed** — noticed default `batch_size=1` (140-hr ETA + coredumps) and raised it to `16` on its own; recognized it's padding-invariant so accuracy is unaffected.
- **Sound hardware judgment** — reasoned the 1.5B model fits one GPU and preserved global batch 192 via gradient accumulation instead of needing multi-GPU.

## 🧰 Wins from the r2r framework (the scaffolding that made the above possible)
- **`docker_run` as a tool** — gave the model a clean, repeatable way to run/probe containers.
- **Configurable timeout + graceful return** *(a limitation already fixed)* — **before** the fix, `docker_run` was capped at 3600s and **crashed the tool call on timeout** (raised `TimeoutExpired`), so the model never saw the failure and could only shrink jobs to "proof-of-life." **After** the fix it's raised to 12h and **returns `{timed_out: true}` + a streamed `log_file`** instead of crashing — which is precisely what let the model *observe* the NaN/coredumps and diagnose them.
- **Persistent memory** — captured the durable `TunableOp=0` lesson for future runs.
- **Grounding discipline** — verbatim quote + flag verification kept the reproduction honest.

## ❌ Negatives (r2r's fault, not the model's)
- **Self-inflicted the NaN.** `PYTORCH_TUNABLEOP_ENABLED=1` came from r2r's **own seeded AMD knowledge** (`amd_knowledge.py`) + executor prompt as a *speed* tweak; on gfx950/bf16 it produced a bad GEMM kernel → overflow → poisoned Lion momentum. Not from the paper or repo.
- **Cost a full wasted attempt** — the broken run trained ~2.5h and ran to the 12h timeout before the model could diagnose and restart.

## 🔧 Limitations & improvements
- **Optimize the workload up front** — eval `batch_size` and the `TunableOp` risk were knowable pre-launch; apply safe, source-preserving defaults on attempt #1.
- **Fix the contradictory knowledge base** — seed still says `TunableOp=1` while learned memory says `=0`; flip the default to `=0` so the mistake can't recur.
- **Fail fast on divergence** — add a NaN/Inf early-stop so a diverged run aborts in minutes, not 12 hours.

---
**One-line takeaway:** The recovery intelligence came from the **Claude model**; r2r's role was providing the tools and the recently-fixed timeout plumbing that let that intelligence act — while one avoidable failure (`TunableOp=1`) was seeded by r2r itself.
