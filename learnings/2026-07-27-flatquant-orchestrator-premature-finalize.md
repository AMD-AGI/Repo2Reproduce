# r2r pipeline issue — orchestrator finalized before the pipeline ran

## Run
- **Paper:** FlatQuant: Flatness Matters for LLM Quantization (ICML 2025) — https://arxiv.org/abs/2410.09426
- **Code:** https://github.com/ruikangliu/FlatQuant
- **Run dir:** `runs/20260726-235556-https-github.com-ruikangliu-FlatQuant/`
- **Mode:** `--amd --execute` (background)
- **Outcome:** aborted early, nothing built or executed. Cost ~$5.39, exited 0 (looked successful).

## Issue
The orchestrator launched the first wave of subagents, got **1 of 3** back (`base-image-selector`), did **not** wait for the other two (`paper-analyst`, `code-cartographer` — still running, no errors), decided it "had enough," and emitted the **final JSON report** — skipping the remaining ~9 pipeline steps (experiment-selector, config-extractor, command-synthesizer, env-porter, kernel-migrator, build-verifier, reconciler, executor, verifier).

**Root cause:** the pipeline (step order + "converge only after both tracks finish") exists **only as prose in `prompts/orchestrator_amd.md`** — it is advisory, not enforced. One free-running orchestrator LLM drives everything; subagents run as async background tasks it is *supposed* to await. Nothing in code requires the steps to run in order, requires pending tasks to complete, or requires the executor's result before finalizing. A template-valid but empty report is accepted, written, and exits 0 → a silent "successful failure."

Non-deterministic: the same setup completed fully for GraLoRA (all 12 subagents ran); FlatQuant bailed after 3.

## Fix direction (later)
Add real enforcement, not just better wording:
- A `Stop`/pre-`save_report` guard that refuses to finalize when required outputs are missing (e.g. execute mode with no executor Output 5) and marks the run failed (non-zero exit), without overwriting a good report.
- Longer term: hoist orchestration into code (sequence/gate subagents in Python, i.e. a DAG/state-machine skeleton with LLM-in-the-node) instead of a prose recipe.
