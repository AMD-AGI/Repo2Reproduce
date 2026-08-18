# Repo2Reproduce

Give it a **paper** (arXiv/PDF URL) and its **GitHub code**. It reads the *whole* paper
and the code, picks the single most reproducible experiment, extracts the exact
configuration, synthesizes the **grounded command(s)** to replicate the result, and — with
`--amd --execute` — **enables the repo on AMD ROCm GPUs and runs it**.

It produces **three core outputs** (Track 1):

1. **Selected experiment** — the *exact* paper text and table used to choose it.
2. **Configuration** — all settings, hyperparameters, data, model, seed, hardware,
   environment — with the paper value and the code default reconciled (conflicts and
   gaps flagged).
3. **Replication commands** — the ordered command(s) to run, with every flag verified
   against the repository's real argparse / config definitions.

With `--amd` it adds **Track 2** (ROCm enablement): it selects a ROCm base image, ports
dependencies, applies a HIP kernel patch when the repo has custom CUDA kernels, and emits
a **verified Dockerfile** (Output 4). **Only `--amd --execute`** actually **builds the
image and runs the reproduction on an AMD ROCm GPU**, capturing the achieved metric
(Output 5).

Everything is **evidence-grounded**: each claim is backed either by a verbatim paper
quote (with a page locator) or a real file path/flag in the code.

## How it works

A Claude Agent SDK orchestrator coordinates specialized subagents.

Two independent Markdown reviewers surround the main pipeline:

- **Pre-run verifier** — reads the paper and repo first, then writes `preflight.md`
  covering local reproducibility, data/setup identity, required comparisons, and blockers.
- **Final verifier** — reads `preflight.md`, the execution logs, and the final report, then
  writes `final_review.md` covering unsupported numbers, silent scope changes, and whether
  the conclusion is fair.

These reviews are advisory and intentionally do not use a strict JSON schema.

**Track 1 (always):**

| Phase | Subagent | Output |
|---|---|---|
| Read the full paper | `paper-analyst` | paper map (experiments, tables, hyperparams) |
| Map the repo | `code-cartographer` | entry points, CLI flags, configs, env |
| Choose the experiment | `experiment-selector` | **Output 1** |
| Extract & reconcile config | `config-extractor` | **Output 2** |
| Synthesize commands | `command-synthesizer` | **Output 3** |
| Audit grounding | `verifier` | gaps, assumptions, confidence |

**Track 2 (`--amd`):**

| Phase | Subagent | Output |
|---|---|---|
| Pick a ROCm base image | `base-image-selector` | `rocm/pytorch:*` tag |
| Port dependencies for ROCm | `env-porter` | ROCm-safe requirements |
| Migrate custom CUDA kernels | `kernel-migrator` | HIP-aware build patch |
| Build + verify the image | `build-verifier` | **Output 4** (verified Dockerfile) |
| Reconcile config/commands | `reconciler` | updated Outputs 2/3 |
| Run on the GPU (`--execute`) | `executor` | **Output 5** (achieved metric) |

**Paper reading is strict:** PDFs are parsed using **only**
[OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) — a
deterministic, local-first, Java-based parser (no torch, no GPU, so it never disturbs a
ROCm/CUDA ML environment).

## Requirements

- **Python 3.10–3.13** and **git**.
- The **`claude` engine binary** on PATH. The Python Agent SDK runs the agent loop by
  launching `claude` as a subprocess; without it the SDK raises `CLINotFoundError`.
- **Java 11+** (OpenDataLoader PDF shells out to `java`). Java 8 will fail with
  `UnsupportedClassVersionError` — see the install step below.
- **LLM access:** an `ANTHROPIC_API_KEY` (public API) **or** an Anthropic-format gateway
  key (see *Configure*).
- **For `--amd --execute` only:** an **AMD ROCm GPU host** with the **ROCm driver**
  loaded (`/dev/kfd` and `/dev/dri` present) and **Docker** installed and runnable by
  your user. Plan-only `--amd` does *not* need a GPU or Docker.

## Install

### 1. Clone and install the Python package

```bash
git clone <this-repo> Repo2ROCm
cd Repo2ROCm
python3 -m pip install -e .
```

### 2. Install the `claude` engine binary

```bash
curl -fsSL https://claude.ai/install.sh | bash   # installs to ~/.local/bin/claude
```

The orchestrator automatically prepends `~/.local/bin` to PATH at runtime (override the
location with `R2R_CLI_BIN_DIR` in `.env`). Verify with `claude --version`.

### 3. Install Java 17 (or any JRE 11+)

```bash
sudo apt-get install -y openjdk-17-jre-headless
```

> **Note:** if `sudo apt-get update` fails on an unrelated third-party repo (e.g. a stale
> `compute-cdn.amd.com` ROCm source returning `404 ... no longer has a Release file`),
> run the `install` command **on its own** (don't chain it with `&& update`) — the Ubuntu
> package lists are already present, so the install succeeds. If needed, disable the broken
> source in `/etc/apt/sources.list.d/` first.

Find the install path and note it for `.env` (the exact dir name varies by distro/arch):

```bash
ls /usr/lib/jvm/                                # e.g. java-17-openjdk-amd64
/usr/lib/jvm/<your-jdk>/bin/java -version       # should print 11+ (e.g. 17.x)
```

## Configure

**One file, one source of truth** — all config, keys, and secrets live in `.env`:

```bash
cp .env.example .env
```

Edit `.env`. At minimum set the LLM provider and the Java home:

**Public Anthropic API:**

```bash
R2R_PROVIDER=anthropic
R2R_AUTH_MODE=api_key
ANTHROPIC_API_KEY=sk-ant-...
R2R_JAVA_HOME=/usr/lib/jvm/<your-jdk>   # a Java 11+ home
```

**An Anthropic-format gateway with custom auth headers** (e.g. AMD's internal LLM gateway):

```bash
R2R_PROVIDER=amd
R2R_AUTH_MODE=custom_headers
R2R_BASE_URL=<your-gateway-anthropic-endpoint>
R2R_AUTH_HEADERS=<Your-Auth-Header>={key};user={user}
R2R_AUTH_KEY=<secret>                  # the secret, inline
R2R_AUTH_USER=<youruser>
R2R_MODEL_REASONING=<reasoning-model-id>
R2R_MODEL_MAPPING=<mapping-model-id>
R2R_JAVA_HOME=/usr/lib/jvm/<your-jdk>   # a Java 11+ home
```

The `{key}`/`{user}` placeholders in the headers are filled at runtime from `R2R_AUTH_KEY`
and `R2R_AUTH_USER` (defaults to `$USER`). Built-in fallback defaults (public Anthropic
API, `opus`/`sonnet` tiers, budget/turn limits) apply only when a var is unset.

### Verify your config before a full run

```bash
python3 -c "from repo2reproduce import config; print(config.active_provider_name()); print(config.provider_env())"
```

An empty dict `{}` is correct for the plain `anthropic` api_key case; a gateway prints
`ANTHROPIC_BASE_URL` + `ANTHROPIC_CUSTOM_HEADERS`.

## Run

Run from the repo root (the CLI auto-loads `.env` from the current directory). If the
`r2r` script isn't on PATH, use `python3 -m repo2reproduce.cli` instead.

### Track 1 — plan the reproduction and Track 2 — enable on AMD ROCm

```bash
# Plan only: pick a ROCm base image, port deps, emit a verified Dockerfile (no GPU needed):
r2r reproduce <paper_url> <repo_url> --amd

# Full: build the image AND run the reproduction on an AMD GPU (needs Docker + ROCm host):
r2r reproduce <paper_url> <repo_url> --amd --execute

# Scope reduction is forbidden by default. Opt in only with explicit human approval:
r2r reproduce <paper_url> <repo_url> --amd --execute --allow-scope-reduction
```

`--execute` implies `--amd`. In execute mode the tool runs containers with AMD GPU
access.

**Pin a run to one GPU** (e.g. on a multi-GPU host) — the value is forwarded into the
container:

```bash
HIP_VISIBLE_DEVICES=0 r2r reproduce <paper_url> <repo_url> --amd --execute
```

## Output

Artifacts land in `runs/<timestamp>-<repo>/`:

- `report.md` / `report.json` — selected experiment, configuration, grounded commands,
  and (with `--amd`) the ROCm Dockerfile + achieved metric.
- `preflight.md` — independent paper/repo feasibility review written before orchestration.
- `final_review.md` — independent review of the execution evidence and final conclusion.
- the parsed paper and the cloned repo.

Execute-mode reports also include `run_status`. Incomplete/cancelled runs exit non-zero
and are excluded from aggregate execution metrics. If the final SDK structured output is
less complete than an earlier `save_report` artifact, the saved report is preserved.

## Notes

- The safety hook blocks `rm -rf`, `git push`, `sudo`, and mass downloads. In plan mode it
  also blocks anything that would launch training/eval; `--execute` relaxes this so the
  reproduction can actually run inside the container.
- Tune cost/turn limits and model tiers in `.env`
  (`R2R_MAX_BUDGET_USD`, `R2R_MAX_TURNS`, `R2R_MODEL_REASONING`, `R2R_MODEL_MAPPING`).

