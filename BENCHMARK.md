# Running the 22-paper benchmark

This guide covers everything from a bare AMD ROCm host to reading the results. Every step
runs through a bash script in `scripts/`; the papers live in `dataset/papers.tsv`.

| Script | Purpose |
| --- | --- |
| `scripts/setup.sh` | Check the host, create `.venv`, install, create `.env`, verify. |
| `scripts/list_papers.sh` | List the dataset with ids, names, porting mode, and default target. |
| `scripts/run_one.sh` | Reproduce one paper (by dataset id/name, or by URLs), or recover a run. |
| `scripts/run_all.sh` | Reproduce all 22 papers sequentially (or a subset). |
| `scripts/summarize.sh` | Table of status, verdict, paper value, and reproduced value per run. |

Every script prints its usage with `--help`. The [dataset README](dataset/README.md) lists
the papers and the file format. [HOWTOUSE.md](HOWTOUSE.md) is the reference for the
underlying `r2r` command.

## 1. Set up the environment

### 1.1 System packages (needs root, once per host)

Repo2Reproduce runs on Linux with AMD Instinct GPUs. On Ubuntu 22.04/24.04:

```bash
sudo apt update
sudo apt install -y git curl perl python3 python3-venv python3-pip openjdk-17-jre-headless
```

| Requirement | Why |
| --- | --- |
| Python 3.10–3.13 with `venv` | Runs the pipeline. |
| Git, curl | Clone repositories, download papers. |
| Java 11+ | OpenDataLoader PDF, the only paper parser. |
| Perl | The pinned official HIPIFY used by `--rocm-porting` runs. |
| Docker | Every experiment runs in its own container. |
| ROCm kernel driver (`/dev/kfd`, `/dev/dri`) | GPU access from the containers. |
| Claude Code CLI (`claude`) | The agent engine. |
| Disk space | Each run holds the repository, model/dataset caches, and outputs; budget tens of GB per paper. |

Install Docker by following <https://docs.docker.com/engine/install/>, then let your user
run it without `sudo`:

```bash
sudo usermod -aG docker "$USER"    # log out and back in afterwards
docker info                        # must succeed without sudo
```

The ROCm driver must already be installed (`rocm-smi` lists the GPUs). The containers
bring their own ROCm user-space, so the host only needs the driver.

### 1.2 Clone and install

```bash
git clone <repo-url> Repo2Reproduce
cd Repo2Reproduce
scripts/setup.sh --install-claude --with-tests
```

`setup.sh`:

1. checks every prerequisite above and prints the fix for anything missing;
2. installs the Claude Code CLI when it is missing (`--install-claude`);
3. creates `.venv` and runs `pip install -e .[dev]`;
4. creates `.env` from `.env.example` if it does not exist;
5. checks `r2r --help` and, with `--with-tests`, runs the unit tests.

It exits non-zero while any check fails; fix the reported item and run it again. Use
`--python python3.12` when the default `python3` is outside 3.10–3.13.

### 1.3 Configure `.env`

Edit `.env` in the checkout. With the public Anthropic API:

```dotenv
R2R_PROVIDER=anthropic
R2R_AUTH_MODE=api_key
ANTHROPIC_API_KEY=<your-key>
R2R_MODEL_REASONING=<model-id>
R2R_JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
R2R_RUNS_ROOT=/data/<you>/r2r-runs
```

With an Anthropic-compatible gateway, use `R2R_AUTH_MODE=custom_headers` with
`R2R_BASE_URL`, `R2R_AUTH_HEADERS`, and `R2R_AUTH_KEY` instead; `.env.example` shows the
format and [HOWTOUSE.md §3](HOWTOUSE.md#3-configure-the-provider) lists all settings.

`R2R_RUNS_ROOT` is where run directories go. Put it **outside the checkout** on a large
disk. When it is unset the scripts use `~/r2r-runs`. `.env` is git-ignored; never commit it.

Papers whose models are gated on Hugging Face (for example LLaMA2-7B for NSNQuant) need a
token in the environment before launch: `export HF_TOKEN=<token>`.

Re-run `scripts/setup.sh` after editing `.env`; it warns about missing provider settings.

## 2. Run

`--gpu` is the physical GPU index as listed by `rocm-smi`. Inside each container that GPU
is device 0. Runs take from tens of minutes to many hours; `--timeout-seconds` (default
14400, set per paper in the dataset) bounds only the final execution, not the agent
stages before it.

### 2.1 List the papers

```bash
scripts/list_papers.sh
```

### 2.2 One paper

By dataset id or name, using the dataset's porting mode and default target:

```bash
scripts/run_one.sh 9 --gpu 0
scripts/run_one.sh convolutional-diffusion --gpu 0     # same paper, by name
```

Run in the background and come back later:

```bash
scripts/run_one.sh nsnquant --gpu 1 --detach
tail -f "$R2R_RUNS_ROOT"/21-nsnquant-*.log
```

A paper that is not in the dataset:

```bash
scripts/run_one.sh --paper https://arxiv.org/abs/<id> --repo https://github.com/<org>/<repo> --gpu 0
```

Print the exact `r2r` command without running it with `--dry-run`. Arguments after `--`
are passed to `r2r` unchanged, for example `-- --max-execution-repairs 1`.

### 2.3 One paper with a specific target experiment

`--target` names the result to reproduce and overrides the dataset's target column. Name
the table, row, column, or figure; leave hyperparameters out, since discovering them is
the configuration audit's job.

```bash
scripts/run_one.sh cogs --gpu 0 \
  --target "Reproduce Table 2, the d=71 modular-addition row."

scripts/run_one.sh convolutional-diffusion --gpu 0 \
  --target "Reproduce Table 1, the CIFAR10 ResNet ELS correlation."

scripts/run_one.sh sageattention --gpu 2 --porting force \
  --target "Reproduce the attention accuracy (cosine similarity and relative L1 against full-precision attention) of the official SageAttention INT8 CUDA kernel, ported to ROCm."
```

`--no-target` ignores the dataset target and lets the agent choose a central, locally
reproducible result. `--porting none|rocm|force` overrides the dataset's porting mode
(`rocm` = `--rocm-porting`, `force` = `--rocm-porting --force-kernel-port`; see
[PORTING.md](PORTING.md)). `--prepare-only` stops after the environment is built and the
execution manifest is frozen, without running the experiment.

### 2.4 All 22 papers, sequentially

```bash
scripts/run_all.sh --gpu 0 --detach
tail -f "$R2R_RUNS_ROOT"/batch-*/batch.log
```

Without `--detach` the batch runs in the foreground; use `tmux` or `screen` so it survives
a closed terminal. Papers run one after another on the same GPU. A failed or blocked paper
does not stop the batch unless you pass `--stop-on-error`.

Useful variants:

```bash
scripts/run_all.sh --gpu 0 --only 5,9,nsnquant        # a subset, in dataset order
scripts/run_all.sh --gpu 0 --skip 10,refersplat       # everything except these
scripts/run_all.sh --gpu 0 --no-targets               # agent chooses every result
scripts/run_all.sh --gpu 0 --prepare-only             # build and freeze, do not execute
scripts/run_all.sh --dry-run                          # print the 22 commands only
```

To use several GPUs, start one batch per GPU on disjoint subsets:

```bash
scripts/run_all.sh --gpu 0 --only 1,2,3,4,5,6,7,8,9,10,11 --detach
scripts/run_all.sh --gpu 1 --only 12,13,14,15,16,17,18,19,20,21,22 --detach
```

### 2.5 Resume an interrupted run

A run that stopped (crash, reboot, interrupted agent) resumes from its first incomplete
stage. Paper, repository, target, and porting mode come from the run directory:

```bash
scripts/run_one.sh --recover "$R2R_RUNS_ROOT/9-convolutional-diffusion-20261022-101500" --gpu 0
```

To retry a whole paper from scratch, run it again; each run gets a new timestamped id.

## 3. Outputs and where they are

```text
$R2R_RUNS_ROOT/
├── <id>-<name>-<UTC timestamp>/        one directory per run (layout below)
├── <id>-<name>-<UTC timestamp>.log     console output of that run
└── batch-<UTC timestamp>/              one directory per run_all.sh batch
    ├── batch.log                       everything the batch printed
    ├── runs.txt                        the run directories, one per line
    ├── exit_codes.tsv                  id, name, run directory, r2r exit status
    └── summary.txt                     the final summary table
```

Most useful files inside a run directory:

| File | What it holds |
| --- | --- |
| `final_report.md` | Start here: the human-readable result, verdict, assumptions, deviations, gaps. |
| `final_report.json` | The same report, machine-readable. |
| `metadata.json` | Overall status and per-stage status/attempts; check it while a run is in progress. |
| `target.json` | The paper result that was selected and frozen. |
| `configuration_audit.json` | Paper-vs-code reconciliation of every scientific setting. |
| `execution_manifest.json` | The frozen launch: image, command, expected outputs, metric extractor. |
| `execution_result.json` | Execution status, runtime, and the extracted metric. |
| `assumptions.json` | Every value the paper did not specify and the fallback used. |
| `workspace/execution/launch.sh` | The exact command run inside Docker. |
| `logs/executor.log` | Full log of the experiment itself. |
| `logs/agent_messages.jsonl` | Every agent message and tool call (debugging). |
| `artifacts/full-run/` | Raw outputs written by the experiment. |
| `execution_repairs/` | Diagnosis and patch for each automatic repair, if any ran. |

[HOWTOUSE.md §8](HOWTOUSE.md#8-run-artifacts) lists every file. Containers run as root, so
files under `artifacts/` belong to root on the host; delete them through Docker as
described there.

## 4. Interpret the results

### 4.1 Summary table

`run_all.sh` prints the summary at the end. Print it any time, for any runs:

```bash
scripts/summarize.sh                                   # every run under the runs root
scripts/summarize.sh --batch "$R2R_RUNS_ROOT"/batch-20261022-101500
scripts/summarize.sh "$R2R_RUNS_ROOT"/9-convolutional-diffusion-*
scripts/summarize.sh --tsv > results.tsv               # for a spreadsheet
```

Example:

```text
run                                     status     execution  verdict         paper  reproduced          rel.err  stopped at
--------------------------------------  ---------  ---------  --------------  -----  ------------------  -------  ----------
9-convolutional-diffusion-20261021-...  completed  succeeded  match           0.95   0.9534522891044617  0.36%
21-nsnquant-20261022-...                completed  timed_out  run_incomplete  ≈3×...
```

| Column | Meaning |
| --- | --- |
| `status` | Run status from `metadata.json` (below). |
| `execution` | Status of the deterministic execution: `succeeded`, `failed`, `timed_out`, `blocked`. |
| `verdict` | The final verdict (below); empty until finalization has run. |
| `paper` / `reproduced` | Paper target value and the value this run measured. |
| `rel.err` | Relative error in percent, when the comparison is a scalar. |
| `stopped at` | For unfinished runs, the first stage that has not succeeded and its state. |

### 4.2 Run status

| `status` | Meaning | What to do |
| --- | --- | --- |
| `completed` | All stages ran and a final report exists. | Read the verdict. |
| `prepared` | `--prepare-only` finished; nothing was executed. | Run again without `--prepare-only`. |
| `blocked` | An agent proved the result cannot run here (e.g. hardware or data not available). The reason is `blocked_reason` in `metadata.json`. | Read the reason; sometimes a different `--target` or `--porting` mode helps. |
| `failed` | A stage errored. `error` in `metadata.json` says which. | Fix the cause (key, disk, Docker) and `--recover`. |
| `initialized`, `recovering` | Still in progress, or interrupted without cleanup. | Wait, or `--recover` if no `r2r` process is alive. |
| `recovery_failed` | A `--recover` attempt failed. | Read `recovery_error`, then recover again. |

The `r2r` exit status, recorded in `exit_codes.tsv`, is `0` for success (or prepared),
`1` when the execution did not succeed, `2` when the run was blocked, and anything else
for a pipeline error. `64` means the script was called incorrectly.

### 4.3 Verdict

The finalization agent compares the reproduced value with the paper's value under a
**5 % relative-error tolerance**, and checks that the paper's claimed advantage over its
strongest baseline still holds.

| Verdict | Meaning |
| --- | --- |
| `match` | Reproduced within 5 % of the paper and, where the paper claims to beat a baseline, still better than that baseline. |
| `miss` | Comparable, but outside 5 % or the baseline advantage is lost. A large favorable deviation is also a miss. |
| `run_incomplete` | The execution did not succeed, produced no value, or did not complete the full protocol (seeds, splits, samples). Not comparable. |
| `needs_review` | Comparable in principle but ambiguous: a figure read-off, a vector or interval without a defined rule, a zero paper value, unit mismatch, or two interpretations that disagree. A person must decide. |

Before trusting any verdict, read `final_report.md`:

- **Selected result** – confirm it is the table cell or figure you intended.
- **Calculation** – the comparison with the numbers substituted.
- **Paper baseline** – which competing method was checked, and whether it applies.
- **Assumptions and fallbacks** – values the paper did not state, and where the fallback
  came from (code default, config, README). Many assumptions weaken a match.
- **Deviations** – where the run differs from the paper or the repository's defaults
  (for example a paper/code conflict resolved in the paper's favor, or a ROCm port).
- **Gaps** – what could not be grounded or checked.

Hardware matters for some results. Speed and memory numbers measured on an AMD MI300-class
GPU are not expected to equal numbers from an NVIDIA A100 or RTX 4090; a `miss` there may
be the hardware rather than the method. For ported kernels (`--porting force`), the
porting section of the report and `artifacts/env/` record the numerical parity check
between the port and its reference.
