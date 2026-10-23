# Repo2Reproduce

Repo2Reproduce is a multistep agent based on Claude-code SDK that reproduces a paper's result from its official repository: it selects a target, reconciles paper and code settings, builds a Docker environment, runs the experiment deterministically, and compares the metric with the paper. It runs on AMD Instinct GPUs and ROCm.

## How it works

Each run goes through these stages, and each stage's output is saved in the run directory:

1. **Bootstrap.** Download and parse the paper, and clone the official repository.
2. **Target selection.** Freeze the result to reproduce: the one you name, or a central,
   locally reproducible result the agent picks.
3. **Configuration audit.** Reconcile every setting the paper states (dataset, model,
   seeds, metric) with what the code actually runs.
4. **Environment preparation.** Build a Docker image, run a short smoke test, and freeze an
   execution manifest.
5. **Execution.** Run the manifest deterministically. A failed run gets up to two automatic
   repairs.
6. **Report.** Compare the measured value with the paper and give a verdict: `match`,
   `miss`, `run_incomplete`, or `needs_review`.

## Requirements

- Linux with AMD Instinct GPUs and the ROCm kernel driver (`rocm-smi` lists the GPUs)
- Docker, usable without `sudo`
- Python 3.10–3.13, Git, curl, Perl, and Java 11+
- The Claude Code CLI (`setup.sh --install-claude` installs it)
- Access to Claude, through the Anthropic API or an Anthropic-compatible gateway
- Tens of GB of disk per paper, for repositories, caches, and outputs

[BENCHMARK.md](BENCHMARK.md#1-set-up-the-environment) has the install commands.

## Installation

```bash
git clone <repo-url> Repo2Reproduce
cd Repo2Reproduce
scripts/setup.sh --install-claude --with-tests
```

`setup.sh` checks the host, creates `.venv`, installs the package, creates `.env` from
`.env.example`, and runs the unit tests. It prints how to fix anything missing.

## Configuration

Settings are read from `.env` in the checkout. [`.env.example`](.env.example) lists every
setting with comments; copy it and fill in your values:

```bash
cp .env.example .env
```

| Setting | Description |
| --- | --- |
| `R2R_AUTH_MODE` | `api_key` for the Anthropic API, or `custom_headers` for a gateway. `R2R_PROVIDER` is a label. |
| `ANTHROPIC_API_KEY` | Your key, with `api_key` mode. |
| `R2R_BASE_URL`, `R2R_AUTH_HEADERS`, `R2R_AUTH_KEY` | Gateway URL, header template, and key, with `custom_headers` mode. |
| `R2R_MODEL_REASONING` | Model ID for every agent stage. |
| `R2R_JAVA_HOME` | Java 11+ installation, used to parse papers. |
| `R2R_RUNS_ROOT` | Where runs are written. Put it outside the checkout on a large disk. |
| `R2R_MAX_BUDGET_USD` | Spending cap in US dollars for a whole run, across all stages (default `100`). |
| `R2R_MAX_TURNS` | Turn limit per agent session (default `200`). |

Papers with gated Hugging Face models, such as LLaMA2-7B for NSNQuant, also need
`export HF_TOKEN=<token>` before launch.

## Run one paper

Select a paper from the [dataset](dataset/README.md) by id or name. The dataset supplies
the paper and repository; without `--target`, the agent picks the result.

```bash
scripts/list_papers.sh                                # ids, names, and paper links
scripts/run_one.sh 9 --gpu 0                          # by id
scripts/run_one.sh convolutional-diffusion --gpu 0    # the same paper, by name
scripts/run_one.sh nsnquant --gpu 1 --detach          # in the background
```

Any paper with public code works too:

```bash
scripts/run_one.sh --paper https://arxiv.org/abs/<id> --repo https://github.com/<org>/<repo> --gpu 0
```

| Flag | Description |
| --- | --- |
| `--gpu N` | Physical AMD GPU index from `rocm-smi` (default `0`). |
| `--target TEXT` | Result to reproduce (default: the agent chooses). |
| `--porting MODE` | `none` (default), `rocm`, or `force`. |
| `--max-budget-usd USD` | Spending cap for the whole run; overrides `R2R_MAX_BUDGET_USD`. |
| `--prepare-only` | Build and freeze the run without executing it. |
| `--detach` | Run in the background; the log is `<runs-root>/<run-id>.log`. |

## Run all papers

`run_all.sh` runs the papers one after another on one GPU. A failed paper does not stop the
batch.

```bash
scripts/run_all.sh --gpu 0 --detach                   # all 22 papers
scripts/run_all.sh --gpu 0 --only 5,9,nsnquant        # a subset
scripts/run_all.sh --gpu 0 --skip 10,refersplat       # all but these
```

To use two GPUs, start one batch per GPU on different papers:

```bash
scripts/run_all.sh --gpu 0 --only 1,2,3,4,5,6,7,8,9,10,11 --detach
scripts/run_all.sh --gpu 1 --only 12,13,14,15,16,17,18,19,20,21,22 --detach
```

| Flag | Description |
| --- | --- |
| `--gpu N` | GPU for every run (default `0`). |
| `--only LIST`, `--skip LIST` | Comma-separated ids or names to run, or to leave out. |
| `--max-budget-usd USD` | Spending cap for each paper's run. |
| `--stop-on-error` | Stop at the first paper that does not succeed. |
| `--detach` | Run the batch in the background. |

## Choose a target

`--target` names the result to reproduce. Name a table, row, or figure; leave out
hyperparameters, because the configuration audit finds them in the paper and code.

```bash
scripts/run_one.sh cogs --gpu 0 --target "Reproduce Table 2, the d=71 modular-addition row."
```

## Port CUDA kernels to ROCm

Some papers ship their contribution as a CUDA kernel. `--porting` controls how strictly
that code must be ported to ROCm:

| Mode | Behavior |
| --- | --- |
| `none` | No porting policy. |
| `rocm` | Port CUDA code with the official HIPIFY and check numerical parity. |
| `force` | The paper's own CUDA kernel must run on ROCm, through HIPIFY with fixes or a rewrite. |

```bash
scripts/run_one.sh sageattention --gpu 0 --porting force
```

See [PORTING.md](PORTING.md) for how porting works and what it achieved.

## Set a budget

`R2R_MAX_BUDGET_USD` in `.env`, or `--max-budget-usd` on the command line, caps the agent
cost of a whole run: target selection, audit, preparation, repairs, the final report, and
any recovery together. Each agent session gets only what the run has left, and the run
stops with status `failed` once the budget is spent. `budget.json` in the run directory
records the cost of each session.

```bash
scripts/run_one.sh cogs --gpu 0 --max-budget-usd 30
scripts/run_one.sh --recover "$R2R_RUNS_ROOT/<run-id>" --gpu 0 --max-budget-usd 45
```

When recovering, the budget is the new total for the run, including what it already spent.

## Read the results

```bash
scripts/summarize.sh                                  # every run under the runs root
scripts/summarize.sh --tsv > results.tsv              # for a spreadsheet
```

Each run directory has `final_report.md` with the selected result, the paper and
reproduced values, the verdict, and every assumption and deviation. See
[BENCHMARK.md](BENCHMARK.md#4-interpret-the-results) for what each status and verdict means.



