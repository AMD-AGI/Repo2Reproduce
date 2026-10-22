# Repo2Reproduce

Repo2Reproduce is a multistep agent based on Claude-code SDK that reproduces a paper's result from its official repository: it selects a target, reconciles paper and code settings, builds a Docker environment, runs the experiment deterministically, and compares the metric with the paper.

## Documentation

- [BENCHMARK.md](BENCHMARK.md): host setup, `.env`, run outputs, and how to read a verdict.
- [PORTING.md](PORTING.md): how CUDA kernels are ported to ROCm.
- [dataset/README.md](dataset/README.md): the 22 benchmark papers.

## Quick start

```bash
scripts/setup.sh --install-claude    # check the host, install, create .env
scripts/list_papers.sh               # ids, names, porting modes, targets
scripts/run_one.sh cogs --gpu 0      # one paper
scripts/run_all.sh --gpu 0 --detach  # all papers, sequentially
scripts/summarize.sh                 # status and verdict of every run
```

## Run one paper

```bash
scripts/run_one.sh 9 --gpu 0                         # by id
scripts/run_one.sh convolutional-diffusion --gpu 0   # by name
scripts/run_one.sh --paper <paper-url> --repo <repo-url> --gpu 0
scripts/run_one.sh --recover "$R2R_RUNS_ROOT/<run-id>" --gpu 0
```

| Flag | Description |
| --- | --- |
| `--gpu N` | Physical AMD GPU index from `rocm-smi` (default `0`). |
| `--target TEXT` | Result to reproduce; overrides the dataset target. |
| `--no-target` | Ignore the dataset target; the agent picks a result. |
| `--porting MODE` | `none`, `rocm`, or `force`; overrides the dataset mode. |
| `--timeout SECONDS` | Limit for the final execution only (default from the dataset, else `14400`). |
| `--runs-root DIR` | Where runs go (default `$R2R_RUNS_ROOT`, else `~/r2r-runs`). |
| `--run-id ID` | Run directory name (default `<id>-<name>-<timestamp>`). |
| `--prepare-only` | Build and freeze the run, but do not execute it. |
| `--detach` | Run in the background; log at `<runs-root>/<run-id>.log`. |
| `--dry-run` | Print the `r2r` command and exit. |
| `--recover DIR` | Resume an interrupted run from its first incomplete stage. |
| `-- ARGS` | Pass the remaining arguments to `r2r`. |

## Run all papers sequentially

```bash
scripts/run_all.sh --gpu 0 --detach
scripts/run_all.sh --gpu 0 --only 5,9,nsnquant
scripts/run_all.sh --gpu 1 --skip 10,refersplat
```

| Flag | Description |
| --- | --- |
| `--gpu N` | GPU for every run in the batch (default `0`). |
| `--only LIST` | Comma-separated ids or names to run. |
| `--skip LIST` | Comma-separated ids or names to leave out. |
| `--no-targets` | Ignore every dataset target. |
| `--prepare-only` | Freeze each run without executing it. |
| `--stop-on-error` | Stop at the first run that does not exit `0`. |
| `--runs-root DIR`, `--detach`, `--dry-run`, `-- ARGS` | Same as for `run_one.sh`. |

## Target

Name a table, row, or figure, not hyperparameters:

```bash
scripts/run_one.sh cogs --gpu 0 --target "Reproduce Table 2, the d=71 modular-addition row."
```

## Porting

| `--porting` | Behavior |
| --- | --- |
| `none` | No porting policy. |
| `rocm` | Port CUDA code with HIPIFY and check numerical parity (`--rocm-porting`). |
| `force` | The paper's own CUDA kernel must run on ROCm (`--rocm-porting --force-kernel-port`). |

```bash
scripts/run_one.sh sageattention --gpu 0 --porting force
```

Run `r2r --help` for the options of the underlying command.
