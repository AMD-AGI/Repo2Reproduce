# Repo2Reproduce

Repo2Reproduce is a Claude Code multistep agent that:

1. parses a research paper and clones its official repository;
2. selects a requested or central reproducible result;
3. reconciles paper requirements with the repository's executable configuration;
4. prepares and smoke-tests an isolated Docker environment;
5. freezes an execution manifest and runs it deterministically; and
6. compares the extracted metric with the paper target.

The working agent is implemented entirely under `repo2reproduce/`.

See [BENCHMARK.md](BENCHMARK.md) for the step-by-step setup and the bash scripts that run
the 22-paper benchmark in [`dataset/`](dataset/README.md), and [HOWTOUSE.md](HOWTOUSE.md)
for the `r2r` command reference: configuration, recovery, and generated artifacts.

Quick start:

```bash
scripts/setup.sh --install-claude          # check the host, install, create .env
scripts/list_papers.sh                     # the 22 benchmark papers
scripts/run_one.sh cogs --gpu 0            # one paper
scripts/run_one.sh cogs --gpu 0 --target "Reproduce Table 2, the d=71 modular-addition row."
scripts/run_all.sh --gpu 0 --detach        # all 22 papers, sequentially
scripts/summarize.sh                       # status and verdict of every run
```

Basic invocation:

```bash
r2r <paper-url> <repository-url> \
  --target "<table, row, or result to reproduce>" \
  --gpu 0
```

The equivalent module command is:

```bash
python -m repo2reproduce <paper-url> <repository-url> \
  --target "<table, row, or result to reproduce>" \
  --gpu 0
```

Run `r2r --help` for all options.
