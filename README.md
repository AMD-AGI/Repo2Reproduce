# Repo2Reproduce

Repo2Reproduce is a Claude Code multistep agent that:

1. parses a research paper and clones its official repository;
2. selects a requested or central reproducible result;
3. reconciles paper requirements with the repository's executable configuration;
4. prepares and smoke-tests an isolated Docker environment;
5. freezes an execution manifest and runs it deterministically; and
6. compares the extracted metric with the paper target.

The working agent is implemented entirely under `repo2reproduce/`.

See [HOWTOUSE.md](HOWTOUSE.md) for requirements, installation, configuration, example
runs, recovery, and generated artifacts.

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
