# How to install and use Repo2Reproduce

## 1. Requirements

The host needs:

- Python 3.10–3.13;
- Git and curl;
- Java 11 or newer for OpenDataLoader PDF;
- the Claude Code `claude` executable;
- Docker and a working ROCm driver for GPU execution.

Verify the execution host:

```bash
python3 --version
git --version
java -version
claude --version
docker info
test -e /dev/kfd
test -d /dev/dri
```

Install Claude Code if it is unavailable:

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Install Repo2Reproduce

From a fresh clone:

```bash
git clone <repo-url> Repo2Reproduce
cd Repo2Reproduce
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Verify both supported entrypoints:

```bash
.venv/bin/r2r --help
.venv/bin/python -m repo2reproduce --help
```

## 3. Configure the provider

Create the local configuration file:

```bash
cp .env.example .env
```

For the public Anthropic API:

```dotenv
R2R_PROVIDER=anthropic
R2R_AUTH_MODE=api_key
ANTHROPIC_API_KEY=<your-key>
R2R_MODEL_REASONING=<available-model-id>
R2R_JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

For an Anthropic-compatible gateway:

```dotenv
R2R_PROVIDER=<provider-name>
R2R_AUTH_MODE=custom_headers
R2R_BASE_URL=<gateway-url>
R2R_AUTH_HEADERS=<header-name>={key}
R2R_AUTH_KEY=<gateway-key>
R2R_MODEL_REASONING=<available-model-id>
R2R_JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

Optional settings:

```dotenv
R2R_AUTH_USER=<gateway-user>
R2R_CLI_BIN_DIR=/home/<user>/.local/bin
R2R_MAX_TURNS=200
R2R_MAX_BUDGET_USD=100.0
```

Do not commit `.env`.

## 4. Run an example

The target should identify the paper result—such as a table, row, dataset, and method—
without supplying the hyperparameters that the configuration-audit stage is intended to
discover.

Example: reproduce the Table 2 `d=23` modular-addition row:

```bash
RUN_ID="cogs-d23-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/r2r \
  "https://arxiv.org/abs/2410.01779" \
  "https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset" \
  --run-id "$RUN_ID" \
  --runs-root /dev/shm/machaita/Repo2ReproduceData \
  --gpu 0 \
  --target "Reproduce Table 2, the d=23 modular-addition row." \
  --timeout-seconds 14400
```

The equivalent module invocation starts with:

```bash
.venv/bin/python -m repo2reproduce <paper-url> <repository-url> ...
```

To detach the run:

```bash
mkdir -p logs
nohup setsid .venv/bin/r2r <paper-url> <repository-url> \
  --run-id "<unique-run-id>" \
  --runs-root /dev/shm/machaita/Repo2ReproduceData \
  --gpu 0 \
  --target "<table, row, or result to reproduce>" \
  > "logs/<unique-run-id>.log" 2>&1 < /dev/null &
```

Always use a unique `--run-id`; initialization rejects a non-empty run directory.

## 5. Prepare without executing

To stop after target selection, configuration reconciliation, environment preparation,
smoke testing, and execution-manifest creation:

```bash
.venv/bin/r2r <paper-url> <repository-url> \
  --target "<result to reproduce>" \
  --gpu 0 \
  --prepare-only
```

`--skip-evidence-validation` disables optional paper-quote and code-evidence diagnostics.
It does not remove scientific gaps recorded by the agent.

Full execution includes a bounded repair loop by default:

```text
execute → failed only → repair + bounded smoke test → execute again
```

Python permits at most two repairs (three execution attempts total). It never
automatically repairs `succeeded`, `timed_out`, or `blocked` results. Disable repair or
change its limit with:

```bash
.venv/bin/r2r <paper-url> <repository-url> --max-execution-repairs 0
```

Repairs may change environment/dependency files, launch scripts, generated configs, and
minimal repository source. The selected experiment, metric, paper target, dataset/protocol,
seeds, sample count, aggregation, and resolved scientific values remain frozen.

## 6. Recover a run

Resume from the first incomplete durable stage:

```bash
.venv/bin/r2r \
  --recover-run /path/to/runs-root/<run-id> \
  --gpu 0
```

The saved paper URL, repository URL, target, model, and timeout are read from the run
directory.

## 7. Run artifacts

Each run is isolated under `<runs-root>/<run-id>/`. Important files include:

```text
input.json
metadata.json
bootstrap.json
paper_audit.json
paper/paper.pdf
paper/paper.md
paper/paper_parse.json
workspace/repo/
target.json
configuration_audit.json
environment_manifest.json
execution_manifest.json
execution_result.json
final_report.json
final_report.md
assumptions.json
stage_attempts/
execution_repairs/repair-1.json
execution_repairs/repair-1.patch
logs/
artifacts/
```

`execution_manifest.json` freezes the command, image, launch script, expected outputs,
metric extractor, and hashes used by deterministic execution.

When a failed execution is repaired, `execution_repairs/repair-N.json` records the
diagnosis, changed-file hashes, smoke evidence, and repository patch evidence. Failed
results, executor logs, and full-run outputs are archived under
`artifacts/execution-attempts/attempt-N/` before retrying.

## 8. Final report and evidence

`final_report.md` is the concise user-facing result. It identifies the exact paper
table/figure location, paper target, applicable baseline, reproduced value, calculation,
verdict, assumptions, deviations, gaps, and paths to the authoritative evidence.

Lengthy details remain in separate files:

- `execution_manifest.json` → exact applied scientific configuration under
  `scientific_configuration`;
- `configuration_audit.json` → paper/code reconciliation and evidence;
- `environment_manifest.json` → image, Dockerfile, and build-log details;
- `workspace/execution/launch.sh` → exact scientific command executed inside Docker;
- `execution_result.json` → outer Docker invocation, status, runtime, and extracted metric;
- `assumptions.json` → complete structured assumptions and paper-unspecified fallbacks;
- `artifacts/` and `logs/executor.log` → raw outputs and complete execution log.

`final_report.json` is the complete machine-readable report. Its verdict is authored by the
post-execution reporting agent from frozen evidence. Python protects frozen values and
checks report structure and assumption coverage, but does not reinterpret the scientific
comparison.
