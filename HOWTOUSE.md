# How to install and use Repo2Reproduce

## 1. Requirements

The host needs:

- Python 3.10–3.13;
- Git and curl;
- Java 11 or newer for OpenDataLoader PDF;
- the Claude Code `claude` executable;
- Docker and a working ROCm driver for GPU execution;
- for `--rocm-porting` runs, `perl` and access to GitHub (the runner clones the pinned
  official [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) release into each run);
- free disk space where runs are written (see section 4): a run holds the cloned
  repository, model and dataset caches, and every output the experiment writes, which can
  reach tens of gigabytes.

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
R2R_RUNS_ROOT=/path/with/space/r2r-runs
R2R_MAX_TURNS=200
R2R_MAX_BUDGET_USD=100.0
```

`R2R_MAX_TURNS` and `R2R_MAX_BUDGET_USD` limit each agent call (every stage attempt)
separately, not the whole run.

Do not commit `.env`.

## 4. Where runs are saved

Every run gets its own directory, `<runs-root>/<run-id>/`. The runs root is chosen in this
order:

1. `--runs-root <dir>` on the command line;
2. `R2R_RUNS_ROOT` in `.env` or the environment;
3. the default, `<checkout>/runs/multi-step` (ignored by git).

Use a runs root **outside the repository checkout**, on a disk with plenty of space.
Agents run with the run's `workspace/` as their working directory, and Claude Code also
loads `CLAUDE.md` files from parent directories. A run below the checkout therefore
inherits the repository's own `CLAUDE.md`, whose instructions (for example the
`mcp__r2r__*` tools) do not apply to this pipeline and end up as spurious gaps in reports.

The run ID defaults to `two-stage-<UTC timestamp>-<repository name>`. Initialization
rejects a run directory that already exists and is not empty, so give each run a unique
`--run-id`.

## 5. Run an example

The target should identify the paper result—such as a table, row, dataset, and method—
without supplying the hyperparameters that the configuration-audit stage is intended to
discover. Omit `--target` to let the agent choose a central, locally reproducible result.

Target selection keeps only requirements it can quote from the paper. Operational
instructions in `--target` that the paper does not state (batch-size grids, repeat counts,
memory caps, extra checks) are recorded as warnings and are not enforced on the launch.

Example: reproduce the Table 2 `d=23` modular-addition row:

```bash
export R2R_RUNS_ROOT=/path/with/space/r2r-runs
RUN_ID="cogs-d23-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/r2r \
  "https://arxiv.org/abs/2410.01779" \
  "https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset" \
  --run-id "$RUN_ID" \
  --gpu 0 \
  --target "Reproduce Table 2, the d=23 modular-addition row." \
  --timeout-seconds 14400
```

The run is written to `$R2R_RUNS_ROOT/$RUN_ID/`. The equivalent module invocation starts
with:

```bash
.venv/bin/python -m repo2reproduce <paper-url> <repository-url> ...
```

`--gpu` is the physical AMD GPU index on the host; inside every container that GPU is
device 0. `--timeout-seconds` (default 14400) limits only the final deterministic
execution of the experiment. Target selection, the configuration audit, environment
preparation, repairs, and finalization run before or after it and are not counted, so the
total wall time is longer.

For a CUDA kernel that must run on ROCm, add `--rocm-porting`. Add `--force-kernel-port` as
well when the paper's own CUDA kernel has to be ported with HIPIFY or a targeted rewrite.
`PORTING.md` describes the policy, the host tools, and the manifest checks.

```bash
.venv/bin/r2r <paper-url> <repository-url> \
  --run-id "<unique-run-id>" \
  --gpu 0 \
  --rocm-porting \
  --force-kernel-port \
  --target "<kernel-centric table, figure, or result>"
```

To detach a run and keep its console output next to the run directory:

```bash
RUNS_ROOT=/path/with/space/r2r-runs
RUN_ID="<unique-run-id>"
mkdir -p "$RUNS_ROOT"
nohup setsid .venv/bin/r2r <paper-url> <repository-url> \
  --run-id "$RUN_ID" \
  --runs-root "$RUNS_ROOT" \
  --gpu 0 \
  --target "<table, row, or result to reproduce>" \
  > "$RUNS_ROOT/$RUN_ID.log" 2>&1 < /dev/null &
```

Progress is recorded in `$RUNS_ROOT/$RUN_ID/metadata.json` (per-stage status, attempts, and
skills) and in `logs/agent_messages.jsonl` inside the run directory.

## 6. Prepare without executing

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

## 7. Recover a run

Resume from the first incomplete durable stage:

```bash
.venv/bin/r2r \
  --recover-run <runs-root>/<run-id> \
  --gpu 0
```

The saved paper URL, repository URL, target, model, timeout, and porting mode are read
from the run directory, so `paper_url`, `repo_url`, `--rocm-porting`, and
`--force-kernel-port` cannot be passed again. Recovery reinstalls the packaged skills.

## 8. Run artifacts

Each run is isolated under `<runs-root>/<run-id>/`:

```text
input.json                      run inputs (paper, repository, target, timeout, porting mode)
metadata.json                   status, per-stage attempts and skills, model, GPU, HIPIFY pin
bootstrap.json                  clone details, commit, subtree snapshot commit
paper_audit.json                parsed-paper hashes
paper/                          paper.pdf, paper.md, paper_parse.json
prompts/                        rendered stage prompts
finalize_prompt.txt             rendered finalization prompt
target.json                     frozen paper target
configuration_audit.json        paper/code configuration ledger
environment_manifest.json       image, Dockerfile, build log
execution_manifest.json         launch script, expected outputs, metric extractor, hashes
execution_result.json           deterministic execution status and extracted metric
final_report.json / .md         structured and human-readable report
assumptions.json                assumptions and paper-unspecified fallbacks
stage_attempts/                 every structured stage output, per attempt
execution_repairs/              repair-N.json and repair-N.patch, when repairs ran
bin/                            docker wrapper; port-scan and port-hipify in porting runs
cache/                          Hugging Face, Torch, and pip caches mounted into containers
tools/HIPIFY-rocm-7.2.4/        pinned official HIPIFY (porting runs)
workspace/repo/                 cloned repository (its own git repo, even for subtrees)
workspace/repo-hip/             HIPIFY-translated copy (porting runs)
workspace/execution/launch.sh   exact command executed inside Docker
workspace/.claude/              Bash hook settings and the installed skills
logs/agent_messages.jsonl       every agent message, tool call, and result
logs/bash_tool_calls.jsonl      host Bash commands and the hook's allow/deny decisions
logs/docker_commands.jsonl      every docker command the agents ran
logs/executor.log               complete log of the deterministic execution
artifacts/env/                  build, smoke-test, and parity logs
artifacts/full-run/             outputs and metric of the full execution
artifacts/port-scan.*           CUDA scan report (porting runs)
artifacts/port-hipify.*         HIPIFY translation report (porting runs)
artifacts/hipify-provenance/    HIPIFY-to-port diffs and overlap summary (hipify_with_fixes)
```

Inside containers the run directory is mounted at `/run`, so `/run/workspace` and
`/run/artifacts` correspond to `workspace/` and `artifacts/` above. Container processes
run as root, so files they create (for example under `artifacts/full-run/`) are owned by
root on the host; remove them through Docker, for example
`docker run --rm -v <run-dir>/artifacts:/a <image> rm -rf /a/full-run`.

`execution_manifest.json` freezes the command, image, launch script, expected outputs,
metric extractor, and hashes used by deterministic execution.

When a failed execution is repaired, `execution_repairs/repair-N.json` records the
diagnosis, changed-file hashes, smoke evidence, and repository patch evidence. Failed
results, executor logs, and full-run outputs are archived under
`artifacts/execution-attempts/attempt-N/` before retrying. Because containers write
`artifacts/full-run` as root, it is archived by renaming it to
`artifacts/full-run.attempt-N`, and `attempt-N/full-run` links to it.

Repository diffs are taken against a local baseline commit. When the repository URL names
a subdirectory (`.../tree/<branch>/<path>`), bootstrap makes that subtree its own git
repository with one commit holding the pristine upstream files, recorded as
`clone.snapshot_commit` in `bootstrap.json`. Recovering an older subtree run rebuilds this
baseline from a fresh clone of the same upstream commit.

## 9. Final report and evidence

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

## 10. Stage prompts and skills

Stage prompts in `repo2reproduce/prompts/` supply run-specific paths, frozen values,
stage boundaries, and output contracts. Reusable, paper-independent reasoning lives in
Claude Code skills under `repo2reproduce/skills/`:

| Stage | Skill |
| --- | --- |
| Configuration audit | `configuration-audit` |
| Environment preparation with `--rocm-porting` | `rocm-kernel-porting` |
| Execution repair | `execution-repair`, plus `rocm-kernel-porting` with `--rocm-porting` |
| Finalization | `paper-result-comparison` |

Target selection, and environment preparation without `--rocm-porting`, use prompts only.
In porting runs the porting skill's no-fallback rule overrides the repair skill's
substitution allowance, and a repair may not drop `rocm_port` or any ported kernel.

Every run installs the packaged skills into `workspace/.claude/skills/` and records each
skill's SHA-256 under `skills` in `metadata.json`; recovery reinstalls them from the
package. Each stage enables only its own skill through the SDK `skills` option, and
prompt-only stages run with no skills listed. A stage fails immediately when Claude Code
does not discover its skill, and its output is accepted only after the agent has loaded
the skill with the Skill tool; otherwise the agent gets one correction request.
