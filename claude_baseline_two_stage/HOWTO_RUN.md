# How to run the staged Claude agent

## Prepare the agent environment

The host needs Python 3.10–3.13, Git, curl, Java 11+, Docker, and a working ROCm driver.
Verify the GPU and container runtime:

```bash
java -version
docker info
test -e /dev/kfd
test -d /dev/dri
```

Install the Claude Code engine if it is not already available:

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
```

Create the Python environment and install Repo2Reproduce with the staged agent and paper
reader dependencies:

```bash
cd /home/machaita/Repo2Reproduce
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Configure the LLM provider and Java location:

```bash
test -f .env || cp .env.example .env
```

Edit `.env` and set the provider credentials, models, and `R2R_JAVA_HOME`. Do not commit
`.env`.

## Example: launch CoMRes

Run directly from the allocated compute node:

```bash
cd /home/machaita/Repo2Reproduce
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs/two-stage-launches

RUN_ID="staged-comres-$(date -u +%Y%m%d-%H%M%S)"
nohup setsid .venv/bin/python -m claude_baseline_two_stage \
  "https://openreview.net/pdf?id=bRa4JLPzii" \
  "https://github.com/yjucho1/CoMRes" \
  --run-id "$RUN_ID" \
  --gpu 0 \
  --timeout-seconds 14400 \
  --target "Reproduce the five-seed ETTh1 L=96 H=96 CoMRes interpolation ensemble-MSE using seeds 22, 99, 123, 456, and 999; produce the five-seed mean and standard deviation; do not reduce the experiment to one seed." \
  > "logs/two-stage-launches/${RUN_ID}.log" 2>&1 < /dev/null &
```

Replace `--gpu 0` with the allocated physical GPU index. Add `--prepare-only` to stop
after the environment build, audit, smoke test, and immutable manifest creation.

To launch from another node, wrap the same command with:

```bash
spur exec <allocation-id> bash -lc '<command>'
```

The orchestrator persists these audit artifacts before full execution:

- `target.json`
- `configuration_audit.json`
- `environment_manifest.json`
- `execution_manifest.json`
- `execution_result.json`
- `final_report.json`
- `final_report.md`

To resume from the first incomplete stage:

```bash
.venv/bin/python -m claude_baseline_two_stage \
  --recover-run /shared_nfs/machita/claude-code-two-stage-runs/<run-id>
```

Use `--prepare-only` to stop after the environment is smoke-tested and the immutable
execution manifest is written.
