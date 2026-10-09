#!/usr/bin/env bash
set -euo pipefail

# Host-side bootstrap and launch for one staged CoMRes reproduction.
# Override any uppercase setting below through the environment.

REPO_ROOT="${REPO_ROOT:-/home/machaita/Repo2Reproduce}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
RUNS_ROOT="${RUNS_ROOT:-/shared_nfs/machita/claude-code-two-stage-runs}"
RUN_ID="${RUN_ID:-staged-comres-$(date -u +%Y%m%d-%H%M%S)}"
GPU_INDEX="${GPU_INDEX:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"
MAX_TURNS="${MAX_TURNS:-200}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-100}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
FOREGROUND="${FOREGROUND:-0}"

PAPER_URL="https://openreview.net/pdf?id=bRa4JLPzii"
REPO_URL="https://github.com/yjucho1/CoMRes"
TARGET="Reproduce the five-seed ETTh1 L=96 H=96 CoMRes interpolation ensemble-MSE using seeds 22, 99, 123, 456, and 999; produce the five-seed mean and standard deviation; do not reduce the experiment to one seed."

cd "${REPO_ROOT}"

for command_name in "${PYTHON_BIN}" git curl java docker; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required host command: ${command_name}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"Python 3.10-3.13 is required; found {sys.version_info.major}.{sys.version_info.minor}"
    )
PY

JAVA_VERSION="$(java -version 2>&1 | awk -F '"' '/version/ {print $2; exit}')"
JAVA_MAJOR="${JAVA_VERSION%%.*}"
if [[ "${JAVA_MAJOR}" == "1" ]]; then
  JAVA_MAJOR="$(printf '%s' "${JAVA_VERSION}" | cut -d. -f2)"
fi
if [[ -z "${JAVA_MAJOR}" || "${JAVA_MAJOR}" -lt 11 ]]; then
  echo "Java 11+ is required; found ${JAVA_VERSION:-unknown}" >&2
  exit 2
fi

if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
  echo "ROCm devices /dev/kfd and /dev/dri are unavailable on this node." >&2
  exit 2
fi
docker info >/dev/null

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v claude >/dev/null 2>&1; then
  echo "Installing the Claude Code engine into ~/.local/bin ..."
  curl -fsSL https://claude.ai/install.sh | bash
  export PATH="${HOME}/.local/bin:${PATH}"
fi
claude --version

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  echo "Missing ${REPO_ROOT}/.env; copy .env.example and configure the provider." >&2
  exit 2
fi

"${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv

root = Path.cwd()
load_dotenv(root / ".env", override=False)

auth_mode = os.environ.get("R2R_AUTH_MODE", "api_key")
if auth_mode == "custom_headers":
    required = ("R2R_BASE_URL", "R2R_AUTH_HEADERS", "R2R_AUTH_KEY")
else:
    required = (os.environ.get("R2R_API_KEY_ENV", "ANTHROPIC_API_KEY"),)

missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit("Missing provider configuration in .env: " + ", ".join(missing))

java_home = os.environ.get("R2R_JAVA_HOME")
if java_home and not (Path(java_home) / "bin" / "java").is_file():
    raise SystemExit(f"R2R_JAVA_HOME does not contain bin/java: {java_home}")
PY

mkdir -p "${REPO_ROOT}/logs/two-stage-launches" "${RUNS_ROOT}"
LOG_PATH="${REPO_ROOT}/logs/two-stage-launches/${RUN_ID}.log"

COMMAND=(
  "${VENV_DIR}/bin/python"
  -m
  claude_baseline_two_stage
  "${PAPER_URL}"
  "${REPO_URL}"
  --runs-root
  "${RUNS_ROOT}"
  --run-id
  "${RUN_ID}"
  --gpu
  "${GPU_INDEX}"
  --timeout-seconds
  "${TIMEOUT_SECONDS}"
  --max-turns
  "${MAX_TURNS}"
  --max-budget-usd
  "${MAX_BUDGET_USD}"
  --target
  "${TARGET}"
)

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  COMMAND+=(--prepare-only)
fi

echo "Run directory: ${RUNS_ROOT}/${RUN_ID}"
echo "Launcher log: ${LOG_PATH}"

if [[ "${FOREGROUND}" == "1" ]]; then
  exec "${COMMAND[@]}"
fi

nohup setsid "${COMMAND[@]}" >"${LOG_PATH}" 2>&1 </dev/null &
echo "Started PID=$!"
