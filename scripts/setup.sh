#!/usr/bin/env bash
# Set up Repo2Reproduce on an AMD ROCm host.
#
#   scripts/setup.sh [--install-claude] [--with-tests] [--python PYTHON]
#
# 1. checks the host prerequisites (Python 3.10-3.13, git, curl, Java 11+, perl, Docker,
#    ROCm devices, the claude CLI) and prints how to fix anything missing;
# 2. creates .venv and installs the package with `pip install -e .[dev]`;
# 3. creates .env from .env.example when it does not exist yet;
# 4. checks that `r2r --help` works, and with --with-tests runs the unit tests.
#
# --install-claude installs the Claude Code CLI with the official installer when missing.
# System packages (Java, Docker, ROCm) need root and are never installed by this script.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

install_claude=0 with_tests=0 python="${PYTHON:-python3}"
while (($#)); do
    case "$1" in
        -h|--help) sed -n '2,/^set -euo/{/^set -euo/d; s/^# \{0,1\}//; p}' "${BASH_SOURCE[0]}"; exit 0 ;;
        --install-claude) install_claude=1; shift ;;
        --with-tests) with_tests=1; shift ;;
        --python) python="${2:?}"; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

problems=0
ok() { printf '  [ok]   %s\n' "$*"; }
bad() { printf '  [FAIL] %s\n' "$*"; problems=$((problems + 1)); }
warn() { printf '  [warn] %s\n' "$*"; }

echo "== 1. Host prerequisites"

if command -v "$python" >/dev/null && "$python" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))'; then
    ok "$("$python" --version) ($python)"
else
    bad "Python 3.10-3.13 not found as '$python' (install python3 + python3-venv, or pass --python python3.12)"
fi
"$python" -c 'import venv, ensurepip' 2>/dev/null || bad "python venv/ensurepip missing (Ubuntu: sudo apt install python3-venv)"

for tool in git curl perl; do
    if command -v "$tool" >/dev/null; then ok "$tool"; else bad "$tool not found (Ubuntu: sudo apt install $tool)"; fi
done

java_bin="java"
java_home="${R2R_JAVA_HOME:-$(dotenv_value R2R_JAVA_HOME)}"
[[ -n "$java_home" && -x "$java_home/bin/java" ]] && java_bin="$java_home/bin/java"
if command -v "$java_bin" >/dev/null; then
    java_version="$("$java_bin" -version 2>&1 | sed -n 's/.*version "\([0-9]*\)[^"]*".*/\1/p' | head -n 1)"
    [[ "$java_version" == 1 ]] && java_version=8
    if [[ -n "$java_version" ]] && ((java_version >= 11)); then
        ok "Java $java_version ($java_bin)"
    else
        bad "Java 11+ required, found '${java_version:-unknown}' (Ubuntu: sudo apt install openjdk-17-jre-headless)"
    fi
else
    bad "Java not found (Ubuntu: sudo apt install openjdk-17-jre-headless; then set R2R_JAVA_HOME in .env)"
fi

if command -v docker >/dev/null; then
    if docker info >/dev/null 2>&1; then
        ok "docker (daemon reachable)"
    else
        bad "docker installed but 'docker info' fails (start the daemon, or: sudo usermod -aG docker \$USER and log in again)"
    fi
else
    bad "docker not found (https://docs.docker.com/engine/install/)"
fi

if [[ -e /dev/kfd && -d /dev/dri ]]; then
    ok "ROCm devices /dev/kfd and /dev/dri"
    if command -v rocm-smi >/dev/null; then
        ok "GPUs visible to rocm-smi: $(rocm-smi --showproductname 2>/dev/null | grep -c 'Card Series' || true)"
    else
        warn "rocm-smi not found; cannot list GPUs (GPU indices for --gpu come from rocm-smi)"
    fi
else
    bad "/dev/kfd or /dev/dri missing: the AMD ROCm kernel driver is not loaded"
fi

claude_dir="${R2R_CLI_BIN_DIR:-$(dotenv_value R2R_CLI_BIN_DIR)}"
claude_dir="${claude_dir:-$HOME/.local/bin}"
export PATH="$claude_dir:$PATH"
if ! command -v claude >/dev/null && ((install_claude)); then
    echo "  installing Claude Code ..."
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi
if command -v claude >/dev/null; then
    ok "claude $(claude --version 2>/dev/null | head -n 1)"
else
    bad "claude CLI not found (rerun with --install-claude, or: curl -fsSL https://claude.ai/install.sh | bash)"
fi

echo
echo "== 2. Python environment ($VENV)"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$python" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -e "$R2R_ROOT[dev]"
ok "installed repo2reproduce $("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("repo2reproduce"))')"

echo
echo "== 3. Configuration (.env)"
if [[ -f "$R2R_ROOT/.env" ]]; then
    ok ".env exists"
else
    cp "$R2R_ROOT/.env.example" "$R2R_ROOT/.env"
    warn "created .env from .env.example; edit it before running (provider, key, model, R2R_RUNS_ROOT)"
fi
for key in R2R_PROVIDER R2R_AUTH_MODE R2R_MODEL_REASONING; do
    [[ -n "$(dotenv_value "$key")" ]] || warn "$key is not set in .env"
done
auth_mode="$(dotenv_value R2R_AUTH_MODE)"
if [[ "$auth_mode" == api_key && -z "$(dotenv_value ANTHROPIC_API_KEY)" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    warn "R2R_AUTH_MODE=api_key but ANTHROPIC_API_KEY is empty"
fi
if [[ "$auth_mode" == custom_headers && "$(dotenv_value R2R_AUTH_KEY)" =~ ^\<.*\>$|^$ ]]; then
    warn "R2R_AUTH_MODE=custom_headers but R2R_AUTH_KEY is still empty or a placeholder"
fi
runs_root="$(default_runs_root)"
case "$runs_root/" in
    "$R2R_ROOT"/*) warn "runs root $runs_root is inside the checkout; set R2R_RUNS_ROOT to a directory outside it" ;;
    *) ok "runs root: $runs_root ($(df -h --output=avail "$(dirname "$runs_root")" 2>/dev/null | tail -n 1 | tr -d ' ') free)" ;;
esac

echo
echo "== 4. Verification"
if "$VENV/bin/r2r" --help >/dev/null; then ok "r2r --help"; else bad "r2r --help failed"; fi
if ((with_tests)); then
    (cd "$R2R_ROOT" && "$VENV/bin/python" -m pytest -q) && ok "unit tests" || bad "unit tests failed"
fi

echo
if ((problems)); then
    echo "$problems problem(s) found; fix them and rerun scripts/setup.sh."
    exit 1
fi
echo "Ready. Next: edit .env if needed, then try:"
echo "  scripts/list_papers.sh"
echo "  scripts/run_one.sh cogs --gpu 0"
