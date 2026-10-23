# Shared helpers for the scripts in this directory. Source it; do not execute it.

R2R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${R2R_DATASET:-$R2R_ROOT/dataset/papers.tsv}"
VENV="${R2R_VENV:-$R2R_ROOT/.venv}"
R2R_BIN="$VENV/bin/r2r"

die() {
    echo "error: $*" >&2
    exit 64
}

# Value of KEY in the checkout's .env, or nothing.
dotenv_value() {
    local key="$1"
    [[ -f "$R2R_ROOT/.env" ]] || return 0
    sed -n "s/^[[:space:]]*${key}=//p" "$R2R_ROOT/.env" | tail -n 1 | sed 's/^["'\'']//; s/["'\'']$//'
}

# Runs root: $R2R_RUNS_ROOT, then .env, then ~/r2r-runs. The pipeline's own default sits
# inside the checkout, where runs would inherit the repository's CLAUDE.md, so avoid it.
default_runs_root() {
    local root="${R2R_RUNS_ROOT:-}"
    [[ -n "$root" ]] || root="$(dotenv_value R2R_RUNS_ROOT)"
    [[ -n "$root" ]] || root="$HOME/r2r-runs"
    echo "${root/#\~/$HOME}"
}

# Print the dataset rows (no comments, no header), tab-separated.
dataset_rows() {
    [[ -f "$DATASET" ]] || die "dataset not found: $DATASET"
    awk -F'\t' '!/^[[:space:]]*#/ && NF > 0 && $1 != "id"' "$DATASET"
}

# Print the dataset row whose id or name matches $1 (name match is case-insensitive).
dataset_lookup() {
    local key="${1,,}"
    dataset_rows | awk -F'\t' -v k="$key" 'tolower($1) == k || tolower($2) == k { print; exit }'
}

require_r2r() {
    [[ -x "$R2R_BIN" ]] || die "$R2R_BIN not found; run scripts/setup.sh first"
}
