#!/usr/bin/env bash
# Summarize finished or in-progress runs as a table.
#
#   scripts/summarize.sh                    every run directory under the runs root
#   scripts/summarize.sh <run-dir>...       the given run directories
#   scripts/summarize.sh --batch <dir>      the runs listed in a run_all.sh batch directory
#   scripts/summarize.sh --tsv ...          tab-separated output instead of a table
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

format=table
dirs=()
while (($#)); do
    case "$1" in
        -h|--help) sed -n '2,/^set -euo/{/^set -euo/d; s/^# \{0,1\}//; p}' "${BASH_SOURCE[0]}"; exit 0 ;;
        --tsv) format=tsv; shift ;;
        --batch)
            [[ -f "${2:?}/runs.txt" ]] || die "not a batch directory: $2"
            mapfile -t -O "${#dirs[@]}" dirs < <(grep -v '^$' "$2/runs.txt"); shift 2 ;;
        *) dirs+=("$1"); shift ;;
    esac
done
if ((${#dirs[@]} == 0)); then
    root="$(default_runs_root)"
    mapfile -t dirs < <(find "$root" -mindepth 2 -maxdepth 2 -name input.json -printf '%h\n' 2>/dev/null | sort)
    ((${#dirs[@]})) || die "no runs found under $root"
fi

python3 -I - "$format" "${dirs[@]}" <<'PY'
import json
import sys
from pathlib import Path

fmt, dirs = sys.argv[1], sys.argv[2:]


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def current_stage(meta):
    for name, stage in (meta.get("stages") or {}).items():
        if isinstance(stage, dict) and stage.get("status") not in ("succeeded", "skipped"):
            return f"{name}:{stage.get('status')}"
    return ""


rows = []
for d in dirs:
    run = Path(d)
    meta = load(run / "metadata.json")
    report = load(run / "final_report.json")
    result = load(run / "execution_result.json")
    comparison = report.get("comparison") or {}
    error = comparison.get("relative_error_percent")
    status = meta.get("status") or ("missing" if not run.is_dir() else "not started")
    rows.append([
        run.name,
        status,
        report.get("execution_status") or result.get("status") or "",
        comparison.get("verdict") or "",
        str(report.get("paper_target_value") or ""),
        str(report.get("achieved_value") or ""),
        "" if error is None else f"{error:.2f}%",
        current_stage(meta) if status not in ("completed", "prepared") else "",
    ])

header = ["run", "status", "execution", "verdict", "paper", "reproduced", "rel.err", "stopped at"]
if fmt == "tsv":
    for row in [header] + rows:
        print("\t".join(cell.replace("\t", " ").replace("\n", " ") for cell in row))
    sys.exit(0)

clip = lambda s, n=28: s if len(s) <= n else s[: n - 3] + "..."
rows = [[clip(c) if i in (4, 5) else c for i, c in enumerate(r)] for r in rows]
widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
for row in [header, ["-" * w for w in widths]] + rows:
    print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())

counts = {}
for r in rows:
    counts[r[3] or r[1]] = counts.get(r[3] or r[1], 0) + 1
print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
PY
