#!/usr/bin/env bash
# Reproduce one paper with Repo2Reproduce.
#
#   scripts/run_one.sh <id|name> [options] [-- extra r2r options]
#   scripts/run_one.sh --paper <url> --repo <url> [options] [-- extra r2r options]
#   scripts/run_one.sh --recover <run-dir> [--gpu N] [--max-budget-usd USD] [--detach]
#
# Options:
#   --target TEXT       result to reproduce (default: the agent chooses)
#   --gpu N             physical AMD GPU index (default: 0)
#   --runs-root DIR     where run directories go (default: $R2R_RUNS_ROOT, .env, ~/r2r-runs)
#   --run-id ID         run directory name (default: <id>-<name>-<UTC timestamp>)
#   --timeout SECONDS   limit for the final execution (default: 14400)
#   --porting MODE      none | rocm | force (default: none)
#   --max-budget-usd USD  spending cap per agent session (default: R2R_MAX_BUDGET_USD in .env, else 100)
#   --prepare-only      stop after the execution manifest is frozen; do not execute
#   --detach            run in the background; console output goes to <runs-root>/<run-id>.log
#   --dry-run           print the r2r command without running it
#
# The console log is always written to <runs-root>/<run-id>.log.
# Exit status is r2r's: 0 succeeded (or prepared), 1 execution failed, 2 blocked;
# any other non-zero value means the pipeline itself errored. 64 is a usage error.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    sed -n '2,/^set -euo/{/^set -euo/d; s/^# \{0,1\}//; p}' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

selector="" paper="" repo="" target="" gpu=0
runs_root="" run_id="" timeout="" porting="" budget="" prepare_only=0 detach=0 dry_run=0 recover=""
extra=()

while (($#)); do
    case "$1" in
        -h|--help) usage 0 ;;
        --paper) paper="${2:?}"; shift 2 ;;
        --repo) repo="${2:?}"; shift 2 ;;
        --target) target="${2?}"; shift 2 ;;
        --gpu) gpu="${2:?}"; shift 2 ;;
        --runs-root) runs_root="${2:?}"; shift 2 ;;
        --run-id) run_id="${2:?}"; shift 2 ;;
        --timeout) timeout="${2:?}"; shift 2 ;;
        --porting) porting="${2:?}"; shift 2 ;;
        --max-budget-usd) budget="${2:?}"; shift 2 ;;
        --prepare-only) prepare_only=1; shift ;;
        --detach) detach=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --recover) recover="${2:?}"; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -*) echo "unknown option: $1" >&2; usage 64 ;;
        *) [[ -z "$selector" ]] || die "only one paper may be given (got '$selector' and '$1')"
           selector="$1"; shift ;;
    esac
done

[[ -n "$runs_root" ]] || runs_root="$(default_runs_root)"

if [[ -n "$recover" ]]; then
    [[ -d "$recover" ]] || die "run directory not found: $recover"
    recover="$(cd "$recover" && pwd)"
    run_id="$(basename "$recover")"
    log="$(dirname "$recover")/$run_id.recover-$(date -u +%Y%m%d-%H%M%S).log"
    cmd=("$R2R_BIN" --recover-run "$recover" --gpu "$gpu")
    [[ -n "$budget" ]] && cmd+=(--max-budget-usd "$budget")
    cmd+=("${extra[@]}")
else
    name="custom" id=""
    if [[ -n "$selector" ]]; then
        row="$(dataset_lookup "$selector")"
        [[ -n "$row" ]] || die "no paper with id or name '$selector' in $DATASET (see scripts/list_papers.sh)"
        IFS=$'\t' read -r id name ds_paper ds_repo _conference <<<"$row"
        [[ -n "$paper" ]] || paper="$ds_paper"
        [[ -n "$repo" ]] || repo="$ds_repo"
    fi
    [[ -n "$paper" && -n "$repo" ]] || { echo "give a dataset id/name, or --paper and --repo" >&2; usage 64; }

    [[ -n "$timeout" ]] || timeout=14400
    [[ -n "$porting" ]] || porting=none
    [[ -n "$run_id" ]] || run_id="${id:+$id-}$name-$(date -u +%Y%m%d-%H%M%S)"
    log="$runs_root/$run_id.log"

    cmd=("$R2R_BIN" "$paper" "$repo" --run-id "$run_id" --runs-root "$runs_root"
         --gpu "$gpu" --timeout-seconds "$timeout")
    [[ -n "$target" ]] && cmd+=(--target "$target")
    case "$porting" in
        none) ;;
        rocm) cmd+=(--rocm-porting) ;;
        force) cmd+=(--rocm-porting --force-kernel-port) ;;
        *) die "--porting must be none, rocm, or force (got '$porting')" ;;
    esac
    [[ -n "$budget" ]] && cmd+=(--max-budget-usd "$budget")
    ((prepare_only)) && cmd+=(--prepare-only)
    cmd+=("${extra[@]}")
fi

echo "run:  ${recover:-$runs_root/$run_id}"
echo "log:  $log"
printf 'cmd: '; printf ' %q' "${cmd[@]}"; echo
((dry_run)) && exit 0

require_r2r
mkdir -p "$(dirname "$log")"
cd "$R2R_ROOT"

if ((detach)); then
    nohup setsid "${cmd[@]}" >"$log" 2>&1 </dev/null &
    echo "started in background (pid $!); follow with: tail -f $log"
    exit 0
fi

set +e
"${cmd[@]}" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e
echo "r2r exited with status $status"
exit "$status"
