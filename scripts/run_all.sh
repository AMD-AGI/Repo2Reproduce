#!/usr/bin/env bash
# Reproduce every paper in dataset/papers.tsv, one after another.
#
#   scripts/run_all.sh [options] [-- extra r2r options]
#
# Options:
#   --gpu N            physical AMD GPU index for every run (default: 0)
#   --runs-root DIR    where run directories go (default: $R2R_RUNS_ROOT, .env, ~/r2r-runs)
#   --only LIST        comma-separated ids or names to run, in dataset order (e.g. 5,9,nsnquant)
#   --skip LIST        comma-separated ids or names to leave out
#   --no-targets       ignore the dataset target column; the agent chooses every result
#   --prepare-only     stop each run after its execution manifest is frozen
#   --stop-on-error    stop the batch at the first run that does not exit 0
#   --dry-run          print each r2r command without running it
#   --detach           run the whole batch in the background (log: <batch-dir>/batch.log)
#
# A batch directory <runs-root>/batch-<UTC timestamp>/ records the batch:
#   batch.log     everything printed by the batch
#   runs.txt      one run directory per line
#   exit_codes.tsv  id, name, run directory, r2r exit status
#   summary.txt   final table (also printed at the end)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

gpu=0 runs_root="" only="" skip="" no_targets=0 prepare_only=0 stop_on_error=0 dry_run=0 detach=0
extra=()
orig_args=("$@")
while (($#)); do
    case "$1" in
        -h|--help) sed -n '2,/^set -euo/{/^set -euo/d; s/^# \{0,1\}//; p}' "${BASH_SOURCE[0]}"; exit 0 ;;
        --gpu) gpu="${2:?}"; shift 2 ;;
        --runs-root) runs_root="${2:?}"; shift 2 ;;
        --only) only="${2:?}"; shift 2 ;;
        --skip) skip="${2:?}"; shift 2 ;;
        --no-targets) no_targets=1; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        --stop-on-error) stop_on_error=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --detach) detach=1; shift ;;
        --) shift; extra=("$@"); break ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
[[ -n "$runs_root" ]] || runs_root="$(default_runs_root)"
runs_root="$(mkdir -p "$runs_root" && cd "$runs_root" && pwd)"

if ((detach)); then
    # Re-launch this script in the background with the same arguments minus --detach.
    relaunch=() passthrough=0
    for a in "${orig_args[@]}"; do
        [[ "$a" == -- ]] && passthrough=1
        ((passthrough)) || [[ "$a" != --detach ]] && relaunch+=("$a")
    done
    batch_dir="$runs_root/batch-$(date -u +%Y%m%d-%H%M%S)"
    mkdir -p "$batch_dir"
    R2R_BATCH_DIR="$batch_dir" nohup setsid "${BASH_SOURCE[0]}" "${relaunch[@]}" \
        >"$batch_dir/batch.log" 2>&1 </dev/null &
    echo "batch started in background (pid $!)"
    echo "follow with: tail -f $batch_dir/batch.log"
    exit 0
fi

# Does $2 (a comma list) contain the id $1 or the name $3?
in_list() {
    local item
    IFS=',' read -ra items <<<"$2"
    for item in "${items[@]}"; do
        item="${item// /}"
        [[ "${item,,}" == "$1" || "${item,,}" == "${3,,}" ]] && return 0
    done
    return 1
}

if ((!dry_run)); then require_r2r; fi
selected=()
while IFS= read -r row; do
    IFS=$'\t' read -r id name _rest <<<"$row"
    [[ -n "$only" ]] && ! in_list "$id" "$only" "$name" && continue
    [[ -n "$skip" ]] && in_list "$id" "$skip" "$name" && continue
    selected+=("$id"$'\t'"$name")
done < <(dataset_rows)
((${#selected[@]})) || die "no papers matched --only/--skip"

batch_dir="${R2R_BATCH_DIR:-$runs_root/batch-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$batch_dir"
: >"$batch_dir/runs.txt"
printf 'id\tname\trun_dir\texit_status\n' >"$batch_dir/exit_codes.tsv"
[[ -n "${R2R_BATCH_DIR:-}" ]] || exec > >(tee -a "$batch_dir/batch.log") 2>&1

echo "batch:     $batch_dir"
echo "runs root: $runs_root"
echo "dataset:   $DATASET"
echo "gpu:       $gpu"

echo "papers:    ${#selected[@]}"

for row in "${selected[@]}"; do
    IFS=$'\t' read -r id name <<<"$row"

    run_id="$id-$name-$(date -u +%Y%m%d-%H%M%S)"
    opts=(--gpu "$gpu" --runs-root "$runs_root" --run-id "$run_id")
    ((no_targets)) && opts+=(--no-target)
    ((prepare_only)) && opts+=(--prepare-only)
    ((dry_run)) && opts+=(--dry-run)

    echo
    echo "=== [$(date -u +%H:%M:%SZ)] paper $id ($name) ==="
    set +e
    "$R2R_ROOT/scripts/run_one.sh" "$id" "${opts[@]}" ${extra[@]+-- "${extra[@]}"}
    status=$?
    set -e
    echo "$runs_root/$run_id" >>"$batch_dir/runs.txt"
    printf '%s\t%s\t%s\t%s\n' "$id" "$name" "$runs_root/$run_id" "$status" >>"$batch_dir/exit_codes.tsv"
    echo "=== paper $id ($name) finished with status $status ==="
    if ((status != 0 && stop_on_error)); then
        echo "stopping: --stop-on-error"
        break
    fi
done

((dry_run)) && exit 0

echo
echo "=== summary ==="
"$R2R_ROOT/scripts/summarize.sh" --batch "$batch_dir" | tee "$batch_dir/summary.txt"
echo
echo "per-run exit status: $batch_dir/exit_codes.tsv"
