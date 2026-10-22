#!/usr/bin/env bash
# List the papers in dataset/papers.tsv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

printf '%-3s %-26s %-7s %s\n' ID NAME PORTING TARGET
dataset_rows | while IFS=$'\t' read -r id name _paper _repo porting _timeout target; do
    [[ "$target" == "-" ]] && target="(agent chooses)"
    (( ${#target} > 70 )) && target="${target:0:67}..."
    printf '%-3s %-26s %-7s %s\n' "$id" "$name" "$porting" "$target"
done
