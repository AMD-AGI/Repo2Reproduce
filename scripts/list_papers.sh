#!/usr/bin/env bash
# List the papers in dataset/papers.tsv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

printf '%-3s %-26s %-31s %s\n' ID NAME CONFERENCE PAPER
dataset_rows | while IFS=$'\t' read -r id name paper _repo conference; do
    printf '%-3s %-26s %-31s %s\n' "$id" "$name" "$conference" "$paper"
done
