#!/usr/bin/env bash
# Sweep the "easy" locality mechanism: point sort key x scan direction.
# Usage:  scripts/sweep_sortings.sh <config.yaml> [extra run.py args...]
# Example: scripts/sweep_sortings.sh configs/tracking/deltanet_tracking-6k.yaml
#
# Each run is tagged in its note so logs are distinguishable:
#   logs/<stamp>_trans_deltanet_<dataset>_<seed>_sort-<key>-<dir>/
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"; shift || true
PY=.venv/bin/python

SORT_KEYS=("eta" "phi" "r" "pt" "kt" "original")
DIRECTIONS=("bi" "uni")

cd src
for key in "${SORT_KEYS[@]}"; do
  for dir in "${DIRECTIONS[@]}"; do
    echo "=== sort_key=$key direction=$dir ==="
    ../$PY run.py --config "../$CONFIG" \
      -o model_kwargs.deltanet_sort_key="$key" \
      -o model_kwargs.deltanet_direction="$dir" \
      --note "sort-${key}-${dir}" "$@"
  done
done
