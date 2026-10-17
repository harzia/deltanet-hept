#!/usr/bin/env bash
# Offline CPU checks on synthetic fixtures. Every failure propagates.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  printf 'Python not found: %s. Set PYTHON=python or create .venv.\n' "$PY" >&2
  exit 1
fi
"$PY" -m pytest -q tests
"$PY" scripts/make_synthetic_tracking.py
"$PY" scripts/make_synthetic_pileup.py
"$PY" tests/test_sharded_equivalence.py
for task in tracking pileup; do
  for experiment in deltanet hept deltanet_hept deltanet_hept_hash; do
    kernel="${experiment//_/-}"
    if [ "$experiment" = deltanet_hept_hash ]; then kernel=deltanet-hept; fi
    "$PY" train.py --config "configs/$task/${experiment}_smoke.yaml" --kernel "$kernel" -o device=cpu
  done
done
printf '\nSMOKE TEST COMPLETE. Logs in ./logs/\n'
