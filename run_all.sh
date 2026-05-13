#!/usr/bin/env bash
set -euo pipefail
for cfg in run_configs/*.yaml; do
  echo "=== Launching $cfg ==="
  accelerate launch pretraining.py --config "$cfg"
done
