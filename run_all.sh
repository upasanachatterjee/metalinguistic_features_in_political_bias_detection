#!/usr/bin/env bash
set -euo pipefail
for cfg in run_configs/*.yaml; do
  echo "=== Launching $cfg ==="
  accelerate launch --config_file accelerate_config.yaml pretraining.py --config "$cfg"
done
