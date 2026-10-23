#!/usr/bin/env bash
# Explicit 3 x 20M campaign training; one learner/CUDA process at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
suffix="${1:-$(date +%Y%m%d-%H%M%S)}"
for map in super_last_stand siege_train the_charmed_child; do
  output="results/${map}-20m-${suffix}"
  bash scripts/run_gpu.sh -u benchmark_number_grid.py \
    --map "$map" --total-timesteps 20000000 --tracking auto \
    --run-name "${map}-20m-${suffix}" --output "$output"
done
