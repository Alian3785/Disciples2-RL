#!/usr/bin/env bash
# User-requested 20M per profile; independent runs, never concurrent on CUDA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="results/capital-rewards-20m-20261016"
test -f "$OUTPUT/preflight.json"
for variant in current python; do
    if [[ -e "$OUTPUT/$variant" ]]; then
        echo "Refusing to overwrite existing run: $OUTPUT/$variant" >&2
        exit 1
    fi
    bash scripts/run_gpu.sh benchmark_number_grid.py \
        --map "$OUTPUT/$variant-map.json" --output "$OUTPUT/$variant" \
        --total-timesteps 20000000 --seed 42 --tracking auto --evaluate \
        2>&1 | tee "$OUTPUT/$variant-console.log"
done
bash scripts/run_gpu.sh scripts/compare_capital_rewards.py report --output "$OUTPUT"
