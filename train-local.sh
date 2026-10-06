#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${DISCIPLES_REFERENCE_VENV:-/home/minigrid/.venvs/disciples2-reference}"
if [[ ! -x "$VENV/bin/python" ]]; then
    printf 'Python environment not found: %s\n' "$VENV" >&2
    exit 1
fi
cd "$ROOT"
export CAMPAIGN_OBSERVATION_VERSION=local5
export MPLBACKEND=Agg
exec "$VENV/bin/python" -u Big_map/train_campaign.py \
    --no-scripted-bot --no-boss-roster --no-comet "$@"
