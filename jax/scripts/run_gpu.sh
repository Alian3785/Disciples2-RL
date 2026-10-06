#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NUMBERGRID_VENV:-$HOME/.venvs/numbergrid-cuda13}"
if [[ ! -f "$VENV/.numbergrid-ready" ]]; then
  bash "$ROOT/scripts/install_runtime.sh"
fi
cd "$ROOT"
# The path also works after moving this checkout, without editing the venv.
export PYTHONPATH="$ROOT:$ROOT/stoa-src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
unset LD_LIBRARY_PATH
exec "$VENV/bin/python" "$@"
