#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NUMBERGRID_VENV:-$HOME/.venvs/numbergrid-cuda13}"
python3 -m venv "$VENV"
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [[ -x "$HOME/.venvs/xland-minigrid-cuda13/bin/uv" ]]; then
  UV="$HOME/.venvs/xland-minigrid-cuda13/bin/uv"
else
  "$VENV/bin/pip" install uv==0.12.23
  UV="$VENV/bin/uv"
fi
args=()
[[ -f "$ROOT/runtime-constraints.txt" ]] && args+=(--constraint "$ROOT/runtime-constraints.txt")
"$UV" pip install --python "$VENV/bin/python" "${args[@]}" -r "$ROOT/numbergrid-runtime.txt"
# Keep sources on the Windows filesystem; no setuptools chmod/build on /mnt/c.
"$VENV/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import sysconfig, sys
Path(sysconfig.get_paths()['purelib'], 'numbergrid-source.pth').write_text(
    sys.argv[1] + '\n' + sys.argv[1] + '/stoa-src\n')
PY
"$UV" pip check --python "$VENV/bin/python"
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$VENV/bin/python" -c 'import jax, jax.numpy as jnp; print("GPU:", jax.devices()); assert jax.devices()[0].platform == "gpu"; print("JIT:", jax.jit(lambda x: x @ x)(jnp.ones((32, 32))).block_until_ready()[0,0])'
touch "$VENV/.numbergrid-ready"
