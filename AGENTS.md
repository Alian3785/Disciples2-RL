# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
This is a Reinforcement Learning project with Gymnasium environments simulating tactical combat inspired by Disciples 2. It uses Stable Baselines 3 (PPO) for training. There are three independent modules:

- **Root (`/workspace`)** — Core battle environment (`battle_env.py`) and training script (`train_sb3.py`)
- **`capitalbattle/`** — 10×10 grid-world campaign layer with battles, buildings, and special unit actions
- **`Big_map/`** — 40×40 grid-world map variant with similar mechanics

### Running tests
Each module has its own `tests/` directory. Tests must be run **from the module's own directory** due to relative imports:

```bash
source .venv/bin/activate

# Root tests
cd /workspace && python -m pytest tests/ -v

# capitalbattle tests
cd /workspace/capitalbattle && python -m pytest tests/ -v

# Big_map tests
cd /workspace/Big_map && python -m pytest tests/ -v
```

### Linting
```bash
ruff check .
```
There are ~400 pre-existing lint warnings (unused imports, ambiguous variable names, etc.). These are not regressions.

### Running training (hello-world)
```bash
cd /workspace
WANDB_DISABLED=true TOTAL_STEPS=2048 python train_sb3.py
```
Set `WANDB_DISABLED=true` to skip Weights & Biases integration (requires API key). Set `TOTAL_STEPS` to a small number for quick validation.

### Key gotchas
- The `capitalbattle/` and `Big_map/` modules depend on `python3-tk` (tkinter) at import time because `data_dicts_compact_lines.py` imports tkinter. This system package must be installed.
- Standalone test files at root level (`test_visibility.py`, `test_lord.py`, `test_asterot.py`, `test_asterot_double.py`) are ad-hoc debugging scripts, not proper pytest tests. They may have import errors against the current codebase.
- There are no long-running services. All scripts are standalone Python programs (training, visualization, testing).
- Visualization scripts (`visualize_pygame.py`, `visualize_console.py`) require trained model checkpoints to run. Console visualization works without a display server; pygame visualization requires X11.
