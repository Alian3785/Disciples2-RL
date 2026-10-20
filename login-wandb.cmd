@echo off
wsl -d Ubuntu-24.04 -u minigrid -- /home/minigrid/.venvs/numbergrid-cuda13/bin/python -m wandb login --verify
pause
