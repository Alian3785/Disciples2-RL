@echo off
wsl.exe -d Ubuntu-24.04 -u minigrid --cd "%~dp0." -- bash scripts/run_gpu.sh -m pytest -q
pause
