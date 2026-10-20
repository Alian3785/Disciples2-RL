@echo off
wsl.exe -d Ubuntu-24.04 -u minigrid --cd "%~dp0." -- bash scripts/run_gpu.sh benchmark_number_grid.py %*
if errorlevel 1 (
  echo Training failed. See the error above.
  pause
  exit /b 1
)
pause
