@echo off
setlocal
wsl.exe -d Ubuntu-24.04 -u minigrid --cd "%~dp0." -- bash ./train-local.sh %*
exit /b %errorlevel%
