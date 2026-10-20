@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\test_gpu.ps1" %*
set "TEST_EXIT=%ERRORLEVEL%"
pause
exit /b %TEST_EXIT%
