@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_local.ps1" %*
exit /b %errorlevel%
