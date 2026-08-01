@echo off
REM One-click launcher that bypasses PowerShell's script execution policy.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
