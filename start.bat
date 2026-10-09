@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title UAV Vision Detection and Tracking

if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment is missing.
  echo Run install.bat first.
  echo.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python main.py
if errorlevel 1 (
  echo.
  echo The application stopped because of an error.
  echo The traceback is printed above.
  echo.
  pause
  exit /b 1
)
exit /b 0
