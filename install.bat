@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo  UAV Vision Detection ^& Tracking
echo  Installer
echo ============================================================
echo.

set "PY="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=py -3"
if "%PY%"=="" python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
if "%PY%"=="" goto nopython

echo Using %PY%
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto fail
python -m pip uninstall -y opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >nul 2>&1
python -m pip install "opencv-python>=4.10,<5" "numpy>=1.26,<3" "PySide6>=6.6"
if errorlevel 1 goto fail
echo.
echo Installing YOLOv8 ^(ultralytics, torch^). This step is large.
echo If it fails, video, demo mode and the vehicle tracker still start.
python -m pip install "ultralytics>=8.3.0"
if errorlevel 1 echo YOLOv8 extra was not installed. See docs\РУКОВОДСТВО.md
echo.
echo Replacing the CPU PyTorch build with the NVIDIA CUDA 13 build.
python -m pip install --upgrade "torch==2.14.1+cu130" "torchvision==0.29.1+cu130" --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 echo CUDA PyTorch was not installed. YOLOv8 stays on CPU.
python tools\verify_install.py
if errorlevel 1 goto fail

echo.
echo Install finished.
echo Start the application with start.bat
echo.
pause
exit /b 0

:nopython
echo Python 3.10 or newer was not found.
echo Install Python from https://www.python.org/downloads/
echo and enable the "py" launcher.
echo.
pause
exit /b 1

:fail
echo.
echo Install failed. The messages above show the reason.
echo.
pause
exit /b 1
