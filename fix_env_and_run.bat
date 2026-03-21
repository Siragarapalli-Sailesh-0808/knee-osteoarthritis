@echo off
REM TF 2.16+ breaks old knee.h5; TF 2.15 uses Keras 2 and loads it correctly.
cd /d "%~dp0"

echo === Step 1: Go to project folder ===
echo %CD%
echo.

if not exist .venv (
  echo Creating venv with Python 3.11...
  py -3.11 -m venv .venv
)

echo === Step 2: Activate venv ===
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo ERROR: Could not activate .venv
  echo Fix: run this file from the folder that contains app.py and .venv
  pause
  exit /b 1
)

echo === Step 3: Install TensorFlow 2.15 + Flask (replaces broken TF 2.21) ===
pip uninstall -y tensorflow tensorflow-intel keras tf-keras 2>nul
pip install "numpy<2" "tensorflow==2.15.1" flask pillow

echo.
echo === Step 4: Start app (leave this window open) ===
python app.py
pause
