@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari python main.py
if errorlevel 1 (
    echo.
    echo Failed. If the env is missing, run:  conda env create -f environment.yml
    echo.
    pause
)
