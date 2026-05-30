@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari python run_micro_sam_nuclei.py
if errorlevel 1 (
    echo.
    echo Failed. Run step 1 first:  scripts\run_plane_split.bat
    echo If the env is missing:  conda env create -f environment.yml
    echo.
    pause
)
