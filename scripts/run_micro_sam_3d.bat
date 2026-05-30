@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari micro_sam.annotator_3d
if errorlevel 1 (
    echo.
    echo Failed. If the env is missing, run:  conda env create -f environment.yml
    echo.
    pause
)
