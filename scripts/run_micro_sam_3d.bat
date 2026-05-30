@echo off
setlocal
REM Blank Annotator 3d (no stack). For step 2 use run_micro_sam_nuclei.bat instead.
cd /d "%~dp0.."

conda run -n micro-sam-napari micro_sam.annotator_3d
if errorlevel 1 (
    echo.
    echo Failed. If the env is missing, run:  conda env create -f environment.yml
    echo.
    pause
)
