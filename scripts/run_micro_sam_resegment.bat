@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari python run_micro_sam_resegment.py
if errorlevel 1 (
    echo.
    echo Failed. Need step 1 segment TIFFs and labels in data\test_cell_labels\
    echo Run step 2 first if you have no labels:  scripts\run_micro_sam_nuclei.bat
    echo.
    pause
)
