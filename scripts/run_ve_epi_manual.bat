@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari python run_ve_epi_manual.py
if errorlevel 1 (
    echo.
    echo Failed. Need step 1 segment + step 2 labels in data\test_cell_labels\
    echo.
    pause
)
