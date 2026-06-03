@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari python run_inspect_cell_labels.py %*
if errorlevel 1 (
    echo.
    echo Failed. Need step 1 segment TIFFs and labels in data\test_cell_labels\
    echo.
    pause
)
