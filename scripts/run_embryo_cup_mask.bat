@echo off
REM Embryo cup mask from cell labels (dilation + hole fill)
cd /d "%~dp0\.."
conda run -n micro-sam-napari python run_embryo_cup_mask.py
pause
