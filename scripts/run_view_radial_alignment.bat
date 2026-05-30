@echo off
REM Cell labels colored by CSV features (micro-sam-napari env)
cd /d "%~dp0\.."
conda run -n micro-sam-napari python run_view_radial_alignment.py
pause
