@echo off
REM VE / EPI feature table + Napari classifier (micro-sam-napari env)
cd /d "%~dp0\.."
conda run -n micro-sam-napari python run_epi_ve_classifier.py
pause
