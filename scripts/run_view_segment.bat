@echo off
setlocal
cd /d "%~dp0.."

conda run -n micro-sam-napari python run_view_segment.py
if errorlevel 1 (
    echo.
    echo Failed. Run step 1 first:  scripts\run_plane_split.bat
    echo.
    pause
)
