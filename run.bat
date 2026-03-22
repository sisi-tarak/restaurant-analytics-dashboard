@echo off
REM ============================================================
REM  Restaurant Analytics Dashboard — Windows Launcher
REM  Double-click this file to start the dashboard
REM ============================================================

REM Change to the folder where this batch file lives
cd /d "%~dp0"

REM Launch Streamlit with explicit 1 GB upload limit
REM This overrides config.toml regardless of working directory
streamlit run app.py ^
    --server.maxUploadSize=1024 ^
    --server.port=8501 ^
    --browser.gatherUsageStats=false

pause
