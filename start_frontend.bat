@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "BOOTSTRAP_PYTHON="

if not exist "%VENV_PYTHON%" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "BOOTSTRAP_PYTHON=python"
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            set "BOOTSTRAP_PYTHON=py -3"
        )
    )

    if "!BOOTSTRAP_PYTHON!"=="" (
        echo [ERROR] Python not found.
        echo Install Python 3.11 or newer, then run this file again.
        pause
        exit /b 1
    )

    echo Creating project environment...
    call !BOOTSTRAP_PYTHON! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
)

call "%VENV_PYTHON%" -c "import cvxpy, duckdb, streamlit, typer, yfinance" >nul 2>nul
if errorlevel 1 (
    echo Installing project dependencies into .venv...
    call "%VENV_PYTHON%" -m pip install -e .
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        echo Run: .venv\Scripts\python.exe -m pip install -e .
        pause
        exit /b 1
    )
)

echo Starting Streamlit dashboard...
echo URL: http://127.0.0.1:8501
echo.

call "%VENV_PYTHON%" -m streamlit run src\qss\dashboard\streamlit_app.py --server.address 127.0.0.1 --server.headless true

if errorlevel 1 (
    echo.
    echo [ERROR] Frontend failed to start.
    pause
    exit /b 1
)

endlocal
