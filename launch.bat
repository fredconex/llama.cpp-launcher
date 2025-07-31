@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Llama.cpp Launcher Auto-Setup
echo ========================================

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and add it to your PATH
    pause
    exit /b 1
)

:: Check if .venv directory exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check if requirements are already installed
echo Checking if requirements are installed...
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo Requirements installed successfully!
) else (
    echo Requirements already installed.
)

:: Launch the application
echo ========================================
echo   Launching Llama.cpp Launcher...
echo ========================================
python launcher.py

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with error code !errorlevel!
    pause
)

endlocal