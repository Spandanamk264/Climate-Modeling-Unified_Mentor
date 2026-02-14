@echo off
echo Starting Climate Change Modeling Setup...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in your PATH.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo or use Anaconda/Miniconda.
    echo.
    echo After installing, run this script again.
    pause
    exit /b
)

:: Create Virtual Environment
if not exist "env" (
    echo Creating virtual environment...
    python -m venv env
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. 
        echo Please ensure you have the 'venv' module installed (standard in Python 3).
        pause
        exit /b
    )
) else (
    echo Virtual environment 'env' already exists.
)


:: Activate and Install
echo Activating environment...
call env\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Launching Climate Modeling Dashboard...
streamlit run src/app.py

pause
