@echo off
REM Start Streamlit app script for Windows

echo Starting Student Performance Predictor Web App...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if model artifacts exist
if not exist "artifacts\model.pkl" (
    echo Model artifacts not found!
    echo Please run training first: scripts\train_model.bat
    pause
    exit /b 1
)

if not exist "artifacts\preprocessor.pkl" (
    echo Preprocessor artifacts not found!
    echo Please run training first: scripts\train_model.bat
    pause
    exit /b 1
)

REM Install requirements if needed
echo Checking requirements...
pip install -r requirements.txt

REM Start Streamlit app
echo Starting Streamlit app...
streamlit run app.py

echo App stopped.
pause
