@echo off
REM Training script for Windows - Student Performance Predictor

echo Starting model training pipeline...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install requirements
echo Installing/updating requirements...
pip install -r requirements.txt

REM Run training
echo Running training pipeline...
python src\pipeline\train_pipeline.py

REM Check if training was successful
if %errorlevel% equ 0 (
    echo Training completed successfully!
    echo Model artifacts saved in artifacts\ directory
) else (
    echo Training failed!
    exit /b 1
)

echo Training pipeline completed.
pause
