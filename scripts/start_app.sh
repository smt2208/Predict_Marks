#!/bin/bash
# Start Streamlit app script

echo "Starting Student Performance Predictor Web App..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model artifacts exist
if [ ! -f "artifacts/model.pkl" ] || [ ! -f "artifacts/preprocessor.pkl" ]; then
    echo "Model artifacts not found!"
    echo "Please run training first: bash scripts/train_model.sh"
    exit 1
fi

# Install requirements if needed
echo "Checking requirements..."
pip install -r requirements.txt

# Start Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py

echo "App stopped."
