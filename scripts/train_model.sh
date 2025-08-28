#!/bin/bash
# Training script for the Student Performance Predictor

echo "Starting model training pipeline..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements
echo "Installing/updating requirements..."
pip install -r requirements.txt

# Run training
echo "Running training pipeline..."
python src/pipeline/train_pipeline.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model artifacts saved in artifacts/ directory"
else
    echo "Training failed!"
    exit 1
fi

echo "Training pipeline completed."
