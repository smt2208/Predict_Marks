# Student Performance Predictor

A machine learning web application that predicts student math scores based on various demographic and academic factors.

## Features

- **Web Interface**: Interactive Streamlit application for easy predictions
- **Multiple ML Models**: Utilizes various regression algorithms including Random Forest, XGBoost, CatBoost, and more
- **Automated Pipeline**: Complete ML pipeline with data ingestion, transformation, and model training
- **Robust Architecture**: Modular design with proper exception handling and logging

## Project Structure

```
├── app.py                          # Streamlit web application
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading and splitting
│   │   ├── data_transformation.py  # Feature preprocessing
│   │   └── model_trainer.py        # Model training and evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Training pipeline
│   │   └── predict_pipeline.py     # Prediction pipeline
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Utility functions
├── artifacts/                      # Generated model artifacts
├── logs/                          # Application logs
└── notebook/                      # Jupyter notebooks for EDA
```

## Dataset Features

The model predicts **math_score** based on the following features:
- **gender**: Student gender (male/female)
- **race_ethnicity**: Ethnic group (group A through E)
- **parental_level_of_education**: Parent's highest education level
- **lunch**: Lunch type (standard/free or reduced)
- **test_preparation_course**: Test prep completion status
- **reading_score**: Reading score (0-100)
- **writing_score**: Writing score (0-100)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Predict_Marks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional):
```bash
python src/pipeline/train_pipeline.py
```

## Usage

### Web Application
Run the Streamlit app:
```bash
streamlit run app.py
```

### Training Pipeline
To retrain the model:
```bash
python src/pipeline/train_pipeline.py
```

### API Usage
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create prediction data
data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=85,
    writing_score=80
)

# Make prediction
pipeline = PredictPipeline()
result = pipeline.predict(data.get_data_as_data_frame())
print(f"Predicted Math Score: {result[0]:.2f}")
```

## Model Performance

The system automatically selects the best performing model from multiple algorithms:
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

Models are evaluated using R² score and the best performer is saved for predictions.

## Technical Details

- **Framework**: Streamlit for web interface
- **ML Libraries**: scikit-learn, XGBoost, CatBoost
- **Data Processing**: pandas, numpy
- **Model Persistence**: dill for serialization
- **Logging**: Python's built-in logging module

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.