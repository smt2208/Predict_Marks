import sys
import pandas as pd
from typing import Union
from src.exception import CustomException
from src.utils import load_object
import os 
from src.config import config

class PredictPipeline:
    """
    Pipeline for making predictions using trained model and preprocessor
    """
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame) -> Union[list, None]:
        """
        Make predictions on input features
        
        Args:
            features (pd.DataFrame): Input features as DataFrame
            
        Returns:
            Union[list, None]: Prediction results
        """
        try:
            model_path = config.MODEL_FILE_PATH
            preprocessor_path = config.PREPROCESSOR_FILE_PATH
            
            # Check if model files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    """
    Custom data class for handling user input data
    """
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: Union[int, float],
        writing_score: Union[int, float]
    ):
        # Validate inputs
        self._validate_inputs(
            gender, race_ethnicity, parental_level_of_education,
            lunch, test_preparation_course, reading_score, writing_score
        )
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = float(reading_score)
        self.writing_score = float(writing_score)

    def _validate_inputs(self, gender, race_ethnicity, parental_level_of_education,
                        lunch, test_preparation_course, reading_score, writing_score):
        """
        Validate input parameters
        """
        # Normalize string inputs (strip & lower) before validation
        def norm(v):
            return v.strip().lower() if isinstance(v, str) else v
        gender = norm(gender)
        race_ethnicity = norm(race_ethnicity)
        parental_level_of_education = norm(parental_level_of_education)
        lunch = norm(lunch)
        test_preparation_course = norm(test_preparation_course)

        # Valid categories from config (already lower-case in config assumptions)
        valid_genders = [v.lower() for v in config.VALID_GENDERS]
        valid_races = [v.lower() for v in config.VALID_RACE_ETHNICITY]
        valid_education = [v.lower() for v in config.VALID_EDUCATION_LEVELS]
        valid_lunch = [v.lower() for v in config.VALID_LUNCH_TYPES]
        valid_test_prep = [v.lower() for v in config.VALID_TEST_PREP]

        # Validate categorical variables
        if gender not in valid_genders:
            raise ValueError(f"Gender must be one of {valid_genders}")
        if race_ethnicity not in valid_races:
            raise ValueError(f"Race/ethnicity must be one of {valid_races}")
        if parental_level_of_education not in valid_education:
            raise ValueError(f"Parental education must be one of {valid_education}")
        if lunch not in valid_lunch:
            raise ValueError(f"Lunch must be one of {valid_lunch}")
        if test_preparation_course not in valid_test_prep:
            raise ValueError(f"Test preparation course must be one of {valid_test_prep}")

        # Validate numerical variables
        try:
            reading_score = float(reading_score)
            writing_score = float(writing_score)
        except (ValueError, TypeError):
            raise ValueError("Reading and writing scores must be numeric")

        if not (config.MIN_SCORE <= reading_score <= config.MAX_SCORE):
            raise ValueError(f"Reading score must be between {config.MIN_SCORE} and {config.MAX_SCORE}")
        if not (config.MIN_SCORE <= writing_score <= config.MAX_SCORE):
            raise ValueError(f"Writing score must be between {config.MIN_SCORE} and {config.MAX_SCORE}")

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convert custom data to pandas DataFrame
        
        Returns:
            pd.DataFrame: Input data as DataFrame
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)  