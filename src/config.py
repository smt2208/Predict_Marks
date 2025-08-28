"""
Configuration settings for the Student Performance Predictor application
"""
import os

class Config:
    """Base configuration class"""
    
    # Project paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "notebook", "data")
    
    # Model files
    MODEL_FILE_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
    PREPROCESSOR_FILE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    RAW_DATA_FILE_PATH = os.path.join(ARTIFACTS_DIR, "data1.csv")
    TRAIN_DATA_FILE_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
    TEST_DATA_FILE_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
    
    # Data source
    SOURCE_DATA_PATH = os.path.join(DATA_DIR, "data.csv")
    
    # Model training parameters
    TEST_SIZE = 0.3
    RANDOM_STATE = 20
    MODEL_SCORE_THRESHOLD = 0.6
    
    # Valid input categories
    VALID_GENDERS = ["male", "female"]
    VALID_RACE_ETHNICITY = ["group A", "group B", "group C", "group D", "group E"]
    VALID_EDUCATION_LEVELS = [
        "some high school", "high school", "some college", 
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    VALID_LUNCH_TYPES = ["standard", "free/reduced"]
    VALID_TEST_PREP = ["none", "completed"]
    
    # Feature columns
    NUMERICAL_COLUMNS = ["writing_score", "reading_score"]
    CATEGORICAL_COLUMNS = [
        "gender", "race_ethnicity", "parental_level_of_education",
        "lunch", "test_preparation_course"
    ]
    TARGET_COLUMN = "math_score"
    
    # Score ranges
    MIN_SCORE = 0
    MAX_SCORE = 100
    
    # Logging configuration
    LOG_FORMAT = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Default configuration
config = DevelopmentConfig()
