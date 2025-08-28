import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.config import config
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion parameters"""
    train_data_path: str = config.TRAIN_DATA_FILE_PATH
    test_data_path: str = config.TEST_DATA_FILE_PATH
    raw_data_path: str = config.RAW_DATA_FILE_PATH

class DataIngestion:
    """
    Class responsible for ingesting data from source and splitting into train/test sets
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process
        
        Returns:
            tuple: Paths to train and test data files
            
        Raises:
            CustomException: If data ingestion fails
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv(config.SOURCE_DATA_PATH)
            logging.info('Read the dataset as dataframe')
            
            # Validate data
            self._validate_data(df)
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Split the data
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE
            )
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
            logging.info("Ingestion of the data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the ingested data
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Check if required columns exist
            required_columns = config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS + [config.TARGET_COLUMN]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for minimum number of rows
            if len(df) < 10:
                raise ValueError("Dataset too small (less than 10 rows)")
            
            # Check for target column values
            if df[config.TARGET_COLUMN].isnull().all():
                raise ValueError("Target column has no valid values")
            
            # Log data statistics
            logging.info(f"Data validation successful. Shape: {df.shape}")
            logging.info(f"Target column statistics:\n{df[config.TARGET_COLUMN].describe()}")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrain = ModelTrainer()
    print(modeltrain.initiate_model_trainer(train_array, test_array))