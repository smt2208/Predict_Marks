import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training(self):
        """
        Start the complete training pipeline
        """
        try:
            logging.info("Training pipeline started")
            
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train: {train_data_path}, Test: {test_data_path}")
            
            # Step 2: Data Transformation
            logging.info("Starting data transformation")
            train_array, test_array, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
            
            # Step 3: Model Training
            logging.info("Starting model training")
            r2_score = self.model_trainer.initiate_model_trainer(train_array, test_array)
            logging.info(f"Model training completed. R2 Score: {r2_score}")
            
            logging.info("Training pipeline completed successfully")
            return r2_score
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        score = pipeline.start_training()
        print(f"Training completed successfully with R2 Score: {score}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise CustomException(e, sys)