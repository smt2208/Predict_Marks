import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from src.utils import save_object  # There in utils

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.config import config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def get_data_transformer_object(self):  # Function for data transformation
        try:
            numerical_columns = config.NUMERICAL_COLUMNS
            categorical_columns = config.CATEGORICAL_COLUMNS

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            # Combine
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):  # when data transform is triggered
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of training and testing data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = config.TARGET_COLUMN

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")
            # fit and transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Replace any remaining NaNs with feature-wise median
            input_feature_train_arr = np.nan_to_num(
                input_feature_train_arr, nan=np.nanmedian(input_feature_train_arr)
            )
            input_feature_test_arr = np.nan_to_num(
                input_feature_test_arr, nan=np.nanmedian(input_feature_test_arr)
            )

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)