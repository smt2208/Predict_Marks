import numpy as np
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.config import config

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Replace NaNs and infs in training/testing arrays, just in case
            X_train = np.nan_to_num(X_train, nan=np.nanmedian(X_train))
            X_test = np.nan_to_num(X_test, nan=np.nanmedian(X_test))
            y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))
            y_test = np.nan_to_num(y_test, nan=np.nanmedian(y_test))

            # Provide deterministic results where possible
            models = {
                "Random Forest": RandomForestRegressor(random_state=config.RANDOM_STATE),
                "Decision Tree": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
                "Gradient Boosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=config.RANDOM_STATE, verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=config.RANDOM_STATE),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=config.RANDOM_STATE),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params
            )

            if not model_report:
                raise CustomException("Model evaluation returned empty report", sys)

            best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])
            best_model = models[best_model_name]

            if best_model_score < config.MODEL_SCORE_THRESHOLD:
                raise CustomException(
                    f"No model met threshold {config.MODEL_SCORE_THRESHOLD}. Best: {best_model_name}={best_model_score:.3f}",
                    sys,
                )

            logging.info(
                f"Best model: {best_model_name} with R2={best_model_score:.4f} (threshold {config.MODEL_SCORE_THRESHOLD})"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, #saving the model
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)