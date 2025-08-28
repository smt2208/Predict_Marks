import os
import sys

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for name, model in models.items():
            try:
                search_space = param.get(name, {})
                if search_space:
                    gs = GridSearchCV(model, search_space, cv=3, n_jobs=-1, scoring="r2")
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)
                    logging.info(f"{name}: best params {gs.best_params_}")
                # Fit final model
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[name] = test_model_score
            except Exception as inner_e:
                logging.warning(f"Model {name} failed during evaluation: {inner_e}")
        return report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) 
