import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """Saves any Python object (e.g. a trained model or preprocessor) to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)

        # Make sure the target directory exists before saving
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and tunes each model in `models` using GridSearchCV with the
    hyperparameters in `param`, evaluates each on the test set, and
    returns a dict of {model_name: test_r2_score}.
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Get this model's hyperparameter grid (empty dict if none specified)
            para = param[model_name]

            # Run grid search with 3-fold cross-validation to find the best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Refit the model using the best parameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on both train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute R2 scores for both
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report, keyed by model name
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Loads a pickled object (model or preprocessor) back from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)