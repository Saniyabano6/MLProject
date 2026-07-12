import sys
from dataclasses import dataclass
from src.utils import save_object
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os


@dataclass
class DataTransformationConfig:
    # Path where the fitted preprocessor object (scaler/encoder pipeline) will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        # Load config so we know where to save the preprocessor object
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        '''This function builds and returns the preprocessing pipeline
        (imputation + scaling for numeric columns, imputation + encoding + scaling for categorical columns)'''
        try:
            # Numeric columns to be scaled
            numericals_columns = ["writing_score", "reading_score"]

            # Categorical columns to be encoded
            categorical_columns = [
                "gender",
                "race_ethnicity",  # double-check this matches your CSV's actual column name
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numeric columns:
            # 1. Fill missing values with the median
            # 2. Scale values to have mean 0, std 1
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Pipeline for categorical columns:
            # 1. Fill missing values with the most frequent value
            # 2. One-hot encode the categories
            # 3. Scale the resulting encoded columns (with_mean=False needed for sparse OHE output)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numericals_columns}')
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            # Combine both pipelines into a single ColumnTransformer:
            # applies num_pipeline to numeric columns, cat_pipeline to categorical columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numericals_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the train and test CSVs produced by the data ingestion step
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Build the preprocessing pipeline
            preprocessing_obj = self.get_transformer_object()

            target_column_name = "math_score"
            numericals_columns = ["writing_score", "reading_score"]

            # Separate features (X) and target (y) for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features (X) and target (y) for test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            # Fit the preprocessor on TRAINING data only, then transform it
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Only TRANSFORM the test data (do NOT re-fit) — prevents data leakage
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with the target column into a single array
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            # Save the fitted preprocessor to disk so it can be reused later
            # (e.g. during inference on new/unseen data)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed train/test arrays and the path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)