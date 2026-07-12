import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    # These are the file paths (inputs) that we give to the data ingestion component
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")


class DataIngestion:
    def __init__(self):
        # Load the config so we know where to save the raw/train/test data
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestionn(self):
        logging.info("Enter the data ingestion method or component")

        try:
            # Read the raw dataset from disk
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as DataFrame')

            # Create the 'artifacts' folder if it doesn't exist yet
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save an unsplit copy of the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split into 80% train, 20% test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the split datasets to their own CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the file paths so the next pipeline stage can load them
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__== "__main__":
    # Step 1: Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestionn()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
