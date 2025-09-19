from src.Training_Pipeline.logger import logging
from src.Training_Pipeline.exception import CustomException
from src.Training_Pipeline.components.data_ingestion import DataIngestion
from src.Training_Pipeline.components.data_ingestion import DataIngestionConfig
from src.Training_Pipeline.components.data_transformation import DataTransformationConfig, DataTransformation
from src.Training_Pipeline.utils import read_csv_data, save_object
# from src.Training_Pipeline.components.model_trainer import ModelTrainerConfig, ModelTrainer
import sys


if __name__=="__main__":
    logging.info("The execution has started")


    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

        # data_transformation_config= DataTransformationConfig()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)