from src.Training_Pipeline.logger import logging
from src.Training_Pipeline.exception import CustomException
from src.Training_Pipeline.components.data_ingestion import DataIngestion
from src.Training_Pipeline.components.data_ingestion import DataIngestionConfig
import sys


if __name__=="__main__":
    logging.info("The execution has started")


    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)