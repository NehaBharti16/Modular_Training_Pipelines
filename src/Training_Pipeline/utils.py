import os
import sys
from src.Training_Pipeline.logger import logging
from src.Training_Pipeline.exception import CustomException
import pandas as pd

def read_csv_data(file_path: str) ->pd.DataFrame:
    logging.info("Reading csv file started")
    try:
        df = pd.read_csv(file_path)
        print(df.head())
        logging.info(f"CSV file loaded successfully with shape {df.shape}")
        return df
    except Exception as ex:
        raise CustomException(ex)