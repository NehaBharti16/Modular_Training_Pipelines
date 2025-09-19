import os
import sys
from src.Training_Pipeline.logger import logging
from src.Training_Pipeline.exception import CustomException
import pandas as pd
import numpy as np
import pickle

def read_csv_data(file_path: str) ->pd.DataFrame:
    logging.info("Reading csv file started")
    try:
        df = pd.read_csv(file_path)
        print(df.head())
        logging.info(f"CSV file loaded successfully with shape {df.shape}")
        return df
    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)