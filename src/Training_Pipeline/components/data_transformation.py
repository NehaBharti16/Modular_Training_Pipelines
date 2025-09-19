import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

from src.Training_Pipeline.utils import save_object

from src.Training_Pipeline.logger import logging
from src.Training_Pipeline.exception import CustomException
import os



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, input_feature_df):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = [col for col in input_feature_df.columns if pd.api.types.is_numeric_dtype(input_feature_df[col])]
            categorical_columns = [col for col in input_feature_df.columns if not pd.api.types.is_numeric_dtype(input_feature_df[col])]
            
            num_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5, weights='uniform', missing_values=np.nan)),
                ("scaler", RobustScaler()),
                ("power", PowerTransformer(method='yeo-johnson'))
            ]) 

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ]) 

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            

            target_column_name = "Median_House_Value"
            # numerical_columns = [col for col in train_df.columns if pd.api.types.is_numeric_dtype(train_df[col])]
            # Dividing the train dataset into input and target features

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Dividing the test dataset into input and target features

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(str(e), sys)    

