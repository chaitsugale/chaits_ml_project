import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object

from exception import CustomException
from logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path :str = os.path.join('artifacts','preprocesor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            num_columns = ["writing_score", "reading_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Columns : {num_columns}")

            cat_pipeline=Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns : {cat_columns}")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Otaining preprocessing object")

            preprocessing_obj = self.get_transformer_object()

            target_column_name = "math_score"
            num_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            a = np.array(input_feature_train_df)
            logging.info(f"Shape : {a.shape}")

            

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            
            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )



            
        except Exception as e:
            raise CustomException(e,sys)
            
            
