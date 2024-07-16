import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from dataclasses import dataclass
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder
import torch


import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from dataclasses import dataclass
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder
import torch

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataPreprocessor:
    def __init__(self, train_csv_path, test_csv_path, target_column, categorical_columns, exclude_columns=[]):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.exclude_columns = exclude_columns
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_columns}
        self.data_transform = DataTransformationConfig()


    def load_and_encode_data(self):
        # Load data
        train_data = pd.read_csv(self.train_csv_path, index_col=False)
        test_data = pd.read_csv(self.test_csv_path, index_col=False)

        # Separate categorical and numerical columns
        train_categorical = train_data[self.categorical_columns].copy()
        train_numerical = train_data.drop(columns=self.categorical_columns)

        test_categorical = test_data[self.categorical_columns].copy()
        test_numerical = test_data.drop(columns=self.categorical_columns)

        # Apply LabelEncoder to categorical columns
        for col in self.categorical_columns:
            train_categorical[col] = self.label_encoders[col].fit_transform(train_categorical[col])
            test_categorical[col] = self.label_encoders[col].transform(test_categorical[col])

        # Merge dataframes back together
        df_train = pd.concat([train_numerical, train_categorical], axis=1)
        df_test = pd.concat([test_numerical, test_categorical], axis=1)
        logging.info("Data transformation applied successfully")

        label_mapping = {'High': 2, 'Medium': 1, 'Low': 0}
        df_train['EngagementLevel'] = df_train['EngagementLevel'].map(label_mapping)
        df_test['EngagementLevel'] = df_test['EngagementLevel'].map(label_mapping)

        # Save the encoders
        save_object(self.data_transform.preprocessor_obj_file_path, self.label_encoders)
        logging.info(f"Transformation module saved as pickle file at {self.data_transform.preprocessor_obj_file_path}")
        
        return df_train, df_test

    def to_tensors(self, df_train, df_test):
        x_train = df_train.drop(columns=[self.target_column] + ['PlayerID'])
        y_train = df_train[self.target_column]

        x_test = df_test.drop(columns=[self.target_column] + ['PlayerID'])
        y_test = df_test[self.target_column]

        X_train_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)  # Assuming target is categorical
        X_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
        
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
