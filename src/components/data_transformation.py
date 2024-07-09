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
    def __init__(self, train_csv_path, test_csv_path, target_column, exclude_columns=[]):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.target_column = target_column
        self.exclude_columns = exclude_columns
        self.label_encoder = LabelEncoder()
        self.data_transform = DataTransformationConfig()

    def load_and_encode_data(self):
        # Load data
        train_data = pd.read_csv(self.train_csv_path, index_col=False)
        test_data = pd.read_csv(self.test_csv_path, index_col=False)

        # Apply LabelEncoder to all columns
        df_train = train_data.apply(self.label_encoder.fit_transform, axis=0)
        df_test = test_data.apply(self.label_encoder.fit_transform, axis=0)

        # Save the encoder
        save_object(self.data_transform.preprocessor_obj_file_path, self.label_encoder)
        
        return df_train, df_test

    def to_tensors(self, df_train, df_test):
        x_train = df_train.drop(columns=[self.target_column] + self.exclude_columns)
        y_train = df_train[self.target_column]

        x_test = df_test.drop(columns=[self.target_column] + self.exclude_columns)
        y_test = df_test[self.target_column]

        X_train_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)  # Assuming target is categorical
        X_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
        
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
