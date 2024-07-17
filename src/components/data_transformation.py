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
    def __init__(self, train_csv_path, test_csv_path, target_column):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.target_column = target_column
        self.data_transform = DataTransformationConfig()


    def load_and_encode_data(self):
        # Load data
        df_train = pd.read_csv(self.train_csv_path, index_col=False)
        df_test = pd.read_csv(self.test_csv_path, index_col=False)

        label_mapping = {'High': 2, 'Medium': 1, 'Low': 0}
        gameGenre = {
            'Strategy': 0, 
            'Sports' :1, 
            'Action':2, 
            'RPG':3, 
            'Simulation':4
        }

        location = {
            'Other':0, 
            'USA':1, 
            'Europe':2, 
            'Asia':3
        }

        difficulty = {
            'Medium':1, 
            'Easy':0, 
            'Hard':2
        }

        gender = {
            'Male':0,
            'Female':1
        }

        df_train['EngagementLevel'] = df_train['EngagementLevel'].map(label_mapping)
        df_test['EngagementLevel'] = df_test['EngagementLevel'].map(label_mapping)

        df_train['Location'] = df_train['Location'].map(location)
        df_test['Location'] = df_test['Location'].map(location)

        df_train['GameGenre'] = df_train['GameGenre'].map(gameGenre)
        df_test['GameGenre'] = df_test['GameGenre'].map(gameGenre)

        df_train['GameDifficulty'] = df_train['GameDifficulty'].map(difficulty)
        df_test['GameDifficulty'] = df_test['GameDifficulty'].map(difficulty)  

        df_train['Gender'] = df_train['Gender'].map(gender)
        df_test['Gender'] = df_test['Gender'].map(gender)      
        logging.info("Data transformation applied successfully")
        
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
