import pandas as pd
import torch
import pickle


def load_object(path):
    with open(path, 'rb') as file:
        loaded_le = pickle.load(file)
        print("preprocessor loaded")
    return  loaded_le

def prepare_data_for_model(data):
    """
    Prepares incoming data for model prediction.
    
    Parameters:
    - data (dict): Incoming data for prediction
    - encoders_path (str): Path to the saved encoders
    - target_column (str): The target column name
    - categorical_columns (list): List of categorical columns to be encoded
    - exclude_columns (list): List of columns to exclude from the features
    
    Returns:
    - X_tensor (torch.Tensor): Features tensor for model prediction
    """


    # Convert incoming data to a DataFrame
    data_df = pd.DataFrame([data])

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

    data_df['Location'] = data_df['Location'].map(location)

    data_df['GameGenre'] = data_df['GameGenre'].map(gameGenre)

    data_df['GameDifficulty'] = data_df['GameDifficulty'].map(difficulty)

    data_df['Gender'] = data_df['Gender'].map(gender)    

    # Convert to numpy array and then to tensor
    X_numpy = data_df.to_numpy()
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

    return X_tensor

