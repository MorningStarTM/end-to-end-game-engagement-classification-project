from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from src.components.models import EngageModel
import pickle
import pandas as pd

app = FastAPI()

level = ['Medium', 'High', 'Low']

p_path = 'artifacts\\preprocessor.pkl'

model = EngageModel(11,3,"artifacts\\model-1.pth")
model.load_model()

def load_object(path):
    with open(path, 'rb') as file:
        loaded_le = pickle.load(file)
        print("preprocessor loaded")
    return  loaded_le


def prepare_data_for_model(data, encoders_path):
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

    exclude_columns = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
    categorical_columns = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
    target_column = "target"
    # Load the saved label encoders
    label_encoders = load_object(encoders_path)

    # Convert incoming data to a DataFrame
    data_df = pd.DataFrame([data])

    # Separate categorical and numerical columns
    data_categorical = data_df[categorical_columns].copy()
    data_numerical = data_df.drop(columns=categorical_columns)

    # Apply LabelEncoder to categorical columns
    for col in categorical_columns:
        if col in label_encoders:
            data_categorical[col] = label_encoders[col].transform(data_categorical[col])
        else:
            raise ValueError(f"No encoder found for column: {col}")

    # Merge dataframes back together
    data_transformed = pd.concat([data_numerical, data_categorical], axis=1)

    # Drop the target column if it exists in the incoming data
    if target_column in data_transformed.columns:
        data_transformed = data_transformed.drop(columns=[target_column])

    # Convert to numpy array and then to tensor
    
    X_numpy = data_transformed.to_numpy()
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

    return X_tensor


class ScoringItem(BaseModel):
    Age : int
    Gender : str
    Location : str
    GameGenre : str
    PlayTimeHours : float
    InGamePurchases : int
    GameDifficulty : str
    SessionsPerWeek : int
    AvgSessionDurationMinutes : int
    PlayerLevel : int
    AchievementsUnlocked : int

#Gender	Location	GameGenre	
# #PlayTimeHours	InGamePurchases	GameDifficulty	SessionsPerWeek		PlayerLevel	AchievementsUnlocked

@app.get('/')
async def scoring_endpoint(item:ScoringItem):
    data_point = item.dict()
    print(data_point)
    x_tensor = prepare_data_for_model(data_point, p_path)
    pred = model.predict(x_tensor)
    print(pred.item())
    return level[pred.item()]
    