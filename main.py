from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from src.components.models import EngageModel
from src.model_deployment import prepare_data_for_model
import pickle
import pandas as pd

app = FastAPI()

level = ['Medium', 'High', 'Low']

p_path = 'artifacts\\preprocessor.pkl'

model = EngageModel(11,3,"artifacts\\model-1.pth")
model.load_model()




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

@app.get('/predict')
async def scoring_endpoint(item:ScoringItem):
    data_point = item.dict()
    print(data_point)
    x_tensor = prepare_data_for_model(data_point, p_path)
    pred = model.predict(x_tensor)
    print(pred.item())
    return level[pred.item()]
    