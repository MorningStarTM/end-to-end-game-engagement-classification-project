from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from src.components.models import EngageModel
import pickle
import pandas as pd

app = FastAPI()

level = ['Medium', 'High', 'Low']

model = EngageModel(11,3,"notebook\\engage.pth")
model.load_model()

with open('artifacts\\preprocessor.pkl', 'rb') as file:
    loaded_le = pickle.load(file)
print("preprocessor loaded")

class ScoringItem(BaseModel):
    Age : int
    Gender : int
    Location : int
    GameGenre : int
    PlayTimeHours : float
    InGamePurchases : int
    GameDifficulty : int
    SessionsPerWeek : int
    AvgSessionDurationMinutes : int
    PlayerLevel : int
    AchievementsUnlocked : int

#Gender	Location	GameGenre	
# #PlayTimeHours	InGamePurchases	GameDifficulty	SessionsPerWeek		PlayerLevel	AchievementsUnlocked

@app.get('/')
async def scoring_endpoint(item:ScoringItem):
    data_point = item.dict().values()
    #df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    #df = df.apply(loaded_le., axis=0)
    #data_arr = df.to_numpy()
    data_arr = np.array([list(data_point)])
    x = torch.tensor(data_arr, dtype=torch.float32)
    pred = model.predict(x)
    print(pred.item())
    return level[pred.item()]
    