import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict
from src.database import PlayerDB
from src.const import *
import uvicorn
from pydantic import BaseModel
import torch
import numpy as np
from src.components.models import EngageModel
from src.model_deployment import prepare_data_for_model

app = FastAPI()

level = ['Medium', 'High', 'Low']
p_path = 'artifacts\\preprocessor.pkl'

model = EngageModel(11,3, "artifacts\\testModel.pth")
model.load_model()
db = PlayerDB(DB_NAME)



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


@app.get('/predict')
async def scoring_endpoint(item:ScoringItem):
    try:
        data_point = item.dict()
        #print(data_point)
        x_tensor = prepare_data_for_model(data_point)
        pred = model.predict(x_tensor)
        print(pred.item())
        return level[pred.item()]
    except Exception as e:
        raise HTTPException(status_code=STATUS_BAD_REQUEST, detail=e)


@app.get("/players/count")
async def count_players():
    success, result = db.count_players()
    if success:
        return {"PlayerCount": result}
    else:
        raise HTTPException(status_code=STATUS_BAD_REQUEST, detail=result)

# Endpoint for creating a player record
@app.post("/players/")
async def create_player(player_data: Dict):
    success, message = db.create_player(list(player_data.values()))
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=STATUS_BAD_REQUEST, detail=message)

# Endpoint for getting player records
@app.get("/players/{player_id}")
async def read_player(player_id: int):
    result = db.read_player(player_id)
    if result["error"] is None:
        return {"player": result["player"]}
    else:
        raise HTTPException(status_code=STATUS_NOT_FOUND, detail=result["error"])

# Endpoint for updating a player record
@app.put("/players/{player_id}")
async def update_player(player_id: int, updated_data: Dict):
    success, message = db.update_player(player_id, list(updated_data.values()))
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=STATUS_BAD_REQUEST, detail=message)

# Endpoint for deleting a player record
@app.delete("/players/{player_id}")
async def delete_player(player_id: int):
    success, message = db.delete_player(player_id)
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=STATUS_BAD_REQUEST, detail=message)

# Endpoint for adding players from a CSV file
@app.post("/players/csv/")
async def add_players_from_csv(csv_file: UploadFile = File(...)):
    try:
        file_path = f"temp/{csv_file.filename}"
        with open(file_path, "wb") as f:
            f.write(await csv_file.read())
        success, message = db.add_players_from_csv(file_path)
        if success:
            return {"message": "Players added from CSV successfully"}
        else:
            raise HTTPException(status_code=STATUS_BAD_REQUEST, detail=message)
    except Exception as e:
        raise HTTPException(status_code=STATUS_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")
    


if __name__ == "__main__":
    uvicorn.run("main_db:app", host="127.0.0.1", port=8000, reload=True)
