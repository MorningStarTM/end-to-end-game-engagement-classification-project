import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict
from src.database import PlayerDB
app = FastAPI()


db = PlayerDB("players.db")

@app.post("/players/")
async def create_player(player_data: Dict):
    player_id = db.create_player(list(player_data.values()))
    return {"message": "Player added successfully", "PlayerID": player_id}

@app.get("/players/{player_id}")
async def read_player(player_id: int):
    player = db.read_player(player_id)
    if player:
        return {"Player": player}
    else:
        raise HTTPException(status_code=404, detail="Player not found")

@app.put("/players/{player_id}")
async def update_player(player_id: int, updated_data: Dict):
    db.update_player(player_id, list(updated_data.values()))
    return {"message": "Player data updated successfully"}

@app.delete("/players/{player_id}")
async def delete_player(player_id: int):
    db.delete_player(player_id)
    return {"message": "Player deleted successfully"}

@app.post("/players/csv/")
async def add_players_from_csv(csv_file: UploadFile = File(...)):
    file_path = f"temp/{csv_file.filename}"
    with open(file_path, "wb") as f:
        f.write(await csv_file.read())
    db.add_players_from_csv(file_path)
    return {"message": "Players added from CSV successfully"}

@app.get("/players/count/")
async def count_players():
    count = db.count_players()
    return {"PlayerCount": count}
