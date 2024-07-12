import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict
from src.database import PlayerDB
from src.const import *
app = FastAPI()


db = PlayerDB(DB_NAME)


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