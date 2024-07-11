import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict
from src.database import PlayerDB
app = FastAPI()


db = PlayerDB("players.db")


#endpoint for create player record 
@app.post("/players/")
async def create_player(player_data: Dict):
    success, message = db.create_player(list(player_data.values()))
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=400, detail=message)
    


#endpoint for get player records
@app.get("/players/{player_id}")
async def read_player(player_id: int):
    player = db.read_player(player_id)
    if player:
        return {"Player": player}
    else:
        raise HTTPException(status_code=404, detail="Player not found")



# endpoint for update player
@app.put("/players/{player_id}")
async def update_player(player_id: int, updated_data: Dict):
    success, message = db.update_player(player_id, list(updated_data.values()))
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=400, detail=message)



# endpoint for delete record
@app.delete("/players/{player_id}")
async def delete_player(player_id: int):
    success, message = db.delete_player(player_id)
    if success:
        return {"message": message}
    else:
        raise HTTPException(status_code=400, detail=message)


# endpoint for add data from csv file
@app.post("/players/csv/")
async def add_players_from_csv(csv_file: UploadFile = File(...)):
    file_path = f"temp/{csv_file.filename}"
    with open(file_path, "wb") as f:
        f.write(await csv_file.read())
    db.add_players_from_csv(file_path)
    return {"message": "Players added from CSV successfully"}


# end point for get total count of data
@app.get("/players/count")
async def count_players():
    success, result = db.count_players()
    
    if success:
        return {"PlayerCount": result}
    else:
        raise HTTPException(status_code=400, detail=result)
