import sqlite3
import os
import pandas
import pandas as pd
import numpy as np
from src.logger import logging


class PlayerDB:
    def __init__(self, db_file):
        self.db_file = os.path.join('db',db_file)
        self.conn = None
        self.cursor = None
        self._initialize_database()


    def _initialize_database(self):
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS Players (
            PlayerID INTEGER PRIMARY KEY,
            Age INTEGER,
            Gender TEXT,
            Location TEXT,
            GameGenre TEXT,
            PlayTimeHours REAL,
            InGamePurchases INTEGER,
            GameDifficulty TEXT,
            SessionsPerWeek INTEGER,
            AvgSessionDurationMinutes REAL,
            PlayerLevel INTEGER,
            AchievementsUnlocked INTEGER,
            EngagementLevel TEXT
        );
        '''
        self.cursor.execute(create_table_query)
        self.conn.commit()
        logging.info("database initialized")

    def create_player(self, player_data):
        try:
            insert_query = '''
            INSERT INTO Players (PlayerID, Age, Gender, Location, GameGenre, 
                                PlayTimeHours, InGamePurchases, GameDifficulty, 
                                SessionsPerWeek, AvgSessionDurationMinutes, 
                                PlayerLevel, AchievementsUnlocked, EngagementLevel)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            '''
            self.cursor.execute(insert_query, player_data)
            self.conn.commit()
            return (True, "Player data added successfully.")
        
        except sqlite3.Error as er:
            return (False, f"Error {er}")



    def read_player(self, player_id):
        try:
            select_query = '''
            SELECT * FROM Players WHERE PlayerID = ?;
            '''
            self.cursor.execute(select_query, (player_id,))
            player = self.cursor.fetchone()
            return {"player":player, "error":None}
        except sqlite3.Error as er:
            return {"player":None, "error":f"Error: {er}"}

    def update_player(self, player_id, updated_data):
        try:
            update_query = '''
            UPDATE Players SET Age=?, Gender=?, Location=?, GameGenre=?, 
                            PlayTimeHours=?, InGamePurchases=?, GameDifficulty=?, 
                            SessionsPerWeek=?, AvgSessionDurationMinutes=?, 
                            PlayerLevel=?, AchievementsUnlocked=?, EngagementLevel=?
            WHERE PlayerID=?;
            '''
            updated_data.append(player_id)
            self.cursor.execute(update_query, updated_data)
            self.conn.commit()
            return (True, "Player data updated successfully.")
        
        except sqlite3.Error as er:
            return (False, f"Error: {er}")


    
    def delete_player(self, player_id):
        try:
            delete_query = '''
            DELETE FROM Players WHERE PlayerID=?;
            '''
            self.cursor.execute(delete_query, (player_id,))
            self.conn.commit()

            if self.cursor.rowcount == 0:
                return (False, "No player found with the given ID.")
            
            logging.info(f"Player {player_id} deleted")
            return (True, "Player data deleted successfully.")
            
        except sqlite3.Error as er:
            return (False, f"Error: {er}")


    def add_players_from_csv(self, csv_file):

        """
        This function for add data from csv file into database

        Args:
            csv_file (str)
        """
        # Read CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_file)

            # Insert DataFrame rows into SQLite database
            df.to_sql('Players', self.conn, if_exists='append', index=False)

            print("Players data added from CSV successfully.")
            logging.info(f"data added via csv file")
        except sqlite3.Error as er:
            return (er.sqlite_errorcode, er.sqlite_errorname)

    def count_players(self):
        try:
            count_query = '''
            SELECT COUNT(*) FROM Players;
            '''
            self.cursor.execute(count_query)
            count = self.cursor.fetchone()[0]
            return (True, count)
        except sqlite3.Error as er:
            return (False, f"Error : {er}")



    
    def close_connection(self):
        self.conn.close()
        print("Database connection closed.")

    