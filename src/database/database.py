import sqlite3
import os
import pandas




class PlayerDB:
    def __init__(self, db_file):
        self.db_file = db_file
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


    def create_player(self, player_data):
        insert_query = '''
        INSERT INTO Players (PlayerID, Age, Gender, Location, GameGenre, 
                             PlayTimeHours, InGamePurchases, GameDifficulty, 
                             SessionsPerWeek, AvgSessionDurationMinutes, 
                             PlayerLevel, AchievementsUnlocked, EngagementLevel)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        self.cursor.execute(insert_query, player_data)
        self.conn.commit()
        print("Player data added successfully.")


    def read_player(self, player_id):
        select_query = '''
        SELECT * FROM Players WHERE PlayerID = ?;
        '''
        self.cursor.execute(select_query, (player_id,))
        player = self.cursor.fetchone()
        return player
    

    def update_player(self, player_id, updated_data):
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
        print("Player data updated successfully.")

    
    def delete_player(self, player_id):
        delete_query = '''
        DELETE FROM Players WHERE PlayerID=?;
        '''
        self.cursor.execute(delete_query, (player_id,))
        self.conn.commit()
        print("Player data deleted successfully.")

    
    def close_connection(self):
        self.conn.close()
        print("Database connection closed.")

    