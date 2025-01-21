import pandas as pd
import pickle
import os
from naiveElo import EloModel, Player, Match


def load_elo_model(filename: str):
    file_path = os.path.join(os.path.dirname(__file__), filename) #Added to handle file paths correctly
    with open(file_path, 'rb') as file: #Fixed file opening
        return pickle.load(file) #Fixed pickle loading

eloModel = load_elo_model('elo_model_atp.pkl')
help(eloModel)

player_data = []
for player_id, player in eloModel.players.items():
    player_data.append({'Player Name': player.name, 'Overall ELO': player.elo.get('overall', None), **player.elo})

df = pd.DataFrame(player_data)
top100 = df.sort_values(by='Overall ELO', ascending=False).head(100)
top100['Overall ELO'].mean()
top100

df[df['Player Name'] == 'Adrian Mannarino']