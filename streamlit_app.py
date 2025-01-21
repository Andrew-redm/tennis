import streamlit as st
import pandas as pd
import pickle
from accessDB import get_upcoming_matches
from naiveElo import EloModel, Player, Match
import os

def load_elo_model(filename: str):
    file_path = os.path.join(os.path.dirname(__file__), filename) #Added to handle file paths correctly
    with open(file_path, 'rb') as file: #Fixed file opening
        return pickle.load(file) #Fixed pickle loading

elo_models = {
    'atp': None,
    'wta': None
}

def calculate_expected_score(p1, p2) -> float:
    if p1 is None or p2 is None:
        return None
    return 1 / (1 + 10 ** ((p2 - p1) / 400))


def get_player_elo(player_id, tour, surface='overall'):
    elo = elo_models[tour]
    player = elo.get_player(player_id)
    if not player:
        return None
    if surface in player.elo:
        return player.elo[surface]
    return player.elo['overall']

def prepare_data(tour, elo):
    matches = get_upcoming_matches(tour) #Get matches only once
    matches['Player1 ELO'] = matches['ID1'].apply(lambda id: get_player_elo(id, tour)) #Vectorized operation
    matches['Player2 ELO'] = matches['ID2'].apply(lambda id: get_player_elo(id, tour)) #Vectorized operation
    matches['P1 Model'] = (matches.apply(lambda row: calculate_expected_score(row['Player1 ELO'], row['Player2 ELO']), axis=1)).round(2) #Vectorized operation
    matches['P2 Model'] = 1 - matches['P1 Model'] #Vectorized operation
    matches['Player1 sELO'] = matches.apply(lambda row: get_player_elo(row['ID1'], tour, row['Surface']), axis=1) #Vectorized operation
    matches['Player2 sELO'] = matches.apply(lambda row: get_player_elo(row['ID2'], tour, row['Surface']), axis=1) #Vectorized operation
    matches['P1 sModel'] = (matches.apply(lambda row: calculate_expected_score(row['Player1 sELO'], row['Player2 sELO']), axis=1)).round(2) #Vectorized operation
    matches['P2 sModel'] = 1 - matches['P1 sModel'] #Vectorized operation
    matches['P1 Market'] = 1 / matches['P1_Odds'] #Vectorized operation
    matches['P2 Market'] = 1 / matches['P2_Odds'] #Vectorized operation
    return matches[['Player1', 'P1 Model', 'P1 sModel', 'P1 Market', 'Player2' , 'P2 Model', 'P2 sModel', 'P2 Market', 'Tournament']]

def format_player_name(player_name, model_value, market_price):
    if model_value > market_price + 0.03:
        return f'<b>{player_name}</b>'
    else:
        return player_name

def main():
    st.title("Upcoming Matches")
    elo_models['atp'] = load_elo_model('elo_model_atp.pkl')
    elo_models['wta'] = load_elo_model('elo_model_wta.pkl')
    st.sidebar.title("Filter Matches")
    selected_tour = str.lower(st.sidebar.selectbox("Tour", ['atp', 'wta']))
    elo = elo_models[selected_tour]
    matches = prepare_data(selected_tour, elo)
    if matches.empty:
        st.write("No matches found")
        return


    tournaments = matches['Tournament'].unique()
    selected_tournament = st.sidebar.selectbox("Tournaments", tournaments, key="Tournament_selectbox")
    matches['Tour']=selected_tour
    filtered_matches = matches[(matches['Tour'] == selected_tour) & (matches['Tournament'] == selected_tournament)]
    if filtered_matches.empty:
        st.write("No matches found for the selected tournament")
        return

    st.write(f"Matches for {selected_tournament}")
    match_df = filtered_matches.drop(columns=['Tournament', 'Tour'])
    match_df['Player1'] = match_df.apply(lambda row: format_player_name(row['Player1'], row['P1 Model'], row['P1 Market']), axis=1)
    match_df['Player2'] = match_df.apply(lambda row: format_player_name(row['Player2'], row['P2 Model'], row['P2 Market']), axis=1)
    match_df_html = match_df.to_html(escape=False, index=False)
    st.markdown(match_df_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
