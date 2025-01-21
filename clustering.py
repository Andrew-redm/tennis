from datetime import datetime
from accessDB import get_player_matches_in_daterange, top_n_in_date_range, get_player_name, get_player_id
import pandas as pd
import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def count_sets(result_string):
    if 'ret.' in result_string or 'w/o' in result_string:  # Check for retirement
        return result_string.count('-')
    else:
        return result_string.count('-')
    
def player_centric_stats(df, player_id):
    player_df = df.copy()
    player_lost = player_df['ID2'] == player_id
    for col in df.columns:
        if col.endswith('_1'):
            col2 = col[:-1] + '2'
            player_df.loc[player_lost, [col, col2]] = player_df.loc[player_lost, [col2, col]].values

    rename_dict = {col: col[:-2] + '_player' for col in df.columns if col.endswith('_1')}
    rename_dict.update({col: col[:-2] + '_opponent' for col in df.columns if col.endswith('_2')})
    player_df = player_df.rename(columns=rename_dict)
    return player_df

start_date = '2023-06-01'
tour = 'atp'
players = top_n_in_date_range(tour, 150, '2025-01-01')
playerMatches = get_player_matches_in_daterange(tour, players, start_date)
playerMatches['MT'] = playerMatches['MT'].dt.time
playerMatches['MT_minutes'] = playerMatches['MT'].apply(lambda x: x.hour * 60 + x.minute + x.second / 60 if x else None)
playerMatches = playerMatches[~playerMatches['MT_minutes'].isna()]

all_player_data = []

for player_id in tqdm.tqdm(players):
    player_matches = playerMatches[(playerMatches['ID1_G'] == player_id) | (playerMatches['ID2_G'] == player_id)]

    if player_matches is not None and not player_matches.empty: 
        all_match_stats = []
        playerStats = player_centric_stats(player_matches, player_id)

    #changes to final df
    playerStats['unfinished'] = playerStats['RESULT_G'].str.contains('ret')
    playerStats = playerStats[~playerStats['unfinished']]
    playerStats['num_sets'] = playerStats['RESULT_G'].apply(count_sets)
    playerStats['ace_per_set'] = playerStats['ACES_player']/playerStats['num_sets']
    playerStats['aced_per_set'] = playerStats['ACES_opponent']/playerStats['num_sets']
    playerStats['df_per_set'] = playerStats['DF_player']/playerStats['num_sets']
    playerStats['bp_earned_per_set'] = playerStats['BPOF_player']/playerStats['num_sets']
    playerStats['breaks_per_set'] = playerStats['BP_player']/playerStats['num_sets']
    playerStats['bp_conversion'] = playerStats['breaks_per_set']/playerStats['bp_earned_per_set']
    playerStats['bp_saved_per_set'] = playerStats['BPOF_opponent']/playerStats['num_sets']
    playerStats['broken_per_set'] = playerStats['BP_opponent']/playerStats['num_sets']
    playerStats['SPW_rate_first'] = playerStats['W1S_player']/playerStats['W1SOF_player']
    playerStats['SPW_rate_second'] = playerStats['W2S_player']/playerStats['W2SOF_player']
    playerStats['RPW_rate_first'] = 1-playerStats['W1S_opponent']/playerStats['W1SOF_opponent']
    playerStats['RPW_rate_second'] = 1-playerStats['W2S_opponent']/playerStats['W2SOF_opponent']
    playerStats['MT_minutes'] = playerStats['MT'].apply(lambda x: x.hour * 60 + x.minute + x.second / 60 if x else None)

    player_stats = {
            'Player ID': player_id,
            'Win Rate': len(playerStats[playerStats['ID1']==player_id])/len(playerStats),
            'Time_per_set': float((playerStats['MT_minutes']/playerStats['num_sets']).mean()),
            'Aces_per_set': float(playerStats['ace_per_set'].mean()),
            'Aced_per_set': float(playerStats['aced_per_set'].mean()),
            'df_per_set': float(playerStats['df_per_set'].mean()),
            'bp_earned_per_set': float(playerStats['bp_earned_per_set'].mean()),
            'breaks_per_set': float(playerStats['breaks_per_set'].mean()),
            'bp_efficiency': float(playerStats['bp_conversion'].mean()),
            'bp_saved_per_set': float(playerStats['bp_saved_per_set'].mean()),
            'broken_per_set': float(playerStats['broken_per_set'].mean()),
            'SPW_rate_first': float(playerStats['SPW_rate_first'].mean()),
            'SPW_rate_second': float(playerStats['SPW_rate_second'].mean()),
            'RPW_rate_first': float(playerStats['RPW_rate_first'].mean()),
            'RPW_rate_second': float(playerStats['RPW_rate_second'].mean())
        }
    
    all_player_data.append(player_stats)

final_df = pd.DataFrame(all_player_data)
final_df[final_df['Player ID'] == 39309]

final_df['RPW_rate_first'].hist(bins=20)
final_df['RPW_rate_second'].hist(bins=20)

features = ['Win Rate', 'Aces_per_set', 'Aced_per_set', 
            'bp_earned_per_set', 'breaks_per_set', 
            'bp_efficiency', 'bp_saved_per_set', 'broken_per_set', 
            'SPW_rate_first', 'SPW_rate_second', 'RPW_rate_first', 
            'RPW_rate_second']
X = final_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 20  # You'll need to determine the optimal k
kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducibility
clusters = kmeans.fit_predict(X_scaled)

final_df['cluster'] = clusters

print("Cluster Centers:")
print(kmeans.cluster_centers_)
print(final_df.groupby('cluster')[features].mean())

seed_random = 1

fitted_kmeans = {}
labels_kmeans = {}
df_scores = []
k_values_to_try = np.arange(2, 25)
for n_clusters in k_values_to_try:
    
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=seed_random,
                    )
    labels_clusters = kmeans.fit_predict(X_scaled)
    
    fitted_kmeans[n_clusters] = kmeans
    labels_kmeans[n_clusters] = labels_clusters
    
    silhouette = silhouette_score(X, labels_clusters)
    ch = calinski_harabasz_score(X, labels_clusters)
    db = davies_bouldin_score(X, labels_clusters)
    tmp_scores = {"n_clusters": n_clusters,
                  "silhouette_score": silhouette,
                  "calinski_harabasz_score": ch,
                  "davies_bouldin_score": db,
                  "inertia": kmeans.inertia_,
                  
                  }
    df_scores.append(tmp_scores)

df_scores = pd.DataFrame(df_scores)
df_scores.set_index("n_clusters", inplace=True)

df_scores['inertia'].plot(kind='line')