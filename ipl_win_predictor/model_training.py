# model_training.py

import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from data_preprocessing import load_and_clean_data

def prepare_dataset(matches_path, deliveries_path):
    match_df, delivery_df = load_and_clean_data(matches_path, deliveries_path)

    # Keep only 1st innings score
    total_score_df = (
        delivery_df.groupby(['match_id', 'inning', 'batting_team'])
        .sum()['total_runs']
        .reset_index()
    )
    total_score_df = total_score_df[total_score_df['inning'] == 1]

    # Merge with match_df
    match_df = match_df.merge(
        total_score_df[['match_id', 'total_runs']],
        left_on='id', right_on='match_id'
    )

    # Replace old team names
    replacements = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Gujarat Lions': 'Gujarat Titans',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    match_df['team1'] = match_df['team1'].replace(replacements)
    match_df['team2'] = match_df['team2'].replace(replacements)

    teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Titans',
        'Royal Challengers Bangalore', 'Kolkata Knight Riders',
        'Kings XI Punjab', 'Chennai Super Kings',
        'Rajasthan Royals', 'Delhi Capitals'
    ]

    # Filter valid teams & non-DL matches
    match_df = match_df[
        (match_df['team1'].isin(teams)) &
        (match_df['team2'].isin(teams)) &
        (match_df['dl_applied'] == 0)
    ]

    match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
    delivery_df = delivery_df[delivery_df['inning'] == 2]
    delivery_df = match_df.merge(delivery_df, on='match_id')

    delivery_df.rename(
        columns={'total_runs_x': 'total_runs', 'total_runs_y': 'Ball_score'},
        inplace=True
    )

    # Feature engineering
    delivery_df['Score'] = delivery_df.groupby('match_id')['Ball_score'].cumsum()
    delivery_df['target_left'] = (delivery_df['total_runs'] + 1) - delivery_df['Score']
    delivery_df['Remaining Balls'] = 120 - ((delivery_df['over'] - 1) * 6 + delivery_df['ball'])

    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna('0')
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
    delivery_df['Wickets'] = 10 - delivery_df.groupby('match_id')['player_dismissed'].cumsum()

    delivery_df['crr'] = delivery_df.apply(
        lambda row: (row['Score'] * 6) / (120 - row['Remaining Balls'])
        if row['over'] > 0 else 0, axis=1
    )
    delivery_df['rrr'] = (delivery_df['target_left'] * 6) / delivery_df['Remaining Balls']

    delivery_df['result'] = delivery_df.apply(
        lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1
    )

    model_df = delivery_df[
        ['batting_team', 'bowling_team', 'city', 'Score', 'Wickets',
         'Remaining Balls', 'target_left', 'crr', 'rrr', 'result']
    ].dropna()

    model_df = model_df[model_df['Remaining Balls'] != 0]

    return model_df, teams, list(model_df['city'].unique())

def train_and_save_model(matches_path, deliveries_path):
    model_df, teams, cities = prepare_dataset(matches_path, deliveries_path)

    X = model_df.iloc[:, :-1]
    y = model_df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trf = ColumnTransformer(
        [('trf', OneHotEncoder(sparse_output=False, drop='first'),
          ['batting_team', 'bowling_team', 'city'])],
        remainder='passthrough'
    )

    pipe = Pipeline(steps=[
        ('step1', trf),
        ('step2', LogisticRegression(solver='liblinear'))
    ])

    pipe.fit(x_train, y_train)
    print("Accuracy:", accuracy_score(y_test, pipe.predict(x_test)))

    # Save files
    joblib.dump(pipe, "model.joblib")
    joblib.dump(teams, "team.joblib")
    joblib.dump(cities, "city.joblib")

if __name__ == "__main__":
    train_and_save_model("ipl_win_predictor/data/matches.csv", "ipl_win_predictor/data/deliveries.csv")
