# predictor.py

import pickle as pkl
import pandas as pd

# Load saved models & data
model = pkl.load(open("model.pkl", "rb"))
teams = pkl.load(open("team.pkl", "rb"))
cities = pkl.load(open("city.pkl", "rb"))

def predict_win_probability(batting_team, bowling_team, city, target, score, overs, wickets):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'Score': [score],
        'Wickets': [wickets_left],
        'Remaining Balls': [balls_left],
        'target_left': [runs_left],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = model.predict_proba(input_df)
    return result[0][0], result[0][1]  # (loss, win)
