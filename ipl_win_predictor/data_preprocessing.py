# data_preprocessing.py

import pandas as pd

def load_and_clean_data(matches_path, deliveries_path):
    # Load data
    match_df = pd.read_csv(matches_path)
    delivery_df = pd.read_csv(deliveries_path)

    # Drop unnecessary columns
    match_df.drop(
        ['umpire1', 'umpire2', 'umpire3', 'player_of_match',
         'toss_winner', 'toss_decision'],
        axis=1, inplace=True
    )

    delivery_df.drop(
        ['bowler', 'is_super_over', 'wide_runs', 'bye_runs',
         'legbye_runs', 'noball_runs', 'penalty_runs',
         'dismissal_kind', 'fielder'],
        axis=1, inplace=True
    )

    return match_df, delivery_df

# Example usage

matches_path = "data/matches.csv"
deliveries_path = "data/deliveries.csv"

match_df, delivery_df = load_and_clean_data(matches_path, deliveries_path)

print(match_df.head())
print(delivery_df.head())

