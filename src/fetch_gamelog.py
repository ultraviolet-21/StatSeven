import requests
import io
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def gamelog(team: str) -> pd.DataFrame:

    url = f"https://www.hockey-reference.com/teams/{team}/2026_gamelog.html" 

    response = requests.get(url)
    response.raise_for_status()

    stats_df = pd.read_html(io.StringIO(response.text), match = "Date")[0]

    cleaned_df = stats_df[stats_df[('Unnamed: 2_level_0', 'Date')] != "Date"][:-1:]

    cleaned_df = cleaned_df.dropna(subset = [('Score', 'Rslt')])

    cleaned_df = parse_numerical(cleaned_df)

    add_points(cleaned_df)

    cleaned_df[("Unnamed: 2_level_0", "Date")] = pd.to_datetime(cleaned_df[("Unnamed: 2_level_0", "Date")])

    return cleaned_df

def parse_numerical(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            continue

    return df

def add_points(df: pd.DataFrame) -> pd.DataFrame:
    length = df.shape[0]
    if df.iloc[0][("Score", "Rslt")] == 'W':
        points = [2]
    elif df.iloc[0][("Score", "Rslt")] == 'L' and (df.iloc[0][("Score", "OT")] == "OT" or df.iloc[0][("Score", "OT")] == "SO"):
        points = [1]
    else:
        points = [0]
    
    for i in range(1, length):
        if df.iloc[i][("Score", "Rslt")] == 'W':
            points.append(points[-1]+2)
        elif df.iloc[i][("Score", "Rslt")] == 'L' and (df.iloc[i][("Score", "OT")] == "OT" or df.iloc[i][("Score", "OT")] == "SO"):
            points.append(points[-1]+1)
        else:
            points.append(points[-1])

    df[('Unnamed: 5_level_0', 'Points')] = points
    
team_stats = [('Team', 'SOG'), ('Team', 'PIM'), ('Team', 'PPG'), ('Team', 'PPO'), ('Team', 'SHG')]

opp_stats = [('Opponent', 'SOG'), ('Opponent', 'PIM'), ('Opponent', 'PPG'), ('Opponent', 'PPO'), ('Opponent', 'SHG')]


#df = gamelog("SJS")





#use multi-output regression here 
#Random Forest regressor
#take team and opponent stats, predict team and opponent score
#then user defined
'''
Use 80% train / 20% test if you want a simple split.
Prefer 5-fold cross-validation to maximize training data and get stable performance metrics.
Keep the test set separate if you want a “final check” on real user input.

'''