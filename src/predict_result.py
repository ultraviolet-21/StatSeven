import numpy as np
import pandas as pd

from fetch_gamelog import *

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from statistics import mean

def random_value(array):
    n = array.size
    weights = [1 + 0.5 * (i / n) for i in range(n)] #gives slightly more weight to games later in the season
    s = sum(weights)
    p = [w / s for w in weights]
    return np.random.choice(array, p = p) 

def estimate_goals(team: str, df) -> float:
    cols = [('Team', 'SOG'), ('Team', 'PIM'), ('Team', 'PPG'), ('Team', 'PPO'), ('Team', 'SHG'), ('Faceoffs', 'FO%'),
       ('Advanced (5-on-5)', 'oZS%'), ('Advanced (5-on-5)', 'PDO')]
    X = df[cols]
    y = df[("Score", "GF")]

    #Scale all inputs to be between 0 and 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Set up k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    #Using Linear Regression
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error')
    model.fit(X_scaled, y)

    vector = []
    for attr in cols:
        vector.append(random_value(df[attr]))

    if vector[3] < vector[2]: #PPG cannot be greater than PPO
        vector[3] = vector[2]
  
    vector_scaled = scaler.transform(np.array(vector).reshape(1, -1))
    temp = model.predict(vector_scaled)
    if temp < (vector[2] + vector[4]): #total goals can't be less than PPG + SHG
        temp = vector[2] + vector[4]
    return temp
    
def get_score(team: str, opp: str, team_df: pd.DataFrame, opp_df: pd.DataFrame, verbose = True) -> str:
    #prints output by default, but this can be overrriden

    score1 = estimate_goals(team, team_df)
    if type(score1) == np.ndarray:
        score1 = score1[0]
   
    score2 = estimate_goals(opp, opp_df)
    if type(score2) == np.ndarray:
        score2 = score2[0]

    goals_for = round(score1)
    goals_against = round(score2)

    if goals_for != goals_against:
        if verbose:
            print(f"Score: {team} {goals_for}, {opp} {goals_against}")
        else:
            pass
    else: #predicts tie (i.e., OT)
        if score1 > score2:
            goals_for += 1
        else:
            goals_against += 1
        if verbose:
            print(f"Score: {team} {goals_for}, {opp} {goals_against} (OT)")

    if score1 > score2:
        return team
    else:
        return opp

def simulate_series(team: str, opp: str, team_df: pd.DataFrame, opp_df: pd.DataFrame, verbose = True) -> None:
    #prints output by default, but this can be overridden

    team_wins = 0
    opp_wins = 0
    games = 0
    for i in range(7):
        games += 1
        if verbose:
            winner = get_score(team, opp, team_df, opp_df)
        else:
            winner = get_score(team, opp, team_df, opp_df, verbose = False)
        if winner == team:
            team_wins += 1
        else:
            opp_wins += 1
        if team_wins >= 4 or opp_wins >= 4:
            break #series ends if one team has 4 wins

    if verbose:
        if team_wins > opp_wins:
            print(f"{team} wins {team_wins}-{opp_wins}")
        else:
            print(f"{opp} wins {opp_wins}-{team_wins}")

    return winner, games

def n_simulations(team, opp, team_gamelog, opp_gamelog, n = 25):
    if team_gamelog is None:
        team_gamelog = gamelog(team)
    if opp_gamelog is None:
        opp_gamelog = gamelog(opp)

    team_wins = 0
    num_games1 = [] #contains the number of games it took for team to win series
    opp_wins = 0
    num_games2 = []
    for i in range(n):
        winner, games = simulate_series(team, opp, team_gamelog, opp_gamelog, verbose = False)
        if winner == team:
            team_wins += 1
            num_games1.append(games)
        else:
            opp_wins += 1
            num_games2.append(games)

    avg1 = round(mean(num_games1))
    avg2 = round(mean(num_games2))
    print(f"{team} wins in {round(team_wins/n * 100, 2)}% of simulations, in {avg1} games on average.")
    print(f"{opp} wins in {round(opp_wins/n * 100, 2)}% of simulations, in {avg2} games on average.")