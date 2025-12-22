''' 

1. LOAD MODEL (read from models directory)
2. CREATE FEATURES (using feature engineer loaded from models directory)
3. MAKE PREDICTION (using Loaded model on created features)
4. DISPLAY RESULTS (show predicted winner, show confidence / probability, show key states that influenced prediction)

'''

import argparse 
import joblib # for loading models


import pandas as pd
from datetime import datetime
import os

def load_model(model_type="auto"):

    if model_type == "auto":
        # Ensure files exist and then load
        if os.path.exists('models/nfl_random_forest.pkl'):
            model_path = joblib.load('models/nfl_random_forest.pkl')
            model_name = "Random Forest"
        elif os.path.exists('models/nfl_logistic_regression.pkl'):
            model_path = 'models/nfl_logistic_regression.pkl'
            model_name = "Logistic Regression"
        else:
            raise FileNotFoundError("No model found in models directory")
    elif model_type == "random_forest":
        model_path = "models/nfl_random_forest.pkl"
        model_name = "Random Forest"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Random Forest model not found in models directory")
    elif model_type == "logistic_regression":
        model_path = "models/nfl_logistic_regression.pkl"
        model_name = "Logistic Regression"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Logistic Regression model not found in models directory")
    else:
        raise ValueError("Invalid model_type. Choose 'auto', 'random_forest', or 'logistic_regression'.")

    print(f"Loading {model_name} model from {model_path}...")
    model = joblib.load(model_path)

    print("Loading feature engineer...")
    feature_engineer_path = "models/nfl_feature_engineer.pkl"
    feature_engineer = joblib.load(feature_engineer_path)

    print("Loading feature columns...") # saved so that we can use the same features as during training
    feature_columns = joblib.load("models/nfl_feature_columns.pkl")

    return model, feature_engineer, feature_columns


def predict_game(home_team, away_team, model, feature_engineer, feature_columns, game_date=None):
    """
    1. Create features for both home and away teams based on recent performance
        - Look up home team's recent 5 games (from feature_engineer.schedule)
        - Calculate: win_rate, avg_ponints_scored, avg_points_allowed, ...
        - Do the same for the away team
        - Combine into single feature vector for prediction
    2. Use the trained model to predict win probability
        - Feed feature vector into the trained Random Forest / Logistic Regression model
    3. Return the prediction and confidence
        - Binary predictio: 0 (away team wins) or 1 (home team wins)
        - Probabilities: [away_team_win_prob, home_team_win_prob]
        - Confidence: max(probabilities) --> this is basically the confidence in the prediction TODO: Decide how to calculate
    """

    print("\n " + "="*60)
    print(f"Predicting outcome for {home_team} (home) vs {away_team} (away)")
    print("="*60 + "\n")

    # STEP 1: Create Features
    print("\n[Step 1] Creating features for both teams...")

    try:         
        features_dict = feature_engineer.create_prediction_features(
            home_team, away_team, current_date=game_date
        )

        features_df = pd.DataFrame([features_dict])

        X = features_df[feature_columns] # Selects only the features used during training

        print("feature created successfully!")
    except ValueError as e:
        print(f"Error: {e}")
        return None, None, None
    
    # STEP 2: Make Prediction
    print("\n[Step 2] Making prediction using the loaded model...")

    prediction = model.predict(X)[0] # 0 for away team win, 1 for home team win

    probabilities = model.predict_proba(X)[0] # [away_team_win_prob, home_team_win_prob]
    home_team_win_prob = probabilities[1]
    away_team_win_prob = probabilities[0]

    print("\n" + "="*60)
    print("[Prediction Results:")
    print("="*60 + "\n")

    print(f"\nHome Team: {home_team}")
    print(f"Away Team: {away_team}")
    
    if game_date:
        print(f"Game Date: {game_date.strftime('%Y-%m-%d')}") # TODO: Verify this will work
    
    print(f"Prediction: {'Home Team Wins' if prediction == 1 else 'Away Team Wins'}")
    print(f"Confidence {max(away_team_win_prob, home_team_win_prob):.2%}")

    print(f" - Home Team Win Probability: {home_team_win_prob:.2%}")
    print(f" - Away Team Win Probability: {away_team_win_prob:.2%}")

    print("\n Key Stats (used by the model)")
    print(f" {home_team} recent win rate: {features_dict['home_win_rate']:.2%}")
    print(f" {away_team} recent win rate: {features_dict['away_win_rate']:.2%}")

    print(f" {home_team} avg points scored: {features_dict['home_avg_points_scored']:.1f}")
    print(f" {away_team} avg points scored: {features_dict['away_avg_points_scored']:.1f}")

    print(f" {home_team} avg points allowed: {features_dict['home_avg_points_allowed']:.1f}")
    print(f" {away_team} avg points allowed: {features_dict['away_avg_points_allowed']:.1f}")

    print(f" {home_team} rest days: {features_dict['home_rest_days']}")
    print(f" {away_team} rest days: {features_dict['away_rest_days']}")

    print("\n" + "="*60)
    print("[End of Prediction]")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": home_team if prediction == 1 else away_team,
        "home_win_prob": home_team_win_prob,
        "away_win_prob": away_team_win_prob,
        "confidence": max(home_team_win_prob, away_team_win_prob),
        "features": features_dict # just for debugging purposes

    }

def main():
    # TODO: Add command line arguments for home_team, away_team, and game_date, and model_type and parse them
