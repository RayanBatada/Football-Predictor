import argparse
import joblib

import pandas as pd
from datetime import datetime
import os

 

def load_model(model_type="auto"):
    
    if model_type == "auto":
        if os.path.exists("models/nfl_random_forest_model.pkl"):
            model_path = "models/nfl_random_forest_model.pkl"
            model_name = "Random Forest"
        elif os.path.exists("models/nfl_logistic_regression_model.pkl"):
            model_path = "models/nfl_logistic_regression_model.pkl"
            model_name = "Logistic Regression"
        else:
            raise FileNotFoundError("No trained model found in models directory.")
    elif model_type == "random_forest":
        model_path = "models/nfl_random_forest_model.pkl"
        model_name = "Random Forest"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Random Forest model not found in models directory.")
    elif model_type == "logistic_regression":
        model_path = "models/nfl_logistic_regression_model.pkl"
        model_name = "Logistic Regression"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Logistic Regression model not found in models directory.")
    else:
        raise ValueError(f"Unknown model type specified: {model_type}")

    print(f"Loading {model_name} model from {model_path}...")
    model = joblib.load(model_path)

    print("Loading feature engineer...")
    feature_engineer_path = "models/nfl_feature_engineer.pkl"
    feature_engineer = joblib.load(feature_engineer_path)

    print("Loading feature columns...")
    feature_columns = joblib.load("models/nfl_feature_columns.pkl")

    return model, feature_engineer, feature_columns


def predict_game(home_team, away_team, model, feature_engineer, feature_columns, game_date=None):
    """
    1. Create features for both home and away teams based on recent performance
    - Look up home team's recent 5 games (from feature_engineer.schedules)
    - Calculate: win_rate, avg_points_scored, avg_points_allowed, ...
    - Do the same for away team
    - Combine into single feature vector for prediction
    2. Use the trained model to predict win probability
    - Feed feature vector into the trained Random Forest / Logistic Regression model
    3. Return the prediction and confidence
    - Binary prediction: 0 (away team wins) or 1 (home team wins)
    - Probabilities: [away_team_win_prob, home_team_win_prob]
    - confidence: max(probabilities) --- TODO: Decide how calculate
    """

    print("\n" + "="*60)
    print(f"Predicting outcome for {home_team} (home) vs {away_team} (away)")
    print("="*60 + "\n")

    # STEP 1: Create Features
    print("\n[Step 1] Creating features for both teams...")

    try:
        features_dict = feature_engineer.create_prediction_features(
            home_team, away_team, current_date=game_date
        )
        
        features_df = pd.DataFrame([features_dict])
        
        X = features_df[feature_columns]
        
        print("Features created successfully")
    except ValueError as e:
        print(f"Error creating features: {e}")
        return None, None, None
    
    
    # STEP 2: Make Prediction
    print("\n[Step 2] Making prediction using the trained model...")

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]  # [away_team_win_prob, home_team_win_prob]
    away_win_prob = probabilities[0]
    home_win_prob = probabilities[1]


    print("\n" + "="*60)
    print("[Prediction Results]")
    print("="*60 + "\n")

    print(f"\nHome Team: {home_team}")
    print(f"Away Team: {away_team}")

    if game_date:
        print(f"Game Date: {game_date.strftime('%Y-%m-%d')}") # TODO: Verify this will work

    print(f"Prediction: {'Home Team Wins' if prediction == 1 else 'Away Team Wins'}")
    print(f"Confidence: {max(away_win_prob, home_win_prob):.2%}")

    print("\nWin Probabilities:")
    print(f" - Home Win Probability: {home_win_prob:.2%}")
    print(f" - Away Win Probability: {away_win_prob:.2%}")

    print("\n Key Stats (used by model)")
    print(f"  {home_team} recent win rate: {features_dict['home_win_rate']:.2%}")
    print(f"  {away_team} recent win rate: {features_dict['away_win_rate']:.2%}")

    print(f"  {home_team} avg points scored: {features_dict['home_avg_point_scored']:.1f}")
    print(f"  {away_team} avg points scored: {features_dict['away_avg_point_scored']:.1f}")

    print(f"  {home_team} points differential: {features_dict['home_point_diff']:.1f}")
    print(f"  {away_team} points differential: {features_dict['away_point_diff']:.1f}")

    print(f"  {home_team} rest days: {features_dict['home_rest_days']}")
    print(f"  {away_team} rest days: {features_dict['away_rest_days']}")

    print("\n" + "="*60 + "\n")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": home_team if prediction == 1 else away_team,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "confidence": max(away_win_prob, home_win_prob),
        "features": features_dict # Mostly for debugging purposes
    }
    

def interactive_mode(model_type="auto"):
    print("Welcome to the NFL Game Prediction Tool!")
    print("You can predict the outcome of an NFL game by entering the home team and away team names.")
    print("If you want to use a specific model type, you can specify it with the --model_type argument.")
    print("If you want to specify a game date, you can do so with the --game_date argument.")
    print("To exit, type 'exit'.")

    """
    1. Model -> Training takes a while and requires a lot of data so after trained, save it for quick reference
    2. Feature Engineer -> To train model, had to load all raw data. Represents the DataFrame of all historical games and schedules
    -- Without this, you'd have to re-downlaod all the NFL data from the PAI every time you want to predict a game
    3. Feature Columns -> The columns the model was trained on. The model learns associations (col 0 = win_rate, col 1 = avg points, ...)
    -- If we were to pass columns of new data in a different order, then the model may interpret say avg_points as win_rate --> GARBAGE predictions
    -- Ensures our Features match the order the model was trained on during prediction
    """
    model, feature_engineer, feature_columns = load_model(model_type)

    while True:
        home_team = input("\nEnter the home team: ")
        if home_team.lower() == "exit":
            break

        away_team = input("Enter the away team: ")
        if away_team.lower() == "exit":
            break

        try:
            predict_game(home_team, away_team, model, feature_engineer, feature_columns)
        except ValueError:
            print("Error: Invalid input. Please ensure the team names are correct and try again.")
            continue

 

def main():
    parser = argparse.ArgumentParser(description="Predict the outcome of an NFL game.")
    parser.add_argument("home_team", nargs="?", type=str, help="The home team.")
    parser.add_argument("away_team", nargs="?", type=str, help="The away team.")
    parser.add_argument("--model_type", type=str, choices=["auto", "random_forest", "logistic_regression"], default="auto", help="Type of model to use for prediction.")
    parser.add_argument("--game_date", default=None, type=str, help="Date of the game in YYYY-MM-DD format.")

    args = parser.parse_args()

    if not args.home_team or not args.away_team:
        interactive_mode(model_type=args.model_type)
        return

    model, feature_engineer, feature_columns = load_model(args.model_type)
    
    game_date = None
    if args.game_date:
        game_date = datetime.strptime(args.game_date, '%Y-%m-%d')

    predict_game(args.home_team, args.away_team, model, feature_engineer, feature_columns, game_date)


if __name__ == "__main__":
    main()