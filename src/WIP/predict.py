"""
NFL Game Prediction Tool

Improvements:
- Fixed path mismatch (now uses trained_models/ consistently)
- Better error handling and user feedback
- Support for scaled models (logistic regression, ensemble)
- Improved output formatting
- More detailed prediction analysis
"""

import argparse
import joblib
import pandas as pd
from datetime import datetime
import os


def load_model(model_type="auto"):
    """Load trained model and associated files"""
    
    model_dir = "trained_models"  # FIXED: Was "models/"
    
    if model_type == "auto":
        # Auto-detect available models
        available_models = []
        for m in ["random_forest", "logistic_regression", "gradient_boosting", "xgboost", "ensemble"]:
            model_path = os.path.join(model_dir, f"nfl_{m}_model.pkl")
            if os.path.exists(model_path):
                available_models.append((m, model_path))
        
        if not available_models:
            raise FileNotFoundError(
                f"No trained models found in {model_dir}/\n"
                "Please run train_model.py first to train a model."
            )
        
        # Use the first available model
        model_type, model_path = available_models[0]
        model_name = model_type.replace("_", " ").title()
        
        print(f"\nAuto-detected model: {model_name}")
        if len(available_models) > 1:
            print(f"Other available models: {[m[0] for m in available_models[1:]]}")
    else:
        model_path = os.path.join(model_dir, f"nfl_{model_type}_model.pkl")
        model_name = model_type.replace("_", " ").title()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_name} model not found at {model_path}\n"
                "Please train this model first using train_model.py"
            )

    print(f"Loading {model_name} model from {model_path}...")
    model = joblib.load(model_path)

    # Load scaler if it exists (for logistic regression and ensemble)
    scaler_path = os.path.join(model_dir, f"nfl_{model_type}_scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        print("Loading feature scaler...")
        scaler = joblib.load(scaler_path)

    # Load feature engineer
    print("Loading feature engineer...")
    feature_engineer_path = os.path.join(model_dir, "nfl_feature_engineer.pkl")
    if not os.path.exists(feature_engineer_path):
        raise FileNotFoundError(
            f"Feature engineer not found at {feature_engineer_path}\n"
            "Please retrain your model."
        )
    feature_engineer = joblib.load(feature_engineer_path)

    # Load feature columns
    print("Loading feature columns...")
    feature_columns_path = os.path.join(model_dir, "nfl_feature_columns.pkl")
    if not os.path.exists(feature_columns_path):
        raise FileNotFoundError(
            f"Feature columns not found at {feature_columns_path}\n"
            "Please retrain your model."
        )
    feature_columns = joblib.load(feature_columns_path)

    print("✓ All model components loaded successfully!\n")

    return model, feature_engineer, feature_columns, scaler, model_type


def predict_game(home_team, away_team, model, feature_engineer, feature_columns, 
                 scaler=None, model_type="random_forest", game_date=None):
    """
    Predict the outcome of an NFL game
    
    Returns:
        dict: Prediction results including winner, probabilities, and key stats
    """

    print("\n" + "=" * 70)
    print(f"  PREDICTING: {away_team.upper()} @ {home_team.upper()}")
    print("=" * 70)

    # Step 1: Create Features
    print("\n[Step 1] Generating features for both teams...")

    try:
        features_dict = feature_engineer.create_prediction_features(
            home_team, away_team, current_date=game_date
        )
        features_df = pd.DataFrame([features_dict])
        X = features_df[feature_columns]
        
        print("  ✓ Features created successfully")
        
    except ValueError as e:
        print(f"\n  ✗ Error creating features: {e}")
        print("\n  Possible issues:")
        print("    - Team name misspelled (check exact spelling)")
        print("    - Not enough historical data for one or both teams")
        print("    - Teams haven't played enough games this season")
        return None

    # Step 2: Make Prediction
    print("\n[Step 2] Making prediction...")

    try:
        # Handle ensemble models differently
        if hasattr(model, 'is_ensemble') and model.is_ensemble:
            rf_pred = model.fitted_estimators[0].predict_proba(X)[:, 1]
            gb_pred = model.fitted_estimators[1].predict_proba(X)[:, 1]
            lr_pred = model.fitted_estimators[2].predict_proba(scaler.transform(X))[:, 1]
            
            home_team_win_prob = (rf_pred[0] + gb_pred[0] + lr_pred[0]) / 3
            away_team_win_prob = 1 - home_team_win_prob
            prediction = 1 if home_team_win_prob > 0.5 else 0
            
        else:
            # Apply scaling if needed
            if scaler is not None:
                X = scaler.transform(X)
            
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            home_team_win_prob = probabilities[1]
            away_team_win_prob = probabilities[0]

    except Exception as e:
        print(f"\n  ✗ Error making prediction: {e}")
        return None

    # Step 3: Display Results
    print("\n" + "=" * 70)
    print("  PREDICTION RESULTS")
    print("=" * 70)

    predicted_winner = home_team if prediction == 1 else away_team
    confidence = max(home_team_win_prob, away_team_win_prob)
    
    # Confidence level description
    if confidence >= 0.70:
        confidence_level = "VERY HIGH"
    elif confidence >= 0.60:
        confidence_level = "HIGH"
    elif confidence >= 0.55:
        confidence_level = "MODERATE"
    else:
        confidence_level = "LOW (TOSS-UP)"

    print(f"\n  🏆 PREDICTED WINNER: {predicted_winner.upper()}")
    print(f"  📊 CONFIDENCE: {confidence:.1%} ({confidence_level})")
    print(f"\n  Breakdown:")
    print(f"    • {home_team} (Home): {home_team_win_prob:.1%}")
    print(f"    • {away_team} (Away): {away_team_win_prob:.1%}")

    if game_date:
        print(f"\n  📅 Game Date: {game_date.strftime('%A, %B %d, %Y')}")

    # Display key factors
    print(f"\n" + "=" * 70)
    print("  KEY FACTORS INFLUENCING PREDICTION")
    print("=" * 70)

    print(f"\n  Recent Performance:")
    print(f"    {home_team:.<25} Win Rate: {features_dict['home_win_rate']:.1%}")
    print(f"    {away_team:.<25} Win Rate: {features_dict['away_win_rate']:.1%}")

    print(f"\n  Offensive Power (Avg Points Scored):")
    print(f"    {home_team:.<25} {features_dict['home_avg_points_scored']:.1f} PPG")
    print(f"    {away_team:.<25} {features_dict['away_avg_points_scored']:.1f} PPG")

    print(f"\n  Defensive Strength (Avg Points Allowed):")
    print(f"    {home_team:.<25} {features_dict['home_avg_points_allowed']:.1f} PPG")
    print(f"    {away_team:.<25} {features_dict['away_avg_points_allowed']:.1f} PPG")

    print(f"\n  Point Differential:")
    print(f"    {home_team:.<25} {features_dict['home_point_diff']:+.1f}")
    print(f"    {away_team:.<25} {features_dict['away_point_diff']:+.1f}")

    print(f"\n  Rest & Recovery:")
    print(f"    {home_team:.<25} {features_dict['home_rest_days']} days rest")
    print(f"    {away_team:.<25} {features_dict['away_rest_days']} days rest")

    if features_dict['home_injury_count'] > 0 or features_dict['away_injury_count'] > 0:
        print(f"\n  ⚕️  Injury Report:")
        print(f"    {home_team:.<25} {features_dict['home_injury_count']} key injuries")
        print(f"    {away_team:.<25} {features_dict['away_injury_count']} key injuries")

    # Home/Away splits
    print(f"\n  Home/Away Performance:")
    print(f"    {home_team} at home:......... {features_dict['home_home_win_rate']:.1%}")
    print(f"    {away_team} on road:......... {features_dict['away_away_win_rate']:.1%}")

    # Recent form
    print(f"\n  Recent Momentum:")
    print(f"    {home_team:.<25} {features_dict['home_weighted_form']:.1%}")
    print(f"    {away_team:.<25} {features_dict['away_weighted_form']:.1%}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70 + "\n")

    # Betting recommendation (for entertainment purposes only)
    edge = abs(home_team_win_prob - 0.5) * 2  # Convert to 0-1 scale
    if edge > 0.3:
        print(f"  💡 Strong prediction - Model shows {edge:.0%} edge")
    elif edge > 0.15:
        print(f"  💡 Moderate prediction - Model shows {edge:.0%} edge")
    else:
        print(f"  ⚠️  Close matchup - Consider a toss-up")
    
    print(f"\n  ⚠️  Note: Past performance doesn't guarantee future results.")
    print(f"           Use this prediction as one factor among many.\n")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": predicted_winner,
        "home_win_prob": home_team_win_prob,
        "away_win_prob": away_team_win_prob,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "features": features_dict
    }


def interactive_mode(model_type="auto"):
    """Run the prediction tool in interactive mode"""
    
    print("\n" + "=" * 70)
    print("  🏈 NFL GAME PREDICTION TOOL 🏈")
    print("=" * 70)
    print("\n  Predict the outcome of any NFL matchup!")
    print("  Type 'exit' or 'quit' at any time to stop.\n")
    print("=" * 70 + "\n")

    try:
        model, feature_engineer, feature_columns, scaler, loaded_model_type = load_model(model_type=model_type)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return

    while True:
        print("\n" + "-" * 70)
        home_team = input("Enter HOME team (or 'exit' to quit): ").strip()
        if home_team.lower() in ['exit', 'quit']:
            print("\n👋 Thanks for using the NFL Game Prediction Tool. Goodbye!\n")
            break

        away_team = input("Enter AWAY team (or 'exit' to quit): ").strip()
        if away_team.lower() in ['exit', 'quit']:
            print("\n👋 Thanks for using the NFL Game Prediction Tool. Goodbye!\n")
            break

        # Optional: Ask for game date
        date_input = input("Enter game date (YYYY-MM-DD) or press Enter for today: ").strip()
        game_date = None
        if date_input:
            try:
                game_date = datetime.strptime(date_input, "%Y-%m-%d")
            except ValueError:
                print("⚠️  Invalid date format. Using today's date instead.")

        # Make prediction
        try:
            predict_game(
                home_team, away_team, model, feature_engineer, 
                feature_columns, scaler, loaded_model_type, game_date
            )
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            print("Please try again with different teams.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict the outcome of an NFL game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py                                    # Interactive mode
  python predict.py "Kansas City Chiefs" "Buffalo Bills"
  python predict.py "Chiefs" "Bills" --game_date 2024-12-25
  python predict.py "Chiefs" "Bills" --model_type xgboost
        """
    )
    
    parser.add_argument(
        "home_team", 
        nargs="?", 
        type=str, 
        help="The home team name"
    )
    parser.add_argument(
        "away_team", 
        nargs="?", 
        type=str, 
        help="The away team name"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "random_forest", "logistic_regression", "gradient_boosting", "xgboost", "ensemble"],
        help="Type of model to use (default: auto-detect)"
    )
    parser.add_argument(
        "--game_date",
        type=str,
        default=None,
        help="Date of the game in YYYY-MM-DD format (default: today)"
    )

    args = parser.parse_args()

    # Interactive mode if no teams provided
    if not args.home_team or not args.away_team:
        interactive_mode(model_type=args.model_type)
        return

    # Single prediction mode
    try:
        model, feature_engineer, feature_columns, scaler, loaded_model_type = load_model(
            model_type=args.model_type
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return

    game_date = None
    if args.game_date:
        try:
            game_date = datetime.strptime(args.game_date, "%Y-%m-%d")
        except ValueError:
            print("⚠️  Invalid date format. Using today's date instead.")

    predict_game(
        args.home_team, args.away_team, model, feature_engineer,
        feature_columns, scaler, loaded_model_type, game_date
    )


if __name__ == "__main__":
    main()