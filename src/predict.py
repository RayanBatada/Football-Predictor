'''
NFL Game Prediction Tool - Fixed for Ensemble Models
'''

import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_model(model_type="auto"):
    """Load trained model"""
    
    model_dir = "models"
    
    if model_type == "auto":
        available = []
        for m in ["ensemble", "xgboost", "gradient_boosting", "random_forest"]:
            if os.path.exists(os.path.join(model_dir, f"nfl_{m}_model.pkl")):
                available.append(m)
        
        if not available:
            raise FileNotFoundError("No models found. Run: python train_model.py")
        
        model_type = available[0]
        print(f"\n🤖 Using: {model_type.replace('_', ' ').title()}")
    
    # Load model
    model = joblib.load(os.path.join(model_dir, f"nfl_{model_type}_model.pkl"))
    
    # Load scaler if exists
    scaler_path = os.path.join(model_dir, f"nfl_{model_type}_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    # Load feature engineer and columns
    fe = joblib.load(os.path.join(model_dir, "nfl_feature_engineer.pkl"))
    cols = joblib.load(os.path.join(model_dir, "nfl_feature_columns.pkl"))
    
    print(f"✓ Loaded ({len(cols)} features)\n")
    
    return model, fe, cols, scaler, model_type


def predict_game(home, away, model, fe, cols, scaler=None, model_type="ensemble", date=None):
    """Predict game outcome"""
    
    print("\n" + "="*70)
    print(f"  🏈 {away.upper()} @ {home.upper()}")
    print("="*70)
    
    # Generate features
    print("\n[1/2] Generating features...")
    try:
        features = fe.create_prediction_features(home, away, current_date=date)
        X = pd.DataFrame([features])[cols]
        print(f"  ✓ {len(cols)} features")
    except ValueError as e:
        print(f"\n  ✗ {e}")
        return None
    
    # Make prediction
    print("\n[2/2] Predicting...")
    
    try:
        # FIXED: Proper handling of ensemble models
        if model_type == "ensemble":
            # Ensemble needs special handling
            all_probs = []
            
            # Get predictions from each estimator
            for name, estimator in model.named_estimators_.items():
                if name == 'lr' and scaler is not None:
                    # Logistic regression needs scaling
                    X_scaled = scaler.transform(X)
                    probs = estimator.predict_proba(X_scaled)[0]
                else:
                    probs = estimator.predict_proba(X)[0]
                all_probs.append(probs)
            
            # Average the probabilities
            avg_probs = np.mean(all_probs, axis=0)
            away_prob, home_prob = avg_probs[0], avg_probs[1]
            prediction = 1 if home_prob > 0.5 else 0
            
        elif scaler is not None:
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            away_prob, home_prob = probs[0], probs[1]
        else:
            prediction = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            away_prob, home_prob = probs[0], probs[1]
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Display results
    print("\n" + "="*70)
    print("  📊 RESULTS")
    print("="*70)
    
    winner = home if prediction == 1 else away
    confidence = max(home_prob, away_prob)
    
    # Show game date if provided
    if date:
        print(f"\n  📅 Game Date: {date.strftime('%A, %B %d, %Y')}")
    
    # Confidence level
    if confidence >= 0.70:
        level, emoji = "VERY HIGH", "🔥"
    elif confidence >= 0.65:
        level, emoji = "HIGH", "✓"
    elif confidence >= 0.60:
        level, emoji = "MODERATE", "→"
    else:
        level, emoji = "LOW", "⚠️"
    
    print(f"\n  {emoji} WINNER: {winner.upper()}")
    print(f"  📈 CONFIDENCE: {confidence:.1%} ({level})")
    print(f"\n  Probabilities:")
    print(f"    {home} (Home): {home_prob:.1%}")
    print(f"    {away} (Away): {away_prob:.1%}")
    
    # Key stats
    print(f"\n" + "="*70)
    print("  KEY STATS")
    print("="*70)
    
    print(f"\n  Recent Form:")
    print(f"    {home}: {features['home_win_rate']:.1%} win rate")
    print(f"    {away}: {features['away_win_rate']:.1%} win rate")
    
    print(f"\n  Offense (PPG):")
    print(f"    {home}: {features['home_avg_points_scored']:.1f}")
    print(f"    {away}: {features['away_avg_points_scored']:.1f}")
    
    print(f"\n  Point Differential:")
    print(f"    {home}: {features['home_point_diff']:+.1f}")
    print(f"    {away}: {features['away_point_diff']:+.1f}")
    
    # Enhanced features if available
    if 'home_momentum' in features:
        print(f"\n  Momentum:")
        if features['home_momentum'] > 0.5:
            print(f"    {home}: 🔥 HOT!")
        if features['away_momentum'] > 0.5:
            print(f"    {away}: 🔥 HOT!")
    
    if 'home_close_game_record' in features:
        print(f"\n  Clutch (Close Games):")
        print(f"    {home}: {features['home_close_game_record']:.1%}")
        print(f"    {away}: {features['away_close_game_record']:.1%}")
    
    if features.get('is_division_game', 0) == 1:
        print(f"\n  🏆 DIVISION GAME!")
    
    print("\n" + "="*70 + "\n")
    
    return {"winner": winner, "confidence": confidence}


def interactive():
    print("\n🏈 NFL PREDICTION TOOL\n")
    
    try:
        model, fe, cols, scaler, mt = load_model()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    while True:
        print("-"*70)
        home = input("Home team (or 'exit'): ").strip()
        if home.lower() in ['exit', 'quit', 'q']:
            break
        
        away = input("Away team (or 'exit'): ").strip()
        if away.lower() in ['exit', 'quit', 'q']:
            break
        
        # Ask for game date
        date_input = input("Game date (YYYY-MM-DD) or press Enter for today: ").strip()
        game_date = None
        if date_input:
            try:
                game_date = datetime.strptime(date_input, "%Y-%m-%d")
            except ValueError:
                print("⚠️  Invalid date format. Using today's date.")
        
        predict_game(home, away, model, fe, cols, scaler, mt, game_date)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team", nargs="?", help="Home team")
    parser.add_argument("away_team", nargs="?", help="Away team")
    parser.add_argument("--model_type", default="auto", 
                       choices=["auto", "ensemble", "xgboost", "random_forest"])
    parser.add_argument("--game_date", help="Game date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if not args.home_team or not args.away_team:
        interactive()
        return
    
    try:
        model, fe, cols, scaler, mt = load_model(args.model_type)
        date = datetime.strptime(args.game_date, "%Y-%m-%d") if args.game_date else None
        predict_game(args.home_team, args.away_team, model, fe, cols, scaler, mt, date)
    except FileNotFoundError as e:
        print(f"\n✗ {e}")


if __name__ == "__main__":
    main()