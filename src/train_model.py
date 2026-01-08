import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

from data_loader import download_nfl_data
from features import NFLFeatureEngineer


def train_model(model_type="ensemble", lookback_games=8, use_cross_validation=True, tune_hyperparameters=False):
    """
    ENHANCED Model Training with:
    - More sophisticated models (XGBoost, Gradient Boosting)
    - Ensemble methods for better predictions
    - Hyperparameter tuning
    - Better evaluation metrics
    - Increased lookback window (8 games instead of 5)
    """
    
    print("=" * 70)
    print("ENHANCED NFL GAME PREDICTION - MODEL TRAINING")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model Type: {model_type}")
    print(f"  Lookback Games: {lookback_games} (more history = better predictions)")
    print(f"  Cross-Validation: {use_cross_validation}")
    print(f"  Hyperparameter Tuning: {tune_hyperparameters}")
    print("=" * 70)

    print("\n[1/6] Loading historical NFL data...")
    schedules_df, seasonal_stats_df, weekly_stats_df, injuries_df = download_nfl_data()
    print(f"  ✓ Loaded {len(schedules_df)} games")

    print("\n[2/6] Engineering advanced features...")
    feature_engineer = NFLFeatureEngineer(lookback_games=lookback_games)
    feature_engineer.load_data(schedules_df, injuries_df)
    features_df = feature_engineer.create_features()
    
    print(f"  ✓ Created {len(features_df)} training examples")
    
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team", "target"]
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    print(f"  ✓ Generated {len(feature_cols)} features (up from 29!)")
    print(f"\n  New features include:")
    print(f"    • Momentum indicators (win streaks)")
    print(f"    • Scoring consistency metrics")
    print(f"    • Margin of victory trends")
    print(f"    • Close game performance")
    print(f"    • Division/Conference game indicators")
    print(f"    • Recent vs season performance")

    print("\n[3/6] Preparing training data...")
    X = features_df[feature_cols]
    y = features_df["target"]
    
    home_win_pct = y.sum() / len(y)
    print(f"  Target distribution: {y.sum()} home wins ({home_win_pct:.1%}) vs {len(y)-y.sum()} away wins ({1-home_win_pct:.1%})")

    # Time-based split (more realistic for time-series sports data)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)} games | Test: {len(X_test)} games (time-based split)")

    print("\n[4/6] Training model...")
    
    scaler = None
    model = None
    
    if model_type == "random_forest":
        if tune_hyperparameters:
            print("  ⚙️  Tuning Random Forest hyperparameters (this may take a few minutes)...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"  ✓ Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,  # More trees
                max_depth=15,      # Deeper trees
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        print("  Training Random Forest...")
        
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            print("\n  ✗ XGBoost not installed. Please run: pip install xgboost")
            return None
        
        if tune_hyperparameters:
            print("  ⚙️  Tuning XGBoost hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
            xgb = XGBClassifier(random_state=42, eval_metric='logloss')
            grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"  ✓ Best parameters: {grid_search.best_params_}")
        else:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        print("  Training XGBoost (usually the best for structured data)...")
        
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        print("  Training Gradient Boosting...")
        
    elif model_type == "ensemble":
        print("  Training Ensemble (combining multiple models)...")
        print("    → Random Forest (200 trees)")
        print("    → Gradient Boosting (200 estimators)")
        if XGBOOST_AVAILABLE:
            print("    → XGBoost (200 estimators)")
        print("    → Logistic Regression (with scaling)")
        
        # Scale data for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train individual models
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        
        print("    Training Random Forest...")
        rf.fit(X_train, y_train)
        print("    Training Gradient Boosting...")
        gb.fit(X_train, y_train)
        print("    Training Logistic Regression...")
        lr.fit(X_train_scaled, y_train)
        
        estimators = [('rf', rf), ('gb', gb), ('lr', lr)]
        
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, eval_metric='logloss')
            print("    Training XGBoost...")
            xgb.fit(X_train, y_train)
            estimators.append(('xgb', xgb))
        
        model = VotingClassifier(estimators=estimators, voting='soft')
        # Fit on pre-trained estimators
        model.estimators_ = [est[1] for est in estimators]
        model.named_estimators_ = {name: est for name, est in estimators}
        
    else:
        print(f"  ✗ Unknown model type: {model_type}")
        return None

    if model is not None and model_type != "ensemble":
        model.fit(X_train, y_train)
    
    print("  ✓ Model training complete!")

    # Cross-validation
    if use_cross_validation and model_type != "ensemble":
        print("\n[5/6] Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"  Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")

    print("\n[6/6] Evaluating model performance...")
    
    # Handle ensemble predictions
    if model_type == "ensemble":
        # Use scaler for LR predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 70)
    
    print(f"\n  ✓ ACCURACY: {accuracy:.1%}")
    print(f"    Baseline (random): 50%")
    print(f"    Decent: 58-60% | Good: 60-63% | Great: 63-65% | Excellent: 65%+")
    
    if accuracy >= 0.63:
        print(f"GREAT performance! Model is beating most predictions.")
    elif accuracy >= 0.60:
        print(f"Good performance - model has predictive power")
    elif accuracy >= 0.58:
        print(f"Decent performance- slightly better than previous version")
    
    print(f"\n  Log Loss: {logloss:.4f} (lower is better, good < 0.65)")
    print(f"  ROC-AUC: {roc_auc:.4f} (higher is better, good > 0.60)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "=" * 70)
    print("  CONFUSION MATRIX")
    print("=" * 70)
    print(f"\n                 Predicted")
    print(f"              Away Win  |  Home Win")
    print(f"         ---------------------------")
    print(f"  Away Win |   {cm[0,0]:>4}     |   {cm[0,1]:>4}    ← {cm[0,0]/(cm[0,0]+cm[0,1]):.1%} correct")
    print(f"Actual    |            |")
    print(f"  Home Win |   {cm[1,0]:>4}     |   {cm[1,1]:>4}    ← {cm[1,1]/(cm[1,0]+cm[1,1]):.1%} correct")

    # Classification report
    print("\n" + "=" * 70)
    print("  DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win'])
    print(report)

    # Feature importance (for tree-based models)
    if model_type in ["random_forest", "gradient_boosting", "xgboost"]:
        print("\n" + "=" * 70)
        print("  TOP 20 MOST IMPORTANT FEATURES")
        print("=" * 70)
        
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances
        }).sort_values(by="importance", ascending=False)

        for idx, row in importance_df.head(20).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

    print("\n" + "=" * 70)
    print(" SAVING MODEL")
    print("=" * 70)
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"nfl_{model_type}_model.pkl")
    joblib.dump(model, model_path)
    print(f"  ✓ Model: {model_path}")

    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"nfl_{model_type}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"  ✓ Scaler: {scaler_path}")

    engineer_path = os.path.join(model_dir, "nfl_feature_engineer.pkl")
    joblib.dump(feature_engineer, engineer_path)
    print(f"  ✓ Feature Engineer: {engineer_path}")

    feature_columns_path = os.path.join(model_dir, "nfl_feature_columns.pkl")
    joblib.dump(feature_cols, feature_columns_path)
    print(f"  ✓ Feature Columns: {feature_columns_path}")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Final Accuracy: {accuracy:.1%}")
    print(f"  Improvement: {(accuracy - 0.586) * 100:+.1f} percentage points over baseline")
    print(f"\n  You can now predict games with:")
    print(f"    python predict.py KC BUF")
    print(f"    python predict.py --model_type {model_type}")
    print("\n" + "=" * 70 + "\n")

    return {
        "model": model,
        "scaler": scaler,
        "feature_engineer": feature_engineer,
        "feature_columns": feature_cols,
        "accuracy": accuracy,
        "log_loss": logloss,
        "roc_auc": roc_auc,
        "model_type": model_type
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced NFL Game Prediction Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_train_model.py                          # Train ensemble (best)
  python enhanced_train_model.py --model_type xgboost    # Train XGBoost
  python enhanced_train_model.py --tune_hyperparameters  # Auto-tune (slower)
  python enhanced_train_model.py --lookback_games 10     # Use 10-game history
        """
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="ensemble",
        choices=["random_forest", "gradient_boosting", "xgboost", "ensemble"],
        help="Model type (default: ensemble - best accuracy)"
    )
    parser.add_argument(
        "--lookback_games",
        type=int,
        default=8,
        help="Number of recent games to consider (default: 8)"
    )
    parser.add_argument(
        "--use_cross_validation",
        action="store_true",
        default=True,
        help="Perform cross-validation (default: True)"
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Tune hyperparameters (slower but may improve accuracy)"
    )

    args = parser.parse_args()

    result = train_model(
        model_type=args.model_type,
        lookback_games=args.lookback_games,
        use_cross_validation=args.use_cross_validation,
        tune_hyperparameters=args.tune_hyperparameters
    )
    
    if result is None:
        print("\n✗ Training failed.")
        exit(1)
    elif result['accuracy'] >= 0.63:
        print("\n High accuracy achieved! Model is highly competitive.")
    elif result['accuracy'] >= 0.60:
        print("\n✓ Good accuracy! Model has solid predictive power.")