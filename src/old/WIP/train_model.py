import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, brier_score_loss, roc_auc_score
)
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

from WIP.data_loader import download_nfl_data
from WIP.features import NFLFeatureEngineer


def train_model(model_type="random_forest", loopback_games=5, use_cross_validation=False, 
                use_time_split=True, tune_hyperparameters=False):
    """
    Train NFL Game Prediction Model with improved features and evaluation
    
    Improvements:
    - Fixed print statements (= * 60 instead of =, 60)
    - Added XGBoost support
    - Added ensemble models
    - Implemented cross-validation
    - Added time-based train/test split
    - Better evaluation metrics (log loss, Brier score, ROC-AUC)
    - Feature scaling for logistic regression
    - Consistent path naming (trained_models/)
    """

    print("=" * 60)
    print("Starting NFL Game Prediction Model Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Model Type: {model_type}")
    print(f"  Lookback Games: {loopback_games}")
    print(f"  Use Cross Validation: {use_cross_validation}")
    print(f"  Use Time-Based Split: {use_time_split}")
    print(f"  Tune Hyperparameters: {tune_hyperparameters}")
    print("=" * 60)

    print("\nStep 1: Loading historical NFL data...")
    schedules_df, seasonal_stats_df, weekly_stats_df, injuries_df = download_nfl_data()

    print(f"  - Loaded {len(schedules_df)} games from schedules data.")
    print(f"  - Loaded {len(injuries_df)} injury records.")

    print("\nStep 2: Creating features...")
    feature_engineer = NFLFeatureEngineer(lookback_games=loopback_games)
    feature_engineer.load_data(schedules_df, injuries_df)
    features_df = feature_engineer.create_features()

    print(f"  - Created features for {len(features_df)} games.")
    
    # Separate metadata from features
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team", "target"]
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    print(f"  - Number of features: {len(feature_cols)}")

    print("\nStep 3: Splitting data into training and testing sets...")
    
    X = features_df[feature_cols]
    y = features_df["target"]

    print(f"\n  Target Distribution:")
    print(f"    Home wins: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
    print(f"    Away wins: {len(y) - y.sum()} ({(len(y) - y.sum()) / len(y) * 100:.1f}%)")

    home_win_percent = y.sum() / len(y) * 100
    if 40 < home_win_percent < 60:
        print("    ✓ Target variable is reasonably balanced.")
    else:
        print("    ⚠ Warning: Target variable is imbalanced. Consider resampling techniques.")

    # Time-based split (more realistic for time-series data like sports)
    if use_time_split:
        # Add gameday back for sorting
        X_with_date = X.copy()
        X_with_date['gameday'] = features_df['gameday'] if 'gameday' in features_df.columns else features_df.index
        
        # Sort by date
        sort_idx = features_df['gameday'].argsort() if 'gameday' in features_df.columns else features_df.index
        X = X.iloc[sort_idx]
        y = y.iloc[sort_idx]
        
        # 80/20 split by time
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n  Using time-based split:")
        print(f"    Training: First {split_idx} games (80%)")
        print(f"    Testing: Last {len(X_test)} games (20%)")
    else:
        # Random split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n  Using random stratified split:")
        print(f"    Training set: {len(X_train)} games ({len(X_train) / len(X) * 100:.1f}%)")
        print(f"    Testing set: {len(X_test)} games ({len(X_test) / len(X) * 100:.1f}%)")

    print("\nStep 4: Training the model...")
    
    scaler = None
    model = None
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        print("  Using Random Forest with optimized hyperparameters")
        
    elif model_type == "logistic_regression":
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            n_jobs=-1
        )
        print("  Using Logistic Regression with feature scaling")
        
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        print("  Using Gradient Boosting")
        
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        print("  Using XGBoost (often the best for structured data)")
        
    elif model_type == "ensemble":
        print("  Building ensemble of Random Forest, Gradient Boosting, and Logistic Regression")
        
        # Scale data for logistic regression component
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
        lr = LogisticRegression(max_iter=500, random_state=42)
        
        # Train RF and GB on original data
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train_scaled, y_train)
        
        # Create voting classifier
        model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
        
        # For ensemble, we need to handle predictions differently
        # We'll store the individual models
        model.fitted_estimators = [rf, gb, lr]
        model.is_ensemble = True
        model.scaler = scaler
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    if model is not None and not (model_type == "ensemble"):
        model.fit(X_train, y_train)
        print("  ✓ Model training complete.")

    # Cross-validation
    if use_cross_validation and model_type != "ensemble":
        print("\n  Performing cross-validation...")
        cv_splitter = TimeSeriesSplit(n_splits=5) if use_time_split else 5
        
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv_splitter, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"    Cross-validation scores: {cv_scores}")
        print(f"    Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    print("\nStep 5: Evaluating the model...")
    
    # Make predictions
    if model_type == "ensemble":
        # Handle ensemble predictions specially
        rf_pred = model.fitted_estimators[0].predict_proba(X_test)[:, 1]
        gb_pred = model.fitted_estimators[1].predict_proba(X_test)[:, 1]
        lr_pred = model.fitted_estimators[2].predict_proba(scaler.transform(X_test))[:, 1]
        
        y_pred_proba = (rf_pred + gb_pred + lr_pred) / 3
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"\n  Accuracy: {accuracy * 100:.2f}%")
    print(f"    → Model correctly predicts {accuracy:.1%} of games")
    print(f"    → Baseline (random): 50% | Good: 60-65% | Great: 65-70%")
    print(f"    → >70% may indicate overfitting")
    
    print(f"\n  Log Loss: {logloss:.4f}")
    print(f"    → Measures probability quality (lower is better)")
    print(f"    → Good: <0.65 | Great: <0.60")
    
    print(f"\n  Brier Score: {brier:.4f}")
    print(f"    → Measures calibration (lower is better)")
    print(f"    → Good: <0.25 | Great: <0.23")
    
    print(f"\n  ROC-AUC: {roc_auc:.4f}")
    print(f"    → Measures ranking ability (higher is better)")
    print(f"    → Random: 0.50 | Decent: >0.55 | Good: >0.60")

    # Classification report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    report = classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win'])
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"\n                 Predicted")
    print(f"              Away Win  |  Home Win")
    print(f"         ---------------------------")
    print(f"  Away Win |   {cm[0,0]:>4}     |   {cm[0,1]:>4}")
    print(f"Actual    |            |")
    print(f"  Home Win |   {cm[1,0]:>4}     |   {cm[1,1]:>4}")
    
    # Feature importance (for tree-based models)
    if model_type in ["random_forest", "gradient_boosting", "xgboost"]:
        print(f"\n{'='*60}")
        print("TOP 15 MOST IMPORTANT FEATURES")
        print(f"{'='*60}\n")
        
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances
        }).sort_values(by="importance", ascending=False)

        for idx, row in importance_df.head(15).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"  {row['feature']:.<35} {row['importance']:.4f} {bar}")

    print(f"\n{'='*60}")
    print("Step 6: Saving trained model...")
    print(f"{'='*60}\n")
    
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"nfl_{model_type}_model.pkl")
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")

    # Save scaler if used
    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"nfl_{model_type}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"  ✓ Scaler saved to {scaler_path}")

    # Save feature engineer
    engineer_path = os.path.join(model_dir, f"nfl_feature_engineer.pkl")
    joblib.dump(feature_engineer, engineer_path)
    print(f"  ✓ Feature engineer saved to {engineer_path}")

    # Save feature columns
    feature_columns_path = os.path.join(model_dir, f"nfl_feature_columns.pkl")
    joblib.dump(feature_cols, feature_columns_path)
    print(f"  ✓ Feature columns saved to {feature_columns_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")
    print(f"  Final Accuracy: {accuracy * 100:.1f}%")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\n  You can now use predict.py to make predictions on future games.")
    print(f"{'='*60}\n")

    return {
        "model": model,
        "scaler": scaler,
        "feature_engineer": feature_engineer,
        "feature_columns": feature_cols,
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier_score": brier,
        "roc_auc": roc_auc,
        "model_type": model_type,
        "confusion_matrix": cm,
        "classification_report": report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL Game Prediction Model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression", "gradient_boosting", "xgboost", "ensemble"],
        help="Type of model to train (default: random_forest)"
    )
    parser.add_argument(
        "--lookback_games",
        type=int,
        default=5,
        help="Number of past games to consider for feature engineering (default: 5)"
    )
    parser.add_argument(
        "--use_cross_validation",
        action="store_true",
        help="Whether to use cross-validation during training (default: False)"
    )
    parser.add_argument(
        "--use_time_split",
        action="store_true",
        default=True,
        help="Use time-based train/test split instead of random (default: True)"
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Perform hyperparameter tuning (slower but may improve accuracy)"
    )

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        loopback_games=args.lookback_games,
        use_cross_validation=args.use_cross_validation,
        use_time_split=args.use_time_split,
        tune_hyperparameters=args.tune_hyperparameters
    )