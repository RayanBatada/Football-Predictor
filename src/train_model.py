'''import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import os


from data_loader import download_nfl_data
from features import NFLFeatureEngineer


def train_model(model_type="random_forest", loopback_games=5, use_cross_validation=False):
    """
    We're solving a Binary Classifcation problem:
    - Input (X): Features about both teams (performance, injuries, rest, etc.)
    - Output (y): 0 (away team wins) or 1 (home team wins)
    - Goal: Learn patterns from historical games to predict future outcomes (via Random Forest or Logistic Regression)

    Steps:
    1. Load Data (schedules, injuries, etc.)
    2. Create features using NFLFeatureEngineer (turn raw data into model-ready features)
    3. Split data into training and testing sets (for evaluating model accuracy)
    4. Train  model (Random Forest or Logistic Regression - learn patterns from training data)
    5. Evaluate model performance (check accuracy on unseen data)
    6. Save trained model to disk (persist for future predictions)
    """

    print("=", 60)
    print("Starting NFL Game Prediction Model Training")
    print("=", 60)
    print("\nConfiguration")
    print(f"Model Type: {model_type}")
    print(f"Lookback Games: {loopback_games}")
    print(f"Use Cross Validation: {use_cross_validation}")
    print("=", 60)

    print("\nStep 1: Loading historical NFL data...")

    schedules_df, seasonal_stats_df, weekly_stats_df, injuries_df = download_nfl_data()

    # Each game is one example the model uses (random forest or logistic regression) to learn patterns
    print(f" - Loaded {len(schedules_df)} games from schedules data.")
    print(f" - Loaded {len(injuries_df)} injury records.")

    print("\nStep 2: Creating features...")
    feature_engineer = NFLFeatureEngineer(lookback_games=loopback_games)
    feature_engineer.load_data(schedules_df, injuries_df)
    features_df = feature_engineer.create_features()

    print(f" - Created features for {len(features_df)} games.")
    print(f" - Features per example: {len(features_df.columns) - 1}")  # minus 6 for target variable and identifiers

    print("\nStep 3: Splitting data into training and testing sets...")
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team", "target"] # columns to exclude from features
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]

    X = features_df[feature_cols]
    y = features_df["target"]

    print(f" - Feature columns: {feature_cols}") # print feature columns (which columns are being used as features)
    print(f"  - Target distribution:")
    print(f" .. Home wins: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)") # print home wins
    print(f"  .. Away wins: {len(y) - y.sum()} ({(len(y) - y.sum()) / len(y) * 100:.1f}%)") # print away wins

    home_win_percent = y.sum() / len(y) * 100
    if 0.4 < home_win_percent < 0.6:
        print("  - Target variable is reasonably balanced.")
    else:
        print("  - Warning: Target variable is imbalanced. Consider resampling techniques.")

    # 80% training set, 20% testing set split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f" - Training set size: {len(X_train)} examples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f" - Testing set size: {len(X_test)} examples ({len(X_test) / len(X) * 100:.1f}%)")

    print("\nStep 4: Training the model...")
    model = None
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100, # number of trees in the forest
            max_depth=10,    # maximum depth of each tree (prevents overfitting)
            random_state=42, # for reproducibility 
            n_jobs=-1 # Use all available CPU cores for faster training
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=200, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

    if model is not None:
        model.fit(X_train, y_train)
        print(" - Model training complete.")
    else:
        print(" - Model training failed.")
        return
    
    if use_cross_validation:
        ...

    print("\nStep 5: Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {  accuracy * 100:.1f}% -- This means the model correctly predicts {accuracy:.1%} of games in the test set.")
    print("(Baseline is 50% is as good as randomly guessing, good 60-65 (useful predictions), great score is 65-70%, anything above 70% may suggest overfitting)")

    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f" Predicted ->       Away win      |     Home win")
    print(f" Actual             -----------------------------")
    print(f" Away win      |     {cm[0,0] :>8}           |     {cm[0,1]:>8}")
    print(f" Home win      |     {cm[1,0] :>8}           |     {cm[1,1]:>8}")

    if model_type == "random_forest":
        # Random forest calculates importance by measuring how much each feature decreases impurity across all trees
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances
        }).sort_values(by="importance", ascending=False)

        for idx, row in importance_df.head(10).iterrows():
            print(f" - Feature: {row['feature']}, Importance: {row['importance']:.4f}")
    
    

    print("\nStep 6: Saving trained model to disk...")
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model to avoid retraining every time
    model_path = os.path.join(model_dir, f"nfl_{model_type}_model.pkl")
    joblib.dump(model, model_path)
    print(f" - Model saved to {model_path}")

    # Save feature engineer for future feature transformations (feature transformations are features that need to be applied to new data before making predictions)
    engineer_path = os.path.join(model_dir, f"nfl_feature_engineer.pkl") # save feature engineer (feature engineer is used for transforming features) as well
    joblib.dump(feature_engineer, engineer_path)
    print(f" - Feature engineer saved to {engineer_path}")

    # Save feature columns used in the model to ensure consistent feature ordering during inference (model expects features in same order as training)
    feature_columns_path = os.path.join(model_dir, f"nfl_feature_columns.pkl")
    joblib.dump(feature_cols, feature_columns_path)
    print(f" - Feature columns saved to {feature_columns_path}")

    print("\n" + "=" * 60)
    print("\nModel training and saving complete!")
    print("\n" + "=" * 60)
    print(f"Your model achieved {accuracy * 100:.1f}% accuracy on the test set.")
    print("You can now use src/train_model.py to make predictions on future games.")

    return {
        "model": model,
        "feature_engineer": feature_engineer,
        "feature_columns": feature_cols,
        "accuracy": accuracy,
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
        choices=["random_forest", "logistic_regression"],
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

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        loopback_games=args.lookback_games,
        use_cross_validation=args.use_cross_validation

    )'''
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

from data_loader import download_nfl_data
from features import NFLFeatureEngineer


def train_model(model_type="random_forest", lookback_games=5, use_cross_validation=False):
    """
    We're solving a Binary Classification problem:
    - Input (X): Features about both teams (performance, injuries, rest, etc.)
    - Output(y): 0 (away team wins) or 1 (home team wins)
    - Goal: Learn patterns from historical games to predict future outcomes (via Random Forest or Logist Regression)
    
    Steps:
    1. Load historical data (schedules, injuries)
    2. Create features using NFLFeatureEngineer (turn raw data into model-ready features)
    3. Split data into training and testing sets (for evaluating model accuracy)
    4. Train model (learn patterns between features and outcomes)
    5. Evaluate model (check accuracy on unseen test data)
    6. Save trained model to disk (persist for future use)
    """
    
    print("=", 60)
    print("Starting NFL Game Prediction Model Training")
    print("=", 60)
    print("\nConfiguration:")
    print(f"- Model: {model_type}")
    print(f"- Lookback Games for Features: {lookback_games}")
    print(f"- Use Cross-Validation: {use_cross_validation}")

    print("\nStep 1: Loading historical NFL data...")
    schedules_df, seasonal_stats_df, weekly_stats_df, injuries_df = download_nfl_data()

    # Each game is one example the model uses to learn on
    print(f" - Loaded {len(schedules_df)} games from schedules.")
    print(f" - Loaded {len(injuries_df)} injury reports.")

    print("\nStep 2: Creating features...")
    feature_engineer = NFLFeatureEngineer(lookback_games=lookback_games)
    feature_engineer.load_data(schedules_df, injuries_df)
    features_df = feature_engineer.create_features()
    
    print(f" - Created {len(features_df)} training examples")
    print(f" - Features per examples: {len(features_df.columns) - 6} (excluding metadata and target)")
    
    print("\nStep 3: Splitting data into training and testing sets...")
    metadata_cols = ["game_id", "season", "week", "home_team", "away_team", "target"] # Separate metadata and target from actual features
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]

    X = features_df[feature_cols]
    y = features_df["target"]
    
    print(f" - Feature columns: {feature_cols}")
    print(f"  - Target distribution:")
    print(f" .. Home wins: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
    print(f" .. Away wins: {len(y) - y.sum()} ({(len(y) - y.sum()) / len(y) * 100:.1f}%)")

    home_win_percent = y.sum() / len(y)
    if 0.4 <= home_win_percent <= 0.6:
        print(" - Target variable is reasonably balanced.")
    else:
        print(" - Warning: Target variable is imbalanced. Consider resampling techniques.")

    # 80% training set, 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f" - Training set: {len(X_train)} examples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f" - Testing set: {len(X_test)} examples ({len(X_test) / len(X) * 100:.1f}%)")
    
    print("\nStep 4: Training the model...")
    model = None # TODO: TEACHING - SESSIon - RAYAN - CHANGES MADE HERE (and below if statement)
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100, # Use 100 trees in the forest (more trees = better performance but slower), too many can be overkill, too few can underfit
            max_depth=10,    # Limit depth of each tree to prevent overfitting (ensures learning general patterns)
            random_state=42,
            n_jobs=-1         # Use all CPU cores for faster training
        )
    else:
        print("TODO - Implement Logistic Regression")
        ...

    if model is not None:
        model.fit(X_train, y_train)
        print("CHECKPOINT - Model training complete.")
    else:
        print("Error: Model training failed due to unsupported model type.")
        return
    
    if use_cross_validation:
        ...
    
    print("\nStep 5: Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # TODO: TEACHING - SESSION - RAYAN - CHANGES MADE HERE (can't have accuracy_score be name...)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Test Accuracy: {accuracy:.1%} -- This means the model correctly predicts {accuracy:.1%} of games on unseen data.")
    print("(Baseline is 50% is as good as randomly guessing, good 60-65% (useful predictions), great score is 65-70%, anything > 70% may suggest overfitting)")

    report = classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"])
    print("\nClassification Report:")
    print(report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f" Predicted ->    Awary Win   |   Home Win")
    print(f" Actual         -------------------------")
    print(f" Away Win      |   {conf_matrix[0,0] :>8}   |   {conf_matrix[0,1]:>8} ")
    print(f" Home Win      |   {conf_matrix[1,0] :>8}   |   {conf_matrix[1,1]:>8} ")
    '''
    CORRECTLY PREDICTED AWAY WINS            | PREDICTED HOME WIN BUT ACTUALLY AWAY WIN
    PREDICTED AWAY WIN BUT ACTUALLY HOME WIN | CORRECTLY PREDICTED HOME WINS
    '''

    if model_type == "random_forest":
        # Random Forest calculates importance by measuring how much each feature decreases (Gini) impurity across all trees
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances
        }).sort_values(by="importance", ascending=False)

        for idx, row in importance_df.head(10).iterrows():
            print(f" - Feature: {row['feature']}, Importance: {row['importance']:.4f}")
    
    print("\nStep 6: Saving the trained model to disk...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model itself to avoid needing to retrain (learned parameters, tree structures/weights)
    model_path = os.path.join(model_dir, f"nfl_{model_type}_model.pkl")
    joblib.dump(model, model_path)
    print(f" - Model saved to: {model_path}")

    # Save feature engineer to ensure consistent feature creation during inference (historical data tracked for feature creation)
    engineer_path = os.path.join(model_dir, f"nfl_feature_engineer.pkl")
    joblib.dump(feature_engineer, engineer_path)
    print(f" - Feature engineer saved to: {engineer_path}")

    # Save feature columns to ensure consistent feature ordering during inference (model expects features in exact same order as training)
    feature_columns_path = os.path.join(model_dir, f"nfl_feature_columns.pkl")
    joblib.dump(feature_cols, feature_columns_path)
    print(f" - Feature columns saved to: {feature_columns_path}")


    print("\n" + "=" * 60)
    print("\nModel training pipeline complete.")
    print("=" * 60 + "\n")
    print(f"Your model achieved {accuracy:.1%} accuracy on the test set.")
    print("You can now use src/predict.py to make predictions on future games")


    return {
        "model": model,
        "feature_engineer": feature_engineer,
        "feature_columns": feature_cols,
        "accuracy": accuracy,
        "model_type": model_type,
        "confusion_matrix": conf_matrix,
        "classification_report": report
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL Game Prediction Model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Type of model to train (default: random_forest)"
    )
    parser.add_argument(
        "--lookback_games",
        type=int,
        default=5,
        help="Number of past games to consider for feature creation (default: 5)"
    )
    parser.add_argument(
        "--use_cross_validation",
        action="store_true",
        help="Whether to use cross-validation during training (default: False)"
    )

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        lookback_games=args.lookback_games,
        use_cross_validation=args.use_cross_validation
    )