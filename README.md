# 🏈 NFL Game Prediction Model

A machine learning project that predicts NFL game outcomes using historical data, team performance metrics, and advanced feature engineering.

## 📊 Project Overview

This project uses historical NFL game data to train a Random Forest classifier that predicts whether the home team will win. The model achieves **~59% accuracy** on test data, which is better than random guessing (50%) and competitive for sports prediction.

### Key Features
- **Historical Data**: Uses data from 2021-2025 NFL seasons
- **Advanced Features**: 29+ engineered features including win rates, point differentials, rest days, strength of schedule, home/away splits, and injury counts
- **Multiple Models**: Support for Random Forest, Logistic Regression, Gradient Boosting, XGBoost, and Ensemble models
- **Easy Predictions**: Simple command-line interface for predicting any matchup

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- conda (recommended) or pip

### Installation

1. **Clone or download this repository**

2. **Create a conda environment** (recommended):
```bash
conda create -n football python=3.9
conda activate football
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
nfl_data_py==0.3.0
matplotlib==3.8.2
seaborn==0.13.0
tqdm==4.66.1
joblib==1.3.2
```

## 🎯 Usage

### Method 1: Automated Training & Prediction (Recommended)

Run the complete pipeline:
```bash
./src/run.sh
```

This will:
1. Clean old models
2. Download NFL data
3. Train the model
4. Enter interactive prediction mode

### Method 2: Manual Training

Train the model manually:
```bash
# Navigate to src directory
cd src

# Train with default settings (Random Forest)
python train_model.py

# Train with different model types
python train_model.py --model_type logistic_regression
python train_model.py --model_type gradient_boosting
python train_model.py --model_type xgboost
python train_model.py --model_type ensemble

# Customize lookback window
python train_model.py --lookback_games 10

# Enable cross-validation
python train_model.py --use_cross_validation
```

### Method 3: Making Predictions

After training, make predictions:

**Interactive Mode:**
```bash
python src/predict.py
```

**Single Prediction:**
```bash
python src/predict.py "Kansas City Chiefs" "Buffalo Bills"
# Or use team abbreviations
python src/predict.py KC BUF
```

**With Specific Date:**
```bash
python src/predict.py KC BUF --game_date 2025-01-15
```

**Use Different Model:**
```bash
python src/predict.py KC BUF --model_type xgboost
```

## 🏟️ Team Abbreviations

Use these standard NFL team abbreviations for predictions:

**AFC East:** BUF, MIA, NE, NYJ  
**AFC North:** BAL, CIN, CLE, PIT  
**AFC South:** HOU, IND, JAX, TEN  
**AFC West:** DEN, KC, LV, LAC  

**NFC East:** DAL, NYG, PHI, WAS  
**NFC North:** CHI, DET, GB, MIN  
**NFC South:** ATL, CAR, NO, TB  
**NFC West:** ARI, LA (or LAR), SF, SEA  

## 📈 Model Performance

### Random Forest (Default)
- **Accuracy:** ~58-60%
- **Baseline:** 50% (random guessing)
- **Good:** 60-65%
- **Great:** 65-70%

### Feature Importance (Top 10)
1. Point Differential
2. Average Points Scored
3. Strength of Schedule
4. Win Rate
5. Home/Away Win Rate
6. Weighted Recent Form
7. Scoring Trend
8. Rest Days
9. Injury Count
10. Head-to-Head Record

## 🔧 Project Structure

```
Football Predictor/
├── src/
│   ├── data_loader.py        # Downloads NFL data via nfl_data_py
│   ├── features.py            # Feature engineering (basic)
│   ├── train_model.py         # Model training pipeline
│   ├── predict.py             # Prediction interface
│   ├── run.sh                 # Automated pipeline script
│   ├── test_data.py           # Data exploration
│   ├── explore_nfl_data.py    # API exploration tool
│   └── WIP/                   # Advanced implementations
│       ├── features.py        # Advanced feature engineering
│       ├── train_model.py     # Enhanced training with more models
│       ├── predict.py         # Enhanced predictions
│       └── check_data_quality.py
├── models/                     # Saved trained models
│   ├── nfl_random_forest_model.pkl
│   ├── nfl_feature_engineer.pkl
│   └── nfl_feature_columns.pkl
├── data/                       # Cached data (auto-generated)
├── notebooks/                  # Jupyter notebooks for analysis
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎓 How It Works

### 1. Data Collection
- Downloads historical NFL game data (2021-2025) using `nfl_data_py`
- Includes schedules, scores, team stats, and injury reports
- Data is cached locally for faster subsequent runs

### 2. Feature Engineering
For each game, the model creates 29+ features describing both teams:

**Performance Metrics:**
- Win rate (last 5 games)
- Average points scored
- Average points allowed
- Point differential
- Weighted recent form

**Situational Factors:**
- Rest days since last game
- Bye week indicator
- Short rest indicator (<6 days)
- Home/away splits

**Advanced Metrics:**
- Strength of schedule
- Scoring trends
- Head-to-head history
- Injury count

### 3. Model Training
- Uses scikit-learn's Random Forest Classifier
- 80/20 train-test split
- Prevents overfitting with max_depth and min_samples_leaf
- Evaluates with accuracy, precision, recall, F1-score

### 4. Prediction
- Takes any matchup (home team vs away team)
- Generates features using recent historical data
- Outputs win probability and confidence level
- Shows key factors influencing the prediction

## 🐛 Troubleshooting

### Common Issues

**1. "No trained model found"**
```bash
# Solution: Train the model first
python src/train_model.py
```

**2. "Not enough historical data for team"**
- Check team abbreviation spelling (e.g., "DET" not "DT")
- Use standard 2-3 letter abbreviations
- Team must have played at least 3 games in the dataset

**3. "Module not found" errors**
```bash
# Make sure you're in the right environment
conda activate football
pip install -r requirements.txt
```

**4. Run script permission denied**
```bash
# Make the script executable
chmod +x src/run.sh
```

**5. Injury data not loading (404 error)**
- This is expected - injury data endpoint has limited availability
- Model works without injury data (injury_count will be 0)
- Features are still accurate without this data

### Data Freshness Warning
If you see "WARNING: Using stale data", it means:
- The most recent game in the dataset is >30 days old
- Predictions will be based on outdated team performance
- Re-run data_loader.py to fetch fresh data

## 📊 Example Output

```
============================================================
  PREDICTING: BUF @ KC
============================================================

[Step 1] Generating features for both teams...
  ✓ Features created successfully

[Step 2] Making prediction...

============================================================
  PREDICTION RESULTS
============================================================

  🏆 PREDICTED WINNER: KC
  📊 CONFIDENCE: 64.2% (HIGH)

  Breakdown:
    • KC (Home): 64.2%
    • BUF (Away): 35.8%

============================================================
  KEY FACTORS INFLUENCING PREDICTION
============================================================

  Recent Performance:
    KC......................... Win Rate: 80.0%
    BUF........................ Win Rate: 60.0%

  Offensive Power (Avg Points Scored):
    KC......................... 28.4 PPG
    BUF........................ 25.1 PPG

  Defensive Strength (Avg Points Allowed):
    KC......................... 18.2 PPG
    BUF........................ 22.3 PPG

  Point Differential:
    KC......................... +10.2
    BUF........................ +2.8
```

## 🔬 Advanced Usage

### Explore the NFL Data API
```bash
python src/explore_nfl_data.py
```

This interactive tool lets you explore:
- Game schedules and scores
- Team statistics
- Player statistics
- Rosters and depth charts
- Injury reports
- Draft picks

### Custom Feature Engineering
Edit `src/features.py` to add new features:
- Turnover margins
- Time zone differences
- Weather conditions
- Coaching experience
- QB injury status

### Model Comparison
Train multiple models and compare:
```bash
python src/train_model.py --model_type random_forest
python src/train_model.py --model_type xgboost
python src/train_model.py --model_type ensemble
```

## 📝 Notes

- **Accuracy Expectations**: 58-60% accuracy is competitive for NFL prediction
- **Home Field Advantage**: Model accounts for home/away performance splits
- **Injuries**: Limited injury data due to API restrictions
- **Updates**: Data is cached; delete `data/` folder to force refresh
- **Ethics**: This is for educational purposes only, not gambling advice

## 🤝 Contributing

Potential improvements:
- [ ] Add weather data integration
- [ ] Include player-specific stats (QB rating, RB yards, etc.)
- [ ] Implement betting line predictions
- [ ] Add real-time game predictions
- [ ] Create web interface
- [ ] Add playoff probability calculations
- [ ] Include coaching matchup analysis

## 📄 License

This project is for educational purposes only.

## 🙏 Acknowledgments

- Data provided by [nfl_data_py](https://github.com/nflverse/nfl_data_py)
- Built with scikit-learn, pandas, and NumPy
- NFL data courtesy of the NFL and nflverse project

---

**Have fun predicting games! 🏈**

For questions or issues, please check the troubleshooting section or open an issue.
