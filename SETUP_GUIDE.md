# 🚀 Setup Guide - NFL Game Prediction Model

Complete guide for setting up and running the NFL game prediction model.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Directory Structure](#directory-structure)
- [First Run](#first-run)
- [Common Issues & Solutions](#common-issues--solutions)
- [Team Name Reference](#team-name-reference)

---

## Prerequisites

### Required Software
- **Python**: Version 3.8 or higher
- **conda** (recommended) or **pip**
- **Git** (optional, for cloning)

### Check Your Python Version
```bash
python --version
# or
python3 --version
```

If Python is not installed or version is < 3.8:
- **macOS**: `brew install python3`
- **Ubuntu/Debian**: `sudo apt-get install python3`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

---

## Installation Steps

### Step 1: Get the Code

**Option A: Clone from repository**
```bash
git clone <repository-url>
cd Football-Predictor
```

**Option B: Download ZIP**
1. Download and extract the ZIP file
2. Navigate to the folder in terminal

### Step 2: Create Virtual Environment

**Using conda (Recommended)**
```bash
# Create new environment
conda create -n football python=3.9

# Activate environment
conda activate football

# Verify activation (you should see "(football)" in your prompt)
```

**Using venv (Alternative)**
```bash
# Create virtual environment
python3 -m venv football-env

# Activate on macOS/Linux
source football-env/bin/activate

# Activate on Windows
football-env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Make sure you're in the project directory
cd Football-Predictor

# Install all required packages
pip install -r requirements.txt
```

**What gets installed:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning
- `nfl_data_py` - NFL data API
- `matplotlib` - Plotting
- `seaborn` - Advanced plotting
- `tqdm` - Progress bars
- `joblib` - Model serialization

### Step 4: Verify Installation

```bash
# Test imports
python -c "import pandas, numpy, sklearn, nfl_data_py; print('✓ All packages installed successfully!')"
```

If you see the success message, you're ready to go!

---

## Directory Structure

After setup, your project should look like this:

```
Football-Predictor/
├── src/                         # Source code
│   ├── data_loader.py          # Downloads NFL data
│   ├── features.py             # Feature engineering
│   ├── train_model.py          # Model training
│   ├── predict.py              # Make predictions
│   ├── run.sh                  # Automation script
│   └── WIP/                    # Work in progress code
├── models/                      # Trained models (created after first run)
│   ├── nfl_random_forest_model.pkl
│   ├── nfl_feature_engineer.pkl
│   └── nfl_feature_columns.pkl
├── data/                        # Cached data (auto-generated)
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
└── SETUP_GUIDE.md              # This file
```

---

## First Run

### Quick Start (Automated)

The easiest way to get started:

```bash
# Navigate to src directory
cd src

# Make run script executable (macOS/Linux only)
chmod +x run.sh

# Run the complete pipeline
./run.sh
```

This will:
1. ✓ Clean old models
2. ✓ Download NFL data (takes 30-60 seconds first time)
3. ✓ Train the model (takes 2-3 minutes)
4. ✓ Launch interactive prediction mode

### Step-by-Step (Manual)

If you prefer to run each step manually:

**1. Train the model**
```bash
cd src
python train_model.py
```

Expected output:
```
============================================================
Starting NFL Game Prediction Model Training
============================================================
Downloading NFL data...
  ✓ Loaded 1411 games
Creating features...
  ✓ Created 1363 training examples
Training model...
  ✓ Model training complete
Test Accuracy: 58.6%
============================================================
```

**2. Make predictions**
```bash
python predict.py
```

Then enter team names when prompted:
```
Enter the home team: KC
Enter the away team: BUF
```

---

## Common Issues & Solutions

### 🔴 Issue 1: "No trained model found"

**Problem:**
```
FileNotFoundError: No trained model found in models directory.
```

**Solution:**
You need to train the model first:
```bash
cd src
python train_model.py
```

---

### 🔴 Issue 2: "Not enough historical data for 'DT'"

**Problem:**
```
ValueError: Not enough historical data for 'DT'. Need at least 3 games.
```

**Cause:** Invalid team abbreviation (DT should be DET)

**Solution:** Use correct 3-letter team abbreviations:
- ✅ Correct: `DET` (Detroit Lions)
- ❌ Wrong: `DT`, `DETROIT`, `det`

See [Team Name Reference](#team-name-reference) below for all valid abbreviations.

---

### 🔴 Issue 3: KeyError in predictions

**Problem:**
```
KeyError: 'home_avg_point_scored'
```

**Cause:** Mismatch between feature names in training and prediction

**Solution:** Use the fixed files provided:
1. Replace `src/predict.py` with the fixed version
2. Replace `src/features.py` with the fixed version
3. Retrain the model: `python train_model.py`

---

### 🔴 Issue 4: Module not found errors

**Problem:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
# Make sure you're in the virtual environment
conda activate football

# Reinstall dependencies
pip install -r requirements.txt
```

---

### 🔴 Issue 5: Permission denied on run.sh

**Problem:**
```
zsh: permission denied: ./run.sh
```

**Solution:**
```bash
chmod +x run.sh
./run.sh
```

---

### 🔴 Issue 6: Wrong Python version

**Problem:**
```
python: command not found
```

**Solution:**
Try `python3` instead:
```bash
python3 train_model.py
python3 predict.py
```

Or update your PATH:
```bash
alias python=python3
```

---

### 🔴 Issue 7: "Could not load injuries: HTTP Error 404"

**Problem:**
```
⚠ Could not load injuries: HTTP Error 404: Not Found
```

**Cause:** NFL injury data endpoint has limited availability

**Solution:** This is expected and **not a problem**!
- The model still works without injury data
- Injury count features will be set to 0
- Predictions remain accurate

---

### 🔴 Issue 8: Slow data download

**Problem:** First run takes a long time to download data

**Solution:** This is normal!
- First download: 30-60 seconds (downloads ~1400 games)
- Subsequent runs: Nearly instant (data is cached)
- To force fresh data: delete the `data/` directory

---

## Team Name Reference

### Valid Team Abbreviations

Use these exact 2-3 letter codes when making predictions:

#### AFC East
- `BUF` - Buffalo Bills
- `MIA` - Miami Dolphins
- `NE` - New England Patriots
- `NYJ` - New York Jets

#### AFC North
- `BAL` - Baltimore Ravens
- `CIN` - Cincinnati Bengals
- `CLE` - Cleveland Browns
- `PIT` - Pittsburgh Steelers

#### AFC South
- `HOU` - Houston Texans
- `IND` - Indianapolis Colts
- `JAX` - Jacksonville Jaguars
- `TEN` - Tennessee Titans

#### AFC West
- `DEN` - Denver Broncos
- `KC` - Kansas City Chiefs
- `LV` - Las Vegas Raiders
- `LAC` - Los Angeles Chargers

#### NFC East
- `DAL` - Dallas Cowboys
- `NYG` - New York Giants
- `PHI` - Philadelphia Eagles
- `WAS` - Washington Commanders

#### NFC North
- `CHI` - Chicago Bears
- `DET` - Detroit Lions
- `GB` - Green Bay Packers
- `MIN` - Minnesota Vikings

#### NFC South
- `ATL` - Atlanta Falcons
- `CAR` - Carolina Panthers
- `NO` - New Orleans Saints
- `TB` - Tampa Bay Buccaneers

#### NFC West
- `ARI` - Arizona Cardinals
- `LA` or `LAR` - Los Angeles Rams
- `SF` - San Francisco 49ers
- `SEA` - Seattle Seahawks

---

## Testing Your Setup

Run these tests to verify everything works:

### Test 1: Data Download
```bash
cd src
python data_loader.py
```
Expected: Should download ~1400 games successfully

### Test 2: Feature Creation
```bash
python -c "from features import NFLFeatureEngineer; print('✓ Features module working')"
```

### Test 3: Model Training
```bash
python train_model.py
```
Expected: Should complete in 2-3 minutes with ~58-60% accuracy

### Test 4: Predictions
```bash
python predict.py KC BUF
```
Expected: Should show prediction with probabilities

---

## Next Steps

Once setup is complete:

1. **Read the main README.md** for detailed usage instructions
2. **Try making predictions** for upcoming games
3. **Experiment with different models:**
   ```bash
   python train_model.py --model_type logistic_regression
   python train_model.py --model_type random_forest --lookback_games 10
   ```
4. **Explore the data** using `explore_nfl_data.py`

---

## Getting Help

If you encounter issues not covered here:

1. **Check error messages carefully** - they often indicate the exact problem
2. **Verify your Python version** is 3.8+
3. **Ensure virtual environment is activated** (you should see `(football)` in prompt)
4. **Try reinstalling dependencies**: `pip install --force-reinstall -r requirements.txt`
5. **Check file paths** - make sure you're in the correct directory

---

## Performance Expectations

### Training Time
- First run with data download: 2-3 minutes
- Subsequent runs (cached data): 1-2 minutes
- On slower machines: up to 5 minutes

### Model Accuracy
- Random Forest: 58-60% (typical)
- Baseline (random): 50%
- Good performance: 60-65%
- Excellent performance: 65%+

### Prediction Speed
- Feature generation: < 1 second
- Prediction: < 0.1 seconds
- Interactive mode: Instant

---

**Setup complete! You're ready to predict NFL games! 🏈**

For detailed usage instructions, see [README.md](README.md).
