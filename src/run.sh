#!/bin/bash

# Simple run script for NFL Predictor

# Clean old models
rm -rf trained_models/

# Train model
python src/train_model.py

# Run predictions
python src/predict.py