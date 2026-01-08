#!/bin/bash

# Simple run script for NFL Predictor
# Run this from the src/ directory

echo "🏈 NFL Game Prediction Pipeline"
echo "================================"
echo ""

# Clean old models
echo "Cleaning old models..."
rm -rf ../models/

# Train model
echo ""
echo "Starting model training..."
python train_model.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training complete!"
    echo ""
    echo "Starting prediction mode..."
    echo ""
    
    # Run predictions
    python predict.py
else
    echo ""
    echo "✗ Training failed. Please check the errors above."
    exit 1
fi
