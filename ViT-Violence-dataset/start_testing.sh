#!/bin/bash
echo "🧪 Starting Comprehensive Violence Detection Testing"
echo "=================================================="

# Activate virtual environment
source .venv/bin/activate

# Change to src directory and start testing
cd src
echo "🔍 Starting model evaluation..."
python test.py
