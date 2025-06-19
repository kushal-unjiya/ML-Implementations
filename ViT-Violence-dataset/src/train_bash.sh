#!/bin/bash

# Comprehensive Violence Detection Training Script
# =================================================

echo "🧹 Cleaning workspace..."

# Remove old experiments, checkpoints, and logs
rm -rf experiments/
rm -rf checkpoints/*/
rm -rf logs/fit/*/
rm -rf logs/comprehensive_training/
rm -f src/experiment_note/*.out
rm -f *.log

echo "Workspace cleaned! Workspace cleaned!"

echo ""
echo "Starting comprehensive training with enhanced logging... Starting comprehensive training with enhanced logging..."
echo "Features: Features:"
echo "   • Clean console output with progress bars"
echo "   • Comprehensive metrics logging"
echo "   • Training curve visualization"
echo "   • Automatic plot generation"
echo "   • CSV and JSON export"
echo ""

# Activate virtual environment and run training
source .venv/bin/activate
cd src
python -W ignore train_comprehensive.py

echo ""
echo "🎉 Training completed! Check the logs/ directory for detailed analysis."