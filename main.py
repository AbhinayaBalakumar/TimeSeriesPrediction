#!/usr/bin/env python3
"""
Main script for time series prediction with LSTM and Prophet models.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from evaluation.evaluator import ModelEvaluator
from utils.config import Config
from utils.logger import setup_logger

# Optional imports - only load if available
try:
    from models.lstm_model import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  TensorFlow not available - LSTM model disabled")

try:
    from models.prophet_model import ProphetPredictor
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not available - Prophet model disabled")

def main():
    parser = argparse.ArgumentParser(description='Time Series Prediction')
    parser.add_argument('--dataset', choices=['stock', 'climate', 'sales'], 
                       default='stock', help='Dataset to use for prediction')
    parser.add_argument('--model', choices=['lstm', 'prophet', 'both'], 
                       default='both', help='Model to train')
    parser.add_argument('--tune', action='store_true', 
                       help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger()
    config = Config(args.dataset)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    logger.info(f"Starting {args.dataset} prediction with {args.model} model(s)")
    
    # Load and prepare data
    data_loader = DataLoader(config)
    raw_data = data_loader.load_data()
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config)
    processed_data = feature_engineer.create_features(raw_data)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    results = {}
    
    # Train models
    if args.model in ['lstm', 'both']:
        if LSTM_AVAILABLE:
            logger.info("Training LSTM model...")
            lstm_model = LSTMPredictor(config)
            
            if args.tune:
                lstm_model.tune_hyperparameters(processed_data)
            
            lstm_predictions = lstm_model.train_and_predict(processed_data)
            lstm_metrics = evaluator.evaluate(lstm_predictions, processed_data['y_test'])
            results['LSTM'] = {'predictions': lstm_predictions, 'metrics': lstm_metrics}
        else:
            logger.warning("LSTM model requested but TensorFlow not available")
        
    if args.model in ['prophet', 'both']:
        if PROPHET_AVAILABLE:
            logger.info("Training Prophet model...")
            prophet_model = ProphetPredictor(config)
            
            if args.tune:
                prophet_model.tune_hyperparameters(processed_data)
                
            prophet_predictions = prophet_model.train_and_predict(processed_data)
            prophet_metrics = evaluator.evaluate(prophet_predictions, processed_data['y_test'])
            results['Prophet'] = {'predictions': prophet_predictions, 'metrics': prophet_metrics}
        else:
            logger.warning("Prophet model requested but Prophet not available")
    
    # Generate reports and visualizations
    if results:
        evaluator.generate_report(results, args.dataset)
    else:
        logger.warning("No models were successfully trained. Install TensorFlow for LSTM or Prophet for time series forecasting.")
        print("⚠️  No models available. Try installing:")
        print("   pip install tensorflow  # For LSTM models")
        print("   pip install prophet     # For Prophet models")
        print("   Or run: python3 run_lightweight.py  # For scikit-learn only")
    
    logger.info("Prediction complete! Check results/ directory for outputs.")

if __name__ == "__main__":
    main()