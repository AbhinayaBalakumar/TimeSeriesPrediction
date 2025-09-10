#!/usr/bin/env python3
"""
Lightweight version of the time series prediction project without heavy dependencies.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from utils.config import Config

def lightweight_lstm_alternative(X_train, y_train, X_test, sequence_length=30):
    """
    Lightweight alternative to LSTM using sliding window approach with Random Forest.
    """
    print("   Using Random Forest with sliding window (LSTM alternative)")
    
    # Create sequences manually
    def create_sequences(X, y, seq_len):
        if len(X) < seq_len:
            return np.array([]), np.array([])
        
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            # Flatten the sequence window
            seq_features = X[i-seq_len:i].flatten()
            X_seq.append(seq_features)
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    if len(X_train_seq) == 0:
        print("   ⚠️  Not enough data for sequence modeling")
        return np.array([]), np.array([])
    
    # Train Random Forest on sequences
    rf_seq = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_seq.fit(X_train_seq, y_train_seq)
    
    if len(X_test_seq) > 0:
        predictions = rf_seq.predict(X_test_seq)
        return predictions, y_test_seq
    else:
        return np.array([]), np.array([])

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

def run_lightweight_demo(dataset_type='stock'):
    """Run lightweight demonstration."""
    print(f"🚀 Lightweight Time Series Prediction Demo - {dataset_type.title()} Data")
    print("=" * 70)
    
    # Setup
    config = Config(dataset_type)
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    
    # Load data
    print(f"📊 Loading {dataset_type} data...")
    try:
        raw_data = data_loader.load_data()
        print(f"✅ Loaded {len(raw_data)} data points")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Feature engineering
    print("🔧 Performing feature engineering...")
    try:
        processed_data = feature_engineer.create_features(raw_data)
        print(f"✅ Created {len(processed_data['feature_names'])} features")
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return
    
    # Models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Tuned Random Forest': None  # Will be set after tuning
    }
    
    results = {}
    
    # Prepare data
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    print(f"📈 Training set: {len(X_train)} samples")
    print(f"📈 Test set: {len(X_test)} samples")
    
    # Train and evaluate models
    for model_name, model in models.items():
        if model is None:
            continue
            
        print(f"\n🔹 Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = evaluate_model(y_test, predictions)
            results[model_name] = {'predictions': predictions, 'metrics': metrics}
            
            print(f"   MAE: {metrics['MAE']:.4f}")
            print(f"   RMSE: {metrics['RMSE']:.4f}")
            print(f"   R²: {metrics['R2']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
    
    # Hyperparameter tuning for Random Forest
    print(f"\n🔧 Hyperparameter tuning Random Forest...")
    try:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        tuned_model = grid_search.best_estimator_
        tuned_predictions = tuned_model.predict(X_test)
        tuned_metrics = evaluate_model(y_test, tuned_predictions)
        results['Tuned Random Forest'] = {'predictions': tuned_predictions, 'metrics': tuned_metrics}
        
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   MAE: {tuned_metrics['MAE']:.4f}")
        print(f"   RMSE: {tuned_metrics['RMSE']:.4f}")
        print(f"   R²: {tuned_metrics['R2']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error in hyperparameter tuning: {e}")
    
    # LSTM Alternative (Sequence-based Random Forest)
    print(f"\n🧠 LSTM Alternative (Sequence-based Random Forest)...")
    try:
        lstm_alt_pred, lstm_alt_actual = lightweight_lstm_alternative(
            X_train, y_train, X_test, sequence_length=30
        )
        
        if len(lstm_alt_pred) > 0:
            lstm_alt_metrics = evaluate_model(lstm_alt_actual, lstm_alt_pred)
            results['LSTM Alternative'] = {'predictions': lstm_alt_pred, 'metrics': lstm_alt_metrics}
            
            print(f"   MAE: {lstm_alt_metrics['MAE']:.4f}")
            print(f"   RMSE: {lstm_alt_metrics['RMSE']:.4f}")
            print(f"   R²: {lstm_alt_metrics['R2']:.4f}")
        else:
            print("   ⚠️  Insufficient data for sequence modeling")
            
    except Exception as e:
        print(f"   ❌ Error in LSTM alternative: {e}")
    
    # Results summary
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    if results:
        best_model = min(results.keys(), key=lambda k: results[k]['metrics']['MAE'])
        best_mae = results[best_model]['metrics']['MAE']
        
        print(f"🏆 Best Model: {best_model}")
        print(f"🎯 Best MAE: {best_mae:.4f}")
        
        print(f"\nAll Results:")
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"   {model_name}:")
            print(f"     MAE: {metrics['MAE']:.4f}")
            print(f"     RMSE: {metrics['RMSE']:.4f}")
            print(f"     R²: {metrics['R2']:.4f}")
    
    # Create visualization
    create_lightweight_visualization(results, dataset_type)
    
    print(f"\n📈 Visualization saved to: results/lightweight_{dataset_type}_results.png")
    print("✅ Lightweight demo completed successfully!")

def create_lightweight_visualization(results, dataset_type):
    """Create visualization for lightweight demo."""
    if not results:
        return
    
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model comparison
    ax1 = axes[0, 0]
    models = list(results.keys())
    maes = [results[model]['metrics']['MAE'] for model in models]
    
    bars = ax1.bar(models, maes, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'][:len(models)])
    ax1.set_title('Model Comparison - MAE', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=9)
    
    # R² comparison
    ax2 = axes[0, 1]
    r2_scores = [results[model]['metrics']['R2'] for model in models]
    
    bars = ax2.bar(models, r2_scores, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'][:len(models)])
    ax2.set_title('Model Comparison - R²', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Predictions plot (best model)
    ax3 = axes[1, 0]
    if results:
        best_model = min(results.keys(), key=lambda k: results[k]['metrics']['MAE'])
        predictions = results[best_model]['predictions']
        
        x = range(len(predictions))
        ax3.plot(x, predictions, label=f'{best_model} Predictions', color='blue', alpha=0.8)
        ax3.set_title(f'Best Model Predictions: {best_model}', fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Predicted Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Metrics summary
    ax4 = axes[1, 1]
    if results:
        metrics_data = []
        for model in models:
            metrics_data.append([
                results[model]['metrics']['MAE'],
                results[model]['metrics']['RMSE'],
                results[model]['metrics']['R2']
            ])
        
        metrics_df = pd.DataFrame(metrics_data, 
                                 columns=['MAE', 'RMSE', 'R²'],
                                 index=models)
        
        # Create a simple text summary
        ax4.axis('off')
        summary_text = f"Dataset: {dataset_type.title()}\n\n"
        summary_text += "Model Performance Summary:\n"
        summary_text += "-" * 25 + "\n"
        
        for model in models:
            metrics = results[model]['metrics']
            summary_text += f"{model}:\n"
            summary_text += f"  MAE: {metrics['MAE']:.4f}\n"
            summary_text += f"  R²:  {metrics['R2']:.4f}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Lightweight Time Series Prediction - {dataset_type.title()} Dataset', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/lightweight_{dataset_type}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all lightweight demos."""
    datasets = ['stock', 'climate', 'sales']
    
    print("🎯 LIGHTWEIGHT TIME SERIES PREDICTION PROJECT")
    print("=" * 60)
    print("Running without TensorFlow or Prophet dependencies")
    print("Using scikit-learn models with advanced feature engineering")
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        try:
            run_lightweight_demo(dataset)
        except Exception as e:
            print(f"❌ Error running {dataset} demo: {e}")
    
    print(f"\n🎉 All lightweight demos completed!")
    print("Check the results/ directory for visualizations")

if __name__ == "__main__":
    main()