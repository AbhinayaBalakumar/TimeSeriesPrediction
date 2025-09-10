"""Model evaluation and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        
    def evaluate(self, predictions, actual):
        """Calculate evaluation metrics."""
        if len(predictions) == 0 or len(actual) == 0:
            return {
                'MAE': float('inf'),
                'RMSE': float('inf'),
                'MAPE': float('inf'),
                'R2': 0
            }
        
        # Ensure same length
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predictions) / np.maximum(np.abs(actual), 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    def generate_report(self, results, dataset_name):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        # Create visualizations
        self._create_prediction_plots(results, dataset_name)
        self._create_metrics_comparison(results, dataset_name)
        
        # Save metrics to JSON
        metrics_summary = {}
        for model_name, result in results.items():
            metrics_summary[model_name] = result['metrics']
        
        with open(f'results/{dataset_name}_metrics.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Create text report
        self._create_text_report(results, dataset_name)
        
        logger.info(f"Report saved to results/{dataset_name}_report.txt")
    
    def _create_prediction_plots(self, results, dataset_name):
        """Create prediction vs actual plots."""
        fig = make_subplots(
            rows=len(results), cols=2,
            subplot_titles=[f'{model} - Time Series' for model in results.keys()] + 
                          [f'{model} - Scatter Plot' for model in results.keys()],
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in results]
        )
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (model_name, result) in enumerate(results.items(), 1):
            predictions = result['predictions']
            
            if len(predictions) == 0:
                continue
                
            # Time series plot
            x_vals = list(range(len(predictions)))
            
            fig.add_trace(
                go.Scatter(x=x_vals, y=predictions, name=f'{model_name} Predictions',
                          line=dict(color=colors[i-1]), mode='lines'),
                row=i, col=1
            )
            
            # Scatter plot (predictions vs actual would need actual values)
            fig.add_trace(
                go.Scatter(x=x_vals, y=predictions, name=f'{model_name} Predictions',
                          mode='markers', marker=dict(color=colors[i-1])),
                row=i, col=2
            )
        
        fig.update_layout(
            title=f'{dataset_name.title()} Prediction Results',
            height=300 * len(results),
            showlegend=True
        )
        
        fig.write_html(f'results/{dataset_name}_predictions.html')
        
        # Also create matplotlib version
        plt.figure(figsize=(15, 5 * len(results)))
        
        for i, (model_name, result) in enumerate(results.items(), 1):
            predictions = result['predictions']
            
            if len(predictions) == 0:
                continue
            
            plt.subplot(len(results), 2, 2*i-1)
            plt.plot(predictions, label=f'{model_name} Predictions', color=colors[i-1])
            plt.title(f'{model_name} - Time Series Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(len(results), 2, 2*i)
            plt.hist(predictions, bins=30, alpha=0.7, color=colors[i-1])
            plt.title(f'{model_name} - Prediction Distribution')
            plt.xlabel('Predicted Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_comparison(self, results, dataset_name):
        """Create metrics comparison visualization."""
        metrics_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in results.items()
        }).T
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        
        for i, metric in enumerate(metrics):
            if metric in metrics_df.columns:
                ax = axes[i]
                bars = ax.bar(metrics_df.index, metrics_df[metric])
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height) and not np.isinf(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics table
        metrics_df.to_csv(f'results/{dataset_name}_metrics.csv')
    
    def _create_text_report(self, results, dataset_name):
        """Create detailed text report."""
        report = f"Time Series Prediction Report - {dataset_name.title()}\n"
        report += "=" * 50 + "\n\n"
        
        for model_name, result in results.items():
            metrics = result['metrics']
            report += f"{model_name} Model Results:\n"
            report += "-" * 30 + "\n"
            
            for metric_name, value in metrics.items():
                if np.isfinite(value):
                    report += f"{metric_name}: {value:.4f}\n"
                else:
                    report += f"{metric_name}: N/A (insufficient data)\n"
            
            report += "\n"
        
        # Model comparison
        report += "Model Comparison:\n"
        report += "-" * 20 + "\n"
        
        if len(results) > 1:
            # Find best model for each metric
            metrics_df = pd.DataFrame({
                model_name: result['metrics'] 
                for model_name, result in results.items()
            }).T
            
            for metric in ['MAE', 'RMSE', 'MAPE']:
                if metric in metrics_df.columns:
                    best_model = metrics_df[metric].idxmin()
                    best_value = metrics_df.loc[best_model, metric]
                    if np.isfinite(best_value):
                        report += f"Best {metric}: {best_model} ({best_value:.4f})\n"
            
            if 'R2' in metrics_df.columns:
                best_model = metrics_df['R2'].idxmax()
                best_value = metrics_df.loc[best_model, 'R2']
                if np.isfinite(best_value):
                    report += f"Best R2: {best_model} ({best_value:.4f})\n"
        
        with open(f'results/{dataset_name}_report.txt', 'w') as f:
            f.write(report)
        
        print(report)