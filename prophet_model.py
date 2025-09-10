"""Prophet model implementation for time series prediction."""

import pandas as pd
import numpy as np
from prophet import Prophet
import optuna
import logging

logger = logging.getLogger(__name__)

class ProphetPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.best_params = None
        
    def prepare_prophet_data(self, data):
        """Prepare data in Prophet format."""
        # Get the original dataframe with dates
        dates_train = data['dates_test'].iloc[:-len(data['y_test'])] if len(data['dates_test']) > len(data['y_test']) else data['dates_test']
        dates_test = data['dates_test']
        
        # Create training dataframe
        train_df = pd.DataFrame({
            'ds': dates_train[-len(data['y_train']):],
            'y': data['y_train']
        })
        
        # Create test dataframe for prediction
        test_df = pd.DataFrame({
            'ds': dates_test
        })
        
        return train_df, test_df
    
    def build_model(self, params=None):
        """Build Prophet model with given parameters."""
        if params is None:
            params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            }
        
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        return model
    
    def train_and_predict(self, data):
        """Train the model and make predictions."""
        try:
            # Prepare data
            train_df, test_df = self.prepare_prophet_data(data)
            
            # Use best parameters if available
            params = self.best_params or {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            }
            
            # Build and train model
            self.model = self.build_model(params)
            
            logger.info("Training Prophet model...")
            self.model.fit(train_df)
            
            # Make predictions
            forecast = self.model.predict(test_df)
            predictions = forecast['yhat'].values
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in Prophet prediction: {str(e)}")
            return np.array([])
    
    def tune_hyperparameters(self, data):
        """Tune hyperparameters using Optuna."""
        logger.info("Starting hyperparameter tuning for Prophet...")
        
        def objective(trial):
            # Sample hyperparameters
            params = {
                'changepoint_prior_scale': trial.suggest_categorical(
                    'changepoint_prior_scale', 
                    self.config.prophet_params['changepoint_prior_scale']
                ),
                'seasonality_prior_scale': trial.suggest_categorical(
                    'seasonality_prior_scale',
                    self.config.prophet_params['seasonality_prior_scale']
                ),
                'holidays_prior_scale': trial.suggest_categorical(
                    'holidays_prior_scale',
                    self.config.prophet_params['holidays_prior_scale']
                ),
                'seasonality_mode': trial.suggest_categorical(
                    'seasonality_mode',
                    self.config.prophet_params['seasonality_mode']
                )
            }
            
            try:
                # Prepare data for cross-validation
                train_df, _ = self.prepare_prophet_data(data)
                
                # Use only a subset for faster tuning
                if len(train_df) > 200:
                    train_df = train_df.iloc[-200:]
                
                # Split for validation
                val_size = int(len(train_df) * 0.2)
                train_fold = train_df.iloc[:-val_size]
                val_fold = train_df.iloc[-val_size:]
                
                # Build and train model
                model = self.build_model(params)
                model.fit(train_fold)
                
                # Predict on validation set
                val_pred_df = pd.DataFrame({'ds': val_fold['ds']})
                forecast = model.predict(val_pred_df)
                
                # Calculate MSE
                mse = np.mean((forecast['yhat'].values - val_fold['y'].values) ** 2)
                return mse
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)
        
        self.best_params = study.best_params
        logger.info(f"Best Prophet parameters: {self.best_params}")
        
        return self.best_params