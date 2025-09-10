"""LSTM model implementation for time series prediction."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import logging

logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.best_params = None
        
    def build_model(self, input_shape, params=None):
        """Build LSTM model with given parameters."""
        if params is None:
            params = {
                'units': 100,
                'dropout': 0.2,
                'learning_rate': 0.001
            }
        
        model = Sequential([
            LSTM(params['units'], return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(params['dropout']),
            
            LSTM(params['units'] // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(params['dropout']),
            
            Dense(50, activation='relu'),
            Dropout(params['dropout'] / 2),
            
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_and_predict(self, data):
        """Train the model and make predictions."""
        X_train_seq = data['X_train_seq']
        y_train_seq = data['y_train_seq']
        X_test_seq = data['X_test_seq']
        
        if len(X_train_seq) == 0:
            logger.error("No training sequences available")
            return np.array([])
        
        # Use best parameters if available, otherwise use defaults
        params = self.best_params or {
            'units': 100,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # Build and train model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self.build_model(input_shape, params)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
        
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=params.get('batch_size', 32),
            epochs=self.config.lstm_params['epochs'],
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Make predictions
        if len(X_test_seq) > 0:
            predictions = self.model.predict(X_test_seq)
            return predictions.flatten()
        else:
            logger.warning("No test sequences available for prediction")
            return np.array([])
    
    def tune_hyperparameters(self, data):
        """Tune hyperparameters using Optuna."""
        logger.info("Starting hyperparameter tuning for LSTM...")
        
        def objective(trial):
            # Sample hyperparameters
            params = {
                'units': trial.suggest_categorical('units', self.config.lstm_params['units']),
                'dropout': trial.suggest_categorical('dropout', self.config.lstm_params['dropout']),
                'learning_rate': trial.suggest_categorical('learning_rate', self.config.lstm_params['learning_rate']),
                'batch_size': trial.suggest_categorical('batch_size', self.config.lstm_params['batch_size'])
            }
            
            X_train_seq = data['X_train_seq']
            y_train_seq = data['y_train_seq']
            
            if len(X_train_seq) == 0:
                return float('inf')
            
            # Split training data for validation
            val_split = 0.2
            val_size = int(len(X_train_seq) * val_split)
            
            X_train_fold = X_train_seq[:-val_size]
            y_train_fold = y_train_seq[:-val_size]
            X_val_fold = X_train_seq[-val_size:]
            y_val_fold = y_train_seq[-val_size:]
            
            # Build and train model
            input_shape = (X_train_fold.shape[1], X_train_fold.shape[2])
            model = self.build_model(input_shape, params)
            
            # Train with early stopping
            callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
            
            model.fit(
                X_train_fold, y_train_fold,
                batch_size=params['batch_size'],
                epochs=20,  # Reduced for tuning
                validation_data=(X_val_fold, y_val_fold),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            return val_loss[0]  # Return MSE loss
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        self.best_params = study.best_params
        logger.info(f"Best LSTM parameters: {self.best_params}")
        
        return self.best_params