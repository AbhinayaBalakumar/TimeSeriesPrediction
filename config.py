"""Configuration management for time series prediction."""

class Config:
    def __init__(self, dataset_type='stock'):
        self.dataset_type = dataset_type
        
        # Common settings
        self.test_size = 0.2
        self.random_state = 42
        self.sequence_length = 60  # For LSTM
        
        # Dataset-specific configurations
        if dataset_type == 'stock':
            self.target_column = 'Close'
            self.date_column = 'Date'
            self.symbols = ['AAPL', 'GOOGL', 'MSFT']
            self.period = '2y'
            
        elif dataset_type == 'climate':
            self.target_column = 'temperature'
            self.date_column = 'date'
            self.features = ['humidity', 'pressure', 'wind_speed']
            
        elif dataset_type == 'sales':
            self.target_column = 'sales'
            self.date_column = 'date'
            self.features = ['marketing_spend', 'seasonality', 'promotions']
        
        # Model hyperparameters
        self.lstm_params = {
            'units': [50, 100, 150],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'epochs': 100
        }
        
        self.prophet_params = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }