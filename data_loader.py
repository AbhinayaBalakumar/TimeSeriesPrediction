"""Data loading utilities for different time series datasets."""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        """Load data based on dataset type."""
        if self.config.dataset_type == 'stock':
            return self._load_stock_data()
        elif self.config.dataset_type == 'climate':
            return self._load_climate_data()
        elif self.config.dataset_type == 'sales':
            return self._load_sales_data()
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")
    
    def _load_stock_data(self):
        """Load stock price data using yfinance."""
        logger.info(f"Loading stock data for {self.config.symbols}")
        
        # Download data for the first symbol (can be extended for multiple symbols)
        symbol = self.config.symbols[0]
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=self.config.period)
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        data.rename(columns={'Date': self.config.date_column}, inplace=True)
        
        logger.info(f"Loaded {len(data)} records for {symbol}")
        return data
    
    def _load_climate_data(self):
        """Generate synthetic climate data for demonstration."""
        logger.info("Generating synthetic climate data")
        
        # Generate 2 years of daily data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Generate synthetic temperature data with seasonal patterns
        day_of_year = dates.dayofyear
        temperature = (
            20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25) +  # Seasonal pattern
            np.random.normal(0, 3, n_samples)  # Random noise
        )
        
        # Generate correlated features
        humidity = 60 + 20 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi) + np.random.normal(0, 5, n_samples)
        pressure = 1013 + 10 * np.cos(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 2, n_samples)
        wind_speed = 10 + 5 * np.random.random(n_samples)
        
        data = pd.DataFrame({
            self.config.date_column: dates,
            self.config.target_column: temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed
        })
        
        logger.info(f"Generated {len(data)} climate records")
        return data
    
    def _load_sales_data(self):
        """Generate synthetic sales data for demonstration."""
        logger.info("Generating synthetic sales data")
        
        # Generate 3 years of monthly data
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='MS')
        n_samples = len(dates)
        
        # Generate synthetic sales data with trend and seasonality
        trend = np.linspace(1000, 2000, n_samples)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
        marketing_spend = 100 + 50 * np.random.random(n_samples)
        promotions = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        sales = (
            trend + 
            seasonality + 
            2 * marketing_spend + 
            100 * promotions + 
            np.random.normal(0, 50, n_samples)
        )
        
        data = pd.DataFrame({
            self.config.date_column: dates,
            self.config.target_column: sales,
            'marketing_spend': marketing_spend,
            'seasonality': seasonality,
            'promotions': promotions
        })
        
        logger.info(f"Generated {len(data)} sales records")
        return data