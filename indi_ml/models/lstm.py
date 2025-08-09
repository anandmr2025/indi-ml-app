"""
LSTM (Long Short-Term Memory) Model for Stock Price Prediction

This module implements a deep learning approach using LSTM neural networks
to predict stock prices based on historical closing price data.

Key Features:
- Preprocesses time series data using MinMaxScaler for normalization
- Creates sequences for time series forecasting with configurable lookback period
- Implements a 2-layer LSTM architecture with dropout for regularization
- Provides training, prediction, and evaluation capabilities
- Includes visualization utilities for model performance analysis

Usage:
    from indi_ml.models.lstm import train_lstm
    model, rmse, predictions, actual, scaler = train_lstm(close_prices, epochs=30, lookback=60)

Author: ML Team
Date: 2024
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging to reduce verbosity
tf.get_logger().setLevel('ERROR')


def _seq(x, lookback=60):
    """
    Create sequences for time series forecasting.
    
    Args:
        x (array): Input time series data
        lookback (int): Number of previous time steps to use for prediction
        
    Returns:
        tuple: (X, y) where X contains sequences and y contains target values
    """
    X, y = [], []
    for i in range(lookback, len(x)):
        X.append(x[i-lookback:i])
        y.append(x[i])
    return np.stack(X), np.stack(y)


def train_lstm(close, epochs=30, lookback=60):
    """
    Train an LSTM model for stock price prediction.
    
    Args:
        close (pd.Series): Historical closing prices
        epochs (int): Number of training epochs
        lookback (int): Number of previous time steps to use for prediction
        
    Returns:
        tuple: (model, rmse, predictions, actual_values, scaler)
            - model: Trained LSTM model
            - rmse: Root Mean Square Error
            - predictions: Model predictions on test set
            - actual_values: Actual values from test set
            - scaler: Fitted scaler for inverse transformation
    """
    # Normalize the data to range [0, 1] for better LSTM performance
    scaler = MinMaxScaler((0, 1))
    data = scaler.fit_transform(close.values.reshape(-1, 1))
    
    # Create sequences for time series forecasting
    X, y = _seq(data, lookback)
    
    # Split data into training (80%) and testing (20%) sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model architecture
    model = tf.keras.Sequential([
        # Input layer with proper shape
        tf.keras.layers.Input(shape=X.shape[1:]),
        # First LSTM layer with 64 units and return sequences for stacking
        tf.keras.layers.LSTM(64, return_sequences=True),
        # Dropout for regularization
        tf.keras.layers.Dropout(0.2),
        # Second LSTM layer with 32 units (no return sequences for final layer)
        tf.keras.layers.LSTM(32),
        # Dropout for regularization
        tf.keras.layers.Dropout(0.2),
        # Dense layer for final prediction (1 unit for single value output)
        tf.keras.layers.Dense(1)
    ])
    
    # Compile model with Adam optimizer and Mean Squared Error loss
    model.compile("adam", "mse")
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    # Make predictions and inverse transform to original scale
    preds = scaler.inverse_transform(model.predict(X_test))
    y_test = scaler.inverse_transform(y_test)
    
    # Calculate Root Mean Square Error
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return model, rmse, preds.flatten(), y_test.flatten(), scaler


def train_lstm_for_pipeline(df_feat, epochs=20, lookback=30):
    """
    Train LSTM model specifically for the pipeline integration.
    
    Args:
        df_feat (pd.DataFrame): DataFrame with 'Close' column and features
        epochs (int): Number of training epochs (reduced for pipeline speed)
        lookback (int): Number of previous time steps to use for prediction
        
    Returns:
        tuple: (model, (rmse, mae, r2), predictions, actuals)
    """
    if 'Close' not in df_feat.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    close_series = df_feat['Close'].dropna()
    
    # Check if we have enough data
    if len(close_series) < lookback + 20:
        print(f"LSTM training failed: Insufficient data. Need at least {lookback + 20} points, got {len(close_series)}")
        return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])
    
    try:
        # Train LSTM model
        model, rmse, predictions, actuals, scaler = train_lstm(
            close_series, 
            epochs=epochs, 
            lookback=lookback
        )
        
        # Calculate additional metrics
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return model, (rmse, mae, r2), predictions, actuals
        
    except Exception as e:
        print(f"LSTM training failed: {e}")
        # Return dummy values
        return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])


if __name__ == "__main__":
    # Example usage and testing
    from indi_ml.ingest import history
    import matplotlib.pyplot as plt
    import numpy as np

    # Load historical data for TCS stock
    df = history("TCS", period="6mo")
    close = df["Close"].dropna()
    
    # Train LSTM model with reduced epochs for quick testing
    model, rmse, preds, actual, scaler = train_lstm(close, epochs=3, lookback=20)
    print(f"LSTM RMSE: {rmse:.4f}")

    # Visualize predictions vs actual values
    minlen = min(len(preds), len(actual))
    plt.plot(np.arange(minlen), actual[:minlen], label="Actual")
    plt.plot(np.arange(minlen), preds[:minlen], label="Forecast")
    plt.legend()
    plt.title("LSTM Forecast vs Actual")
    plt.show()


