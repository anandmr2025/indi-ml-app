"""
arima.py

This script trains an ARIMA model on the closing prices of a stock, forecasts the next 30 days,
and evaluates the forecast using RMSE. It also provides an optional plot comparing forecasted and actual values.

Dependencies:
- statsmodels
- scikit-learn
- numpy
- matplotlib
"""

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(series, significance_level=0.05):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.
    
    Args:
        series (pd.Series): Time series data
        significance_level (float): Significance level for the test
        
    Returns:
        bool: True if stationary, False otherwise
    """
    result = adfuller(series.dropna())
    return result[1] <= significance_level

def find_best_arima_order(series, max_p=3, max_d=2, max_q=3, seasonal=False):
    """
    Find a good ARIMA order using a grid search approach.
    
    Args:
        series (pd.Series): Time series data
        max_p (int): Maximum p value
        max_d (int): Maximum d value  
        max_q (int): Maximum q value
        seasonal (bool): Whether to use seasonal ARIMA (not implemented)
        
    Returns:
        tuple: Best (p, d, q) order
    """
    try:
        best_aic = float('inf')
        best_order = (1, 1, 0)
        
        # Grid search for best parameters
        for p in range(0, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(0, max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    except Exception as e:
        print(f"ARIMA order selection failed, using default order (1,1,0): {e}")
        return (1, 1, 0)

def train_arima(series, order=None, test_size=30, auto_order=True):
    """
    Train an ARIMA model on the given time series and forecast the next test_size points.

    Args:
        series (pd.Series): Time series data (e.g., closing prices).
        order (tuple): ARIMA order (p,d,q). If None and auto_order=True, will be auto-selected.
        test_size (int): Number of points to use for testing.
        auto_order (bool): Whether to automatically select the best ARIMA order.

    Returns:
        tuple: (model, (rmse, mae, r2), forecast, test_values)
    """
    if len(series) <= test_size + 10:
        raise ValueError(f"Series must have more than {test_size + 10} data points for train/test split.")
    
    # Clean the series
    series = series.dropna()
    
    if len(series) <= test_size + 10:
        raise ValueError(f"After cleaning, series has insufficient data points.")
    
    # Split data
    train = series[:-test_size]
    test = series[-test_size:]
    
    # Determine order
    if order is None and auto_order:
        order = find_best_arima_order(train)
    elif order is None:
        order = (1, 1, 0)  # Default order
    
    try:
        # Fit ARIMA model
        model = ARIMA(train, order=order).fit()
        
        # Make forecast
        forecast = model.forecast(steps=test_size)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        r2 = r2_score(test, forecast)
        
        return model, (rmse, mae, r2), forecast, test.values
        
    except Exception as e:
        print(f"ARIMA training failed with order {order}: {e}")
        # Fallback to simple order
        try:
            model = ARIMA(train, order=(1, 1, 0)).fit()
            forecast = model.forecast(steps=test_size)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mae = mean_absolute_error(test, forecast)
            r2 = r2_score(test, forecast)
            return model, (rmse, mae, r2), forecast, test.values
        except Exception as e2:
            print(f"Fallback ARIMA also failed: {e2}")
            # Return dummy values
            return None, (float('inf'), float('inf'), 0.0), np.array([]), test.values

def train_arima_for_pipeline(df_feat):
    """
    Train ARIMA model specifically for the pipeline integration.
    
    Args:
        df_feat (pd.DataFrame): DataFrame with 'Close' column and features
        
    Returns:
        tuple: (model, (rmse, mae, r2), predictions, actuals)
    """
    if 'Close' not in df_feat.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    close_series = df_feat['Close']
    
    # Use 20% of data for testing
    test_size = max(10, int(len(close_series) * 0.2))
    
    try:
        model, (rmse, mae, r2), forecast, actuals = train_arima(
            close_series, 
            test_size=test_size,
            auto_order=True
        )
        
        return model, (rmse, mae, r2), forecast, actuals
        
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        # Return dummy values
        return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])


if __name__ == "__main__":
    # Example usage: ARIMA forecast for TCS closing prices
    from indi_ml.ingest import history
    import matplotlib.pyplot as plt

    df = history("TCS", period="6mo")
    close = df["Close"].dropna()
    if len(close) <= 40:
        raise ValueError("Not enough data for ARIMA train/test split. Please use a longer period.")
    
    model, (rmse, mae, r2), forecast, test = train_arima(close)
    print(f"ARIMA RMSE: {rmse:.4f}")
    print(f"ARIMA MAE: {mae:.4f}")
    print(f"ARIMA R²: {r2:.4f}")

    # Optional: quick plot of forecast vs actual
    if len(forecast) > 0:
        x = range(len(test))
        plt.figure(figsize=(10, 6))
        plt.plot(x, test, label="Actual", linewidth=2)
        plt.plot(x, forecast, label="Forecast", linewidth=2)
        plt.legend()
        plt.title("ARIMA Forecast vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.show()
