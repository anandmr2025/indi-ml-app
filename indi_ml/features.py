"""
features.py

This script provides functions to compute common technical indicators (SMA, EMA, RSI, Bollinger Bands)
and a utility to enrich a DataFrame with these features for financial time series analysis.
"""

import pandas as pd
import numpy as np

def sma(s, n):
    """Simple Moving Average."""
    return s.rolling(n).mean()

def ema(s, n):
    """Exponential Moving Average."""
    return s.ewm(span=n).mean()

def rsi(close, n=14):
    """Relative Strength Index."""
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def bollinger(close, n=20, k=2):
    """Bollinger Bands: returns (lower_band, middle_band, upper_band)."""
    ma = sma(close, n)
    sd = close.rolling(n).std()
    return ma - k * sd, ma, ma + k * sd

def enrich(df: pd.DataFrame, adaptive=True) -> pd.DataFrame:
    """
    Add technical indicators and lagged close prices to the DataFrame.
    Adaptively adjusts indicators based on available data size.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
        adaptive (bool): Whether to adapt indicators to data size
    """
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()
    # Ensure required columns exist
    required_cols = ["Close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df["Close"]
    data_length = len(df)
    
    if adaptive:
        # Adaptive indicator periods based on data size
        sma_short = min(20, max(5, data_length // 10))
        sma_medium = min(50, max(10, data_length // 5))
        sma_long = min(200, max(20, data_length // 2))
        ema_period = min(12, max(5, data_length // 15))
        rsi_period = min(14, max(7, data_length // 10))
        bb_period = min(20, max(10, data_length // 8))
        max_lag = min(5, max(1, data_length // 20))
    else:
        # Original fixed periods
        sma_short, sma_medium, sma_long = 20, 50, 200
        ema_period, rsi_period, bb_period = 12, 14, 20
        max_lag = 5
    
    # Add indicators with adaptive periods
    df["SMA20"] = sma(close, sma_short)
    df["SMA50"] = sma(close, sma_medium) if data_length > sma_medium else sma(close, sma_short)
    df["SMA200"] = sma(close, sma_long) if data_length > sma_long else sma(close, sma_medium)
    df["EMA12"] = ema(close, ema_period)
    df["RSI"] = rsi(close, rsi_period)
    
    lower, mid, upper = bollinger(close, bb_period)
    df["BB_low"], df["BB_mid"], df["BB_up"] = lower, mid, upper
    
    # Adaptive lagged closes
    lags = [1, 2] if data_length < 50 else [1, 2, min(max_lag, data_length // 20)]
    for lag in lags:
        if lag <= data_length // 4:  # Only add lag if we have enough data
            df[f"Close_lag{lag}"] = close.shift(lag)
    
    return df.dropna()

if __name__ == "__main__":
    # Example usage: fetch data and enrich with features
    from indi_ml.ingest import history
    df = history("TCS", period="12mo")
    fe_df = enrich(df)
    print(fe_df.head())
    print(fe_df.tail())

