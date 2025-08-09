"""
ingest.py

This script fetches historical OHLCV (Open, High, Low, Close, Volume) data for Indian stocks
from Yahoo Finance using yfinance, computes daily returns and rolling volatility, and caches
the results as parquet files for efficient reuse. It supports both NSE and BSE tickers.
"""

import yfinance as yf
from nsepython import nsefetch
from pathlib import Path
import pandas as pd

NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"
CACHE = Path("data/raw")  # Directory to cache downloaded data

def _yf_ticker(symbol: str, exch="NSE"):
    """Return Yahoo Finance ticker symbol for NSE or BSE."""
    if exch == "BSE":
        return f"{symbol}{BSE_SUFFIX}"
    return f"{symbol}{NSE_SUFFIX}"

def history(symbol: str,
            exch: str = "NSE",
            period: str = "2y",
            interval: str = "1d",
            force: bool = False) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given symbol, compute returns and volatility,
    and cache the result as a parquet file.
    """
    CACHE.mkdir(parents=True, exist_ok=True)  # Ensure cache directory exists
    cache_f = CACHE / f"{symbol}_{period}_{interval}.parquet"
    if cache_f.exists() and not force:
        # Return cached data if available and not forcing refresh
        return pd.read_parquet(cache_f)

    # Download data from Yahoo Finance
    df = yf.Ticker(_yf_ticker(symbol, exch)).history(
        period=period, interval=interval, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data for {symbol}")

    # Compute daily returns
    df["Returns"] = df["Close"].pct_change()
    # Compute rolling volatility (standard deviation of returns over 20 days)
    df["Volatility"] = df["Returns"].rolling(20).std()
    # Drop rows with NaN values in Returns or Volatility
    df = df.dropna(subset=["Returns", "Volatility"])
    # Cache the processed data
    df.to_parquet(cache_f)
    return df


if __name__ == "__main__":
    # Example usage: fetch 1 month of data for TCS and print the head
    df = history("TCS", period="1mo")  # short period for fast test
    print(df.tail())

