"""Data loading utilities with error handling."""
import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start='2015-01-01', end='2026-06-30'):
    """Download stock data from Yahoo Finance with error handling."""
    try:
        print(f"Downloading {ticker}...")
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
        print(f"  {ticker}: {len(data)} rows")
        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def save_to_csv(data, filepath):
    """Save DataFrame to CSV with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath)
        print(f"Saved to {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")

def load_from_csv(filepath):
    """Load CSV with error handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading {filepath}: {e}")
