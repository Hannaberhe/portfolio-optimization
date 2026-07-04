"""EDA utility functions."""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, name="Series"):
    """Run ADF test and print results."""
    try:
        result = adfuller(series.dropna())
        print(f"ADF Test - {name}:")
        print(f"  P-value: {result[1]:.4f}")
        print(f"  Stationary: {'Yes' if result[1] < 0.05 else 'No'}")
        return result[1]
    except Exception as e:
        print(f"Error in stationarity test: {e}")
        return None

def calculate_sharpe_ratio(returns, rf_rate=0.02):
    """Calculate annualized Sharpe Ratio."""
    try:
        excess = float(returns.mean()) * 252 - rf_rate
        volatility = float(returns.std()) * np.sqrt(252)
        return excess / volatility
    except Exception as e:
        print(f"Error calculating Sharpe: {e}")
        return None

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk."""
    try:
        return float(np.percentile(returns, (1 - confidence) * 100))
    except Exception as e:
        print(f"Error calculating VaR: {e}")
        return None

def plot_price_trend(data, ticker, save_path=None):
    """Plot closing price trend."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], color='blue')
        ax.set_title(f'{ticker} Closing Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
        return fig
    except Exception as e:
        print(f"Error plotting: {e}")
        return None
