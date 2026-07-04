import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

print("Downloading TSLA, BND, SPY...")
tsla = yf.download('TSLA', start='2015-01-01', end='2026-06-30')
bnd = yf.download('BND', start='2015-01-01', end='2026-06-30')
spy = yf.download('SPY', start='2015-01-01', end='2026-06-30')

tsla_close = tsla['Close']
print(f"TSLA rows: {len(tsla)}")
print(f"TSLA Close - min: {float(tsla_close.min()):.2f}, max: {float(tsla_close.max()):.2f}, mean: {float(tsla_close.mean()):.2f}")

tsla_returns = tsla_close.pct_change().dropna()
print(f"TSLA Returns - mean: {float(tsla_returns.mean()):.6f}, std: {float(tsla_returns.std()):.4f}")

result = adfuller(tsla_close.dropna())
print(f"ADF p-value (close): {result[1]:.4f}")

result2 = adfuller(tsla_returns)
print(f"ADF p-value (returns): {result2[1]:.4f}")

rf_rate = 0.02
sharpe = (float(tsla_returns.mean()) * 252 - rf_rate) / (float(tsla_returns.std()) * np.sqrt(252))
print(f"Sharpe Ratio: {sharpe:.4f}")

var_95 = np.percentile(tsla_returns, 5)
print(f"VaR 95%: {float(var_95):.4f}")

# Chart 1
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsla_close.index, tsla_close, color='blue')
ax.set_title('TSLA Closing Price (2015-2026)')
plt.tight_layout()
plt.savefig('reports/tsla_price.png', dpi=150)
print("Chart 1 saved")

# Chart 2
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsla_returns.index, tsla_returns, color='green', alpha=0.7)
ax.set_title('TSLA Daily Returns')
plt.tight_layout()
plt.savefig('reports/tsla_returns.png', dpi=150)
print("Chart 2 saved")

# Chart 3
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsla_close.index, tsla_close/tsla_close.iloc[0], label='TSLA')
ax.plot(bnd['Close'].index, bnd['Close']/bnd['Close'].iloc[0], label='BND')
ax.plot(spy['Close'].index, spy['Close']/spy['Close'].iloc[0], label='SPY')
ax.set_title('Normalized Price Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('reports/all_assets.png', dpi=150)
print("Chart 3 saved")

tsla.to_csv('data/tsla.csv')
bnd.to_csv('data/bnd.csv')
spy.to_csv('data/spy.csv')
print("Done")

# Rolling mean and volatility
import matplotlib.pyplot as plt
import numpy as np

tsla_close = tsla['Close']
tsla_returns = tsla_close.pct_change().dropna()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsla_returns.index, tsla_returns, alpha=0.3, color='gray', label='Daily Return')
ax.plot(tsla_returns.index, tsla_returns.rolling(20).mean(), color='blue', label='20-day Rolling Mean')
ax.plot(tsla_returns.index, tsla_returns.rolling(20).std(), color='red', label='20-day Rolling Std')
ax.set_title('Rolling Mean and Volatility (20-day)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/rolling_stats.png', dpi=150)
print("Rolling stats chart saved")

# Outlier detection
z_scores = (tsla_returns - tsla_returns.mean()) / tsla_returns.std()
outliers = tsla_returns[abs(z_scores) > 3]
print(f"Outliers detected: {len(outliers)} days with z-score > 3")
print("Top 5 outlier dates:")
print(outliers.sort_values().head())

# Rolling mean and volatility
import matplotlib.pyplot as plt
import numpy as np

tsla_close = tsla['Close']
tsla_returns = tsla_close.pct_change().dropna()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsla_returns.index, tsla_returns, alpha=0.3, color='gray', label='Daily Return')
ax.plot(tsla_returns.index, tsla_returns.rolling(20).mean(), color='blue', label='20-day Rolling Mean')
ax.plot(tsla_returns.index, tsla_returns.rolling(20).std(), color='red', label='20-day Rolling Std')
ax.set_title('Rolling Mean and Volatility (20-day)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/rolling_stats.png', dpi=150)
print("Rolling stats chart saved")

# Outlier detection

z_scores = (tsla_returns - tsla_returns.mean()) / tsla_returns.std()
outliers = tsla_returns[abs(z_scores) > 3]
print(f"Outliers detected: {len(outliers)} days with z-score > 3")
print("Top 5 outlier dates:")
print(outliers.sort_values().head())
