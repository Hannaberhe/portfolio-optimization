import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import warnings
warnings.filterwarnings('ignore')

# Load data
tsla = pd.read_csv('data/tsla.csv')
bnd = pd.read_csv('data/bnd.csv')
spy = pd.read_csv('data/spy.csv')

# Get closing prices
prices = pd.DataFrame({
    'TSLA': pd.to_numeric(tsla['Close'], errors='coerce'),
    'BND': pd.to_numeric(bnd['Close'], errors='coerce'),
    'SPY': pd.to_numeric(spy['Close'], errors='coerce')
}).dropna()

print(f"Data: {len(prices)} days")

# Expected returns and covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

print("\nExpected Annual Returns:")
print(mu)

# Covariance heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(S, cmap='coolwarm')
ax.set_xticks(range(3))
ax.set_xticklabels(['TSLA', 'BND', 'SPY'])
ax.set_yticks(range(3))
ax.set_yticklabels(['TSLA', 'BND', 'SPY'])
plt.colorbar(im)
ax.set_title('Covariance Matrix')
plt.tight_layout()
plt.savefig('reports/covariance.png', dpi=150)
print("Covariance chart saved")

# Efficient Frontier
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate=0.02)
clean_weights = ef.clean_weights()
print("\nMax Sharpe Portfolio:")
for k, v in clean_weights.items():
    print(f"  {k}: {v:.2%}")
perf = ef.portfolio_performance(verbose=True)

# Plot Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 6))
ef2 = EfficientFrontier(mu, S)
ef2.max_sharpe(risk_free_rate=0.02)
ret_sharpe, vol_sharpe, _ = ef2.portfolio_performance()

ef3 = EfficientFrontier(mu, S)
ef3.min_volatility()
ret_min, vol_min, _ = ef3.portfolio_performance()

# Generate frontier points
frontier_returns = []
frontier_vols = []
for ret in np.linspace(ret_min, ret_sharpe + 0.1, 50):
    ef_temp = EfficientFrontier(mu, S)
    try:
        ef_temp.efficient_return(ret)
        r, v, _ = ef_temp.portfolio_performance()
        frontier_returns.append(r)
        frontier_vols.append(v)
    except:
        pass

ax.plot(frontier_vols, frontier_returns, 'b-', label='Efficient Frontier')
ax.scatter(vol_sharpe, ret_sharpe, marker='*', color='red', s=200, label='Max Sharpe')
ax.scatter(vol_min, ret_min, marker='*', color='green', s=200, label='Min Volatility')
ax.set_xlabel('Volatility (Risk)')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier')
ax.legend()
plt.tight_layout()
plt.savefig('reports/efficient_frontier.png', dpi=150)
print("Efficient Frontier chart saved")
print("\nDone")
