import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

tsla = pd.read_csv('data/tsla.csv')
bnd = pd.read_csv('data/bnd.csv')
spy = pd.read_csv('data/spy.csv')

prices = pd.DataFrame({
    'TSLA': pd.to_numeric(tsla['Close'], errors='coerce'),
    'BND': pd.to_numeric(bnd['Close'], errors='coerce'),
    'SPY': pd.to_numeric(spy['Close'], errors='coerce')
}).dropna()

# Backtest period: last year
backtest = prices.iloc[-252:]
returns = backtest.pct_change().dropna()
print(f"Backtest: {len(backtest)} days")

# Strategy weights (from Task 4)
strategy_weights = np.array([0.3, 0.2, 0.5])
benchmark_weights = np.array([0.0, 0.4, 0.6])

# Cumulative returns
strategy_cum = (1 + (returns * strategy_weights).sum(axis=1)).cumprod()
benchmark_cum = (1 + (returns * benchmark_weights).sum(axis=1)).cumprod()

# Metrics
def calc_metrics(cum_returns):
    total_ret = cum_returns.iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(cum_returns)) - 1
    daily_rr = cum_returns.pct_change().dropna()
    sharpe = (daily_rr.mean() / daily_rr.std()) * np.sqrt(252) if daily_rr.std() > 0 else 0
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    return total_ret, annual_ret, sharpe, max_dd

print("\n=== PERFORMANCE ===")
print(f"{'Metric':<20} {'Strategy':<15} {'Benchmark':<15}")
s_total, s_ann, s_sharpe, s_dd = calc_metrics(strategy_cum)
b_total, b_ann, b_sharpe, b_dd = calc_metrics(benchmark_cum)
print(f"{'Total Return':<20} {s_total:<15.2%} {b_total:<15.2%}")
print(f"{'Annual Return':<20} {s_ann:<15.2%} {b_ann:<15.2%}")
print(f"{'Sharpe Ratio':<20} {s_sharpe:<15.2f} {b_sharpe:<15.2f}")
print(f"{'Max Drawdown':<20} {s_dd:<15.2%} {b_dd:<15.2%}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(strategy_cum.index, strategy_cum, label='Strategy', color='blue')
ax.plot(benchmark_cum.index, benchmark_cum, label='Benchmark (60/40)', color='gray', linestyle='--')
ax.set_title('Cumulative Returns: Strategy vs Benchmark')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend()
plt.tight_layout()
plt.savefig('reports/backtest.png', dpi=150)
print("Chart saved")
print("Done")
