import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

tsla = pd.read_csv('data/tsla.csv')
tsla_close = pd.to_numeric(tsla['Close'], errors='coerce').dropna().values
print(f"TSLA: {len(tsla_close)} values")

# Fit on all data
model = ARIMA(tsla_close, order=(5, 1, 0))
fitted = model.fit()

# Forecast 252 days (1 year)
forecast_steps = 252
forecast = fitted.forecast(steps=forecast_steps)
forecast_index = range(len(tsla_close), len(tsla_close) + forecast_steps)

# Confidence intervals
forecast_result = fitted.get_forecast(steps=forecast_steps)
conf_int = forecast_result.conf_int(alpha=0.05)

print(f"Forecast generated for {forecast_steps} trading days")
print(f"Final forecasted price: ${forecast[-1]:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(tsla_close[-500:])), tsla_close[-500:], label='Historical', color='blue')
ax.plot(range(len(tsla_close[-500:]), len(tsla_close[-500:]) + forecast_steps), forecast, label='Forecast', color='red')
ax.fill_between(range(len(tsla_close[-500:]), len(tsla_close[-500:]) + forecast_steps),
                conf_int[:, 0], conf_int[:, 1], alpha=0.2, color='red', label='95% CI')
ax.set_title('TSLA Price Forecast (1 Year)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/forecast_future.png', dpi=150)
print("Chart saved")
