import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

tsla = pd.read_csv('data/tsla.csv')
tsla_close = pd.to_numeric(tsla['Close'], errors='coerce').dropna().values
print(f"TSLA: {len(tsla_close)} values")

split = int(len(tsla_close) * 0.8)
train = tsla_close[:split]
test = tsla_close[split:]
print(f"Train: {len(train)}, Test: {len(test)}")

print("Fitting ARIMA...")
model = ARIMA(train, order=(5, 1, 0))
fitted = model.fit()
forecast = fitted.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(test)), test, label='Actual', color='green')
ax.plot(range(len(forecast)), forecast, label='ARIMA', color='red', linestyle='--')
ax.set_title('TSLA: Actual vs ARIMA Forecast')
ax.legend()
plt.tight_layout()
plt.savefig('reports/arima_forecast.png', dpi=150)
print("Done")
