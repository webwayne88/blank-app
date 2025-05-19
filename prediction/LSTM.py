import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import os
import joblib

# 1. Загрузка данных
ticker = 'MSFT'
df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data = df[['Close']].copy()

look_back = 40
forward_days = 10
num_periods = 20

# 2. Масштабирование
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Сохран скейлера
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# 3. Создание обучающих последовательностей
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data) - 7):  # 7 — шаг для прогноза на неделю
        X.append(data[i - window_size:i, 0])
        y.append(data[i:i+7, 0])  # 7-дневный прогноз
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Построение модели
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(7)  # 7-дневный прогноз
])

model.compile(optimizer='adam', loss=MeanSquaredError())

# 5. Обучение
model.fit(X, y, epochs=20, batch_size=32)

# 6. Сохранение модели
model.save("models/lstm_stock_forecast.h5")

# 7. Прогноз на следующую неделю
last_sequence = scaled_data[-window_size:]
last_sequence = last_sequence.reshape((1, window_size, 1))
prediction_scaled = model.predict(last_sequence)
prediction = scaler.inverse_transform(prediction_scaled)

# 8. Визуализация прогноза
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame(prediction.flatten(), index=future_dates, columns=['Forecast'])

plt.figure(figsize=(10,5))
plt.plot(data[-100:], label='История')
plt.plot(forecast_df, label='Прогноз', marker='o')
plt.title(f'Прогноз стоимости акции {ticker} на 7 дней')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.grid()
plt.show()

