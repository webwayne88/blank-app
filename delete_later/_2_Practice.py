# import numpy as np
# import pandas as pd
# import plotly.express as px
# import yfinance as yf
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# import streamlit as st

# # Загрузка данных
# ticker = 'AAPL'
# data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
# data = data[['Close']]  # Используем только столбец 'Close'

# # Разделение данных на обучающую и тестовую выборки
# train_size = int(len(data) * 0.8)
# train_data, test_data = data[:train_size], data[train_size:]

# # Обучение модели ARIMA
# model = ARIMA(train_data, order=(5, 1, 0))  # Порядок (p, d, q)
# fitted_model = model.fit()

# # Прогнозирование на тестовой выборке
# predictions = fitted_model.forecast(steps=len(test_data))

# # Обратное преобразование прогноза (если данные были нормализованы)
# predictions = pd.Series(predictions, index=test_data.index)

# # Вычисление ошибки (RMSE)
# rmse = np.sqrt(mean_squared_error(test_data, predictions))
# st.write(f"RMSE: {rmse}")

# # Создание DataFrame для визуализации
# results_df = pd.DataFrame({
#     'Date': test_data.index,
#     'Real Stock Price': test_data['Close'],
#     'Predicted Stock Price': predictions
# })

# # Визуализация с использованием plotly.express
# fig = px.line(results_df, x='Date', y=['Real Stock Price', 'Predicted Stock Price'],
#               title=f'{ticker} Stock Price Prediction with ARIMA',
#               labels={'value': 'Stock Price', 'variable': 'Legend'},
#               template='plotly_white')

# # Настройка отображения графика
# fig.update_layout(
#     xaxis_title='Date',
#     yaxis_title='Stock Price',
#     legend_title='Legend'
# )

# # Отображение графика в Streamlit
# st.plotly_chart(fig)