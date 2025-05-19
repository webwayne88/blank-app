import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime


st.logo("materials/images/logo.png", size='large')
st.set_page_config(page_title="LSTM Прогноз Акции", layout="centered")

st.title("📈 Прогноз стоимости акции с помощью LSTM")
st.write("Эта модель использует LSTM для прогнозирования цены закрытия на следующую неделю.")

with st.sidebar:
    st.header("Параметры")
    ticker = st.text_input("Тикер акции (например AAPL):", "AAPL").upper()
    n_days = st.slider("Дней для прогноза:", 7, 30, 7)
    button = st.button("🔍 Сделать прогноз")

# Параметры
window_size = 60

# Загрузка модели и скейлера
@st.cache_resource
def load_model_and_scaler():
    model = load_model("models/models/lstm_stock_forecast.h5", compile=False)
    scaler = joblib.load("models/models/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Загрузка данных
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end=datetime.datetime.today())
    return df[['Close']]


if button:
    with st.spinner("Загружаем данные и делаем прогноз..."):
        try:
            df = load_data(ticker)
            data = df[['Close']]
            scaled_data = scaler.transform(data)

            # Подготовка последней последовательности
            last_sequence = scaled_data[-window_size:]
            last_sequence = last_sequence.reshape((1, window_size, 1))

            # Прогноз
            prediction_scaled = model.predict(last_sequence)
            prediction = scaler.inverse_transform(prediction_scaled).flatten()

            # Даты прогноза
            forecast_dates = pd.date_range(start=df.index[-7] + pd.Timedelta(days=1), periods=n_days)
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': prediction})
            forecast_df.set_index('Date', inplace=True)

            # График
            st.subheader("📊 График цены и прогноза")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data[-30:], label='Исторические данные')
            ax.plot(forecast_df, label='Прогноз', linestyle='--')
            ax.set_title(f'Прогноз стоимости акции {ticker} на {n_days} дней')
            ax.set_xlabel("Дата")
            ax.set_ylabel("Цена")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # Таблица
            st.subheader("🧾 Прогнозируемые значения")
            st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))

            # Анализ тренда
            change = prediction[-1] - prediction[0]
            percent_change = (change / prediction[0]) * 100
            trend = "📈 Восходящий" if change > 0 else "📉 Нисходящий" if change < 0 else "➡️ Стабильный"

            st.subheader("📌 Анализ тренда")
            st.markdown(f"""
            - Тренд: **{trend}**
            - Начальная цена прогноза: **${prediction[0]:.2f}**
            - Конечная цена прогноза: **${prediction[-1]:.2f}**
            - Изменение: **${change:.2f} ({percent_change:.2f}%)**
            """)
        except Exception as e:
            st.error(f"Ошибка при прогнозе: {e}")
