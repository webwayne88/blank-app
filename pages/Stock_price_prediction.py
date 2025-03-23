import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

st.set_page_config(page_title = "Stock Price Analysis", layout="wide") 
st.logo(image="materials/images/logo.png", size='large', icon_image="materials/images/icon.png")

ticker = 'AAPL'  
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
data = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(SimpleRNN(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=64)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

results_df = pd.DataFrame({
    'Time': range(len(y_test)),  # Временная ось
    'Real Stock Price': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
    'Predicted Stock Price': predictions.flatten()
})

# Создаем график с помощью plotly.express
fig = px.line(results_df, x='Time', y=['Real Stock Price', 'Predicted Stock Price'],
              title=f'{ticker} Stock Price Prediction',
              labels={'value': 'Stock Price', 'variable': 'Legend'},
              template='plotly_white')

# Настраиваем отображение легенды и осей
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Stock Price',
    legend_title='Legend'
)

# Отображаем график в Streamlit
st.plotly_chart(fig)