import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import GRU, Dropout, SimpleRNN , Dense
from keras.models import Sequential
import plotly.express as px

st.set_page_config(page_title = "LSTM & GRU", layout="wide") 
st.logo(image="materilas/images/logo.png", size='large', icon_image="materilas/images/icon.png")

# Define the ticker symbol
ticker = 'IBM'

# Define the date range
start_date = '2000-01-01'
end_date = '2024-04-01'

@st.cache_data 
def load_data(ticker, start_date, end_date):
    data = pd.DataFrame()
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    return data

data = load_data(ticker, start_date, end_date)
data = data.stack(level=0).reset_index()
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

st.write(data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)
st.write(data_normalized.head())

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 15
X, y = create_sequences(data_normalized, window_size)

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


gru_model = Sequential()

# First GRU layer with dropout
gru_model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
gru_model.add(Dropout(0.2))

# Second GRU layer with dropout
gru_model.add(GRU(50, return_sequences=True))
gru_model.add(Dropout(0.2))

# Third GRU layer with dropout
gru_model.add(GRU(50, return_sequences=True))
gru_model.add(Dropout(0.2))

# Fourth GRU layer with dropout
gru_model.add(GRU(50))
gru_model.add(Dropout(0.2))

# Output layer
gru_model.add(Dense(1))

gru_model.compile(optimizer='adam', loss='mean_squared_error')

st.write(gru_model.summary())

st.write(gru_model.fit(X_train, y_train, epochs=30, batch_size=24, verbose=1))

# Make predictions
predicted_gru = gru_model.predict(X_test)

# Inverse transform the predicted and actual values
predicted_gru = scaler.inverse_transform(predicted_gru)
y_test_actual = scaler.inverse_transform(y_test)

# Plotting
fig = px.line(data, x='Time', y=['Actual', 'Predicted'], 
              title='Actual vs Predicted',
              labels={'value': 'Price', 'variable': 'Legend'},
              template='plotly_white')

# Настраиваем отображение легенды и осей
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Price',
    legend_title='Legend'
)

# Показываем график в Streamlit
st.plotly_chart(fig)