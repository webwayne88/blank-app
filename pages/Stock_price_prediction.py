import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import statistics

st.set_page_config(page_title = "Stock Price Analysis", layout="wide") 
st.logo(image="materilas/images/logo.png", size='large', icon_image="materilas/images/icon.png")

# period = st.selectbox('Выберите период:', ['Неделя', 'Месяц', 'Квартал', 'Год', 'Всё время', 'Выбрать вручную'])
# # Определение дат
# end_date = datetime.now()
# if period == 'Неделя':
#     start_date = end_date - pd.Timedelta(days=7)
# elif period == 'Месяц':
#     start_date = end_date - pd.Timedelta(days=30)
# elif period == 'Квартал':
#     start_date = end_date - pd.Timedelta(days=90)
# elif period == 'Год':
#     start_date = end_date - pd.Timedelta(days=365)
# elif period == 'Всё время':
#     start_date = datetime(2000, 1, 1)
# else:
#     start_date = st.date_input('Выберите начальную дату:', datetime(2020, 1, 1))
#     end_date = st.date_input('Выберите конечную дату:', datetime.now())

with st.sidebar:
    st.header("Input fetuares")
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    tickers = st.text_input("Введите тикеры через запятую (например, AAPL, TSLA, MSFT):", "MSFT")
    tickers = [ticker.strip() for ticker in tickers.split(",")]  # Разделяем введенные тикеры
    start_date = st.date_input("Начальная дата", pd.to_datetime("2024-12-12"))
    end_date = st.date_input("Конечная дата", pd.to_datetime("2025-03-20"))
    time_frame = st.selectbox("Select time frame",
                              ( "Weekly", "Monthly", "Quarterly", "Yearly"),
    )
    # chart_selection = st.selectbox("Select a chart type",
    #                                ("Bar", "Area", "Line"))

@st.cache_data 
def load_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    data = yf.download(tickers, start=start_date, end=end_date,group_by=tickers, auto_adjust=False)
    return data

df = load_data(tickers, start_date, end_date)
df = df.stack(level=0).reset_index()
df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Display the raw data
if st.checkbox('Show raw data'):
    st.subheader(f"Raw Data for {tickers}")
    st.write(df.tail(30))

df=df.set_index('Date')

st.subheader("All-Time Statistics")

def calculate_delta(df, column):
    if len(df) < 2:
        return 0, 0
    current_value = df[column].iloc[-1]
    previous_value = df[column].iloc[-2]
    delta = current_value - previous_value
    delta_percent = (delta / previous_value) * 100 if previous_value != 0 else 0
    return delta, delta_percent

def format_with_commas(number):
    return f"{number:.2f} $"

def create_metric_chart(df, column, color, height=150):
    chart_data = df[[column]].copy()
    st.bar_chart(chart_data, y=column, color=color, height=height)


def display_metric(col, ticker, value, df, column, color):
    with col:
        with st.container(border=True):
            delta, delta_percent = calculate_delta(df, column)
            delta_str = f"{delta:+,.0f} ({delta_percent:+.2f}%)"
            st.metric(ticker, format_with_commas(value), delta=delta_str)
            # create_metric_chart(df, column, color)

tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'CE']
colors = ['#7D44CF', '#44CF7D', '#CF447D', '#447DCF']  # Фиолетовый, зеленый, розовый, синий

if len(colors) < len(tickers):
    for i in range(0, len(tickers) - len(colors)):
        colors.append(colors[i])

# Создание списка метрик с разными цветами
metrics = [(ticker, "Adj Close", color) for ticker, color in zip(tickers, colors)]

cols = st.columns(len(metrics))
for col, (ticker, column, color) in zip(cols, metrics):
    last_value = df[column].iloc[-1]
    display_metric(col, ticker, last_value, df, column, color)
