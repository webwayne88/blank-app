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

# Создание графиков для каждого тикера
st.subheader("Распределение цен открытия")

# Используем столбцы для отображения графиков в сетке 2x2
col1, col2 = st.columns(2)

# Уникальные тикеры и компании
unique_tickers = df['Ticker'].unique()

for i, ticker in enumerate(unique_tickers):
    # Фильтрация данных для текущего тикера
    ticker_data = df[df['Ticker'] == ticker]
    company_name = yf.Ticker(ticker).info['longName']
    # Создание графика с использованием Plotly Express
    fig = px.histogram(ticker_data, x='Open', nbins=30, 
                    title=f"Распределение цен открытия для {company_name} ({ticker})",
                    labels={'Open': 'Цена открытия', 'count': 'Частота'},
                    color_discrete_sequence=[px.colors.qualitative.Plotly[i]])

    # Добавление KDE (ядерной оценки плотности)
    fig.update_traces(opacity=0.75, histnorm='probability density')
    fig.add_vline(x=ticker_data['Open'].mean(), line_dash="dash", line_color="red", 
                  annotation_text=f"Среднее: {ticker_data['Open'].mean():.2f}")

    # Отображение графика в соответствующем столбце
    if i % 2 == 0:
        col1.plotly_chart(fig, use_container_width=True)
    else:
        col2.plotly_chart(fig, use_container_width=True)


fig = px.line(df.reset_index(), x='Date', y='Adj Close', color='Ticker',
              title="Adjusted Close Price",
              labels={'Date': 'Дата', 'Adj Close': 'Adjusted Close Price', 'Ticker': 'Тикер'},
              line_shape="linear", render_mode="svg")

# Настройка макета графика
fig.update_layout(
    xaxis_title="Дата",
    yaxis_title="Adjusted Close Price",
    legend_title="Тикер",
    width=800,
    height=500,
    hovermode="x unified"  # Объединенные подсказки при наведении
)

# Отображение графика в Streamlit
st.plotly_chart(fig, use_container_width=True)

if len(tickers) == 1:
    # 1. Скользящие средние (SMA и EMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # SMA за 50 дней
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # SMA за 50 дней0:
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()  # EMA за 20 дней

    # 2. Относительная сила (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. Средний истинный диапазон (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # 4. Стохастический осциллятор (%K и %D)
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # 5. Индекс направленного движения (ADX)
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / df['ATR'])
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.ewm(span=14, adjust=False).mean()

    # 6. Скользящая средняя конвергенция/дивергенция (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']


    # Визуализация с использованием Plotly Express
    st.subheader("Графики технического анализа")

    # График цены и скользящих средних
    st.write("### Цена и скользящие средние")
    fig1 = px.line(df, x=df.index, y=['Close', 'SMA_50', 'SMA_200', 'EMA_20'], 
                    title=f'Цена и скользящие средние ({ticker})',
                    labels={'value': 'Цена', 'variable': 'Метрики'})
    st.plotly_chart(fig1, use_container_width=True)

    # График RSI
    st.write("### Индекс относительной силы (RSI)")
    fig2 = px.line(df, x=df.index, y='RSI', 
                    title='RSI', 
                    labels={'value': 'RSI', 'variable': 'RSI'})
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Перекупленность")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Перепроданность")
    st.plotly_chart(fig2, use_container_width=True)

    # График MACD
    st.write("### MACD")
    fig3 = px.line(df, x=df.index, y=['MACD', 'Signal_Line'], 
                    title='MACD', 
                    labels={'value': 'Значение', 'variable': 'Метрики'})
    fig3.add_bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram')
    st.plotly_chart(fig3, use_container_width=True)

    # Рекомендация
    # Функция для рекомендации
    # # Функция для расчета ключевых метрик
    def calculate_metrics(df):
        latest_price = df['Close'][-1]
        moving_avg_50 = df['Close'].rolling(window=50).mean()[-1]
        moving_avg_200 = df['Close'].rolling(window=200).mean()[-1]
        return latest_price, moving_avg_50, moving_avg_200

    def get_recommendation(latest_price, moving_avg_50, moving_avg_200):
        if latest_price > moving_avg_50 and latest_price > moving_avg_200:
            return "Покупать"
        elif latest_price < moving_avg_50 and latest_price < moving_avg_200:
            return "Продавать"
        else:
            return "Держать"
    # Расчет и отображение ключевых метрик
    latest_price = df["Close"].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]  # SMA за 50 дней
    sma_200= df['SMA_200'].iloc[-1]
    recommendation = get_recommendation(latest_price, sma_50, sma_200)
    st.subheader(f'Рекомендация: {recommendation}')


