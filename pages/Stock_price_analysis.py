import streamlit as st
import yfinance as yf
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title = "Stock Price Analysis", layout="wide") 

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL").upper()
tickers = ['AAPL']

# Input for date range
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.today() - timedelta(days=1))

# Download stock data using yfinance
@st.cache_data  # Cache the data to improve performance
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, group_by=tickers, auto_adjust=False)
    return data

# Download data
data = load_data(tickers, start_date, end_date)
data = data.stack(level=0).reset_index()
data.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Display the raw data
if st.checkbox('Show raw data'):
    st.subheader(f"Raw Data for {ticker}")
    st.write(data)

# # Функция для расчета ключевых метрик
# def calculate_metrics(data):
#     latest_price = data['Close'][-1]
#     moving_avg_50 = data['Close'].rolling(window=50).mean()[-1]
#     moving_avg_200 = data['Close'].rolling(window=200).mean()[-1]
#     return latest_price, moving_avg_50, moving_avg_200

# Функция для рекомендации
def get_recommendation(latest_price, moving_avg_50, moving_avg_200):
    if latest_price > moving_avg_50 and latest_price > moving_avg_200:
        return "Покупать"
    elif latest_price < moving_avg_50 and latest_price < moving_avg_200:
        return "Продавать"
    else:
        return "Держать"

# Интерфейс Streamlit
st.title('Технический анализ акций')
st.logo(image="materilas/images/logo.png", size='large', icon_image="materilas/images/icon.png")

with st.sidebar:
    # st.title("YouTube Channel Dashboard")
    st.header("Input fetuares")
    
    max_date = data['Date'].max().date()
    default_start_date = max_date - timedelta(days=365)  # Show a year by default
    default_end_date = max_date
    start_date = st.date_input("Start date", default_start_date, min_value=data['Date'].min().date(), max_value=max_date)
    end_date = st.date_input("End date", default_end_date, min_value=data['Date'].min().date(), max_value=max_date)
    time_frame = st.selectbox("Select time frame",
                              ( "Weekly", "Monthly", "Quarterly", "Yearly"),
    )
    chart_selection = st.selectbox("Select a chart type",
                                   ("Bar", "Area", "Line"))

# # Выбор периода
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


# Create Altair line chart for Close price
st.subheader(f"Adj Close Price for {ticker}")
line_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x='Date:T',
    y='Adj Close:Q',
    tooltip=['Date:T', 'Adj Close:Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(line_chart)

# Расчет и отображение ключевых метрик
latest_price, moving_avg_50, moving_avg_200 = 100.00, 200.000, 300.00

# calculate_metrics(data)
st.subheader('Ключевые метрики:')
st.write(f"Последняя цена: {latest_price:.2f}")
st.write(f"50-дневная скользящая средняя: {moving_avg_50:.2f}")
st.write(f"200-дневная скользящая средняя: {moving_avg_200:.2f}")

# Display the change in Close price for the selected period
st.subheader(f"Change in Adj Close Price for {ticker} from {start_date} to {end_date}")
price_change = float(data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0])
st.write(f"The Close price changed by {price_change:.2f} during the selected period.") 

# Рекомендация
recommendation = get_recommendation(latest_price, moving_avg_50, moving_avg_200)
st.subheader('Рекомендация:')
st.write(recommendation)