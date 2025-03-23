import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

st.logo("materials/images/logo.png", size='large')
st.set_page_config(page_title = "Формирование и управление инвестиционным портфелем")

# Данные о распределении активов
portfolio_data = {
    "Очень консервативный": {"Stocks": 25, "Bonds": 73, "Short-term": 2},
    "Консервативный": {"Stocks": 45, "Bonds": 53, "Short-term": 2},
    "Умеренный": {"Stocks": 64, "Bonds": 34, "Short-term": 2},
    "Агрессивный": {"Stocks": 82, "Bonds": 16, "Short-term": 2},
    "Очень агрессивный": {"Stocks": 98, "Bonds": 0, "Short-term": 2},
}

st.subheader("Распределение активов по видам портфеля")
with open('./materials/text/guide.md', 'r') as f:
    markdown_string = f.read()
st.markdown(markdown_string, unsafe_allow_html=True)


# Input features
with st.sidebar:
  st.header('Настройки портфеля')
  investment_amount = st.number_input("Сумма инвестиций ($)", min_value=100, value=1000)
  investment_period = st.number_input("Срок инвестиций (лет)", min_value=1, value=5)
  portfolio_type = st.selectbox("Выберите тип портфеля", list(portfolio_data.keys()))
  # risk = st.selectbox('Уровень риска', ('Очень консервативный', 'Консервативный', 'Умеренный', 'Агрессивный', 'Очень агрессивный'))

# Заголовок приложения
st.title("Формирование и управление инвестиционным портфелем")

#Получение данных для выбранного портфеля
allocations = portfolio_data[portfolio_type]

# Расчет распределения активов
stocks_allocation = allocations["Stocks"] / 100 * investment_amount
bonds_allocation = allocations["Bonds"] / 100 * investment_amount
short_term_allocation = allocations["Short-term"] / 100 * investment_amount

# Таблица с составленным портфелем
portfolio_table = pd.DataFrame({
    "Актив": ["Акции", "Облигации", "Краткосрочные инструменты"],
    "Распределение (%)": [allocations["Stocks"], allocations["Bonds"], allocations["Short-term"]],
    "Сумма ($)": [stocks_allocation, bonds_allocation, short_term_allocation],
})

# Отображение таблицы
st.subheader("Состав портфеля")
st.write(portfolio_table)

# Загрузка исторических данных для расчета доходности
@st.cache_data  # Кэширование данных для ускорения работы
def get_historical_data():
    stocks_data = yf.download("SPY", period="max")  # Индекс S&P 500 для акций
    bonds_data = yf.download("AGG", period="max")  # ETF для облигаций
    short_term_data = yf.download("SHY", period="max")  # ETF для краткосрочных инструментов
    return stocks_data, bonds_data, short_term_data

stocks_data, bonds_data, short_term_data = get_historical_data()

# Расчет годовой доходности для каждого актива
stocks_returns = stocks_data["Close"].pct_change().dropna().mean() * 252  # Годовая доходность акций
bonds_returns = bonds_data["Close"].pct_change().dropna().mean() * 252  # Годовая доходность облигаций
short_term_returns = short_term_data["Close"].pct_change().dropna().mean() * 252  # Годовая доходность краткосрочных инструментов

# Расчет доходности портфеля
portfolio_return = (
    (stocks_returns[0] * allocations["Stocks"] / 100) +
    (bonds_returns[0] * allocations["Bonds"] / 100) +
    (short_term_returns[0] * allocations["Short-term"] / 100)
)

# Прогнозируемая доходность за выбранный срок
future_value = investment_amount * (1 + portfolio_return) ** investment_period

# Отображение результатов
st.subheader("Расчетная доходность портфеля")
st.write(f"Годовая доходность портфеля: {portfolio_return * 100:.2f}%")
if investment_period % 10 == 1:
  st.write(f"Прогнозируемая стоимость портфеля через {investment_period} год: {future_value:.2f} $")
elif investment_period % 10 < 5:
  st.write(f"Прогнозируемая стоимость портфеля через {investment_period} года: {future_value:.2f} $")
else:
  st.write(f"Прогнозируемая стоимость портфеля через {investment_period} лет: {future_value:.2f} $")
# График распределения активов
st.subheader("Распределение активов")
fig = px.pie(
    portfolio_table,
    values="Сумма ($)",
    names="Актив",
    color_discrete_sequence=px.colors.sequential.Purp_r,
)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig)


