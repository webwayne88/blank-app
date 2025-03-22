import streamlit as st
import plotly.express as px

st.logo("materilas/images/logo.png", size='large')
st.set_page_config(page_title = "Introduction") 

# Данные о распределении активов
portfolio_data = {
    "Very Conservative": {"Stocks": 25, "Bonds": 73, "Short-term": 2},
    "Conservative": {"Stocks": 45, "Bonds": 53, "Short-term": 2},
    "Moderate": {"Stocks": 64, "Bonds": 34, "Short-term": 2},
    "Aggressive": {"Stocks": 82, "Bonds": 16, "Short-term": 2},
    "Very Aggressive": {"Stocks": 98, "Bonds": 0, "Short-term": 2},
}

# Заголовок приложения
st.title("Introduction")
# st.subheader("Распределение активов по видам портфеля")
st.markdown("""<span style='color: #63589F;'> This is a red font color</span>""", unsafe_allow_html=True)
with open('./materilas/text/guide.md', 'r') as f:
    markdown_string = f.read()

st.markdown(markdown_string, unsafe_allow_html=True)

# Выбор портфеля
portfolio_type = st.selectbox("Выберите тип портфеля", list(portfolio_data.keys()))

# Получение данных для выбранного портфеля
data = portfolio_data[portfolio_type]
labels = list(data.keys())
sizes = list(data.values())

# Создание pie chart с помощью Plotly Express
fig = px.pie(
    values=sizes,
    names=labels,
    title=f"Распределение активов: {portfolio_type}",
    color_discrete_sequence=px.colors.sequential.Purp_r,  # Оттенки синего
)

# Прозрачный фон
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",  # Прозрачный фон
    plot_bgcolor="rgba(0,0,0,0)",   # Прозрачный фон графика
    title_font=dict(size=20),       # Размер заголовка
)

# Отображение графика в Streamlit
st.plotly_chart(fig)