import streamlit as st
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import kaleido
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import yfinance as yf
from datetime import datetime
import os

st.logo("materials/images/logo.png", size='large')
st.set_page_config(page_title = "Формирование и управление инвестиционным портфелем", layout='wide')

# Данные о распределении активов
portfolio_data = {
    "Очень консервативный": {"Stocks": 25, "Bonds": 73, "Short-term": 2},
    "Консервативный": {"Stocks": 45, "Bonds": 53, "Short-term": 2},
    "Умеренный": {"Stocks": 64, "Bonds": 34, "Short-term": 2},
    "Агрессивный": {"Stocks": 82, "Bonds": 16, "Short-term": 2},
    "Очень агрессивный": {"Stocks": 98, "Bonds": 0, "Short-term": 2},
}

st.subheader("Распределение активов по видам портфеля")
# with open('./materials/text/guide.md', 'r') as f:
#     markdown_string = f.read()
# st.markdown(markdown_string, unsafe_allow_html=True)


# Input features
with st.sidebar:
  st.header('Настройки портфеля')
  investment_amount = st.number_input("Сумма инвестиций ($)", min_value=100, value=1000)
  investment_period = st.number_input("Срок инвестиций (лет)", min_value=1, value=5)
  portfolio_type = st.selectbox("Выберите тип портфеля", list(portfolio_data.keys()))
  risk = st.selectbox('Уровень риска', ('Очень консервативный', 'Консервативный', 'Умеренный', 'Агрессивный', 'Очень агрессивный'))


# Палитра цветов: Purp_r (в обратном порядке, от темного к светлому)
colors = px.colors.sequential.Purp_r

# Определим портфельные стратегии
strategies = {
    "Very Conservative": {"Stocks": 25, "Bonds": 73, "Short-Term": 2},
    "Conservative": {"Stocks": 45, "Bonds": 53, "Short-Term": 2},
    "Moderate": {"Stocks": 64, "Bonds": 34, "Short-Term": 2},
    "Aggressive": {"Stocks": 82, "Bonds": 16, "Short-Term": 2},
    "Very Aggressive": {"Stocks": 98, "Bonds": 0, "Short-Term": 2},
}

# Создание визуализаций
cols = st.columns(len(strategies))

for i, (name, allocation) in enumerate(strategies.items()):
    with cols[i]:
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation.keys()),
            values=list(allocation.values()),
            hole=0.6,
            marker=dict(colors=colors[:len(allocation)]),
            textinfo='none'
        )])

        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            width=180,
            height=180
        )

        st.plotly_chart(fig)
        st.markdown(f"**{name}**")
        for k, v in allocation.items():
            st.markdown(f"{k} = {v}%")

# #Получение данных для выбранного портфеля
# allocations = portfolio_data[portfolio_type]

# # Расчет распределения активов
# stocks_allocation = allocations["Stocks"] / 100 * investment_amount
# bonds_allocation = allocations["Bonds"] / 100 * investment_amount
# short_term_allocation = allocations["Short-term"] / 100 * investment_amount

# # Таблица с составленным портфелем
# portfolio_table = pd.DataFrame({
#     "Актив": ["Акции", "Облигации", "Краткосрочные инструменты"],
#     "Распределение (%)": [allocations["Stocks"], allocations["Bonds"], allocations["Short-term"]],
#     "Сумма ($)": [stocks_allocation, bonds_allocation, short_term_allocation],
# })

# Отображение таблицы
# st.subheader("Состав портфеля")
# st.write(portfolio_table)

# # Загрузка исторических данных для расчета доходности
# @st.cache_data  # Кэширование данных для ускорения работы
# def get_historical_data():
#     stocks_data = yf.download("SPY", period="max")  # Индекс S&P 500 для акций
#     bonds_data = yf.download("AGG", period="max")  # ETF для облигаций
#     short_term_data = yf.download("SHY", period="max")  # ETF для краткосрочных инструментов
#     return stocks_data, bonds_data, short_term_data

# stocks_data, bonds_data, short_term_data = get_historical_data()

# # Расчет годовой доходности для каждого актива
# stocks_returns = stocks_data["Close"].pct_change().dropna().mean() * 252  # Годовая доходность акций
# bonds_returns = bonds_data["Close"].pct_change().dropna().mean() * 252  # Годовая доходность облигаций
# short_term_returns = short_term_data["Close"].pct_change().dropna().mean() * 252  # Годовая доходность краткосрочных инструментов

# # Расчет доходности портфеля
# portfolio_return = (
#     (stocks_returns[0] * allocations["Stocks"] / 100) +
#     (bonds_returns[0] * allocations["Bonds"] / 100) +
#     (short_term_returns[0] * allocations["Short-term"] / 100)
# )

# # Прогнозируемая доходность за выбранный срок
# future_value = investment_amount * (1 + portfolio_return) ** investment_period

# # Отображение результатов
# st.subheader("Расчетная доходность портфеля")
# st.write(f"Годовая доходность портфеля: {portfolio_return * 100:.2f}%")
# if investment_period % 10 == 1:
#   st.write(f"Прогнозируемая стоимость портфеля через {investment_period} год: {future_value:.2f} $")
# elif investment_period % 10 < 5:
#   st.write(f"Прогнозируемая стоимость портфеля через {investment_period} года: {future_value:.2f} $")
# else:
#   st.write(f"Прогнозируемая стоимость портфеля через {investment_period} лет: {future_value:.2f} $")
# # График распределения активов
# st.subheader("Распределение активов")
# fig = px.pie(
#     portfolio_table,
#     values="Сумма ($)",
#     names="Актив",
#     color_discrete_sequence=colors,
# )
# fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
# st.plotly_chart(fig)


# Set up Streamlit page
st.subheader("Portfolio Analysis")  
# st.logo(image="materials/images/logo.png", size='large', icon_image="materials/images/icon.png") 

# Create images directory if it doesn't exist
os.makedirs("materials/images", exist_ok=True)

# Load test data (example portfolio)
def get_portfolio_data():
    portfolio = {
        'Asset': ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'BTC-USD', 'ETH-USD', 'SPY', 'GLD', 'Cash'],
        'Type': ['Stock', 'Stock', 'Stock', 'Stock', 'Stock', 'Crypto', 'Crypto', 'ETF', 'Commodity', 'Cash'],
        'Sector': ['Tech', 'Tech', 'Auto', 'Tech', 'Retail', 'Crypto', 'Crypto', 'ETF', 'Commodity', 'Cash'],
        'Region': ['US', 'US', 'US', 'US', 'US', 'Global', 'Global', 'Global', 'Global', 'Local'],
        'Currency': ['USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD'],
        'Quantity': [10, 5, 3, 7, 2, 0.5, 2, 15, 5, 5000],
        'Entry Price': [150, 300, 700, 2500, 3500, 50000, 3000, 450, 180, 1]
    }
    df = pd.DataFrame(portfolio)
    
    # Get current prices from Yahoo Finance
    current_prices = {}
    for asset in df['Asset']:
        if asset == 'Cash':
            current_prices[asset] = 1
        else:
            try:
                ticker = yf.Ticker(asset)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_prices[asset] = hist['Close'].iloc[-1]
                else:
                    current_prices[asset] = df.loc[df['Asset'] == asset, 'Entry Price'].values[0]
            except:
                current_prices[asset] = df.loc[df['Asset'] == asset, 'Entry Price'].values[0]
    
    df['Current Price'] = df['Asset'].map(current_prices)
    df['Market Value'] = df['Quantity'] * df['Current Price']
    df['PnL (%)'] = (df['Current Price'] / df['Entry Price'] - 1) * 100
    st.write
    return df

# Create visualizations
def create_visualizations(df, output_path="materials/images"):
    # Asset Type Distribution (Pie Chart)
    fig1 = px.pie(
        df.groupby('Type', as_index=False)['Market Value'].sum(),
        names='Type',
        values='Market Value',
        title='Asset Type Distribution',
        color_discrete_sequence=colors,
        hole=0.3
    )
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    fig1.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=50, b=10, l=10, r=10)
    )
    fig1.write_image(f"{output_path}/asset_type.png", scale=2, engine='kaleido')

    # Sector Distribution (Bar Chart)
    sector_df = df.groupby('Sector', as_index=False)['Market Value'].sum().sort_values('Market Value')
    fig2 = px.bar(
        sector_df,
        y='Sector',
        x='Market Value',
        orientation='h',
        title='Sector Distribution',
        color='Market Value',
        color_continuous_scale=colors,
        text_auto='.2s'
    )
    fig2.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Market Value (USD)",
        yaxis_title="Sector",
        coloraxis_showscale=False,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    fig2.update_traces(
        textfont_size=12,
        textangle=0,
        textposition="outside",
        cliponaxis=False
    )
    fig2.write_image(f"{output_path}/sectors.png", scale=2)

    # Asset Performance (Bar Chart)
    fig3 = px.bar(
        df.sort_values('PnL (%)'),
        x='Asset',
        y='PnL (%)',
        title='Asset Performance (%)',
        color='PnL (%)',
        color_continuous_scale=px.colors.diverging.RdYlGn,
        text_auto='.1f'
    )
    fig3.update_layout(
        xaxis_title="Asset",
        yaxis_title="Return (%)",
        coloraxis_showscale=False,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    fig3.update_traces(
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5
    )
    fig3.write_image(f"{output_path}/performance.png", scale=2)

    # Top 5 Assets (Bar Chart)
    top_assets = df.nlargest(5, 'Market Value')
    fig4 = px.bar(
        top_assets,
        x='Asset',
        y='Market Value',
        title='Top 5 Assets by Market Value',
        color='Market Value',
        color_continuous_scale=colors,
        text_auto='.2s'
    )
    fig4.update_layout(
        yaxis_title="Market Value (USD)",
        coloraxis_showscale=False,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    fig4.update_traces(
        texttemplate='$%{text:.2s}',
        textposition='outside'
    )
    fig4.update_yaxes(tickprefix="$")
    fig4.write_image(f"{output_path}/top_assets.png", scale=2)

    # Portfolio Composition (Sunburst Chart) - Bonus visualization
    fig5 = px.sunburst(
        df,
        path=['Type', 'Sector', 'Asset'],
        values='Market Value',
        title='Portfolio Composition',
        color_discrete_sequence=colors
    )
    fig5.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    fig5.write_image(f"{output_path}/composition.png", scale=2)
    st.plotly_chart(fig1) 
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)

# Generate PDF report
def generate_pdf_report(df, output_file="investment_portfolio_report.pdf"):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        name='Title',
        fontSize=18,
        alignment=1,
        spaceAfter=20
    )
    elements.append(Paragraph("Investment Portfolio: Detailed Report", title_style))
    elements.append(Spacer(1, 12))
    
    # General information
    total_value = df['Market Value'].sum()
    elements.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Paragraph(f"<b>Total Portfolio Value:</b> ${total_value:,.2f}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Assets table
    table_data = [['Asset', 'Type', 'Sector', 'Quantity', 'Current Price', 'Market Value', 'Return (%)']]
    for _, row in df.iterrows():
        table_data.append([
            row['Asset'],
            row['Type'],
            row['Sector'],
            row['Quantity'],
            f"${row['Current Price']:,.2f}",
            f"${row['Market Value']:,.2f}",
            f"{row['PnL (%)']:.2f}%"
        ])
    
    asset_table = Table(table_data)
    asset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(asset_table)
    elements.append(Spacer(1, 20))
    
    # Charts
    elements.append(Paragraph("<b>Asset Allocation Visualization</b>", styles['Heading2']))
    elements.append(Image("materials/images/asset_type.png", width=400, height=300))
    elements.append(Spacer(1, 12))
    elements.append(Image("materials/images/sectors.png", width=400, height=300))
    elements.append(Spacer(1, 12))
    elements.append(Image("materials/images/performance.png", width=500, height=350))
    elements.append(Spacer(1, 12))
    elements.append(Image("materials/images/top_assets.png", width=400, height=300))
    elements.append(Spacer(1, 20))
    
    # Risk and return analysis
    sharpe_ratio = 1.2  # Example value (can be calculated more accurately)
    max_drawdown = -15.3  # Example
    
    elements.append(Paragraph("<b>Key Risk and Return Metrics</b>", styles['Heading2']))
    elements.append(Paragraph(f"<b>Sharpe Ratio:</b> {sharpe_ratio:.2f}", styles['Normal']))
    elements.append(Paragraph(f"<b>Maximum Drawdown:</b> {max_drawdown:.1f}%", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Recommendations
    elements.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
    recommendations = [
        "1. Increase bond allocation to reduce volatility.",
        "2. Diversify portfolio by adding international assets.",
        "3. Rebalance the portfolio as technology sector exceeds 40%."
    ]
    for rec in recommendations:
        elements.append(Paragraph(rec, styles['Normal']))
    
    doc.build(elements)

# Streamlit UI
st.title("Investment Portfolio Analysis")

# Get portfolio data
portfolio_df = get_portfolio_data()

# Display portfolio metrics
col1, col2, col3 = st.columns(3)
total_value = portfolio_df['Market Value'].sum()
col1.metric("Total Portfolio Value", f"${total_value:,.2f}")
total_return = ((portfolio_df['Market Value'].sum() - (portfolio_df['Quantity'] * portfolio_df['Entry Price']).sum()) / 
               (portfolio_df['Quantity'] * portfolio_df['Entry Price']).sum() * 100)
col2.metric("Total Return", f"{total_return:.2f}%")
num_assets = len(portfolio_df)
col3.metric("Number of Assets", num_assets)

# Display portfolio table
st.subheader("Portfolio Holdings")
st.dataframe(portfolio_df.style.format({
    'Current Price': '${:,.2f}',
    'Market Value': '${:,.2f}',
    'PnL (%)': '{:.2f}%'
}), use_container_width=True)

# Display charts
st.subheader("Portfolio Visualizations")

# Create visualizations
create_visualizations(portfolio_df)




# col1, col2 = st.columns(2)
# with col1:
#     st.image("materials/images/asset_type.png", caption="Asset Type Distribution")
#     st.image("materials/images/top_assets.png", caption="Top 5 Assets by Market Value")

# with col2:
#     st.image("materials/images/sectors.png", caption="Sector Distribution")
#     st.image("materials/images/performance.png", caption="Asset Performance")

# Generate PDF report
if st.button("Generate PDF Report", type="primary"):
    generate_pdf_report(portfolio_df)
    st.success("Report successfully generated!")
    
    # Offer download link
    with open("investment_portfolio_report.pdf", "rb") as file:
        st.download_button(
            label="Download PDF Report",
            data=file,
            file_name="investment_portfolio_report.pdf",
            mime="application/pdf"
        )