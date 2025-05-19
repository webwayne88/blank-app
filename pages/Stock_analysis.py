import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px


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


def display_ticker_info(ticker, df, recommendation):
    info = yf.Ticker(ticker).info
    # st.write(info.keys())
    with st.container():
        for row in range(1):
            cols = st.columns(2)
            with cols[0]:
                with st.expander(f"**{ticker}**", expanded=True):
                    delta, delta_percent = calculate_delta(df, 'Close')
                    delta_str = f"{delta:+,.0f} ({delta_percent:+.2f}%)"
                    company_name = yf.Ticker(ticker).info['longName']
                    st.metric(f'{company_name} ({ticker})', format_with_commas(df['Close'].iloc[-1]), delta=delta_str)
            with cols[1]:
                with st.expander("**Рекомендация:**", expanded=True):
                    st.metric(f'Что делать прямо сейчас?', recommendation)


def calculate_metrics(df):
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
        return df


def display_ticker_metrics(ticker, df):
        # Визуализация с использованием Plotly Express
        # График цены и скользящих средних
        # st.write("### Цена и скользящие средние")
        fig1 = px.line(df, x=df.index, y=['Close', 'SMA_50', 'SMA_200', 'EMA_20'], 
                        title=f'Цена и скользящие средние ({ticker})',
                        labels={'value': 'Цена', 'variable': 'Метрики'})
        st.plotly_chart(fig1, use_container_width=True)

        # График RSI
        # st.write("### Индекс относительной силы (RSI)")
        fig2 = px.line(df, x=df.index, y='RSI', 
                        title='RSI', 
                        labels={'value': 'RSI', 'variable': 'RSI'})
        fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Перекупленность")
        fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Перепроданность")
        st.plotly_chart(fig2, use_container_width=True)

        # График MACD
        # st.write("### MACD")
        fig3 = px.line(df, x=df.index, y=['MACD', 'Signal_Line'], 
                        title='MACD', 
                        labels={'value': 'Значение', 'variable': 'Метрики'})
        fig3.add_bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram')
        st.plotly_chart(fig3, use_container_width=True)



    # Создание графиков для каждого тикера
     

def display_open_prices(ticker, df):
    company_name = yf.Ticker(ticker).info['longName']
    # Создание графика с использованием Plotly Express
    fig = px.histogram(df, x='Open', nbins=30, 
                    title=f"Распределение цен открытия для {company_name} ({ticker})",
                    labels={'Open': 'Цена открытия', 'count': 'Частота'},)

    # Добавление KDE (ядерной оценки плотности)
    fig.update_traces(opacity=0.75, histnorm='probability density')
    fig.add_vline(x=df['Open'].mean(), line_dash="dash", line_color="red", 
                annotation_text=f"Среднее: {df['Open'].mean():.2f}")
        

    fig = px.line(df.reset_index(), x='Date', y='Close',
                title=f"Цена закрытия {ticker}",
                labels={'Date': 'Дата', 'Close': 'Цена закрытия'},
                line_shape="linear", render_mode="svg")

    # Настройка макета графика
    fig.update_layout(
        xaxis_title="Дата",
        yaxis_title="Цена закрытия",
        width=800,
        height=500,
        hovermode="x unified"  # Объединенные подсказки при наведении
    )

    # Отображение графика в Streamlit
    st.plotly_chart(fig, use_container_width=True)


def get_recommendation(df):
    # Расчет и отображение ключевых метрик
    latest_price = df["Close"].iloc[-1]
    moving_avg_50 = df['SMA_50'].iloc[-1]  # SMA за 50 дней
    moving_avg_200 = df['SMA_200'].iloc[-1]
    if latest_price > moving_avg_50 and latest_price > moving_avg_200:
        return "Покупать"
    elif latest_price < moving_avg_50 and latest_price < moving_avg_200:
        return "Продавать"
    else:
        return "Держать"


def main():
    st.set_page_config(page_title = "Анализ актива") 
    st.logo(image="materials/images/logo.png", size='large', icon_image="materials/images/icon.png")
    st.title('Анализ актива')
    with st.sidebar:
        st.header("Параметры:")
        ticker = st.text_input("Введите тикер: (msft, Appl, GOOG)", "MSFT")
        ticker = ticker.upper()
        # ticker = [ticker.strip() for ticker in tickers.split(",")] 
        start_date = st.date_input("Начальная дата", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("Конечная дата", pd.to_datetime("2025-01-01"))
    
    df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
    df = calculate_metrics(df)

    display_ticker_info(ticker, df, get_recommendation(df))
    st.subheader("Распределение цен открытия")
    display_open_prices(ticker, df)
    st.subheader("Технические графики")
    display_ticker_metrics(ticker, df)

    if st.checkbox("Показать итоговый дата фрейм"):
        st.write(df)

if __name__ == "__main__":
    main()