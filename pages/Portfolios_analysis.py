import streamlit as st
from collections import defaultdict
import plotly.express as px
import pandas as pd

# Шаблоны портфелей
TEMPLATES = {
    "Пустой": {"rows": 5, "portfolios": 2, "data": {}},
    "Консервативный": {
        "rows": 5,
        "portfolios": 1,
        "data": {
            "ticker_0": "BND", "weight_0_0": "40%",
            "ticker_1": "VTI", "weight_0_1": "30%",
            "ticker_2": "GLD", "weight_0_2": "15%",
            "ticker_3": "VNQ", "weight_0_3": "10%",
            "ticker_4": "BIL", "weight_0_4": "5%"
        }
    },
    "Сбалансированный": {
        "rows": 6,
        "portfolios": 1,
        "data": {
            "ticker_0": "VTI", "weight_0_0": "40%",
            "ticker_1": "VXUS", "weight_0_1": "20%",
            "ticker_2": "BND", "weight_0_2": "20%",
            "ticker_3": "GLD", "weight_0_3": "10%",
            "ticker_4": "VNQ", "weight_0_4": "7%",
            "ticker_5": "BIL", "weight_0_5": "3%"
        }
    },
    "Агрессивный": {
        "rows": 6,
        "portfolios": 1,
        "data": {
            "ticker_0": "VTI", "weight_0_0": "50%",
            "ticker_1": "QQQ", "weight_0_1": "25%",
            "ticker_2": "ARKK", "weight_0_2": "15%",
            "ticker_3": "VXUS", "weight_0_3": "5%",
            "ticker_4": "GLD", "weight_0_4": "3%",
            "ticker_5": "BND", "weight_0_5": "2%"
        }
    },
    "Дивидендный": {
        "rows": 8,
        "portfolios": 1,
        "data": {
            "ticker_0": "SCHD", "weight_0_0": "25%",
            "ticker_1": "VYM", "weight_0_1": "20%",
            "ticker_2": "O", "weight_0_2": "15%",
            "ticker_3": "T", "weight_0_3": "10%",
            "ticker_4": "PFE", "weight_0_4": "10%",
            "ticker_5": "MO", "weight_0_5": "8%",
            "ticker_6": "VZ", "weight_0_6": "7%",
            "ticker_7": "KO", "weight_0_7": "5%"
        }
    }
}

def pie_chart(df, port_name):
    # Создаем круговую диаграмму
    fig = px.pie(
        df,
        names='Актив',
        values='Вес (%)',
        title=f'{port_name}',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    # Настраиваем отображение процентов
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Доля: %{percent}<br>"
    )
    
    # Настраиваем легенду
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Отображаем диаграмму
    st.plotly_chart(fig)


def add_template_as_new_portfolio(template_name):
    template = TEMPLATES[template_name]
        # Боковая панель с шаблонами
    with st.sidebar:
        st.header("Шаблоны портфелей")
        selected_template = st.selectbox(
            "Выберите шаблон", 
            list(TEMPLATES.keys()),
            index=0
        )
        
        if st.button("Применить шаблон"):
            apply_template(selected_template)
            st.success(f"Шаблон '{selected_template}' применен!")
        
        st.markdown("---")
        st.info("""
        **Доступные шаблоны:**
        - Консервативный (облигации + ETF)
        - Сбалансированный (60/40 акции/облигации)
        - Агрессивный (ростовые акции)
        - Дивидендный (дивидендные аристократы)
        """)
    # Добавляем новый портфель
    st.session_state.num_portfolios += 1
    new_portfolio_idx = st.session_state.num_portfolios - 1
    
    # Увеличиваем количество строк, если нужно
    if template["rows"] > st.session_state.num_rows:
        st.session_state.num_rows = template["rows"]
    
    # Заполняем данные для нового портфеля
    for key, value in template["data"].items():
        if key.startswith("ticker_"):
            row = int(key.split("_")[-1])
            st.session_state[f"ticker_{row}"] = value
        elif key.startswith("weight_"):
            parts = key.split("_")
            row = int(parts[-1])
            st.session_state[f"weight_{new_portfolio_idx}_{row}"] = value


def portfolio_manager(): 
    # Инициализация состояния сессии
    if 'num_rows' not in st.session_state:
        st.session_state.num_rows = 6
    if 'num_portfolios' not in st.session_state:
        st.session_state.num_portfolios = 2
    
    # Управление количеством строк и портфелей
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Добавить портфель", key="add_portfolio"):
            st.session_state.num_portfolios += 1
    with col2:
        if st.button("Удалить портфель", key="remove_portfolio") and st.session_state.num_portfolios > 1:
            st.session_state.num_portfolios -= 1
            # Очищаем данные удаленного портфеля
            for row in range(st.session_state.num_rows):
                st.session_state.pop(f"weight_{st.session_state.num_portfolios}_{row}", None)
    with col3:
        if st.button("Очистить данные", key="remove_data"):
            for key in list(st.session_state.keys()):
                if key.startswith("ticker_") or key.startswith("weight_"):
                    del st.session_state[key]

    # Создаем заголовки столбцов
    headers = ["Индекс", "Актив"] + [f"Портфель {i+1}" for i in range(st.session_state.num_portfolios)]
    cols = st.columns(len(headers))
    
    for i, header in enumerate(headers):
        with cols[i]:
            st.markdown(f"**{header}**")
    
    # Создаем строки и собираем данные
    portfolio_totals = [0] * st.session_state.num_portfolios
    portfolios = defaultdict(dict)
    
    for row in range(st.session_state.num_rows):
        cols = st.columns(len(headers))
        if row < st.session_state.num_rows - 1:
            with cols[0]:
                st.text(f"Актив {row+1}")
            
            with cols[1]:
                ticker = st.text_input(
                    f"Тикер {row+1}", 
                    key=f"ticker_{row}",
                    label_visibility="collapsed",
                    placeholder="AAPL, MSFT..."
                )
        # elif row == st.session_state.num_rows:
        #     if st.button("Добавить строку", key="add_row"):
        #         st.session_state.num_rows += 1
        
        # Обработка портфелей
        for portfolio_idx in range(st.session_state.num_portfolios):
            with cols[portfolio_idx + 2]:
                if row < st.session_state.num_rows - 1:
                    weight = st.text_input(
                        f"Вес {portfolio_idx+1}_{row}", 
                        key=f"weight_{portfolio_idx}_{row}",
                        label_visibility="collapsed",
                        placeholder="0.0%"
                    )
                    try:
                        clean_weight = float(weight.strip('%')) if weight else 0
                        portfolio_totals[portfolio_idx] += clean_weight
                        
                        if ticker and clean_weight > 0:
                            portfolios[f"Портфель {portfolio_idx+1}"][ticker.upper()] = clean_weight
                    except ValueError:
                        pass
                else:
                    total = portfolio_totals[portfolio_idx]
                    # Определяем цвет в зависимости от суммы весов
                    color = "red" if total > 100 else "green"
                    st.markdown(
                        f"<p style='color:{color}; font-weight:bold;'>Итого: {total:.1f}%</p>", 
                        unsafe_allow_html=True
                        )

                       
    # Кнопка для отображения портфелей
    if st.button("Сформировать портфели"):
        portfolios = {k: v for k, v in portfolios.items() if v}
        
        if portfolios:
            if all(x == 100 for x in portfolio_totals):
                st.success("Портфели успешно сформированы!")
                
                # Отображаем словари портфелей
                st.subheader('Рапсределение активов')
            
                for port_name, port_data in portfolios.items():
                    total_weight = sum(port_data.values())
                    
                    
                    # Создаем DataFrame для визуализации
                    df = pd.DataFrame({
                        'Актив': list(port_data.keys()),
                        'Вес (%)': list(port_data.values())
                    })
                    
                    # Создаем круговую диаграмму
                    pie_chart(df, port_name)
                    
                    # Дополнительная информация
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Количество активов", len(port_data))
                    with col2:
                        st.metric("Количество активов", len(port_data))
            else:
                st.warning("Не все портфели собраны на 100%.")
        else:
            st.warning("Нет данных для формирования портфелей. Добавьте тикеры и веса.")
    if total > 100:
        st.error(f"Сумма весов Портфеля {portfolio_idx+1} превышает 100% ({total:.1f}%)")


def main():
    st.set_page_config(page_title = "Анализ портфелей") 
    st.logo(image="materials/images/logo.png", size='large', icon_image="materials/images/icon.png")

    st.title("Анализ портфелей")
    
    portfolio_manager()


if __name__ == "__main__":
    main()