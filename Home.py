import streamlit as st


st.logo("materilas/images/logo.png", size='large')
st.set_page_config(page_title = "Investio") 

st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
            
    """, unsafe_allow_html=True)
st.title("💼 Welcome to Investio") 
st.info('Приложение все еще находится в состоянии разработки', icon="ℹ️")
st.subheader("Структура:")
st.page_link("pages/Theory.py", label="Theory", icon="📄")
st.page_link("pages/Practice.py", label="Practice", icon="💻")
st.page_link("pages/Guide.py", label="Guide", icon="📘")
st.page_link("pages/Portfolio_optimization.py", label="Portfolio optimization", icon="💼")
st.page_link("pages/Stock_price_analysis.py", label="Stock price analysis", icon="📊")
st.page_link("pages/Stock_price_prediction.py", label="Stock price prediction", icon="📉")

with open('./materilas/text/description.md', 'r') as f:
    markdown_string = f.read()

st.markdown(markdown_string, unsafe_allow_html=True)
