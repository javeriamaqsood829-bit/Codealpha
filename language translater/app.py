import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="LinguaBridge - Language Translator",
    page_icon="🌐",
    layout="wide"
)

# Remove default Streamlit padding so the HTML page fills the screen
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Read the original HTML file and render it as-is
with open("translator.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)
