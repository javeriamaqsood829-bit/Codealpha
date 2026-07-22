# Javeria's AI Chatbot Project
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Javeria's AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS (Dark Gradient Theme) ----------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    h1, h2, h3 {
        color: #f2f2f2;
        font-weight: 700;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 10px;
        margin-bottom: 8px;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: rgba(255, 255, 255, 0.08);
        color: white;
        border-radius: 12px;
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #6a5acd;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Model choose karein",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    )
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful, friendly AI assistant."}
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("Made with ❤️ by **Javeria**")

# ---------------- TITLE ----------------
st.title("🤖 Javeria's AI Chatbot")
st.caption("Ask me anything ✨")

# ---------------- API CLIENT ----------------
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not api_key:
    st.error("⚠️ GROQ_API_KEY nahi mili. .env file ya Streamlit secrets check karein.")
    st.stop()

client = Groq(api_key=api_key)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, friendly AI assistant."}
    ]

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Apna message likhein...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=st.session_state.messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_reply += delta
                placeholder.markdown(full_reply + "▌")
            placeholder.markdown(full_reply)
        except Exception as e:
            full_reply = f"Error aa gaya: {e}"
            placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
