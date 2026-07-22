# 🤖 Javeria's AI Chatbot

ChatGPT-style chatbot built with **Streamlit** and **Groq (Llama models)** — fast, free, aur fully customizable.

## Features
- Real-time streaming responses
- Dark gradient UI theme
- Model switcher (Llama 3.3, Llama 3.1, Mixtral)
- Temperature/creativity control
- Chat history + clear chat button

## Setup

1. Clone the repo
   ```bash
   git clone https://github.com/your-username/my-chatbot.git
   cd my-chatbot
   ```

2. Create virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Groq API key in a `.env` file (copy `.env.example` and rename to `.env`)
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Run the app
   ```bash
   streamlit run app.py
   ```

## Deployment
Deployed free on [Streamlit Community Cloud](https://share.streamlit.io) — add `GROQ_API_KEY` under app **Secrets**.

## Tech Stack
- Streamlit
- Groq API (Llama 3.3 70B)
- Python-dotenv

---
Made with ❤️ by Javeria 
Updated: 2026
