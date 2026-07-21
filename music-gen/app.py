import streamlit as st
import streamlit.components.v1 as components
import base64
import os
import datetime

from generate import generate_music, save_midi

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="wide"
)

# ---------------- DEFAULT GENERATION SETTINGS (fixed, no sidebar) ----------------
INSTRUMENT_NAME = "Piano"
BPM = 100
NOTE_DURATION = 0.5   # Medium speed
NUM_NOTES = 100
TEMPERATURE = 1.0

# ---------------- CUSTOM CSS (Premium Dark Gradient Theme) ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { padding-top: 2rem; max-width: 1000px; }

    h1, h2, h3 { font-family: 'Sora', sans-serif !important; }

    .nav-badge {
        display: inline-block;
        font-size: 12px;
        color: #c8bfff;
        background: rgba(139,124,246,0.15);
        border: 1px solid rgba(139,124,246,0.35);
        padding: 6px 14px;
        border-radius: 999px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 40px;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(135deg, #ffffff, #c8bfff 60%, #8b7cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 14px;
    }
    .hero-sub {
        font-size: 16px;
        color: #b8b3d9;
        line-height: 1.6;
        margin-bottom: 10px;
        max-width: 640px;
    }
    .stats-row {
        display: flex;
        gap: 50px;
        margin: 30px 0 40px;
        flex-wrap: wrap;
    }
    .stat-num {
        font-family: 'Sora', sans-serif;
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(135deg, #8b7cf6, #c8bfff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label { font-size: 12px; color: #8a84b8; margin-top: 2px; }

    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 22px 18px;
        height: 100%;
    }
    .feature-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, rgba(106,90,205,0.3), rgba(139,124,246,0.3));
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
        margin-bottom: 12px;
    }
    .feature-card h4 { font-size: 14px; margin: 0 0 6px; }
    .feature-card p { font-size: 12px; color: #8a84b8; line-height: 1.5; margin: 0; }

    .app-shell {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 30px;
        margin-top: 20px;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #6a5acd, #8b7cf6);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 6px 24px rgba(106,90,205,0.4);
    }
    div[data-testid="stDownloadButton"] button {
        background: rgba(255,255,255,0.06);
        color: white;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
    }
    .history-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .how-step {
        position: relative;
        padding-left: 46px;
        margin-bottom: 20px;
    }
    .how-num {
        position: absolute;
        left: 0; top: 0;
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #6a5acd, #8b7cf6);
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
        font-family: 'Sora', sans-serif;
        font-size: 14px;
    }
    .how-step h4 { font-size: 14px; margin: 0 0 4px; }
    .how-step p { font-size: 12px; color: #8a84b8; margin: 0; line-height: 1.5; }

    .footer-note {
        text-align: center;
        color: #6b6591;
        font-size: 12px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #6a5acd; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "current" not in st.session_state:
    st.session_state.current = None

# ---------------- HERO SECTION ----------------
st.markdown('<div class="nav-badge">⚡ Powered by LSTM Neural Network</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🎵 AI Music Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Compose original, royalty-free melodies in seconds — powered by a '
    'custom-trained neural network. No musical experience needed.</div>',
    unsafe_allow_html=True
)

# ---------------- STATS ----------------
st.markdown("""
<div class="stats-row">
    <div><div class="stat-num">8</div><div class="stat-label">Instruments</div></div>
    <div><div class="stat-num">∞</div><div class="stat-label">Unique tracks</div></div>
    <div><div class="stat-num">100%</div><div class="stat-label">Royalty-free</div></div>
    <div><div class="stat-num">&lt;10s</div><div class="stat-label">Generation time</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
f1, f2, f3, f4 = st.columns(4)
features = [
    ("🎻", "Multi-instrument", "Piano, Violin, Guitar, Flute & more, trained into one model."),
    ("🎨", "Creativity control", "Every generation balances structure with surprise."),
    ("🎹", "Live piano-roll", "Watch your melody play back with a visual timeline."),
    ("⬇️", "Instant MIDI export", "Download every generation as a ready-to-use .mid file."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- APP SHELL (Generate + Player + History) ----------------
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown("### ▶️ Now Playing")
with top_col2:
    generate_clicked = st.button("🎼 Generate New Music", use_container_width=True)

if generate_clicked:
    with st.spinner("Composing your melody... 🎹"):
        os.makedirs("output", exist_ok=True)
        output_stream, notes = generate_music(
            num_notes=NUM_NOTES,
            temperature=TEMPERATURE,
            seed=None,
            instrument_name=INSTRUMENT_NAME,
            bpm=BPM,
            note_duration=NOTE_DURATION
        )
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        midi_path = f"output/generated_{timestamp}.mid"
        save_midi(output_stream, midi_path)

        with open(midi_path, "rb") as f:
            midi_bytes = f.read()

        entry = {
            "name": f"{INSTRUMENT_NAME} · {BPM} BPM · {NUM_NOTES} notes",
            "bytes": midi_bytes,
            "notes_preview": ", ".join(notes[:12]) + ("..." if len(notes) > 12 else ""),
            "timestamp": timestamp
        }
        st.session_state.current = entry
        st.session_state.history.insert(0, entry)
        st.session_state.history = st.session_state.history[:5]

col_main, col_history = st.columns([2, 1])

with col_main:
    if st.session_state.current:
        entry = st.session_state.current
        midi_b64 = base64.b64encode(entry["bytes"]).decode("utf-8")
        st.caption(entry["name"])

        components.html(f"""
            <script src="https://cdn.jsdelivr.net/npm/html-midi-player@1.5.0/dist/midi-player.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/html-midi-player@1.5.0/dist/midi-visualizer.min.js"></script>
            <style>
                midi-player {{ width: 100%; margin-bottom: 8px; }}
                midi-player::part(control-panel) {{
                    background: #1a1a2e;
                    border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.15);
                }}
                midi-visualizer svg {{ background: #16213e; border-radius: 12px; }}
            </style>
            <midi-player
                src="data:audio/midi;base64,{midi_b64}"
                sound-font
                visualizer="#viz_current">
            </midi-player>
            <midi-visualizer id="viz_current" type="piano-roll"></midi-visualizer>
        """, height=260)

        st.download_button(
            label="⬇️ Download MIDI file",
            data=entry["bytes"],
            file_name=f"ai_music_{entry['timestamp']}.mid",
            mime="audio/midi",
            use_container_width=True
        )
    else:
        st.info("👆 Click **Generate New Music** to create your first AI-composed melody.")

with col_history:
    st.markdown("**🕘 RECENT GENERATIONS**")
    if not st.session_state.history:
        st.caption("Nothing generated yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            st.markdown(f'<div class="history-card"><b>{entry["name"]}</b></div>', unsafe_allow_html=True)
            b64 = base64.b64encode(entry["bytes"]).decode("utf-8")
            components.html(f"""
                <script src="https://cdn.jsdelivr.net/npm/html-midi-player@1.5.0/dist/midi-player.min.js"></script>
                <style>
                    midi-player {{ width: 100%; }}
                    midi-player::part(control-panel) {{
                        background: #1a1a2e; border-radius: 10px;
                        border: 1px solid rgba(255,255,255,0.12);
                    }}
                </style>
                <midi-player src="data:audio/midi;base64,{b64}" sound-font></midi-player>
            """, height=70)
            st.download_button(
                "⬇️ Download",
                data=entry["bytes"],
                file_name=f"ai_music_{entry['timestamp']}.mid",
                mime="audio/midi",
                key=f"dl_{i}",
                use_container_width=True
            )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HOW IT WORKS ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## How it works")
h1, h2, h3 = st.columns(3)
steps = [
    ("1", "Model trained on melodies", "An LSTM network learned note transitions, rhythm, and harmony from a curated dataset."),
    ("2", "Click generate", "One click and the network composes a brand-new sequence, note by note."),
    ("3", "Play & download", "Preview it instantly, then export the MIDI file for your own project."),
]
for col, (num, title, desc) in zip([h1, h2, h3], steps):
    with col:
        st.markdown(f"""
        <div class="how-step">
            <div class="how-num">{num}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Built with an LSTM neural network · music21 · Streamlit — '
    'CodeAlpha AI Internship, Task 3</div>',
    unsafe_allow_html=True
)
