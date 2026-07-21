# 🎵 AI Music Composer

A CodeAlpha AI Internship project — **Task 3: Music Generation with AI**.

An LSTM (Long Short-Term Memory) neural network is trained on a dataset
of MIDI melodies to learn musical patterns (note transitions, rhythm,
simple harmony), and then generates brand-new original music note by
note. The result is exported as a downloadable, playable MIDI file.

## How it works

1. **`generate_dataset.py`** — creates a starter set of sample training
   melodies (major/minor scales, several keys) as MIDI files in `data/`.
   *(For richer results, add your own real MIDI files to this folder.)*
2. **`train.py`** — parses all MIDI files in `data/`, extracts note and
   chord sequences, and trains an LSTM model to predict the next note
   given the previous 20 notes. Saves the trained model to `model/`.
3. **`generate.py`** — loads the trained model and samples a brand new
   sequence of notes, which is converted into a MIDI file using `music21`.
4. **`app.py`** — a Streamlit web interface where you can generate new
   music on demand, preview it in-browser (piano roll + playback), and
   download the `.mid` file.

## Features
- **Instrument selection** — Piano, Violin, Acoustic Guitar, Flute, Electric Bass, Xylophone, Church Organ, Synth Lead
- **Tempo control** (BPM slider)
- **Note speed control** (Fast / Medium / Slow / Very slow)
- Adjustable generation length
- "Temperature" control (creativity vs. predictability)
- Optional random seed for reproducible results
- In-browser MIDI playback with piano-roll visualizer
- **Generation history** — last 5 melodies saved in-session with their own mini players
- Raw note-sequence preview (see exactly what the model generated)
- Downloadable `.mid` output for every generation

## Run locally

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# (Optional) regenerate sample training data and retrain the model:
python generate_dataset.py
python train.py

# Launch the app (uses the pre-trained model already included):
streamlit run app.py
```

## Deployment
Deployed free on [Streamlit Community Cloud](https://share.streamlit.io).
The trained model (`model/music_model.keras`) is included in the repo,
so the app generates music instantly without retraining on the server.

## Tech Stack
- TensorFlow / Keras (LSTM)
- music21 (MIDI parsing & generation)
- Streamlit (web interface)
- html-midi-player (in-browser playback)

---
Made with ❤️ by Javeria — CodeAlpha AI Internship, Task 3
