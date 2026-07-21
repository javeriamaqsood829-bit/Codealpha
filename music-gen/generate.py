"""
generate.py
Loads the trained LSTM model + note mappings and generates a brand-new
melody, which is converted into a downloadable MIDI file using music21.
"""

import pickle
import random
import numpy as np
from music21 import stream, note, chord, tempo, instrument
from tensorflow import keras

MODEL_PATH = "model/music_model.keras"
MAPPINGS_PATH = "model/mappings.pkl"

INSTRUMENTS = {
    "Piano": instrument.Piano,
    "Acoustic Guitar": instrument.AcousticGuitar,
    "Violin": instrument.Violin,
    "Flute": instrument.Flute,
    "Electric Bass": instrument.ElectricBass,
    "Xylophone": instrument.Xylophone,
    "Church Organ": instrument.Organ,
    "Synth Lead": instrument.Vocalist,  # generic synth-like fallback
}

_model = None
_mappings = None


def _load():
    """Lazy-load the model and mappings once, reuse afterwards."""
    global _model, _mappings
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    if _mappings is None:
        with open(MAPPINGS_PATH, "rb") as f:
            _mappings = pickle.load(f)
    return _model, _mappings


def generate_music(num_notes=100, temperature=1.0, seed=None,
                    instrument_name="Piano", bpm=100, note_duration=0.5):
    """
    Generate a new sequence of notes/chords using the trained LSTM,
    then convert it into a music21 Stream (ready to export as MIDI).

    temperature: >1.0 = more random/creative, <1.0 = more predictable/safe
    instrument_name: key from INSTRUMENTS dict, sets the MIDI instrument
    bpm: playback tempo
    note_duration: quarter-length duration per note/chord
    """
    model, mappings = _load()
    note_to_int = mappings["note_to_int"]
    int_to_note = mappings["int_to_note"]
    n_vocab = mappings["n_vocab"]
    seq_len = mappings["sequence_length"]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Start from a random snippet of real note indices as a "seed" sequence
    pattern = [random.randint(0, n_vocab - 1) for _ in range(seq_len)]
    generated = []

    for _ in range(num_notes):
        input_seq = np.reshape(pattern, (1, seq_len, 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)[0]

        # Apply temperature for controllable randomness
        prediction = np.log(prediction + 1e-9) / max(temperature, 0.1)
        exp_preds = np.exp(prediction)
        probs = exp_preds / np.sum(exp_preds)

        next_index = np.random.choice(len(probs), p=probs)
        generated.append(int_to_note[next_index])

        pattern.append(next_index)
        pattern = pattern[1:]

    # Convert the generated note/chord strings into a music21 Stream
    output_stream = stream.Stream()
    output_stream.append(tempo.MetronomeMark(number=bpm))

    instr_class = INSTRUMENTS.get(instrument_name, instrument.Piano)
    output_stream.append(instr_class())

    for item in generated:
        if "." in item:
            # It's a chord (multiple pitch classes)
            chord_notes = [note.Note(int(n)) for n in item.split(".")]
            c = chord.Chord(chord_notes)
            c.quarterLength = note_duration
            output_stream.append(c)
        elif item.isdigit():
            # Rare case: a single-pitch-class "chord" with no dot
            n = note.Note(int(item))
            n.quarterLength = note_duration
            output_stream.append(n)
        else:
            n = note.Note(item)
            n.quarterLength = note_duration
            output_stream.append(n)

    return output_stream, generated


def save_midi(output_stream, filepath):
    output_stream.write("midi", fp=filepath)
    return filepath

