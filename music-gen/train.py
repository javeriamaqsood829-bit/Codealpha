"""
train.py
Parses all MIDI files in data/, extracts note/chord sequences, and trains
an LSTM neural network to predict the next note in a sequence.
Saves the trained model + note mappings to model/ so app.py can generate
new music without retraining.
"""

import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow import keras
from tensorflow.keras import layers

SEQUENCE_LENGTH = 20
DATA_DIR = "data"
MODEL_DIR = "model"


def get_notes():
    """Extract notes and chords from every MIDI file in DATA_DIR."""
    notes = []
    midi_files = glob.glob(f"{DATA_DIR}/*.mid")
    print(f"Found {len(midi_files)} MIDI files to parse...")

    for file in midi_files:
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes

        for el in elements:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                notes.append('.'.join(str(n) for n in el.normalOrder))

    print(f"Extracted {len(notes)} notes/chords total.")
    return notes


def prepare_sequences(notes):
    """Convert note strings into integer-encoded training sequences."""
    pitch_names = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(len(notes) - SEQUENCE_LENGTH):
        seq_in = notes[i:i + SEQUENCE_LENGTH]
        seq_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_vocab = len(pitch_names)
    X = np.reshape(network_input, (len(network_input), SEQUENCE_LENGTH, 1))
    X = X / float(n_vocab)
    y = keras.utils.to_categorical(network_output, num_classes=n_vocab)

    return X, y, note_to_int, n_vocab


def build_model(n_vocab):
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(128),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_vocab, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def main():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    notes = get_notes()
    X, y, note_to_int, n_vocab = prepare_sequences(notes)
    int_to_note = {i: n for n, i in note_to_int.items()}

    print(f"Vocabulary size: {n_vocab} unique notes/chords")
    print(f"Training sequences: {len(X)}")

    model = build_model(n_vocab)
    model.summary()

    model.fit(X, y, epochs=60, batch_size=64, verbose=1)

    model.save(f"{MODEL_DIR}/music_model.keras")
    with open(f"{MODEL_DIR}/mappings.pkl", "wb") as f:
        pickle.dump({
            "note_to_int": note_to_int,
            "int_to_note": int_to_note,
            "n_vocab": n_vocab,
            "sequence_length": SEQUENCE_LENGTH
        }, f)

    print("\nTraining complete. Model saved to model/music_model.keras")


if __name__ == "__main__":
    main()
