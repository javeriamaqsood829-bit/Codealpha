"""
generate_dataset.py
Creates a small set of sample MIDI melodies (in different scales/moods)
to use as training data for the music generation LSTM model.

NOTE: For better results, replace/add real MIDI files (classical, jazz, etc.)
into the data/ folder. This script just creates a starter dataset so the
whole pipeline works out of the box.
"""

import random
from music21 import stream, note, chord, scale, tempo
import os

random.seed(42)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALES = {
    "major_C": scale.MajorScale('C'),
    "minor_A": scale.MinorScale('A'),
    "major_G": scale.MajorScale('G'),
    "minor_D": scale.MinorScale('D'),
    "major_F": scale.MajorScale('F'),
}

DURATIONS = [0.25, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0]


def generate_melody(scale_obj, length=120, name="melody"):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=100))
    pitches = scale_obj.getPitches(f'{scale_obj.tonic.name}3', f'{scale_obj.tonic.name}5')

    prev_index = len(pitches) // 2
    for _ in range(length):
        # Randomly walk up/down the scale (more musical than pure random notes)
        step = random.choice([-2, -1, -1, 0, 1, 1, 2])
        prev_index = max(0, min(len(pitches) - 1, prev_index + step))
        p = pitches[prev_index]
        dur = random.choice(DURATIONS)

        # occasionally add a chord (triad) for richness
        if random.random() < 0.15:
            third = pitches[min(prev_index + 2, len(pitches) - 1)]
            fifth = pitches[min(prev_index + 4, len(pitches) - 1)]
            c = chord.Chord([p, third, fifth])
            c.quarterLength = dur
            s.append(c)
        else:
            n = note.Note(p)
            n.quarterLength = dur
            s.append(n)

    filepath = os.path.join(OUTPUT_DIR, f"{name}.mid")
    s.write('midi', fp=filepath)
    print(f"Created {filepath}")


if __name__ == "__main__":
    i = 0
    for name, sc in SCALES.items():
        for variant in range(3):
            generate_melody(sc, length=150, name=f"{name}_{variant}")
            i += 1
    print(f"\nGenerated {i} sample training MIDI files in '{OUTPUT_DIR}/'")
    print("Tip: add your own real MIDI files to this folder for richer training data.")
