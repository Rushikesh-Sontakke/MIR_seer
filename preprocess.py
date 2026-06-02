# preprocess.py
import os
import numpy as np
import pretty_midi
from tqdm import tqdm

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
MIDI_PATH = os.path.join(BASE_PATH, "lmd_matched")
PROCESSED_PATH = os.path.join(BASE_PATH, "processed")

MAX_TIMESTEPS = 2600
FEATURES = 32

def midi_to_sequence(midi_path, max_timesteps=MAX_TIMESTEPS):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        return np.zeros((max_timesteps, FEATURES), dtype=np.float32)

    sequence = []

    # Collect all notes from all instruments with their onset times
    all_notes = []

    for instrument_idx, instrument in enumerate(midi.instruments[:16]):
        for note in instrument.notes:
            all_notes.append((note.start, instrument_idx, note))

    # Sort by onset time so the GRU sees chronological musical progression
    all_notes.sort(key=lambda x: x[0])

    for _, instrument_idx, note in all_notes:
        row = np.zeros(FEATURES)
        channel = instrument_idx * 2
        row[channel] = note.pitch
        row[channel + 1] = note.velocity
        sequence.append(row)

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) == 0:
        sequence = np.zeros((1, FEATURES), dtype=np.float32)

    if len(sequence) < max_timesteps:
        pad = np.zeros((max_timesteps - len(sequence), FEATURES))
        sequence = np.vstack([sequence, pad])
    else:
        sequence = sequence[:max_timesteps]

    return sequence


def main():
    # collect all midi files
    midi_files = []
    for root, dirs, files in os.walk(MIDI_PATH):
        for f in files:
            if f.endswith('.mid') or f.endswith('.midi'):
                midi_files.append(os.path.join(root, f))

    print(f"Found {len(midi_files)} MIDI files")
    print(f"Saving to {PROCESSED_PATH}")

    skipped = 0
    for midi_path in tqdm(midi_files):
        # use the track folder name as the key (MSD track ID)
        track_id = os.path.basename(os.path.dirname(midi_path))
        out_path = os.path.join(PROCESSED_PATH, track_id + ".npy")

        # skip if already processed (safe to re-run)
        if os.path.exists(out_path):
            skipped += 1
            continue

        sequence = midi_to_sequence(midi_path)
        np.save(out_path, sequence)

    print(f"Done. Skipped {skipped} already processed.")


if __name__ == "__main__":
    main()