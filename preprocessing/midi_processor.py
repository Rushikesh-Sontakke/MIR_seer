import numpy as np
import pretty_midi

MAX_TIMESTEPS = 2600
FEATURES = 32


def midi_to_sequence(midi_path, max_timesteps=2600):

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        return np.zeros((max_timesteps, FEATURES), dtype=np.float32)

    sequence = []

    for instrument_idx, instrument in enumerate(midi.instruments[:16]):

        for note in instrument.notes:

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