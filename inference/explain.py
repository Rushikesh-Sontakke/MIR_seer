# inference/explain.py
#
# Segment Forward Propagation (Algorithm 2 from the paper)
#
# Given a trained SeER model, a user, and a song's MIDI file, find the
# 10-second audio segment that best explains why the song was recommended
# to this user.  The segment with the highest predicted score is the one
# whose musical content most strongly matches the user's learned taste.
#
# Key differences from the previous implementation:
#   1. Uses real-time (seconds) for the sliding window, not raw timestep
#      counts.  The paper specifies a 10-second window with a 1-second
#      stride, and because MIDI tempo varies per song, a fixed number of
#      timesteps does NOT correspond to a fixed duration.
#   2. Passes only the segment to the GRU (no zero-padding to 2600).
#      Padding the segment with thousands of zero timesteps causes the
#      GRU hidden state to "wash out" the musical information before the
#      final hidden state is extracted.

import numpy as np
import torch
import pretty_midi


FEATURES = 32


def _midi_to_timed_sequence(midi_path):
    """Parse a MIDI file into a sequence of (onset_time, feature_row) pairs.

    Each note event produces one row of 32 features (16 channels × 2 for
    pitch and velocity), exactly matching the format used in preprocess.py.
    The rows are sorted by onset time so we can slice by real seconds.

    Returns
    -------
    onset_times : np.ndarray, shape (N,)
        The onset time in seconds for each row.
    sequence : np.ndarray, shape (N, 32)
        The feature matrix (same layout as the training data).
    """

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return np.array([]), np.zeros((0, FEATURES), dtype=np.float32)

    events = []  # list of (onset_seconds, feature_row)

    for instrument_idx, instrument in enumerate(midi.instruments[:16]):
        for note in instrument.notes:
            row = np.zeros(FEATURES, dtype=np.float32)
            channel = instrument_idx * 2
            row[channel] = note.pitch
            row[channel + 1] = note.velocity
            events.append((note.start, row))

    if len(events) == 0:
        return np.array([]), np.zeros((0, FEATURES), dtype=np.float32)

    # Sort by onset time
    events.sort(key=lambda e: e[0])

    onset_times = np.array([e[0] for e in events], dtype=np.float32)
    sequence = np.array([e[1] for e in events], dtype=np.float32)

    return onset_times, sequence


@torch.no_grad()
def generate_explanation(model,
                         user_id,
                         midi_path,
                         device,
                         window_seconds=10.0,
                         stride_seconds=1.0):
    """Find the best 10-second segment that explains the recommendation.

    Parameters
    ----------
    model : SeER
        A trained SeER model.
    user_id : int
        Encoded user index.
    midi_path : str
        Path to the original .mid / .midi file.
    device : str
        'cuda' or 'cpu'.
    window_seconds : float
        Duration of the explanation window (default 10 s per the paper).
    stride_seconds : float
        Stride of the sliding window (default 1 s per the paper).

    Returns
    -------
    best_segment : tuple (start_sec, end_sec)
        The start and end time (in seconds) of the best segment.
    best_score : float
        The model's predicted score for that segment.
    """

    model.eval()

    onset_times, sequence = _midi_to_timed_sequence(midi_path)

    if len(onset_times) == 0:
        return None, float('-inf')

    total_duration = float(onset_times[-1])

    best_score = float('-inf')
    best_segment = None

    user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)

    start_sec = 0.0
    while start_sec + window_seconds <= total_duration:

        end_sec = start_sec + window_seconds

        # Select note events whose onset falls within [start_sec, end_sec)
        mask = (onset_times >= start_sec) & (onset_times < end_sec)
        segment = sequence[mask]

        # Skip empty segments (silence / no notes in this window)
        if len(segment) == 0:
            start_sec += stride_seconds
            continue

        # Pass the segment directly to the GRU — no zero-padding.
        # The GRU's final hidden state will represent exactly this
        # musical content, without being diluted by trailing zeros.
        segment_tensor = (
            torch.tensor(segment, dtype=torch.float32)
            .unsqueeze(0)         # (1, num_notes_in_window, 32)
            .to(device)
        )

        score = model(user_tensor, segment_tensor).item()

        if score > best_score:
            best_score = score
            best_segment = (start_sec, end_sec)

        start_sec += stride_seconds

    return best_segment, best_score