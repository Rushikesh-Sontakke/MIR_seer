# main.py
#
# Inference entry point for the SeER music recommender.
# Loads the trained model and recommends songs with full metadata.
#
# Usage:
#   python main.py
#   python main.py --user 42 --top-k 10
#   python main.py --model seer_model2.pth --user 0

import argparse
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER
from inference.recommend import recommend_top_k

# Must match training/train.py and evaluate_model.py
SEQUENCE_LENGTH = 500
FEATURES = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="SeER: generate song recommendations with metadata"
    )
    parser.add_argument(
        "--user", type=int, default=0,
        help="User index to generate recommendations for (default: 0)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top recommendations (default: 5)"
    )
    parser.add_argument(
        "--model", type=str, default="seer_model.pth",
        help="Model filename in project root (default: seer_model.pth)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    DATA_PATH = os.path.join(ROOT_DIR, "data")
    MODEL_PATH = os.path.join(ROOT_DIR, args.model)
    MIDI_ARRAY_PATH = os.path.join(DATA_PATH, "midi_array.txt")

    # ==============================================================
    # LOAD MODEL
    # ==============================================================

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    num_users = state_dict['user_embedding.weight'].shape[0]

    if not (0 <= args.user < num_users):
        print(f"Error: user must be in [0, {num_users - 1}], got {args.user}")
        sys.exit(1)

    model = SeER(num_users=num_users).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from {args.model} ({num_users} users, device={DEVICE})")

    # ==============================================================
    # LOAD SONG SEQUENCES
    # ==============================================================

    print("Loading song sequences from midi_array.txt...")

    with open(MIDI_ARRAY_PATH, 'r') as f:
        midi_array = json.load(f)

    midi_array = np.array(midi_array, dtype=np.float32)

    max_features = SEQUENCE_LENGTH * FEATURES
    if midi_array.shape[1] > max_features:
        midi_array = midi_array[:, :max_features]
    elif midi_array.shape[1] < max_features:
        pad = np.zeros(
            (midi_array.shape[0], max_features - midi_array.shape[1]),
            dtype=np.float32
        )
        midi_array = np.hstack([midi_array, pad])

    print(f"  MIDI array shape: {midi_array.shape}")

    # ==============================================================
    # BUILD LOOKUP TABLES
    # ==============================================================

    song_mapping = pd.read_csv(
        os.path.join(DATA_PATH, "song_to_number_matching.csv")
    )

    # track_id <-> song_id mapping (from unique_tracks.txt)
    track_to_song = {}
    with open(
        os.path.join(DATA_PATH, "unique_tracks.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            parts = line.strip().split("<SEP>")
            if len(parts) >= 2:
                track_to_song[parts[0]] = parts[1]  # track_id -> song_id

    song_to_track = {v: k for k, v in track_to_song.items()}

    # Song metadata
    song_info_path = os.path.join(DATA_PATH, "song_information.csv")
    if os.path.exists(song_info_path):
        song_info = pd.read_csv(song_info_path)[
            ["song_id", "artist_name", "title", "release", "year", "duration"]
        ]
        song_info_dict = song_info.set_index("song_id").to_dict("index")
    else:
        song_info_dict = {}

    # Build song sequences keyed by track_id
    song_sequences = {}

    for _, row in tqdm(song_mapping.iterrows(), total=len(song_mapping)):
        song_idx = int(row["number"])
        song_id = row["song_id"]

        if song_id not in song_to_track:
            continue

        track_id = song_to_track[song_id]
        flat = midi_array[song_idx]
        song_sequences[track_id] = flat.reshape(SEQUENCE_LENGTH, FEATURES)

    print(f"Loaded {len(song_sequences)} songs (500 timesteps each)")

    # ==============================================================
    # RECOMMEND
    # ==============================================================

    recommendations = recommend_top_k(
        model=model,
        user_id=args.user,
        song_sequences=song_sequences,
        device=DEVICE,
        k=args.top_k
    )

    # ==============================================================
    # DISPLAY RESULTS WITH METADATA
    # ==============================================================

    print(f"\n{'=' * 70}")
    print(f"  Top {args.top_k} recommendations for user {args.user}")
    print(f"{'=' * 70}\n")

    for rank, (track_id, score) in enumerate(recommendations, 1):

        # Resolve track_id -> song_id -> metadata
        song_id = track_to_song.get(track_id, None)
        meta = song_info_dict.get(song_id, {}) if song_id else {}

        artist = meta.get("artist_name", "Unknown Artist")
        title = meta.get("title", "Unknown Title")
        release = meta.get("release", "Unknown Album")
        year = meta.get("year", 0)
        duration = meta.get("duration", 0)

        year_str = str(int(year)) if year and year > 0 else "N/A"
        dur_min = int(duration // 60) if duration else 0
        dur_sec = int(duration % 60) if duration else 0

        print(f"  {rank}. {title}")
        print(f"     Artist:  {artist}")
        print(f"     Album:   {release} ({year_str})")
        print(f"     Duration:{dur_min}:{dur_sec:02d}")
        print(f"     Score:   {score:.4f}")
        print(f"     Track:   {track_id}")
        print()


if __name__ == "__main__":
    main()
