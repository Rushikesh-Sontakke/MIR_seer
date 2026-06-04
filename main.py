# main.py
#
# Quick inference entry point.
# Loads the trained model from seer_model.pth and the saved encoder,
# then generates top-K recommendations for a given user.

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER
from utils.encoders import EncoderManager
from inference.recommend import recommend_top_k

# ==============================================================
# PATHS
# ==============================================================

DATA_PATH = os.path.join(ROOT_DIR, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
MODEL_PATH = os.path.join(ROOT_DIR, "seer_model.pth")
ENCODER_PATH = os.path.join(DATA_PATH, "encoder.pkl")

# ==============================================================
# LOAD MODEL
# ==============================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
num_users = state_dict['user_embedding.weight'].shape[0]

model = SeER(num_users=num_users).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded ({num_users} users, device={DEVICE})")

# ==============================================================
# LOAD SONG SEQUENCES
# ==============================================================

print("Loading song sequences...")

song_sequences = {}

for npy_file in tqdm(os.listdir(PROCESSED_PATH)):

    if npy_file.endswith(".npy"):

        track_id = npy_file.replace(".npy", "")

        song_sequences[track_id] = np.load(
            os.path.join(PROCESSED_PATH, npy_file)
        )

print(f"Loaded {len(song_sequences)} songs")

# ==============================================================
# FILTER TO THE 6442-SONG TRAINING CATALOG
# ==============================================================

print("\nLoading training song catalog...")

song_mapping = pd.read_csv(
    os.path.join(DATA_PATH, "song_to_number_matching.csv")
)

allowed_song_ids = set(song_mapping["song_id"])

print(
    f"Songs in training catalog: "
    f"{len(allowed_song_ids)}"
)

song_to_track = {}

with open(
    os.path.join(DATA_PATH, "unique_tracks.txt"),
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        parts = line.strip().split("<SEP>")

        if len(parts) < 2:
            continue

        track_id = parts[0]
        song_id = parts[1]

        song_to_track[song_id] = track_id

allowed_track_ids = set()

for song_id in allowed_song_ids:

    if song_id in song_to_track:
        allowed_track_ids.add(
            song_to_track[song_id]
        )

print(
    f"Matched training tracks: "
    f"{len(allowed_track_ids)}"
)

song_sequences = {
    track_id: sequence
    for track_id, sequence in song_sequences.items()
    if track_id in allowed_track_ids
}

print(
    f"Filtered catalog size: "
    f"{len(song_sequences)} songs"
)
# ==============================================================
# RECOMMEND
# ==============================================================

recommendations = recommend_top_k(
    model=model,
    user_id=0,
    song_sequences=song_sequences,
    device=DEVICE,
    k=5
)

print("\nTop 5 recommendations for user 0:")
for rank, (song_id, score) in enumerate(recommendations, 1):
    print(f"  {rank}. {song_id}  (score: {score:.4f})")