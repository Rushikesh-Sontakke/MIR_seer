# main.py
#
# Quick inference entry point.
# Loads the trained model from seer_model.pth and author midi_array.txt
# (500 × 32 per song — must match training / evaluate_model.py).

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

# ==============================================================
# PATHS
# ==============================================================

DATA_PATH = os.path.join(ROOT_DIR, "data")
MODEL_PATH = os.path.join(ROOT_DIR, "seer_model.pth")
MIDI_ARRAY_PATH = os.path.join(DATA_PATH, "midi_array.txt")

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
# LOAD SONG SEQUENCES (author midi_array — same as training)
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

song_mapping = pd.read_csv(
    os.path.join(DATA_PATH, "song_to_number_matching.csv")
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
        song_to_track[parts[1]] = parts[0]

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

USER_ID = 0
K = 5

recommendations = recommend_top_k(
    model=model,
    user_id=USER_ID,
    song_sequences=song_sequences,
    device=DEVICE,
    k=K
)

print(f"\nTop {K} recommendations for user {USER_ID}:")
for rank, (track_id, score) in enumerate(recommendations, 1):
    print(f"  {rank}. {track_id}  (score: {score:.4f})")
