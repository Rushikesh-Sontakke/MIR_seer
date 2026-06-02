# evaluate_model.py
#
# Standalone evaluation script for the trained SeER model.
#
# This script loads the SAVED encoder from training (data/encoder.pkl)
# to guarantee that user/song indices match the trained model weights.
# It then replicates the exact same 80/20 per-user split used during
# training (same seed, same logic) so the test set is truly unseen.
#
# Usage:
#   python evaluate_model.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER
from utils.encoders import EncoderManager
from inference.evaluate import evaluate_model

# ==============================================================
# PATHS
# ==============================================================

DATA_PATH = os.path.join(ROOT_DIR, "data")
TRIPLETS_PATH = os.path.join(DATA_PATH, "train_triplets.txt")
UNIQUE_TRACKS_PATH = os.path.join(DATA_PATH, "unique_tracks.txt")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
MODEL_PATH = os.path.join(ROOT_DIR, "seer_model.pth")
ENCODER_PATH = os.path.join(DATA_PATH, "encoder.pkl")

# ==============================================================
# CONFIG  (must match training/train.py)
# ==============================================================

TEST_RATIO = 0.2
K = 10
RANDOM_SEED = 42

# ==============================================================
# LOAD TRIPLETS
# ==============================================================

print("=" * 60)
print("SeER Model Evaluation")
print("=" * 60)

print("\n[1/8] Loading triplets...")

triplets = pd.read_csv(
    TRIPLETS_PATH,
    sep='\t',
    header=None,
    names=['user', 'song', 'play_count']
)

print(f"  Total interactions: {len(triplets)}")

# ==============================================================
# MAP SONG IDs -> TRACK IDs
# ==============================================================
# The triplets file uses Echo Nest Song IDs (SO...)
# but the MIDI/processed files use MSD Track IDs (TR...).
# unique_tracks.txt maps between them:
#   TrackID<SEP>SongID<SEP>Artist<SEP>Title

print("\n[2/8] Mapping Song IDs to Track IDs...")

song_to_track = {}

with open(UNIQUE_TRACKS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('<SEP>')
        if len(parts) >= 2:
            track_id, song_id = parts[0], parts[1]
            song_to_track[song_id] = track_id

print(f"  Loaded {len(song_to_track)} song-to-track mappings")

triplets['song'] = triplets['song'].map(song_to_track)

before = len(triplets)
triplets = triplets.dropna(subset=['song'])
print(f"  Mapped {len(triplets)}/{before} interactions to Track IDs")

# ==============================================================
# LOAD MIDI SEQUENCES
# ==============================================================

print("\n[3/8] Loading preprocessed MIDI sequences...")

song_sequences = {}

for npy_file in tqdm(os.listdir(PROCESSED_PATH), desc="  Loading"):
    if npy_file.endswith(".npy"):
        track_id = npy_file.replace(".npy", "")
        song_sequences[track_id] = np.load(
            os.path.join(PROCESSED_PATH, npy_file)
        )

print(f"  Loaded {len(song_sequences)} MIDI tensors")

# ==============================================================
# FILTER (same logic as training)
# ==============================================================

print("\n[4/8] Filtering data (same rules as training)...")

# Keep only songs that have MIDI
valid_song_ids = set(song_sequences.keys())
triplets = triplets[triplets['song'].isin(valid_song_ids)]
print(f"  After MIDI filter: {len(triplets)} interactions")

# Keep only active users (>= 20 songs)
user_counts = triplets.groupby('user')['song'].nunique()
active_users = user_counts[user_counts >= 20].index
triplets = triplets[triplets['user'].isin(active_users)]
print(f"  After user filter: {len(triplets)} interactions")
print(f"  Users: {triplets['user'].nunique()}")
print(f"  Songs: {triplets['song'].nunique()}")

# Convert play counts to ratings
def playcount_to_rating(x):
    if x <= 1:   return 1
    elif x <= 2: return 2
    elif x <= 5: return 3
    elif x <= 10: return 4
    else:        return 5

triplets['rating'] = triplets['play_count'].apply(playcount_to_rating)

# ==============================================================
# LOAD SAVED ENCODER
# ==============================================================

print("\n[5/8] Loading saved encoder...")

if not os.path.exists(ENCODER_PATH):
    print(f"  ERROR: Encoder not found at {ENCODER_PATH}")
    print("  Please run training first:  python -m training.train")
    sys.exit(1)

encoder_manager = EncoderManager.load(ENCODER_PATH)

# Use transform() (not fit_transform) to apply the SAME mapping
# that was learned during training.
triplets = encoder_manager.transform(triplets)

num_users = triplets['user_idx'].nunique()
print(f"  Loaded encoder ({num_users} users)")

# ==============================================================
# TRAIN / TEST SPLIT (same seed & logic as training)
# ==============================================================

print("\n[6/8] Splitting data (80/20 per user, same seed as training)...")

np.random.seed(RANDOM_SEED)

train_rows = []
test_interactions = defaultdict(list)  # user_idx -> [song_ids]
test_rows = []

for user_idx, group in triplets.groupby('user_idx'):

    group = group.sample(frac=1, random_state=RANDOM_SEED)

    n_test = max(1, int(len(group) * TEST_RATIO))

    test_part = group.iloc[:n_test]
    train_part = group.iloc[n_test:]

    train_rows.append(train_part)
    test_rows.append(test_part)

    for _, row in test_part.iterrows():
        test_interactions[user_idx].append(row['song'])

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df = pd.concat(test_rows).reset_index(drop=True)

print(f"  Train interactions: {len(train_df)}")
print(f"  Test interactions:  {len(test_df)}")
print(f"  Test users:         {len(test_interactions)}")

# ==============================================================
# LOAD MODEL
# ==============================================================

print("\n[7/8] Loading trained model...")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {DEVICE}")

# Infer num_users from saved weights (robust even if data changes)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
saved_num_users = state_dict['user_embedding.weight'].shape[0]

model = SeER(num_users=saved_num_users).to(DEVICE)
model.load_state_dict(state_dict)

print(f"  Loaded weights from {MODEL_PATH}")
print(f"  Model num_users: {saved_num_users}")

if saved_num_users != num_users:
    print(f"  WARNING: Model was trained with {saved_num_users} users "
          f"but current data has {num_users} users.")
    print("  This may indicate a data mismatch. Re-train recommended.")

# ==============================================================
# REGRESSION METRICS (RMSE, MAE on test set)
# ==============================================================

print("\n[8/8] Computing metrics...")

print("\n--- Regression Metrics (on test set) ---")

model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="  RMSE/MAE"):

        user_idx = int(row['user_idx'])
        song_id = row['song']
        true_rating = row['rating']

        seq = song_sequences.get(song_id)
        if seq is None:
            continue

        user_t = torch.tensor([user_idx], dtype=torch.long).to(DEVICE)
        seq_t = (
            torch.tensor(seq, dtype=torch.float32)
            .unsqueeze(0)
            .to(DEVICE)
        )

        pred = model(user_t, seq_t).item()

        all_preds.append(pred)
        all_true.append(true_rating)

all_preds = np.array(all_preds)
all_true = np.array(all_true)

rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
mae = np.mean(np.abs(all_preds - all_true))

print(f"\n  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

# ==============================================================
# RANKING METRICS (MAP@K, Precision@K, Recall@K, NDCG@K)
# ==============================================================

print(f"\n--- Ranking Metrics @ K={K} ---")

metrics, per_user = evaluate_model(
    model=model,
    test_interactions=test_interactions,
    song_sequences=song_sequences,
    device=DEVICE,
    k=K
)

print(f"\n{'=' * 40}")
print(f"  Precision@{K} : {metrics[f'precision@{K}']:.4f}")
print(f"  Recall@{K}    : {metrics[f'recall@{K}']:.4f}")
print(f"  MAP@{K}       : {metrics[f'map@{K}']:.4f}")
print(f"  NDCG@{K}      : {metrics[f'ndcg@{K}']:.4f}")
print(f"  Users evaluated: {metrics['num_users_evaluated']}")
print(f"{'=' * 40}")

print("\nEvaluation complete.")
