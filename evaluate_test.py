import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER
from inference.evaluate import evaluate_model

# ==============================================================
# PATHS
# ==============================================================

DATA_PATH = os.path.join(ROOT_DIR, "data")
TRIPLETS_PATH = os.path.join(DATA_PATH, "triplets.txt")
MIDI_ARRAY_PATH = os.path.join(DATA_PATH, "midi_array.txt")
MODEL_PATH = os.path.join(ROOT_DIR, "seer_model.pth")

# ==============================================================
# CONFIG (must match training)
# ==============================================================

SEQUENCE_LENGTH = 500
TEST_RATIO = 0.2
K = 10
RANDOM_SEED = 42
INTERACTION_THRESHOLD = 3  # moved here so it's available throughout

# ==============================================================
# LOAD DATA
# ==============================================================

print("=" * 60)
print("SeER Model Evaluation (Author's Data)")
print("=" * 60)

print("\n[1/6] Loading triplets...")

triplets = pd.read_csv(
    TRIPLETS_PATH,
    sep=" ",
    header=None,
    names=['user', 'song', 'rating']
)

num_users = triplets['user'].max() + 1
num_songs = triplets['song'].max() + 1

print(f"  Interactions: {len(triplets)}")
print(f"  Users: {num_users}")
print(f"  Songs: {num_songs}")

# ==============================================================
# LOAD MIDI ARRAY
# ==============================================================

print("\n[2/6] Loading MIDI array...")

with open(MIDI_ARRAY_PATH, 'r') as f:
    midi_array = json.load(f)

midi_array = np.array(midi_array, dtype=np.float32)

max_features = SEQUENCE_LENGTH * 32
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
# TRAIN / TEST SPLIT (same as training)
# ==============================================================

print("\n[3/6] Splitting data (80/20 per user, same seed as training)...")

np.random.seed(RANDOM_SEED)

test_interactions = defaultdict(list)  # user_idx -> [song_idxs]
test_users, test_songs, test_ratings = [], [], []

for user_idx, group in triplets.groupby('user'):

    group = group.sample(frac=1, random_state=RANDOM_SEED)

    n_test = max(1, int(len(group) * TEST_RATIO))

    test_part = group.iloc[:n_test]

    for _, row in test_part.iterrows():
        test_interactions[row['user']].append(row['song'])
        test_users.append(row['user'])
        test_songs.append(row['song'])
        test_ratings.append(row['rating'])

print(f"  Test interactions: {len(test_users)}")
print(f"  Test users: {len(test_interactions)}")

# ==============================================================
# LOAD MODEL
# ==============================================================

print("\n[4/6] Loading trained model...")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {DEVICE}")

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
saved_num_users = state_dict['user_embedding.weight'].shape[0]

model = SeER(num_users=saved_num_users).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"  Loaded from {MODEL_PATH}")
print(f"  Model num_users: {saved_num_users}")

# ==============================================================
# REGRESSION METRICS (RMSE, MAE)
# ==============================================================

print("\n[5/6] Computing regression metrics...")

all_preds = []
all_true = []

with torch.no_grad():
    for i in tqdm(range(len(test_users)), desc="  RMSE/MAE"):

        user_idx = test_users[i]
        song_idx = test_songs[i]
        true_rating = test_ratings[i]

        # Reshape song features for the GRU
        flat = midi_array[song_idx]
        sequence = flat.reshape(SEQUENCE_LENGTH, 32)

        user_t = torch.tensor([user_idx], dtype=torch.long).to(DEVICE)
        seq_t = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        pred = model(user_t, seq_t).item()

        all_preds.append(pred)
        all_true.append(true_rating)

all_preds = np.array(all_preds)
all_true = np.array(all_true)

rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
mae = np.mean(np.abs(all_preds - all_true))

print(f"\n  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

pred_df = pd.DataFrame({
    "user": test_users,
    "song": test_songs,
    "rating": all_preds,
    "true": all_true
})

pred_df["relevant"] = (
    pred_df["true"] >= INTERACTION_THRESHOLD
).astype(int)

pred_df["rank"] = (
    pred_df.groupby("user")["rating"]
    .rank(method="first", ascending=False)
)

pred_df["rank_true"] = (
    pred_df.groupby("user")["true"]
    .rank(method="first", ascending=False)
)

pred_df.sort_values(
    ["user", "rank"],
    inplace=True
)

print("\n========== DEBUG CHECK ==========")

# Total unique songs in dataset
all_songs_in_data = set(range(num_songs))
print(f"Total songs in dataset (num_songs): {num_songs}")

# Songs appearing in evaluation (pred_df)
songs_in_eval = set(pred_df["song"].unique())
print(f"Songs appearing in pred_df: {len(songs_in_eval)}")

# Coverage check
print(f"Coverage of dataset: {len(songs_in_eval) / num_songs * 100:.2f}%")

# Show difference
missing = all_songs_in_data - songs_in_eval
print(f"Songs NOT in evaluation: {len(missing)}")

print("Example evaluated songs:", list(songs_in_eval)[:10])
print("Example missing songs:", list(missing)[:10])

print("=================================\n")

# ==============================================================
print(f"\n[6/6] Computing ranking metrics (CORRECT full ranking)...")

# -------------------------------------------------
# Build train/test split properly for filtering
# -------------------------------------------------
train_interactions = defaultdict(set)
test_interactions_set = defaultdict(set)

for user_idx, group in triplets.groupby('user'):

    group = group.sample(frac=1, random_state=RANDOM_SEED)

    n_test = max(1, int(len(group) * TEST_RATIO))

    test_part = group.iloc[:n_test]
    train_part = group.iloc[n_test:]

    for _, row in train_part.iterrows():
        train_interactions[row['user']].add(row['song'])

    for _, row in test_part.iterrows():
        test_interactions_set[row['user']].add(row['song'])

# -------------------------------------------------
# FULL ranking evaluation
# -------------------------------------------------
AP_list = []
NDCG_list = []

users = list(test_interactions_set.keys())[:10]

with torch.no_grad():

    for user in tqdm(users, desc="Ranking Eval (10 users)"):

        train_songs = train_interactions[user]
        user_test_songs = test_interactions_set[user]

        candidates = []

        # score ALL songs (except train songs)
        for song in range(num_songs):

            if song in train_songs:
                continue

            flat = midi_array[song]
            seq = flat.reshape(SEQUENCE_LENGTH, 32)

            user_t = torch.tensor([user], dtype=torch.long).to(DEVICE)
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            score = model(user_t, seq_t).item()

            candidates.append((song, score))

        # sort by predicted score
        candidates.sort(key=lambda x: x[1], reverse=True)

        top_k = [s for s, _ in candidates[:K]]

        # relevance (test songs are positives)
        hits = [1 if s in user_test_songs else 0 for s in top_k]

        # -------------------------------------------------
        # MAP@K
        # -------------------------------------------------
        num_hits = 0
        precision_sum = 0

        for i, h in enumerate(hits):
            if h == 1:
                num_hits += 1
                precision_sum += num_hits / (i + 1)

        ap = precision_sum / max(1, len(user_test_songs))
        AP_list.append(ap)

        # -------------------------------------------------
        # NDCG@K
        # -------------------------------------------------
        dcg = 0.0
        for i, h in enumerate(hits):
            if h == 1:
                dcg += 1 / np.log2(i + 2)

        ideal_hits = min(len(user_test_songs), K)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

        ndcg = dcg / idcg if idcg > 0 else 0
        NDCG_list.append(ndcg)

# -------------------------------------------------
# FINAL RESULTS
# -------------------------------------------------
print(f"\n{'=' * 40}")
print(f"MAP@{K}  : {np.mean(AP_list):.4f}")
print(f"NDCG@{K} : {np.mean(NDCG_list):.4f}")
print(f"{'=' * 40}")