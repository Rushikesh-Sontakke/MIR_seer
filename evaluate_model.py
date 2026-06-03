# evaluate_model.py
#
# Evaluation script for the SeER model trained on the author's data.
#
# Usage:
#   python evaluate_model.py

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

# # ==============================================================
# # RANKING METRICS
# # ==============================================================

# print(f"\n[6/6] Computing ranking metrics @ K={K}...")

# # Build song_sequences dict for the evaluate_model function
# # Only include songs that appear in the interaction data
# interaction_songs = set(triplets['song'].unique())
# song_sequences = {}
# for song_idx in interaction_songs:
#     flat = midi_array[int(song_idx)]
#     song_sequences[song_idx] = flat.reshape(SEQUENCE_LENGTH, 32)

# print(f"  Candidate pool: {len(song_sequences)} songs")

# metrics, per_user = evaluate_model(
#     model=model,
#     test_interactions=test_interactions,
#     song_sequences=song_sequences,
#     device=DEVICE,
#     k=K
# )

# print(f"\n{'=' * 40}")
# print(f"  Precision@{K} : {metrics[f'precision@{K}']:.4f}")
# print(f"  Recall@{K}    : {metrics[f'recall@{K}']:.4f}")
# print(f"  MAP@{K}       : {metrics[f'map@{K}']:.4f}")
# print(f"  NDCG@{K}      : {metrics[f'ndcg@{K}']:.4f}")
# print(f"  Users evaluated: {metrics['num_users_evaluated']}")
# print(f"{'=' * 40}")

# print("\nEvaluation complete.")
# ==============================================================
# AUTHOR-STYLE RANKING METRICS
# ==============================================================

print(f"\n[6/6] Computing ranking metrics (author style)...")

INTERACTION_THRESHOLD = 3

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

# --------------------------------------------------------------
# MAP@K
# --------------------------------------------------------------

def map_at_k(pred_df, k=10):

    AP = 0.0

    for user in pred_df["user"].unique():

        user_df = pred_df[pred_df["user"] == user]

        top_items = user_df["relevant"].values[:k]

        p_list = []

        for j in range(1, len(top_items) + 1):

            prefix = top_items[:j]

            precision_j = np.sum(prefix) / len(prefix)

            p_list.append(precision_j)

        p_list = np.array(p_list)

        sum_val = np.sum(p_list * top_items)

        num_relevant = np.sum(user_df["relevant"])

        if num_relevant > 0:
            AP += sum_val / num_relevant

    return AP / pred_df["user"].nunique()


# --------------------------------------------------------------
# NDCG@K
# --------------------------------------------------------------

def ndcg_at_k(pred_df, k=10):

    topk_true = pred_df[
        pred_df["rank_true"] <= k
    ].copy()

    topk_true["idcg_unit"] = topk_true[
        "rank_true"
    ].apply(
        lambda x: np.log(2) / np.log(1 + x)
    )

    topk_true["idcg"] = (
        topk_true.groupby("user")["idcg_unit"]
        .transform("sum")
    )

    test_topk = topk_true[
        topk_true["rank"] <= k
    ].copy()

    test_topk["dcg_unit"] = test_topk[
        "rank"
    ].apply(
        lambda x: np.log(2) / np.log(1 + x)
    )

    test_topk["dcg"] = (
        test_topk.groupby("user")["dcg_unit"]
        .transform("sum")
    )

    test_topk["ndcg"] = (
        test_topk["dcg"] /
        test_topk["idcg"]
    )

    return (
        np.sum(
            test_topk.groupby("user")["ndcg"].max()
        )
        / pred_df["user"].nunique()
    )


map10 = map_at_k(pred_df, K)
ndcg10 = ndcg_at_k(pred_df, K)

print(f"\n{'=' * 40}")
print(f"  MAP@{K}  : {map10:.4f}")
print(f"  NDCG@{K} : {ndcg10:.4f}")
print(f"{'=' * 40}")