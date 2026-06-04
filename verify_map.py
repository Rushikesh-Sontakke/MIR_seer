"""
Verify MAP@10 score by re-implementing both evaluation methods and comparing.

Method A: Author-style (your current code) - ranks only within the test set per user
Method B: Full-catalog (standard IR) - ranks all 6442 candidate songs per user
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

from models.seer import SeER

DATA_PATH = os.path.join(ROOT_DIR, "data")
SEQUENCE_LENGTH = 500
TEST_RATIO = 0.2
RANDOM_SEED = 42
K = 10

# ---- Load data ----
print("Loading data...")
triplets = pd.read_csv(os.path.join(DATA_PATH, "triplets.txt"), sep=" ", header=None, names=['user', 'song', 'rating'])
with open(os.path.join(DATA_PATH, "midi_array.txt"), 'r') as f:
    midi_array = json.load(f)
midi_array = np.array(midi_array, dtype=np.float32)
max_features = SEQUENCE_LENGTH * 32
if midi_array.shape[1] > max_features:
    midi_array = midi_array[:, :max_features]
elif midi_array.shape[1] < max_features:
    pad = np.zeros((midi_array.shape[0], max_features - midi_array.shape[1]), dtype=np.float32)
    midi_array = np.hstack([midi_array, pad])

# ---- Split ----
print("Splitting...")
np.random.seed(RANDOM_SEED)
train_songs_per_user = defaultdict(set)
test_interactions = defaultdict(list)

for user_idx, group in triplets.groupby('user'):
    group = group.sample(frac=1, random_state=RANDOM_SEED)
    n_test = max(1, int(len(group) * TEST_RATIO))
    test_part = group.iloc[:n_test]
    train_part = group.iloc[n_test:]
    for _, row in train_part.iterrows():
        train_songs_per_user[row['user']].add(row['song'])
    for _, row in test_part.iterrows():
        test_interactions[row['user']].append((row['song'], row['rating']))

# ---- Load model ----
print("Loading model...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load(os.path.join(ROOT_DIR, "seer_model.pth"), map_location=DEVICE, weights_only=True)
model = SeER(num_users=state_dict['user_embedding.weight'].shape[0]).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

# ---- METHOD A: Author-style (rank within test set only) ----
print("\n" + "=" * 60)
print("METHOD A: Rank within test set per user (your current code)")
print("=" * 60)

THRESHOLD = 3
aps_a = []
sample_users_a = []

with torch.no_grad():
    for user_idx in tqdm(list(test_interactions.keys())[:32180], desc="Method A"):
        items = test_interactions[user_idx]
        if len(items) == 0:
            continue
        
        # Score each test item
        scored = []
        for song_idx, true_rating in items:
            flat = midi_array[song_idx]
            seq = flat.reshape(SEQUENCE_LENGTH, 32)
            u_t = torch.tensor([user_idx], dtype=torch.long).to(DEVICE)
            s_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(u_t, s_t).item()
            scored.append((song_idx, pred, true_rating))
        
        # Sort by predicted score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Compute AP@K with threshold
        relevant_flags = [1 if s[2] >= THRESHOLD else 0 for s in scored]
        num_relevant = sum(relevant_flags)
        
        if num_relevant == 0:
            # Your code adds 0 to AP when num_relevant > 0, skips otherwise
            # But it still divides by total users at the end
            aps_a.append(0.0)
            continue
        
        top_k = relevant_flags[:K]
        ap = 0.0
        hits = 0
        for j in range(len(top_k)):
            if top_k[j] == 1:
                hits += 1
                ap += hits / (j + 1)
        ap /= num_relevant
        aps_a.append(ap)
        
        if user_idx < 5:
            sample_users_a.append({
                'user': user_idx,
                'test_items': len(items),
                'relevant (>=3)': num_relevant,
                'top_k_flags': top_k,
                'ap': ap
            })

map_a = np.mean(aps_a)
print(f"\nMAP@{K} (Method A): {map_a:.4f}")
print(f"Users evaluated: {len(aps_a)}")

print("\nSample users:")
for s in sample_users_a:
    print(f"  User {s['user']}: {s['test_items']} test items, {s['relevant (>=3)']} relevant, "
          f"top-{K} flags={s['top_k_flags']}, AP={s['ap']:.4f}")
