# training/train.py
#
# Trains the SeER model using the AUTHOR'S preprocessed data
# (triplets.txt + midi_array.txt) from the SeER_Keras repository.
#
# This avoids the need for raw MIDI preprocessing and ensures
# results are directly comparable to the paper.
#
# Usage:
#   python -m training.train

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER


# ==============================================================
# CONSTANTS
# ==============================================================

DATA_PATH = os.path.join(ROOT_DIR, "data")

TRIPLETS_PATH = os.path.join(DATA_PATH, "triplets.txt")
MIDI_ARRAY_PATH = os.path.join(DATA_PATH, "midi_array.txt")

MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "seer_model.pth")
MODEL_INFO_PATH = os.path.join(ROOT_DIR, "seer_model_info.json")

# ==============================================================
# HYPERPARAMETERS
# ==============================================================

SEQUENCE_LENGTH = 500       # median MIDI length from the paper
BATCH_SIZE = 1000              # 500 in the paper
LEARNING_RATE =  1e-3
EPOCHS = 15
TEST_RATIO = 0.2
RANDOM_SEED = 42


# ==============================================================
# DATASET
# ==============================================================

class SeERDataset(Dataset):

    def __init__(self, users, songs, ratings, midi_array, sequence_length):

        self.users = users
        self.songs = songs
        self.ratings = ratings
        self.midi_array = midi_array
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        user_idx = self.users[idx]
        song_idx = self.songs[idx]
        rating = self.ratings[idx]

        # The midi_array is flattened: (num_songs, seq_len * 32)
        # Reshape the song's row into (seq_len, 32)
        flat_features = self.midi_array[song_idx]
        sequence = flat_features.reshape(self.seq_len, 32)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(rating, dtype=torch.float32)
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    # ----------------------------------------------------------
    # LOAD TRIPLETS (author's preprocessed format)
    # ----------------------------------------------------------

    print("=" * 60)
    print("SeER Training (Author's Data)")
    print("=" * 60)

    print("\n[1/5] Loading triplets...")

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

    # ----------------------------------------------------------
    # LOAD MIDI ARRAY (author's preprocessed format)
    # ----------------------------------------------------------

    print("\n[2/5] Loading MIDI array...")

    with open(MIDI_ARRAY_PATH, 'r') as f:
        midi_array = json.load(f)

    midi_array = np.array(midi_array, dtype=np.float32)

    # Truncate to sequence_length * 32 features
    max_features = SEQUENCE_LENGTH * 32
    if midi_array.shape[1] > max_features:
        midi_array = midi_array[:, :max_features]
    elif midi_array.shape[1] < max_features:
        # Pad if shorter
        pad = np.zeros(
            (midi_array.shape[0], max_features - midi_array.shape[1]),
            dtype=np.float32
        )
        midi_array = np.hstack([midi_array, pad])

    print(f"  MIDI array shape: {midi_array.shape}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")

    # ----------------------------------------------------------
    # TRAIN / TEST SPLIT (per user)
    # ----------------------------------------------------------

    print("\n[3/5] Splitting data (80/20 per user)...")

    np.random.seed(RANDOM_SEED)

    train_users, train_songs, train_ratings = [], [], []
    test_users, test_songs, test_ratings = [], [], []

    for user_idx, group in triplets.groupby('user'):

        group = group.sample(frac=1, random_state=RANDOM_SEED)

        n_test = max(1, int(len(group) * TEST_RATIO))

        test_part = group.iloc[:n_test]
        train_part = group.iloc[n_test:]

        train_users.extend(train_part['user'].tolist())
        train_songs.extend(train_part['song'].tolist())
        train_ratings.extend(train_part['rating'].tolist())

        test_users.extend(test_part['user'].tolist())
        test_songs.extend(test_part['song'].tolist())
        test_ratings.extend(test_part['rating'].tolist())

    print(f"  Train: {len(train_users)} interactions")
    print(f"  Test:  {len(test_users)} interactions (held out)")

    # ----------------------------------------------------------
    # DATASET & LOADER
    # ----------------------------------------------------------

    print("\n[4/5] Creating dataset...")

    train_dataset = SeERDataset(
        train_users, train_songs, train_ratings,
        midi_array, SEQUENCE_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_dataset = SeERDataset(
        test_users, test_songs, test_ratings,
        midi_array, SEQUENCE_LENGTH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ----------------------------------------------------------
    # DEVICE & MODEL
    # ----------------------------------------------------------

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {DEVICE}")

    print("\n[5/5] Training model...")

    model = SeER(
        num_users=num_users
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ----------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------

    current_run_best_loss = float('inf')
    global_best_loss = float('inf')

    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, 'r') as f:
                info = json.load(f)
                global_best_loss = info.get('best_val_loss', float('inf'))
            print(f"\n[!] Found previous best model with Val Loss: {global_best_loss:.4f}")
            print(f"[!] Will only overwrite {MODEL_SAVE_PATH} if we beat this score.\n")
        except:
            pass

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for users, sequences, ratings in progress_bar:

            users = users.to(DEVICE)
            sequences = sequences.to(DEVICE)
            ratings = ratings.to(DEVICE)

            # Forward
            predictions = model(users, sequences)

            loss = criterion(
                predictions.squeeze(),
                ratings
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients (standard for RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches

        print(
            f"\nEpoch {epoch + 1} "
            f"Average Train Loss: {avg_loss:.4f}"
        )

        # ----------------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------------
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for users, sequences, ratings in test_loader:
                users = users.to(DEVICE)
                sequences = sequences.to(DEVICE)
                ratings = ratings.to(DEVICE)

                predictions = model(users, sequences)
                loss = criterion(predictions.squeeze(), ratings)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        print(f"Epoch {epoch + 1} Average Val Loss: {avg_val_loss:.4f}")

        # Track best for current run
        if avg_val_loss < current_run_best_loss:
            current_run_best_loss = avg_val_loss
            
            # Compare to global best across all runs
            if avg_val_loss < global_best_loss:
                global_best_loss = avg_val_loss
                print(f"  -> 🌟 NEW GLOBAL BEST! Overwriting {MODEL_SAVE_PATH}\n")
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                with open(MODEL_INFO_PATH, 'w') as f:
                    json.dump({"best_val_loss": global_best_loss}, f)
            else:
                print(f"  -> Best this run, but previous run was better ({global_best_loss:.4f}). Not overwriting.\n")
        else:
            print("")

    print("\nTraining complete.")
    print(f"Best Val Loss this run: {current_run_best_loss:.4f}")
    print(f"All-time Best Val Loss: {global_best_loss:.4f}")

    return model


if __name__ == "__main__":
    main()  