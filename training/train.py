# training/train.py
#
# Trains the SeER model on the Lakh MIDI + Echo Nest triplets data.
#
# Key changes from the original version:
#   1. All logic is wrapped in a main() function behind an
#      `if __name__ == "__main__"` guard, so importing from this
#      module no longer triggers the full training pipeline.
#   2. An 80/20 per-user train/test split is performed BEFORE
#      training, so the model never sees the held-out test data.
#   3. The fitted EncoderManager is saved to data/encoder.pkl
#      alongside the model weights, ensuring the evaluation script
#      uses the exact same user/song index mapping.
#
# Usage:
#   python -m training.train          (from project root)
#   python training/train.py          (also works)

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER
from utils.encoders import EncoderManager


# ==============================================================
# CONSTANTS
# ==============================================================

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.abspath(BASE_PATH)

TRIPLETS_PATH = os.path.join(DATA_PATH, "train_triplets.txt")
UNIQUE_TRACKS_PATH = os.path.join(DATA_PATH, "unique_tracks.txt")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
ENCODER_PATH = os.path.join(DATA_PATH, "encoder.pkl")

MODEL_SAVE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "seer_model.pth")
)

# ==============================================================
# HYPERPARAMETERS
# ==============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
TEST_RATIO = 0.2
RANDOM_SEED = 42


# ==============================================================
# RATING CONVERSION
# ==============================================================

def playcount_to_rating(x):

    if x <= 1:
        return 1

    elif x <= 2:
        return 2

    elif x <= 5:
        return 3

    elif x <= 10:
        return 4

    else:
        return 5


# ==============================================================
# DATASET
# ==============================================================

class SeERDataset(Dataset):

    def __init__(self, dataframe, song_sequences):

        self.df = dataframe.reset_index(drop=True)

        self.song_sequences = song_sequences

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        user_idx = row['user_idx']

        track_id = row['song']

        rating = row['rating']

        sequence = self.song_sequences[track_id]

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
    # LOAD TRIPLETS
    # ----------------------------------------------------------

    print("=" * 60)
    print("SeER Training")
    print("=" * 60)

    print("\n[1/7] Loading triplets...")

    triplets = pd.read_csv(
        TRIPLETS_PATH,
        sep='\t',
        header=None,
        names=['user', 'song', 'play_count']
    )

    # Optional smaller subset for faster experiments
    # triplets = triplets.sample(500000, random_state=42)

    print(f"  Loaded {len(triplets)} interactions")

    # ----------------------------------------------------------
    # MAP SONG IDs -> TRACK IDs
    # ----------------------------------------------------------
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

    # Map the 'song' column from Song IDs to Track IDs
    triplets['song'] = triplets['song'].map(song_to_track)

    # Drop rows where the song had no track mapping
    before = len(triplets)
    triplets = triplets.dropna(subset=['song'])
    print(f"  Mapped {len(triplets)}/{before} interactions to Track IDs")

    # ----------------------------------------------------------
    # LOAD MIDI SEQUENCES
    # ----------------------------------------------------------

    print("\n[3/8] Loading preprocessed MIDI sequences...")

    song_sequences = {}

    for npy_file in tqdm(os.listdir(PROCESSED_PATH)):

        if npy_file.endswith(".npy"):

            track_id = npy_file.replace(".npy", "")

            sequence = np.load(
                os.path.join(PROCESSED_PATH, npy_file)
            )

            song_sequences[track_id] = sequence

    print(f"  Loaded {len(song_sequences)} MIDI tensors")

    # ----------------------------------------------------------
    # INTERSECTION
    # ----------------------------------------------------------

    print("\n[4/8] Filtering interactions to songs with MIDI...")

    valid_song_ids = set(song_sequences.keys())

    triplets = triplets[
        triplets['song'].isin(valid_song_ids)
    ]

    print(f"  Remaining interactions: {len(triplets)}")
    print(f"  Remaining songs: {triplets['song'].nunique()}")

    # ----------------------------------------------------------
    # FILTER SPARSE USERS
    # ----------------------------------------------------------

    print("\n[5/8] Filtering inactive users (<20 songs)...")

    user_counts = (
        triplets.groupby('user')['song']
        .nunique()
    )

    active_users = user_counts[
        user_counts >= 20
    ].index

    triplets = triplets[
        triplets['user'].isin(active_users)
    ]

    print(f"  Remaining users: {triplets['user'].nunique()}")
    print(f"  Remaining interactions: {len(triplets)}")

    # ----------------------------------------------------------
    # PLAY COUNT -> RATING
    # ----------------------------------------------------------

    print("\n  Converting play counts to ratings...")

    triplets['rating'] = triplets['play_count'].apply(
        playcount_to_rating
    )

    # ----------------------------------------------------------
    # ENCODE USERS & SONGS
    # ----------------------------------------------------------

    print("\n[6/8] Encoding users & songs...")

    encoder_manager = EncoderManager()

    triplets = encoder_manager.fit(triplets)

    num_users = triplets['user_idx'].nunique()

    print(f"  Encoded users: {num_users}")

    # Save encoder for evaluation
    encoder_manager.save(ENCODER_PATH)
    print(f"  Encoder saved to {ENCODER_PATH}")

    # ----------------------------------------------------------
    # TRAIN / TEST SPLIT  (per user, 80/20)
    # ----------------------------------------------------------

    print("\n[7/8] Splitting data (80/20 per user)...")

    np.random.seed(RANDOM_SEED)

    train_rows = []
    test_rows = []

    for user_idx, group in triplets.groupby('user_idx'):

        group = group.sample(frac=1, random_state=RANDOM_SEED)

        n_test = max(1, int(len(group) * TEST_RATIO))

        test_part = group.iloc[:n_test]
        train_part = group.iloc[n_test:]

        train_rows.append(train_part)
        test_rows.append(test_part)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    print(f"  Train interactions: {len(train_df)}")
    print(f"  Test interactions:  {len(test_df)} (held out)")

    # ----------------------------------------------------------
    # DATASET & LOADER  (train split only)
    # ----------------------------------------------------------

    print("\n  Creating dataset from TRAIN split only...")

    train_dataset = SeERDataset(
        train_df,
        song_sequences
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # ----------------------------------------------------------
    # DEVICE & MODEL
    # ----------------------------------------------------------

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n  Using device: {DEVICE}")

    print("\n[8/8] Training model...")

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

            # Clip gradients to prevent exploding gradients (common with RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            num_batches += 1

            progress_bar.set_postfix(
                loss=loss.item()
            )

        avg_loss = total_loss / num_batches

        print(
            f"\nEpoch {epoch + 1} "
            f"Average Loss: {avg_loss:.4f}\n"
        )

    # ----------------------------------------------------------
    # SAVE MODEL
    # ----------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Encoder saved to: {ENCODER_PATH}")

    return model


if __name__ == "__main__":
    main()