import torch
from torch.utils.data import Dataset
import numpy as np


class SeERDataset(Dataset):

    def __init__(self, dataframe, song_sequences):

        self.df = dataframe
        self.song_sequences = song_sequences

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        user = row['user_idx']
        song = row['song_idx']
        rating = row['rating']

        sequence = self.song_sequences.get(
            song,
            np.zeros((2600, 32), dtype=np.float32)
        )

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(rating, dtype=torch.float32)
        )