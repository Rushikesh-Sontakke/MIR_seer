import torch
import torch.nn as nn


class SeER(nn.Module):

    # Raw MIDI pitch/velocity values range 0–127.
    # Dividing by this constant normalizes inputs to [0, 1].
    INPUT_SCALE = 127.0

    def __init__(self,
                 num_users,
                 latent_dim=150,
                 input_size=32,
                 hidden_size=150):

        super().__init__()

        self.user_embedding = nn.Embedding(num_users, latent_dim)

        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.prediction = nn.Linear(latent_dim + hidden_size, 1)

    def encode_songs(self, song_sequences):
        """Run the LSTM on song sequences and return the song vectors."""
        normed = (song_sequences / self.INPUT_SCALE).float()
        normed = self.dropout(normed)

        # LSTM returns (output, (h_n, c_n))
        _, (hidden, _) = self.lstm(normed)

        return hidden.squeeze(0)

    def forward(self, user_ids, song_sequences):
        user_vec = self.user_embedding(user_ids)
        song_vec = self.encode_songs(song_sequences)

        # Concatenate user and song vectors (Author's architecture)
        concat = torch.cat([user_vec, song_vec], dim=1)

        # Dense layer to predict rating
        prediction = self.prediction(concat)

        return prediction