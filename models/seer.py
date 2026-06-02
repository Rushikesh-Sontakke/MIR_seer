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

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

    def encode_songs(self, song_sequences):
        """Run the GRU on song sequences and return the song vectors.

        Normalizes inputs before passing through the GRU.
        Use this instead of calling self.gru directly.
        """

        normed = (song_sequences / self.INPUT_SCALE).float()

        _, hidden = self.gru(normed)

        return hidden.squeeze(0)

    def forward(self, user_ids, song_sequences):

        user_vec = self.user_embedding(user_ids)

        song_vec = self.encode_songs(song_sequences)

        prediction = torch.sum(user_vec * song_vec, dim=1)

        return prediction