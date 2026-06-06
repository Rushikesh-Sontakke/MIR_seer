import torch
import torch.nn as nn


class SeER(nn.Module):

    def __init__(self,
                 num_users,
                 latent_dim=150,
                 input_size=32,       # matches actual midi_array feature width
                 hidden_size=150):

        super().__init__()

        self.user_embedding = nn.Embedding(num_users, latent_dim)

        # Matches Keras: dropout=0.2, recurrent_dropout=0.2 on the LSTM itself
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.0           # only applies to stacked LSTMs (num_layers > 1)
        )

        # PyTorch's single-layer LSTM has no built-in recurrent_dropout.
        # VariationalDropout applies the same mask across all timesteps,
        # which is the correct equivalent of Keras recurrent_dropout.
        self.recurrent_dropout = nn.Dropout(p=0.2)

        self.prediction = nn.Linear(latent_dim + hidden_size, 1)

    def encode_songs(self, song_sequences):
        """Run the LSTM on song sequences and return the final hidden state."""

        # No input normalisation — original Keras model had none
        x = song_sequences.float()

        # Apply variational (recurrent-style) dropout to the input stream.
        # In training mode this drops the same features every timestep,
        # matching Keras recurrent_dropout behaviour.
        x = self.recurrent_dropout(x)

        # LSTM returns (output, (h_n, c_n))
        _, (hidden, _) = self.lstm(x)

        return hidden.squeeze(0)

    def forward(self, user_ids, song_sequences):
        user_vec = self.user_embedding(user_ids)
        song_vec = self.encode_songs(song_sequences)

        concat = torch.cat([user_vec, song_vec], dim=1)

        prediction = self.prediction(concat)

        return prediction