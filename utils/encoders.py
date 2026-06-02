import os
import pickle
from sklearn.preprocessing import LabelEncoder


class EncoderManager:

    def __init__(self):

        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()

    def fit(self, dataframe):

        dataframe['user_idx'] = self.user_encoder.fit_transform(
            dataframe['user']
        )

        dataframe['song_idx'] = self.song_encoder.fit_transform(
            dataframe['song']
        )

        return dataframe

    def transform(self, dataframe):
        """Apply a previously fitted encoder to new data.

        Unlike fit(), this does NOT re-learn the label mappings.
        Labels not seen during fit() will raise a ValueError.
        """

        dataframe['user_idx'] = self.user_encoder.transform(
            dataframe['user']
        )

        dataframe['song_idx'] = self.song_encoder.transform(
            dataframe['song']
        )

        return dataframe

    def save(self, path):
        """Persist the fitted encoder to disk."""

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load a previously saved EncoderManager."""

        with open(path, 'rb') as f:
            return pickle.load(f)