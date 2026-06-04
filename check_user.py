"""Quick script to check user interactions and verify MAP score."""
import pandas as pd
import numpy as np

# Load triplets
triplets = pd.read_csv("data/triplets.txt", sep=" ", header=None, names=['user', 'song', 'rating'])

# User 0 interactions
user0 = triplets[triplets['user'] == 0].sort_values('rating', ascending=False)
print("=" * 50)
print(f"USER 0 INTERACTIONS ({len(user0)} songs)")
print("=" * 50)
print(user0.to_string(index=False))

# General stats
print(f"\n{'=' * 50}")
print("DATASET STATS")
print(f"{'=' * 50}")
print(f"Total interactions: {len(triplets)}")
print(f"Users: {triplets['user'].nunique()}")
print(f"Songs: {triplets['song'].nunique()}")
print(f"\nRating distribution:")
print(triplets['rating'].value_counts().sort_index())
print(f"\nAvg interactions per user: {len(triplets) / triplets['user'].nunique():.1f}")
print(f"Avg rating: {triplets['rating'].mean():.2f}")

# Check how many users have >= 3 rated songs (relevant for MAP)
user_counts = triplets.groupby('user').size()
print(f"\nUsers with >= 20 interactions: {(user_counts >= 20).sum()}")
print(f"Users with >= 10 interactions: {(user_counts >= 10).sum()}")
print(f"Users with < 10 interactions: {(user_counts < 10).sum()}")
print(f"Min interactions: {user_counts.min()}, Max: {user_counts.max()}")

# Rating threshold analysis
high_rated = triplets[triplets['rating'] >= 3]
print(f"\nInteractions with rating >= 3: {len(high_rated)} ({100*len(high_rated)/len(triplets):.1f}%)")
print(f"Interactions with rating < 3: {len(triplets) - len(high_rated)} ({100*(len(triplets)-len(high_rated))/len(triplets):.1f}%)")
