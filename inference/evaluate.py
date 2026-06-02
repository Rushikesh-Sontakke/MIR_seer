# inference/evaluate.py
#
# Evaluation metrics for the SeER recommendation model.
#
# The paper ("Sequence-based Explainable Hybrid Song Recommendation")
# evaluates using standard top-K ranking metrics:
#   - Precision@K
#   - Recall@K
#   - MAP@K   (Mean Average Precision)
#   - NDCG@K  (Normalized Discounted Cumulative Gain)
#
# Evaluation protocol (Section 4.1 of the paper):
#   For each user, split their interactions into train and test sets.
#   Use the trained model to score ALL candidate songs for that user,
#   rank them, take the top K, and compare against the held-out test
#   songs.  A test song is considered "relevant" if the user interacted
#   with it (i.e., it appears in the test set).

import numpy as np
import torch
from collections import defaultdict


# ==============================================================
# INDIVIDUAL METRIC FUNCTIONS
# ==============================================================

def precision_at_k(recommended, relevant, k):
    """Fraction of top-K recommendations that are relevant."""

    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items that appear in top-K."""

    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))

    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """Average Precision at K for a single user.

    AP@K = (1 / min(K, |relevant|)) * sum_{i=1}^{K} P(i) * rel(i)

    where P(i) is precision at position i, and rel(i) is 1 if the
    item at position i is relevant, 0 otherwise.
    """

    if len(relevant) == 0:
        return 0.0

    relevant_set = set(relevant)
    top_k = recommended[:k]

    hits = 0
    sum_precision = 0.0

    for i, item in enumerate(top_k):
        if item in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(k, len(relevant))


def ndcg_at_k(recommended, relevant, k):
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: 1 if the item is in the relevant set,
    0 otherwise.

    DCG@K  = sum_{i=1}^{K} rel(i) / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} 1 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    """

    if len(relevant) == 0:
        return 0.0

    relevant_set = set(relevant)
    top_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Ideal DCG
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ==============================================================
# FULL EVALUATION PIPELINE
# ==============================================================

@torch.no_grad()
def evaluate_model(model,
                   test_interactions,
                   song_sequences,
                   device,
                   k=10):
    """Run full top-K evaluation over all test users.

    Optimized approach:
      1. Run the GRU once per song to pre-compute all song vectors.
      2. Extract all user embeddings in one lookup.
      3. Score every (user, song) pair with a single matrix multiply.
      4. Rank and compute metrics per user.

    This reduces ~N_users × N_songs forward passes down to N_songs
    GRU passes + one matrix multiplication.

    Parameters
    ----------
    model : SeER
        A trained SeER model (will be set to eval mode).
    test_interactions : dict
        Mapping from user_idx (int) -> list of song_ids (str) that
        the user interacted with in the held-out test set.
    song_sequences : dict
        Mapping from song_id (str) -> np.ndarray of shape
        (timesteps, 32).  This is the full set of candidate songs.
    device : str
        'cuda' or 'cpu'.
    k : int
        Number of top recommendations to evaluate (default 10,
        matching the paper's MAP@10).

    Returns
    -------
    metrics : dict
        Aggregate metrics averaged over all test users:
        'precision@k', 'recall@k', 'map@k', 'ndcg@k'.
    per_user : dict
        Per-user metric breakdowns (for debugging / analysis).
    """

    model.eval()

    all_song_ids = list(song_sequences.keys())
    num_songs = len(all_song_ids)

    # ----------------------------------------------------------
    # STEP 1: Pre-compute all song vectors via GRU (once per song)
    # ----------------------------------------------------------
    print(f"  Pre-computing GRU embeddings for {num_songs} songs...")

    GRU_BATCH = 64  # songs per batch through the GRU
    song_vecs = []

    for batch_start in range(0, num_songs, GRU_BATCH):
        batch_end = min(batch_start + GRU_BATCH, num_songs)

        batch_seqs = torch.stack([
            torch.tensor(song_sequences[all_song_ids[i]], dtype=torch.float32)
            for i in range(batch_start, batch_end)
        ]).to(device)  # (batch, timesteps, 32)

        song_vec_batch = model.encode_songs(batch_seqs)  # (batch, hidden_size)
        song_vecs.append(song_vec_batch.cpu())

        if (batch_start // GRU_BATCH + 1) % 100 == 0:
            print(f"    {batch_end}/{num_songs} songs embedded")

    # (num_songs, hidden_size)
    song_matrix = torch.cat(song_vecs, dim=0)
    print(f"  Song matrix: {song_matrix.shape}")

    # ----------------------------------------------------------
    # STEP 2: Extract user embeddings for all test users
    # ----------------------------------------------------------
    test_user_ids = list(test_interactions.keys())
    num_users = len(test_user_ids)

    user_id_tensor = torch.tensor(test_user_ids, dtype=torch.long).to(device)
    # (num_users, latent_dim)
    user_matrix = model.user_embedding(user_id_tensor).cpu()
    print(f"  User matrix: {user_matrix.shape}")

    # ----------------------------------------------------------
    # STEP 3: Score all (user, song) pairs via matrix multiply
    # ----------------------------------------------------------
    print(f"  Computing score matrix ({num_users} users × {num_songs} songs)...")

    # (num_users, num_songs) = (num_users, latent) @ (latent, num_songs)
    score_matrix = torch.mm(user_matrix, song_matrix.t()).numpy()

    # ----------------------------------------------------------
    # STEP 4: Rank and compute metrics per user
    # ----------------------------------------------------------
    print(f"  Computing ranking metrics @ K={k}...")

    per_user_metrics = {}
    precisions = []
    recalls = []
    aps = []
    ndcgs = []

    for i, user_idx in enumerate(test_user_ids):

        relevant_songs = test_interactions[user_idx]
        if len(relevant_songs) == 0:
            continue

        # Get top-K song indices for this user (descending score)
        user_scores = score_matrix[i]
        top_k_indices = np.argpartition(user_scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(user_scores[top_k_indices])[::-1]]
        recommended = [all_song_ids[idx] for idx in top_k_indices]

        # Compute metrics
        p = precision_at_k(recommended, relevant_songs, k)
        r = recall_at_k(recommended, relevant_songs, k)
        ap = average_precision_at_k(recommended, relevant_songs, k)
        n = ndcg_at_k(recommended, relevant_songs, k)

        precisions.append(p)
        recalls.append(r)
        aps.append(ap)
        ndcgs.append(n)

        per_user_metrics[user_idx] = {
            'precision': p,
            'recall': r,
            'ap': ap,
            'ndcg': n,
            'num_relevant': len(relevant_songs),
        }

        if (i + 1) % 10000 == 0:
            print(
                f"    Processed {i + 1}/{num_users} users  "
                f"(running MAP@{k}: {np.mean(aps):.4f})"
            )

    metrics = {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'map@{k}': np.mean(aps),
        f'ndcg@{k}': np.mean(ndcgs),
        'num_users_evaluated': len(precisions),
    }

    return metrics, per_user_metrics
