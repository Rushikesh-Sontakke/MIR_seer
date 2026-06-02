import torch


@torch.no_grad()
def recommend_top_k(model,
                    user_id,
                    song_sequences,
                    device,
                    k=10):

    model.eval()

    scores = []

    for song_id, sequence in song_sequences.items():

        user_tensor = torch.tensor([user_id]).to(device)

        seq_tensor = torch.tensor(sequence).unsqueeze(0).to(device)

        prediction = model(user_tensor, seq_tensor)

        scores.append((song_id, prediction.item()))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[:k]