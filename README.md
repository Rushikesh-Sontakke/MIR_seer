# SeER: Sequence-based Explainable Hybrid Song Recommender

PyTorch implementation of **SeER** from [arXiv:1907.01640](https://arxiv.org/abs/1907.01640) — a hybrid deep learning model that combines collaborative filtering with LSTM sequence models on MIDI content for personalized, explainable song recommendation.

## Architecture

```
user_id → Embedding(150) ──────────────────┐
                                           ├─ concat(300) → Linear(1) → predicted rating
song_midi → LSTM(32→150, dropout=0.2) ─────┘
```

**Key features:**
- Hybrid CF + content-based filtering via MIDI sequences
- Segment Forward Propagation explainability (Algorithm 2 from the paper)
- Solves item cold-start using song content

## Results

| Metric | Value | Paper |
|--------|-------|-------|
| RMSE | 1.2448 | ~1.24 |
| MAE | 0.9898 | — |
| MAP@10 | 0.4616 | ~0.45 |
| NDCG@10 | 0.9929 | 0.9867 |

Evaluated on 32,180 users × 6,442 songs (941K interactions) from the Million Song Dataset.

## Project Structure

```
MIR_seer/
├── data/                          # Dataset files
│   ├── midi_array.txt             # MIDI features (6442 × 16000)
│   ├── Time_array.txt             # Timestamps for explainability
│   ├── triplets.txt               # User-song-rating interactions
│   ├── song_information.csv       # Song metadata (artist, title, etc.)
│   ├── song_to_number_matching.csv
│   ├── unique_tracks.txt          # Track ID ↔ Song ID mapping
│   └── lmd_matched/               # Raw MIDI files (for explainability)
├── models/
│   └── seer.py                    # SeER model architecture
├── inference/
│   ├── evaluate.py                # Full-ranking evaluation metrics
│   ├── explain.py                 # Segment Forward Propagation
│   └── recommend.py               # Top-K recommendation
├── training/
│   └── train.py                   # Training loop
├── main.py                        # Recommend songs with metadata
├── evaluate_model.py              # Run evaluation (RMSE, MAP, NDCG)
├── explain_song.py                # Explain a recommendation
├── seer_model.pth                 # Trained model weights
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Recommend songs
```bash
python main.py --user 0 --top-k 5
python main.py --user 42 --top-k 10
```

### Evaluate model
```bash
python evaluate_model.py                    # Author-style metrics
python evaluate_model.py --full-ranking     # Full-ranking (slower, stricter)
```

### Explain a recommendation
```bash
python explain_song.py --user 0 --track TRAEHHJ12903CF492F
python explain_song.py --user 0 --midi path/to/song.mid
```

### Train from scratch
```bash
python -m training.train
```

## Reference

```bibtex
@article{damak2021seer,
  title={Sequence-Based Explainable Hybrid Song Recommendation},
  author={Damak, Khalil and Nasraoui, Olfa and Sanders, W. Scott},
  journal={Frontiers in Big Data},
  volume={4},
  year={2021}
}
```
