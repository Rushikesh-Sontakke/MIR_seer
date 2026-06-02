import os

MIDI_PATH = "data/lmd_matched"
ECHO_FILE = "data/train_triplets.txt"

# ---------------------------------------------------
# STEP 1: Collect MIDI song IDs
# ---------------------------------------------------

midi_song_ids = set()

for root, dirs, files in os.walk(MIDI_PATH):
    for f in files:
        if f.endswith(".mid") or f.endswith(".midi"):

            track_id = os.path.basename(os.path.dirname(
                os.path.join(root, f)
            ))

            midi_song_ids.add(track_id)

print("MIDI songs:", len(midi_song_ids))

# ---------------------------------------------------
# STEP 2: Find intersection
# ---------------------------------------------------

intersection = set()

with open(ECHO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        user_id, song_id, play_count = line.strip().split("\t")

        if song_id in midi_song_ids:
            intersection.add(song_id)

print("Intersection songs:", len(intersection))

# ---------------------------------------------------
# STEP 3: Keep only valid interactions
# ---------------------------------------------------

filtered_interactions = []

with open(ECHO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        user_id, song_id, play_count = line.strip().split("\t")

        if song_id in intersection:
            filtered_interactions.append(
                (user_id, song_id, int(play_count))
            )

print("Filtered interactions:", len(filtered_interactions))