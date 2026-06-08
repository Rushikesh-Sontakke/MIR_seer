# explain_song.py
#
# Segment Forward Propagation (paper Algorithm 2): find the 10-second
# MIDI window that best explains why a song was recommended to a user.
#
# Usage:
#   python explain_song.py --user 0 --track TRAEHHJ12903CF492F
#   python explain_song.py --user 0 --midi path/to/song.mid
#
# Requires: seer_model.pth, and a .mid/.midi file (e.g. under data/lmd_matched/).

import argparse
import os
import sys
import glob

import torch

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.seer import SeER
from inference.explain import generate_explanation

MODEL_PATH = os.path.join(ROOT_DIR, "seer_model.pth")
LMD_PATH = os.path.join(ROOT_DIR, "data", "lmd_matched")


def find_midi_for_track(track_id):
    """Lakh MIDI layout: data/lmd_matched/lmd_matched/A/B/C/<TRACK_ID>/*.mid"""
    
    if len(track_id) < 5:
        return None
        
    # The Lakh MIDI dataset nests files based on the 3rd, 4th, and 5th characters of the track ID
    folder_a = track_id[2]
    folder_b = track_id[3]
    folder_c = track_id[4]
    
    # Handle the fact that it might be extracted into a nested 'lmd_matched' folder
    lmd_base = LMD_PATH
    if os.path.isdir(os.path.join(LMD_PATH, "lmd_matched")):
        lmd_base = os.path.join(LMD_PATH, "lmd_matched")
        
    pattern = os.path.join(lmd_base, folder_a, folder_b, folder_c, track_id, "*.mid*")
    matches = glob.glob(pattern)
    
    # Fallback to the flat structure just in case
    if not matches:
        pattern = os.path.join(LMD_PATH, track_id, "*.mid*")
        matches = glob.glob(pattern)
        
    if not matches:
        return None
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="SeER explanation: best 10s segment for a user + song"
    )
    parser.add_argument(
        "--user", type=int, required=True,
        help="User index (0 .. 32179), same as triplets.txt / main.py"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--track", type=str,
        help="MSD track folder name (e.g. from main.py recommendations)"
    )
    group.add_argument(
        "--midi", type=str,
        help="Direct path to a .mid or .midi file"
    )
    parser.add_argument(
        "--window", type=float, default=10.0,
        help="Explanation window length in seconds (default: 10)"
    )
    parser.add_argument(
        "--stride", type=float, default=1.0,
        help="Sliding window stride in seconds (default: 1)"
    )
    args = parser.parse_args()

    if args.midi:
        midi_path = os.path.abspath(args.midi)
        if not os.path.isfile(midi_path):
            print(f"Error: MIDI file not found: {midi_path}")
            sys.exit(1)
    else:
        midi_path = find_midi_for_track(args.track)
        if midi_path is None:
            print(
                f"Error: no MIDI under {LMD_PATH}/{args.track}/\n"
                "Extract Lakh MIDI (extract.py) or pass --midi PATH"
            )
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    num_users = state_dict["user_embedding.weight"].shape[0]

    if not (0 <= args.user < num_users):
        print(f"Error: user must be in [0, {num_users - 1}]")
        sys.exit(1)

    model = SeER(num_users=num_users).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"User:     {args.user}")
    print(f"MIDI:     {midi_path}")
    print(f"Device:   {device}")
    print(f"Window:   {args.window}s, stride {args.stride}s")
    print("Scanning segments...")

    segment, score = generate_explanation(
        model=model,
        user_id=args.user,
        midi_path=midi_path,
        device=device,
        window_seconds=args.window,
        stride_seconds=args.stride,
    )

    if segment is None:
        print("No valid segment found (empty or unreadable MIDI).")
        sys.exit(1)

    start_sec, end_sec = segment
    print()
    print("Best explanation segment:")
    print(f"  Time:  {start_sec:.2f}s – {end_sec:.2f}s  ({end_sec - start_sec:.1f}s)")
    print(f"  Score: {score:.4f}  (predicted rating for this segment)")
    print()
    print("Listen: trim/play that range from the MIDI or export with pretty_midi / a DAW.")


if __name__ == "__main__":
    main()
