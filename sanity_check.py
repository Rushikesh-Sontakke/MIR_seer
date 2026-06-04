import pandas as pd
song_mapping = pd.read_csv(
    r"C:\Users\Rushi\Downloads\MIR_seer\data\song_to_number_matching.csv"
)

song_info = pd.read_csv(
    r"C:\Users\Rushi\Downloads\MIR_seer\data\song_information.csv"
)

test_songs = [766, 4990, 4784, 1264]

for num in test_songs:

    song_id = song_mapping[
        song_mapping["number"] == num
    ]["song_id"].iloc[0]

    row = song_info[
        song_info["song_id"] == song_id
    ]

    print()
    print("Number:", num)
    print("Song ID:", song_id)

    if len(row):
        print("Artist:", row.iloc[0]["artist_name"])
        print("Title :", row.iloc[0]["title"])