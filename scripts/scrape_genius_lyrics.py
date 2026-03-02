import os
from pathlib import Path
from dotenv import load_dotenv
import lyricsgenius
import time
import json
from tqdm import tqdm

# Get the Genius API key from the .env file
BASE_DIR = Path("getting_data.ipynb").resolve().parent
env_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=env_path)
genius_api_key = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")

# Create a Genius object to use for scraping
genius = lyricsgenius.Genius(
    genius_api_key,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)", "(Demo)", "(Instrumental)", "(Acoustic)"],
    remove_section_headers=True,
    verbose=True,  # silences the per-song print statements
    timeout=15,
    retries=3,
)

# Define list of artists to scrape
artists = [
    "Sabrina Carpenter",
    "Audrey Hobert",
    "Reneé Rapp",
    "Holly Humberstone",
    "Maisie Peters",
    "Carly Rae Jepsen",
    "Lorde",
    "Addison Rae",
    "Billie Eilish",
    "Olivia Rodrigo",
    "Maggie Rogers",
]
# Define an empty list to fill out with all the records
records = []

# Iterate through every artist
for artist_name in tqdm(artists, desc="Artists", position=0):
    print(f"Fetching discography for: {artist_name}")

    # Ensure the artist can be found
    try:
        artist = genius.search_artist(artist_name, max_songs=None)
    except Exception as e:
        print(f"  Failed to fetch artist {artist_name}: {e}")
        continue
    
    # For every song the artist has, add its info to the records list
    for song in tqdm(
        artist.songs, desc=f"Songs by {artist_name}", position=1, leave=True
    ):
        if not song.lyrics:
            continue

        records.append(
            {
                "artist": artist_name,
                "title": song.title,
                "album": song.album["name"] if song.album else None,
                "release_date": song.album["release_date_for_display"]
                if song.album
                else None,
                "lyrics": song.lyrics,
            }
        )

        time.sleep(1.25)  # stay polite to the API

    print(
        f"  Collected {len([r for r in records if r['artist'] == artist_name])} songs"
    )

# Save records as a JSON file
with open("data/songs.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=4)
