import os
from pathlib import Path
from dotenv import load_dotenv
import lyricsgenius
import time
import json
from tqdm import tqdm

# Get the Genius API key from the .env file
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)
genius_api_key = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")

# Create a Genius object to use for scraping
genius = lyricsgenius.Genius(
    genius_api_key,
    skip_non_songs=True,
    excluded_terms=[
        "(Remix)",
        "(Live)",
        "(Demo)",
        "(Instrumental)",
        "(Acoustic)",
        "Remix",
        "remix",
        "Spotify",
        "Disney",
    ],
    remove_section_headers=True,
    verbose=True,  # silences the per-song print statements
    timeout=15,
    retries=3,
)

# Define list of artists to scrape
artists = [
    "Charli xcx",
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
    "Gracie Abrams",
    "Tate McRae",
    "Chappell Roan",
    "Olivia Dean",
    "Rachel Chinouriri",
    "Ethel Cain",
    "Kacey Musgraves",
    "Ariana Grande",
]
# Define an empty list to fill out with all the records
records = []

# Define an empty dictionary to store all the artists and their thumbnail pictures
artist_pics = {}

# Iterate through every artist
for artist_name in tqdm(artists, desc="Artists", position=0):
    print(f"Fetching discography for: {artist_name}")

    # Ensure the artist can be found
    try:
        artist = genius.search_artist(
            artist_name,
            max_songs=300,
        )
    except Exception as e:
        print(f"  Failed to fetch artist {artist_name}: {e}")
        continue

    # Save artist pic if not saved before
    if artist_name not in artist_pics:
        artist_pics[artist_name] = artist.image_url

    # For every song the artist has, add its info to the records list
    for song in tqdm(
        artist.songs, desc=f"Songs by {artist_name}", position=1, leave=True
    ):
        # We can't use songs with no lyrics, and we will filter out songs with no albums, as these are usually covers or songs that do not belong to the artist
        if not song.lyrics or not song.album:
            continue

        records.append(
            {
                "artist": artist_name,
                "title": song.title,
                "album": song.album["name"],
                "release_date": song.album["release_date_for_display"],
                "lyrics": song.lyrics,
                "album_cover_url": song.album["cover_art_url"],
            }
        )

        time.sleep(1.20)  # stay polite to the API

    print(
        f"  Collected {len([r for r in records if r['artist'] == artist_name])} songs"
    )

# Save records as a JSON file
with open("data/raw/lyrics_raw.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

# Save artist thumbnails as a JSON file
with open("data/artist_thumbnails.json", "w", encoding="utf-8") as f:
    json.dump(artist_pics, f, ensure_ascii=False, indent=4)
