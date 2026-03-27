import os
import sys
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import lyricsgenius

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Paths
RAW_LYRICS_PATH = ROOT / "data" / "raw" / "lyrics_raw.json"
THUMBNAILS_PATH = ROOT / "data" / "artist_thumbnails.json"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path: Path) -> list | dict:
    if not path.exists():
        print(f"{path} not found, starting fresh.")
        return [] if path == RAW_LYRICS_PATH else {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list | dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Saved file")


def build_genius_client() -> lyricsgenius.Genius:

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
    if not api_key:
        raise EnvironmentError(
            "GENIUS_CLIENT_ACCESS_TOKEN not found in .env Add it and try again."
        )
    return lyricsgenius.Genius(
        api_key,
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
        verbose=True,
        timeout=15,
        retries=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE SCRAPE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def scrape_artist(
    genius: lyricsgenius.Genius,
    artist_name: str,
    max_songs: int | None = None,
    sleep: float = 1.2,
) -> tuple[list[dict], str | None]:

    print(f"\n── Fetching: {artist_name} ──────────────────────────────────────")

    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs or 300)
    except Exception as e:
        print(f" [error] Could not fetch artist: {e}")
        return [], None

    thumbnail = getattr(artist, "image_url", None)
    records = []

    for song in tqdm(artist.songs, desc=f"Songs by {artist_name}", leave=True):
        # Skip songs with no lyrics, or no albums (covers, featueres, etc.)
        if not song.lyrics or not song.album:
            continue

        records.append(
            {
                "artist": artist_name,
                "title": song.title,
                "album": song.album["name"],
                "release_date": song.album.get("release_date_for_display"),
                "lyrics": song.lyrics,
                "album_cover_url": song.album.get("cover_art_url"),
            }
        )

        time.sleep(sleep)

    print(f"  Collected {len(records)} songs for {artist_name}.")
    return records, thumbnail


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def update_dataset(
    new_artists: list[str],
    max_songs: int | None = None,
    sleep: float = 1.2,
    force_rescrape: bool = False,
    rebuild_pipeline: bool = False,
) -> None:

    # Load existing data
    existing_records = load_json(RAW_LYRICS_PATH)
    existing_thumbnails = load_json(THUMBNAILS_PATH)

    existing_artists = {r["artist"] for r in existing_records}
    print(
        f"\nExisting corpus: {len(existing_records)} songs "
        f"across {len(existing_artists)} artists."
    )

    # Determine which artists actually need scraping
    to_scrape = []
    for name in new_artists:
        # Case-insensitive check against existing artists
        already_present = any(name.lower() == a.lower() for a in existing_artists)
        if already_present and not force_rescrape:
            print(f"  [skip] '{name}' already in dataset. Pass --force to re-scrape.")
        else:
            to_scrape.append(name)

    if not to_scrape:
        print("\nNothing to scrape. Exiting.")
        return

    print(f"\nWill scrape {len(to_scrape)} artist(s): {to_scrape}")

    # Scrape
    genius_client = build_genius_client()
    new_records = []
    new_thumbnails = {}

    for artist_name in tqdm(to_scrape, desc="Artists", position=0):
        records, thumbnail = scrape_artist(
            genius=genius_client,
            artist_name=artist_name,
            max_songs=max_songs,
            sleep=sleep,
        )

        if force_rescrape and any(
            artist_name.lower() == a.lower() for a in existing_artists
        ):
            # Remove old records for this artist before appending fresh ones
            existing_records = [
                r
                for r in existing_records
                if r["artist"].lower() != artist_name.lower()
            ]
            print(f"  Removed old records for '{artist_name}' (force re-scrape).")

        new_records.extend(records)
        if thumbnail:
            new_thumbnails[artist_name] = thumbnail

    if not new_records:
        print("\nNo new songs collected. Check artist names and API key.")
        return

    # Merge and save
    merged_records = existing_records + new_records
    merged_thumbnails = {**existing_thumbnails, **new_thumbnails}

    print(
        f"\nDataset size: {len(existing_records)} → {len(merged_records)} songs "
        f"(+{len(new_records)} new)"
    )
    print(f"Thumbnails: {len(existing_thumbnails)} → {len(merged_thumbnails)} artists")

    save_json(merged_records, RAW_LYRICS_PATH)
    save_json(merged_thumbnails, THUMBNAILS_PATH)

    # Optionally rebuild the pipeline
    if rebuild_pipeline:
        print("\n── Rebuilding pipeline ──────────────────────────────────────")
        import subprocess

        result = subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "build.py")],
            check=False,
        )
        if result.returncode == 0:
            print("Pipeline rebuilt successfully.")
        else:
            print("[warn] Pipeline build exited with errors. Check output above.")
    else:
        print(
            "\nDataset updated. Run `python pipeline/build.py` to reprocess "
            "embeddings and rebuild the topic model."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add new artists to the lyrics dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper/update_dataset.py --artists "Taylor Swift" "SZA"
  python scraper/update_dataset.py --artists "Lana Del Rey" --max_songs 50
  python scraper/update_dataset.py --artists "Beyoncé" --force --rebuild
        """,
    )
    parser.add_argument(
        "--artists",
        nargs="+",
        required=True,
        help="One or more artist names to add (quote names with spaces).",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=None,
        help="Maximum songs per artist. Default: full discography.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.2,
        help="Seconds to wait between song requests. Default: 1.2.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape artists already present in the dataset.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Run pipeline/build.py automatically after updating data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_dataset(
        new_artists=args.artists,
        max_songs=args.max_songs,
        sleep=args.sleep,
        force_rescrape=args.force,
        rebuild_pipeline=args.rebuild,
    )
