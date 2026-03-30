import os
import re
import sys
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT & PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

THEMES_OUTPUT_PATH = ROOT / "data" / "processed" / "song_themes.json"
ARTIST_THEMES_PATH = ROOT / "data" / "processed" / "artist_themes.json"
PROGRESS_CACHE_PATH = ROOT / "data" / "cache" / "theme_progress.json"

# ─────────────────────────────────────────────────────────────────────────────
# THEMES
# ─────────────────────────────────────────────────────────────────────────────

THEMES = [
    # Romantic spectrum
    "romantic love",
    "lust / desire",
    "heartbreak / breakup",
    "longing / unrequited love",
    "toxic relationship",
    # Emotional states
    "happiness / joy",
    "anger / revenge",
    "grief / loss",
    "loneliness",
    "mental health / anxiety",
    # Identity & growth
    "self-confidence / empowerment",
    "self-discovery / coming of age",
    "girlhood / female experience",
    "body image",
    "LGBTQ",
    # Life & lifestyle
    "party / hedonism",
    "fame / celebrity life",
    "holiday / seasonal",
]

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────


def check_ollama_running() -> bool:
    """
    Return True if the Ollama server is reachable.
    """
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_ollama_client(model: str = "llama3.2") -> tuple[OpenAI, str]:
    """
    Return a configured OpenAI client pointed at the local Ollama server,
    and the model name to use.

    Raises a clear error if Ollama isn't running.
    """

    if not check_ollama_running():
        raise RuntimeError(
            "Ollama server is not running.\n"
            "Start it with: ollama serve\n"
            "Then re-run this script."
        )

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    return client, model


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SONG CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a music analyst specializing in lyrical themes.
Your job is to assign themes to songs from a fixed taxonomy.
You must ONLY use themes from the provided list — do not invent new ones.
Always respond with a valid JSON array of strings and nothing else.
Do not include any explanation, preamble, or markdown formatting."""


def build_classification_prompt(
    lyrics: str, title: str, artist: str, max_themes: int = 3
) -> str:
    """
    Build the user prompt for a single song classification.
    """

    # Use the first 200 words of a song to avoid hitting token limit
    lyrics_snippet = " ".join(lyrics.split()[:200])
    themes_formatted = "\n".join(f"  - {t}" for t in THEMES)

    return f"""Assign 1 to {max_themes} themes to this song from the list below.

AVAILABLE THEMES:
{themes_formatted}

SONG: "{title}" by {artist}

LYRICS:
{lyrics_snippet}

RULES:
- Choose only from the available themes above
- Return between 1 and {max_themes} themes
- Return a JSON array of strings only
- Example valid response: ["heartbreak / breakup", "moving on / healing"]

Your response:"""


def parse_llm_response(raw: str) -> list[str]:
    """
    Parse the LLM's response into a list of valid theme strings.

    Returns:
        list[str]: List of themes in the THEMES list.
    """

    # Strip the markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return []
        # Only keep themes that are in the THEMES taxonomy (case-insensitive match)
        valid = [
            t
            for t in parsed
            if isinstance(t, str)
            and any(t.lower() == theme.lower() for theme in THEMES)
        ]
        return valid
    except json.JSONDecodeError:
        return []


def classify_song(
    client: OpenAI,
    model: str,
    lyrics: str,
    title: str,
    artist: str,
    max_themes: int = 3,
    retries: int = 2,
) -> list[str]:
    """
    Classify a single song using the Ollama model.

    Args:
        client (OpenAI): Configured OpenAI client using Ollama.
        model (str): Model name.
        lyrics (str): Raw or lightly processed lyrics string.
        title (str): Song title.
        artist (str): Artist name.
        max_themes (int, optional): Maximum number of themes to assign. Defaults to 3.
        retries (int, optional): Number of retry attempts on failure. Defaults to 2.

    Returns:
        list[str]: List of theme strings from the THEMES list. Will return empty array if classification fails.
    """

    prompt = build_classification_prompt(lyrics, title, artist, max_themes)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            themes = parse_llm_response(raw)

            if themes:
                return themes

            # if parsing succeeded but returned no valid themes, retry
            if attempt < retries:
                print(
                    f"  [retry {attempt + 1}] No valid themes parsed for '{title}'. Raw: {raw[:80]}"
                )

        except Exception as e:
            if attempt < retries:
                print(f"  [retry {attempt + 1}] API error for '{title}': {e}")
                time.sleep(2)
            else:
                print(f"  [fail] '{title}' by {artist}: {e}")

    return []


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS CLASSIFICATION (with progress saving)
# ─────────────────────────────────────────────────────────────────────────────


def load_progress() -> dict:
    """
    Load previously saved classification progress to resume classification if run is interrupted.
    """
    if not PROGRESS_CACHE_PATH.exists():
        return {}
    with open(PROGRESS_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_progress(progress: dict) -> None:
    PROGRESS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def make_progress_key(artist: str, title: str) -> str:
    return f"{artist}|||{title}"


def _load_existing_theme_records() -> dict:
    """
    Load song_themes.json as a lookup dict keyed by (artist, title).
    Returns an empty dict if the file doesn't exist yet.
    """
    if not THEMES_OUTPUT_PATH.exists():
        return {}
    themed_df = pd.read_json(THEMES_OUTPUT_PATH)
    return {
        (row["artist"], row["title"]): row.get("themes", [])
        for _, row in themed_df.iterrows()
    }


def _save_theme_records(df: pd.DataFrame, lookup: dict) -> None:
    """
    Merge theme lookup back into df and save to THEMES_OUTPUT_PATH.
    df should be the FULL corpus of te DatFrame, not just a subset.
    """
    df = df.copy()
    df["themes"] = df.apply(lambda r: lookup.get((r["artist"], r["title"]), []), axis=1)
    THEMES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(THEMES_OUTPUT_PATH, orient="records", indent=2)
    classified = df["themes"].apply(lambda x: len(x) > 0).sum()
    print(f"Saved -> {THEMES_OUTPUT_PATH}   ({classified}/{len(df)} songs classified)")


def classify_corpus(
    df: pd.DataFrame,
    lyrics_column: str = "preprocessed_lyrics",
    model: str = "llama3.2",
    max_themes: int = 3,
    sleep_between: float = 0.3,
    save_every: int = 50,
    resume: bool = True,
    artists: list[str] | None = None,
) -> pd.DataFrame:
    """
    Classify themes for songs in the corpus using Ollama.

    Features:
      - Resumable: saves progress every `save_every` songs.
      - Skips songs already classified in a previous run (if resume=True).
      - Optional `artists` filter: only classify songs by specific artists.
      - Merges new results into the existing song_themes.json rather than
        overwriting it, so adding artists never destroys prior work.

    Args:
        df (pd.DataFrame): Full corpus DataFrame with lyrics and metadata.
        lyrics_column (str, optional): Column containing text to classify. Defaults to "preprocessed_lyrics".
        model (str, optional): Ollama model name. Defaults to "llama3.2".
        max_themes (int, optional): Maximum themes per song. Defaults to 3.
        sleep_between (float, optional): Seconds between API calls. Defaults to 0.3.
        save_every (int, optional): Persist progress cache every N songs. Defaults to 50.
        resume (bool, optional): Skip songs already in the progress cache or saved JSON. Defaults to True.
        artists (list[str] | None, optional): If provided, only classify songs by these artists. All other songs are left as is in the output. Defaults to None.

    Returns:
        pd.DataFrame: Full corpus DataFrame with 'themes' column populated.
        Results are merged into and saved to data/processed/song_themes.json.
    """
    client, model = get_ollama_client(model)

    # ── Determine which rows to classify ──────────────────────────────────
    if artists is not None:
        target_mask = df["artist"].isin(artists)
        target_df = df[target_mask].copy()
        print(f"Artist filter active: {artists}")
        print(
            f"Targeting {len(target_df)} songs across {target_df['artist'].nunique()} artists."
        )
    else:
        target_df = df.copy()
        print(f"Classifying full corpus: {len(target_df)} songs.")

    print(f"Model: {model}  |  Themes: {len(THEMES)}  |  Resume: {resume}\n")

    # ── Load existing classified results (from saved JSON + progress cache)

    existing_lookup = _load_existing_theme_records() if resume else {}
    progress = load_progress() if resume else {}

    # Merge both sources into one lookup so we skip anything already done
    combined_done = {**existing_lookup}
    for key, themes in progress.items():
        parts = key.split("|||")
        if len(parts) == 2:
            combined_done[(parts[0], parts[1])] = themes

    skipped = 0
    n_done_before = len(
        [r for r in target_df.itertuples() if (r.artist, r.title) in combined_done]
    )

    if n_done_before > 0:
        print(f"Skipping {n_done_before} already classified songs.")

    # ── Classification loop ──────────────────────────────────
    calls_since_save = 0

    for i, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Classifying"):
        key = make_progress_key(row["artist"], row["title"])
        lookup_key = (row["artist"], row["title"])

        if resume and lookup_key in combined_done:
            skipped += 1
            continue

        lyrics = row.get(lyrics_column, "")
        if not isinstance(lyrics, str) or len(lyrics.strip()) < 20:
            combined_done[lookup_key] = []
            progress[key] = []
            continue

        themes = classify_song(
            client=client,
            model=model,
            lyrics=lyrics,
            title=row["title"],
            artist=row["artist"],
            max_themes=max_themes,
        )
        combined_done[lookup_key] = themes
        progress[key] = themes
        calls_since_save += 1

        # Persist progress periodically in case of interruption
        if calls_since_save % save_every == 0:
            save_progress(progress)
            _save_theme_records(df, combined_done)
            print(
                f"  [checkpoint] Saved progress at {calls_since_save} new classifications."
            )

        time.sleep(sleep_between)

    # ── Final save ────────────────────────────────────────────────────────
    save_progress(progress)
    _save_theme_records(df, combined_done)

    if skipped > 0:
        print(f"\nSkipped {skipped} already-classified songs.")
    print(f"New classifications this run: {calls_since_save}")

    # Return the full df with themes merged in
    result_df = df.copy()
    result_df["themes"] = result_df.apply(
        lambda r: combined_done.get((r["artist"], r["title"]), []),
        axis=1,
    )
    return result_df


def update_theme_classifications():
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this script to classify the entire song corpus.
    
    Usage:
        # Classify full corpus
        python analysis/themes_llm.py
        
        # Resume an interrupted run
        python analysis/themes_llm.py
        
        # Only classify newly added artists (auto-detected)
        python analysis/themes_llm.py --new_artists-only
        
        # Explicitly classify specific artists (after adding them)
        python analysis/themes_llm.py --artists "SZA" "Clairo"
        
        # Reclassify specific artists from scratch
        python analysis/themes_llm.py --artists "Carli xcx" --no-resume
        
        # Start completely fresh
        python analysis/themes_llm.py --no-resume
    """
    # ── Load arguments ───────────────────────────────────────────────────────
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify song themes using Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--artists",
        nargs="+",
        default=None,
        help="Only classify songs by these artists. Quote names with spaces.",
    )

    parser.add_argument(
        "--new-artists-only",
        action="store_true",
        help=(
            "Auto-detect artists not yet in song_themes.json "
            "and only classify those. Use this after adding new artists."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore saved progress and re-classify from scratch.",
    )

    parser.add_argument(
        "--model", default="llama3.2", help="Ollama model to use (default: llama3.2)."
    )

    parser.add_argument(
        "--lyrics-col",
        default="preprocessed_lyrics",
        help="DataFrame column to use as lyrics input.",
    )

    args = parser.parse_args()

    # ── Load corpus ───────────────────────────────────────────────────────
    data_path = ROOT / "data" / "processed" / "final_songs.json"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run pipeline/build.py first.")
        sys.exit(1)

    df = pd.read_json(data_path)
    print(f"Loaded {len(df)} songs across {df['artist'].nunique()} artists.\n")

    # ── Execute the right function ───────────────────────────────────────────────────────
    if args.new_artists_only:
        # Auto-detect and classify only missing artists
        update_theme_classifications(
            df=df, lyrics_columns=args.lyrics_col, model=args.model
        )

    elif args.artists:
        # Classify specific named artists
        unknown = [a for a in args.artists if a not in df["artist"].unique()]

        if unknown:
            print(f"WARNING: These artists were not found in the corpus: {unknown}")
            print(f"Known artists: {sorted(df['artist'].unique())}\n")

        valid_artists = [a for a in args.artists if a in df["artist"].unique()]

        if not valid_artists:
            print("No valid artists to classify. Exiting.")
            sys.exit(1)

        classify_corpus(
            df=df,
            lyrics_column=args.lyrics_col,
            model=args.model,
            resume=not args.no_resume,
            artists=valid_artists,
        )

    else:
        # Default: classify full corpus
        classify_corpus(
            df=df,
            lyrics_column=args.lyrics_col,
            model=args.model,
            resume=not args.no_resume,
            artists=None,
        )
