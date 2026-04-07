# import os
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
CLUSTER_CACHE_PATH = ROOT / "data" / "processed" / "artist_clusters.json"

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
    "holiday",
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

# One-line definitions shown to the model to prevent vague over-classification
THEME_DEFINITIONS = {
    "romantic love": "songs celebrating being in love or a loving relationship",
    "lust / desire": "songs about physical attraction, wanting someone sexually",
    "heartbreak / breakup": "songs about the pain of a relationship ending",
    "longing / unrequited love": "songs about missing someone or loving someone who doesn't love back",  # — only assign if longing is the DOMINANT theme, not just present",
    "self-confidence / empowerment": "songs about feeling good with oneself, believing in yourself, independence, owning your identity",
    "grief / loss": "songs about death, mourning, or losing someone permanently",
    "nostalgia / coming of age": "songs about looking back on youth, growing up, or the passage of time",
    "faith / spirituality": "songs explicitly referencing God, prayer, religion, or spiritual belief",
    "party / hedonism": "songs about going out, dancing, drinking, having fun with no strings",
    "friendship / sisterhood": "songs addressed to or about close friends, chosen family, or female solidarity",
    "toxic relationship": "songs about a relationship that is unhealthy, controlling, or mutually destructive",
    "moving on / healing": "songs about recovering from heartbreak, growing after pain, choosing yourself",
    "identity / self-discovery": "songs about figuring out who you are, your place in the world",
    "holiday": "songs explicitly about Christmas, winter holidays, New Year",  # — assign this if ANY holiday references appear",
    "fame / celebrity life": "songs about the experience of being famous, public life, or the music industry",
    "mental health / anxiety": "songs explicitly about depression, anxiety, therapy, or mental illness",
    "LGBTQ+": "songs with explicit or strongly implied same-sex attraction, queer and gender identity/experience  — assign this if ANY queer coding is present",
    "body image": "songs explicitly about physical appearance, body acceptance, or how the singer feels about their body",
}


def build_classification_prompt(
    lyrics: str, title: str, artist: str, max_themes: int = 2
) -> str:
    """
    Build the user prompt for a single song classification.
    """

    # Use the first 200 words of a song to avoid hitting token limit
    lyrics_snippet = " ".join(lyrics.split()[:200])
    themes_formatted = "\n".join(
        f"  - {t}: {THEME_DEFINITIONS.get(t, '')}" for t in THEMES
    )

    return f"""Assign 1 to {max_themes} themes to this song using the definitions below.

THEMES WITH DEFINITIONS:
{themes_formatted}

SONG: "{title}" by {artist}

LYRICS:
{lyrics_snippet}

INSTRUCTIONS:
1. Choose the MOST SPECIFIC themes that apply — prefer specific over general.
2. Return between 1 and {max_themes} themes as a JSON array of strings only.
3. Example: ["LGBTQ+", "romantic love"]

Your response:"""


# 1. First check for SPECIFIC themes: "holiday / seasonal", "LGBTQ+", "body image",
# "faith / spirituality", "fame / celebrity life". If ANY of these apply even
# partially, consider including them BEFORE adding general themes.
# 2. Only assign "longing / unrequited love" if longing is the song's PRIMARY subject,
# not just a passing feeling.


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
                temperature=0.1,
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


def update_theme_classifications(
    df: pd.DataFrame,
    lyrics_column: str = "preprocessed_lyrics",
    model: str = "llama3.2",
    max_themes: int = 3,
    sleep_between: float = 0.3,
) -> pd.DataFrame:

    existing_lookup = _load_existing_theme_records()
    classified_artists = {artist for (artist, _) in existing_lookup.keys()}
    all_artists = set(df["artist"].unique())
    new_artists = sorted(all_artists - classified_artists)

    if not new_artists:
        print("All artists already classified. Nothing to do.")
        print("To re-classify specific artists, use classify_corpus(...).")
        result_df = df.copy()
        result_df["themes"] = result_df.apply(
            lambda r: existing_lookup.get((r["artist"], r["title"], []), axis=1)
        )
        return result_df

    print(f"Found {len(new_artists)} new artist(s) to classify:")
    for a in new_artists:
        n = len(df[df["artist"] == a])
        print(f"   - {a}  ({n} songs)")
    print()

    return classify_corpus(
        df=df,
        lyrics_column=lyrics_column,
        model=model,
        max_themes=max_themes,
        sleep_between=sleep_between,
        resume=True,
        artists=new_artists,  # only classify new artists
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARTIST-LEVEL THEME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────


def get_artist_theme_distribution(
    df: pd.DataFrame, artist: str, normalize: bool = True
) -> pd.DataFrame:
    """
    Return the theme distribution for a single artist.

    Args:
        df (pd.DataFrame): DataFrame with a "themes" column.
        artist (str): Artist name to filter by
        normalize (bool, optional): If True, returns proportions. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with theme, count/proportion (sorted descending).
    """
    artist_df = df[df["artist"] == artist].copy()
    if artist_df.empty:
        raise ValueError(f"Artist '{artist}' not found in DataFrame.")

    # Explode themes list into one row per theme
    exploded = artist_df.explode("themes").dropna(subset=["themes"])
    exploded = exploded[exploded["themes"] != ""]

    counts = exploded["themes"].value_counts().reset_index()
    counts.columns = ["theme", "count"]

    if normalize:
        total = counts["count"].sum()
        counts["proportion"] = (counts["count"] / total).round(4)

    return counts


def get_all_artist_theme_distributions(
    df: pd.DataFrame, normalize: bool = True, min_songs: int = 1
) -> pd.DataFrame:
    """
    Compute theme distributions for all artists and return as a wide-format DataFrame for heatmap.
    Rows = artists, Columns = themes, Values = proportions (or counts).

    Args:
        df (pd.DataFrame): DataFrame with "themes" column
        normalize (bool, optional): If True,, use proportions. Defaults to True.
        min_songs (int, optional): Only include artists with at least this many classified songs. Defaults to 1.

    Returns:
        pd.DataFrame: Wide-format DataFrame of shape (n_artists, n_themes).
    """
    artists = sorted(df["artist"].unique())
    rows = {}

    for artist in artists:
        artist_df = df[df["artist"] == artist]
        n_classified = artist_df["themes"].apply(lambda x: len(x) > 0).sum()

        if n_classified < min_songs:
            continue

        dist = get_artist_theme_distribution(df, artist, normalize=normalize)
        rows[artist] = dict(
            zip(dist["theme"], dist["proportion" if normalize else "count"])
        )

        wide = pd.DataFrame(rows).T.fillna(0).round(4)

    # Ensuring all themes are present as columns even if no artist used them
    for theme in THEMES:
        if theme not in wide.columns:
            wide[theme] = 0.0

    # Order columns by taxonomy order
    wide = wide[[t for t in THEMES if t in wide.columns]]

    return wide


def get_corpus_theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of theme frequencies across the full corpus.

    Args:
        df (pd.DataFrame): DataFrame with full corpus and themes.

    Returns:
        pd.DataFrame: DataFrame with columns: theme, song_count, proportion. Sorted by descending song_count.
    """
    exploded = df.explode("themes").dropna(subset=["themes"])
    exploded = exploded[exploded["themes"] != ""]

    counts = exploded["themes"].value_counts().reset_index()
    counts.columns = ["theme", "song_count"]
    counts["proportion"] = (counts["song_count"] / len(df)).round(4)

    return counts


def get_songs_by_theme(
    df: pd.DataFrame, theme: str, artist: str | None = None
) -> pd.DataFrame:
    """
    Return all songs assigned to a given theme.

    Args:
        df (pd.DataFrame): DataFrame with "themes: column.
        theme (str): Theme string, must be in the THEMES taxonomy.
        artist (str | None, optional): Optional artist filter. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with columns: artist, title, album, themes. Sorted by artist, then title.
    """
    if theme not in THEMES:
        raise ValueError(
            f"'{theme}' is not in the theme taxonomy. Valid themes: {THEMES}"
        )

    mask = df["themes"].apply(lambda t: theme in t if isinstance(t, list) else False)
    result = df[mask].copy()

    if artist:
        result = result[result["artist"] == artist]

    return (
        result[["artist", "title", "album", "themes"]]
        .sort_values(["artist", "title"])
        .reset_index(drop=True)
    )


def get_theme_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a theme co-occurrence matrix

    Args:
        df (pd.DataFrame): DataFrame with "themes" column.

    Returns:
        pd.DataFrame: Square DataFrame of shape (n_themes, n_themes). Values = number of songs where both themes appear together.
    """
    # Binary indicator matrix: songs x themes
    indicator = pd.DataFrame(
        {
            theme: df["themes"].apply(
                lambda t: 1 if (isinstance(t, list) and theme in t) else 0
            )
            for theme in THEMES
        }
    )

    # Co-occurrence = dot-product of binary matrix with its transpose
    cooccurrence = indicator.T.dot(indicator)

    return cooccurrence


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAVED RESULTS
# ─────────────────────────────────────────────────────────────────────────────


def load_theme_results(
    df: pd.DataFrame,
    path: Path = THEMES_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load previously saved theme classifications and merge into df.

    Use this in Streamlit instead of re-running classify_corpus().
    classify_corpus() should only be run once, then results are loaded
    from disk on every subsequent run.

    Args:
        df:   The main lyrics DataFrame (from session_state).
        path: Path to saved song_themes.json.

    Returns:
        df with "themes" column added. Songs without a saved classification
        get an empty list — they will not cause errors downstream.

    Raises:
        FileNotFoundError if theme classifications haven't been run yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Theme classifications not found at {path}.\n"
            "Run theme classification first:\n"
            "    python analysis/themes_llm.py"
        )

    lookup = _load_existing_theme_records()
    result_df = df.copy()
    result_df["themes"] = result_df.apply(
        lambda r: lookup.get((r["artist"], r["title"]), []),
        axis=1,
    )

    n_classified = result_df["themes"].apply(lambda x: len(x) > 0).sum()
    n_unclassified = len(result_df) - n_classified

    print(f"Loaded themes for {n_classified}/{len(result_df)} songs.")
    if n_unclassified > 0:
        missing_artists = result_df[result_df["themes"].apply(lambda x: len(x) == 0)][
            "artist"
        ].unique()
        print(
            f"  {n_unclassified} songs have no themes yet "
            f"(artists: {list(missing_artists)}).\n"
            "  Run `python analysis/themes_llm.py` to classify them."
        )

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# PER-ARTIST CLUSTER LABELING (for artists with more than 30 songs)
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_SYSTEM_PROMPT = """You are a music analyst labeling lyrical themes.
Generate short, neutral, descriptive labels for groups of songs.

CRITICAL RULES — you must follow all of these without exception:
- Labels must describe lyrical CONTENT and THEMES only (e.g. what the songs are about)
- Labels must NEVER reference the artist's race or ethnicity
- Labels must NEVER use culturally coded or racially loaded descriptors.
  Forbidden examples: sassy, soulful, fierce, urban, hood, exotic, sultry, feisty, 
  ratchet, ghetto, street
- Keep labels to 2-6 words
- Return ONLY the label, no explanation"""

# - Labels must be the same quality and register regardless of the artist's identity
# - Labels must describe what the LYRICS are about, not the artist's personality or style


def _get_cluster_label(
    client,
    model: str,
    artist: str,
    rep_songs: pd.DataFrame,
    used_labels: list[str],
) -> str:
    """
    Ask the LLM to generate a single cluster label from representative songs.
    Shared by both the small-artist centroid path and the full clustering path.
    """
    songs_text = ""
    for _, row in rep_songs.iterrows():
        snippet = " ".join(str(row.get("lyrics_for_llm", "")).split()[:80])
        songs_text += f'\n---\n"{row["title"]}":\n{snippet}'

    used_labels_str = (
        (
            "\nALREADY USED LABELS (choose something meaningfully different): "
            + ", ".join(f'"{la}"' for la in used_labels)
        )
        if used_labels
        else ""
    )

    prompt = (
        f"These songs by {artist} form a distinct lyrical group.\n"
        f"Write a vivid, specific 2-6 word label describing what unites them.\n"
        f"{songs_text}\n"
        f"{used_labels_str}\n\n"
        f"Rules:\n"
        f"- Describe the LYRICAL CONTENT specifically (situations, emotions, "
        f"storylines in the words) — not the genre or production style\n"
        # f'- Be evocative and specific: prefer "falling for your best friend" '
        # f'over "romantic feelings", "escaping a bad relationship" over "breakup"\n'
        f"- 2-6 words, no quotes in your response\n"
        f"- Must be different from any already used labels\n"
        f"Your label:"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=30,  # slightly more room for descriptive labels
            temperature=0.5,  # higher than before — encourages vivid phrasing
        )
        label = resp.choices[0].message.content.strip().strip('"').strip("'")

        # Blocklist check
        # BLOCKLIST = {
        #     "sassy", "soulful", "fierce", "urban", "hood", "raw", "gritty",
        #     "exotic", "sultry", "feisty", "ratchet", "ghetto", "street",
        #     "edgy", "spicy", "authentic", "real", "spiritual",
        # }
        # if set(label.lower().split()): #& BLOCKLIST:
        #     print(f"  [flagged] '{label}' — retrying")
        #     resp = client.chat.completions.create(
        #         model=model,
        #         messages=[
        #             {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
        #             {"role": "user", "content": (
        #                 f"{prompt}\n\nIMPORTANT: Your previous response '{label}' "
        #                 f"used a culturally coded term. Describe only what the "
        #                 f"lyrics are literally about."
        #             )},
        #         ],
        #         max_tokens=30,
        #         temperature=0.0,
        #     )
        #     label = resp.choices[0].message.content.strip().strip('"').strip("'")

        return label

    except Exception as e:
        print(f"  [warn] label failed: {e}")
        return "unlabeled cluster"


def _label_artist_clusters_offline(
    artist: str,
    df: pd.DataFrame,
    embeddings,
    client: "OpenAI",
    model: str,
    n_clusters: int | None = None,
    min_songs_for_clustering: int = 30,
) -> dict:
    """
    Cluster one artist's songs with KMeans and label each cluster with
    the LLM. Returns a plain dict so results can be serialized to JSON.
    """

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    artist_mask = df["artist"] == artist
    artist_df = df[artist_mask].copy().reset_index(drop=True)
    artist_indices = df.index[artist_mask].tolist()
    artist_vecs = embeddings[artist_indices]
    n_songs = len(artist_df)

    # ── Too few songs — skip clustering ───────────────────────────────────
    if n_songs < min_songs_for_clustering:
        print(
            f"  {artist}: only {n_songs} songs — "
            f"skipping clustering, labeling from centroid."
        )
        centroid = artist_vecs.mean(axis=0)
        distances = np.linalg.norm(artist_vecs - centroid, axis=1)
        rep_idx = np.argsort(distances)[:5]
        rep_songs = artist_df.iloc[rep_idx]

        label = _get_cluster_label(
            client=client,
            model=model,
            artist=artist,
            rep_songs=rep_songs,
            used_labels=[],
        )
        songs = [
            {
                "title": row["title"],
                "album": row.get("album", ""),
                "cluster_id": 0,
                "cluster_label": label,
            }
            for _, row in artist_df.iterrows()
        ]
        return {
            "artist": artist,
            "n_clusters": 1,
            "clusterable": False,
            "songs": songs,
            "cluster_labels": {"0": label},
        }

    # ── Auto-select k via silhouette score ────────────────────────────────
    if n_clusters is None:
        best_k, best_score = 2, -1
        for k in range(2, min(5, n_songs // 10 + 1)):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(artist_vecs)
            score = silhouette_score(artist_vecs, labels)
            if score > best_score:
                best_score, best_k = score, k
        n_clusters = best_k
        print(f"   {artist}: k={n_clusters} (silhouette={best_score:.3f})")

    # ── KMeans ────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_ids = km.fit_predict(artist_vecs)
    artist_df["cluster_id"] = cluster_ids

    # ── LLM cluster labeling ──────────────────────────────────────────────
    cluster_labels = {}
    for cid in range(n_clusters):
        mask = artist_df["cluster_id"] == cid
        c_songs = artist_df[mask]
        c_vecs = artist_vecs[mask.values]
        centroid = km.cluster_centers_[cid]
        distances = np.linalg.norm(c_vecs - centroid, axis=1)
        rep_idx = np.argsort(distances)[:5]
        rep_songs = c_songs.iloc[rep_idx]

        label = _get_cluster_label(
            client=client,
            model=model,
            artist=artist,
            rep_songs=rep_songs,
            used_labels=list(cluster_labels.values()),  # pass already-used labels
        )
        cluster_labels[str(cid)] = label

    print(f"  {artist} → {list(cluster_labels.values())}")

    songs = [
        {
            "title": row["title"],
            "album": row.get("album", ""),
            "cluster_id": int(row["cluster_id"]),
            "cluster_label": cluster_labels[str(int(row["cluster_id"]))],
        }
        for _, row in artist_df.iterrows()
    ]

    return {
        "artist": artist,
        "n_clusters": n_clusters,
        "clusterable": True,
        "songs": songs,
        "cluster_labels": cluster_labels,
    }


def precompute_artist_clusters(
    df: pd.DataFrame,
    embeddings,
    model: str = "llama3.2",
    min_songs_for_clustering: int = 30,
    artists: list[str] | None = None,
    force: bool = False,
) -> None:
    """
    Cluster every eligible artist's songs, label the clusters with the LLM,
    and save all results to data/processed/artist_clusters.json.
    """
    client, model = get_ollama_client(model)

    # Load any existing results
    existing = {}
    if CLUSTER_CACHE_PATH.exists() and not force:
        with open(CLUSTER_CACHE_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        existing = {entry["artist"]: entry for entry in saved}
        print(f"Loaded {len(existing)} existing cluster results.")

    target_artists = sorted(artists if artists is not None else df["artist"].unique())

    results = dict(existing)

    for artist in tqdm(target_artists, desc="Clustering artists"):
        if artist in results and not force:
            print(f"   [skip] {artist} already clustered.")
            continue

        result = _label_artist_clusters_offline(
            artist=artist,
            df=df,
            embeddings=embeddings,
            client=client,
            model=model,
            min_songs_for_clustering=min_songs_for_clustering,
        )
        results[artist] = result

        # Save after every artist so progress isn't lost on interruption
        CLUSTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CLUSTER_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)

    print(f"\nDone. Cluster data saved → {CLUSTER_CACHE_PATH}")
    print(f"Processed {len(target_artists)} artists.")


def load_artist_clusters(path: Path = CLUSTER_CACHE_PATH) -> dict[str, dict]:
    """
    Reads pre-saved JSON
    """

    if not path.exists():
        print(
            f"[info] No cluster data found at {path}. "
            "Run `python analysis/themes_llm.py --clusters` to generate it."
        )
        return {}

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    return {entry["artist"]: entry for entry in records}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run theme classification and/or cluster precomputation from the CLI.
 
    All commands require Ollama to be running: `ollama serve`
 
    Usage examples:
        # Classify full corpus (first time)
        python analysis/themes_llm.py
 
        # Resume an interrupted classification run
        python analysis/themes_llm.py
 
        # Only classify newly added artists (auto-detected)
        python analysis/themes_llm.py --new-artists-only
 
        # Classify specific artists
        python analysis/themes_llm.py --artists "Taylor Swift" "SZA"
 
        # Re-classify specific artists from scratch
        python analysis/themes_llm.py --artists "Charli xcx" --no-resume
 
        # Precompute artist clusters (run after classification is complete)
        python analysis/themes_llm.py --clusters
 
        # Recompute clusters for specific artists only
        python analysis/themes_llm.py --clusters --artists "Carly Rae Jepsen"
 
        # Force recompute all clusters from scratch
        python analysis/themes_llm.py --clusters --force
    """
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Classify song themes and precompute clusters using Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clusters",
        action="store_true",
        help=(
            "Precompute artist clusters instead of classifying themes. "
            "Run this after theme classification is complete. "
            "Saves results to data/processed/artist_clusters.json."
        ),
    )
    parser.add_argument(
        "--artists",
        nargs="+",
        default=None,
        help="Restrict operation to these artists only.",
    )
    parser.add_argument(
        "--new-artists-only",
        action="store_true",
        help="Auto-detect artists not yet classified and only process those.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore saved progress — start fresh.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="(Clusters only) Force recompute even for already-saved artists.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2).",
    )
    parser.add_argument(
        "--lyrics-col",
        default="preprocessed_lyrics",
        help="DataFrame column to use as lyrics input.",
    )
    parser.add_argument(
        "--min-songs",
        type=int,
        default=30,
        help="Minimum songs for an artist to be clustered (default: 30).",
    )
    args = parser.parse_args()

    # ── Load corpus ───────────────────────────────────────────────────────
    data_path = ROOT / "data" / "processed" / "final_songs.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run pipeline/build.py first.")
        sys.exit(1)

    df = pd.read_json(data_path)
    print(f"Loaded {len(df)} songs across {df['artist'].nunique()} artists.\n")

    # ── Cluster precomputation mode ───────────────────────────────────────
    if args.clusters:
        embeddings_path = ROOT / "data" / "cache" / "llm_embedding.npy"
        if not embeddings_path.exists():
            print(f"ERROR: {embeddings_path} not found. Run pipeline/build.py first.")
            sys.exit(1)

        # Load themes into df so cluster labeling can reference them
        if THEMES_OUTPUT_PATH.exists():
            df = load_theme_results(df)
        else:
            print(
                "WARNING: song_themes.json not found — "
                "run theme classification before clustering for best results."
            )

        embeddings = np.load(str(embeddings_path))

        precompute_artist_clusters(
            df=df,
            embeddings=embeddings,
            model=args.model,
            min_songs_for_clustering=args.min_songs,
            artists=args.artists,
            force=args.force,
        )

    # ── Theme classification mode ─────────────────────────────────────────
    elif args.new_artists_only:
        update_theme_classifications(
            df=df,
            lyrics_column=args.lyrics_col,
            model=args.model,
        )

    elif args.artists:
        unknown = [a for a in args.artists if a not in df["artist"].unique()]
        if unknown:
            print(f"WARNING: Artists not found in corpus: {unknown}")

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
        classify_corpus(
            df=df,
            lyrics_column=args.lyrics_col,
            model=args.model,
            resume=not args.no_resume,
            artists=None,
        )
