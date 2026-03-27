import numpy as np
import pandas as pd
import umap
from embed import embed_dataset
from filter_songs import filter_songs
from preprocess import preprocess_lyrics
from pathlib import Path
import sys

# Anchor all paths to the project root, regardless of working directory
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PRO = ROOT / "data" / "processed"
DATA_CAC = ROOT / "data" / "cache"

# Make sure pipeline/ modules are importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def embed_umap(embedding: np.array) -> np.array:

    fit = umap.UMAP(n_neighbors=5)
    umap_embedding = fit.fit_transform(embedding)

    return umap_embedding


def main():

    # Ensure output directories exist
    DATA_PRO.mkdir(parents=True, exist_ok=True)
    DATA_CAC.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # Filter songs and save filtered_songs.json file
    df = pd.read_json(DATA_RAW / "lyrics_raw.json")
    df = filter_songs(df)
    print(f"Saving a total of {len(df)} songs after filtering.")
    df.to_json(DATA_PRO / "filtered_songs.json", indent=4, orient="records")

    # ----------------------------------------------------
    # Preprocess lyrics and save final_songs.json file
    df = preprocess_lyrics(df)

    pics = pd.read_json(ROOT / "data" / "artist_thumbnails.json", orient="index")
    pics.columns = ["artist_thumbnail_url"]
    pics.index.name = "artist"
    df = df.join(pics, on="artist")

    print("Saving preprocessed lyrics.")
    df.to_json(DATA_PRO / "final_songs.json", indent=4, orient="records")

    # ----------------------------------------------------
    # Embed lyrics
    embeddings = embed_dataset(df)
    print("Saving embedded lyrics.")
    np.save(DATA_CAC / "llm_embedding.npy", embeddings)

    # ----------------------------------------------------
    # Create and save UMAP embedding
    print("Creating UMAP embedding.")
    umap_embedding = embed_umap(embeddings)
    np.save(DATA_CAC / "umap_embedding.npy", umap_embedding)


if __name__ == "__main__":
    main()
