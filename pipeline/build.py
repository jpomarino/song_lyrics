import numpy as np
import pandas as pd
from embed import embed_dataset
from filter_songs import filter_songs
from preprocess import preprocess_lyrics




def main():
    # ----------------------------------------------------
    # Filter songs and save filtered_songs.json file
    df = pd.read_json("../data/raw/lyrics_raw.json")
    df = filter_songs(df)

    print(f"Saving a total of {len(df)} songs after filtering.")

    df.to_json("../data/processed/filtered_songs.json", indent=4, orient="records")

    # ----------------------------------------------------
    # Preprocess lyrics and save final_songs.json file
    df = preprocess_lyrics(df)

    print("Saving preprocessed lyrics.")

    df.to_json("../data/processed/final_songs.json", indent=4, orient="records")

    # ----------------------------------------------------
    # Create word embeddings for each song and save llm_embedding.npy file
    embeddings = embed_dataset(df)

    print("Saving embedded lyrics.")

    # Save embeddings as an npy file
    np.save("../data/cache/llm_embedding.npy", embeddings)


if __name__ == "__main__":
    main()
