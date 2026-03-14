import re

import pandas as pd

MAX_WORDS_PER_CHUNK = 200
CHUNK_OVERLAP = 30

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = re.sub(r"\[.*?\]", "", text)  # remove [Verse 1], [Chorus] etc.
    text = re.sub(r"\s+", " ", text)  # collapse multiple whitespace
    return text.strip()


def preprocess_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    # Clean up lyrics for text processing
    df["preprocessed_lyrics"] = df["lyrics"].apply(clean_text)
    return df


def main():
    # Import the json file with all the songs
    df = pd.read_json("../data/processed/filtered_songs.json")

    # Preprocess the lyrics
    df = preprocess_lyrics(df)

    # Save records as a JSON file
    df.to_json("../data/processed/final_songs.json", indent=4, orient="records")


if __name__ == "__main__":
    main()
