import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MAX_WORDS_PER_CHUNK = 200
CHUNK_OVERLAP = 30

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def chunk_lyrics(
    lyrics: str, max_words: int = MAX_WORDS_PER_CHUNK, overlap: int = CHUNK_OVERLAP
) -> list[str]:

    words = lyrics.split()

    if len(words) <= max_words:
        return words

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start += max_words - overlap

    return chunks


def embed_song(
    lyrics: str, model: SentenceTransformer, embedding_dim: int = EMBEDDING_DIM
):

    # If the lyrics are not a string, or have no words: return zeros
    if not isinstance(lyrics, str) or len(lyrics.strip()) == 0:
        return np.zeros(embedding_dim)

    # Chunk the songs due to the model's token limit
    chunks = chunk_lyrics(lyrics)
    chunk_embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize each chunk to normalize vector magnitudes for different sized chunks
    )

    # Average the chunks weighted by the length of the chunks
    word_counts = np.array([len(c.split()) for c in chunks], dtype=float)
    weights = word_counts / word_counts.sum()
    averaged = np.average(chunk_embeddings, axis=0, weights=weights)

    # Re-normalize the averaged vector
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm

    return averaged


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print("Loading embedding model.")
    return SentenceTransformer(model_name)


def embed_dataset(df: pd.DataFrame) -> np.array:
    # Load embedding model
    model = load_model()

    # Empty numpy array to store the embeddings
    embeddings = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)

    # Embed every song
    for i, lyrics in enumerate(tqdm(df["preprocessed_lyrics"], desc="Embedding")):
        embeddings[i] = embed_song(lyrics, model)

    return embeddings


def main():

    # Import the json file with all the songs
    df = pd.read_json("../data/processed/final_songs.json")

    # Embed the dataset
    embeddings = embed_dataset(df)

    # Save embeddings as an npy file
    np.save("../data/cache/llm_embedding.npy", embeddings)


if __name__ == "__main__":
    main()
