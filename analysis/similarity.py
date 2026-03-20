import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# embeddings = np.load("../data/cache/llm_embedding.npy")
# df = pd.read_json("../data/processed/final_songs.json")


def get_similar_songs(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    top_n: int = 10,
    exclude_index: int | None = None,
) -> pd.DataFrame:

    # Reshape input
    query_vec = np.array(query_vec).reshape(1, -1)

    # Compute cosine similarity between query and every song
    sims = cosine_similarity(query_vec, embeddings)[0]  # shape: (n_songs,)

    # Exclude a specific index if requested (e.g. the query song itself)
    if exclude_index is not None:
        sims[exclude_index] = -np.inf

    # Get top_n indices sorted by descending similarity
    top_indices = np.argsort(sims)[::-1][:top_n]

    results = df.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices].round(4)
    results.insert(0, "rank", range(1, len(results) + 1))
    results = results.reset_index(drop=True)

    return results


def get_artist_similarit_matrix():
    pass


def get_song_similarity_matrix():
    pass


def find_similar_to_song():
    pass


if __name__ == "__main__":
    pass
    # query = np.ones(384)
    # similar_songs = get_similar_songs(query, embeddings, df)

    # print(similar_songs[['artist', 'title']].head())
