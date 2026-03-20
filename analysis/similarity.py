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


# ─────────────────────────────────────────────────────────────────────────────
# ARTIST-LEVEL SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────
def get_artist_similarity_matrix(
    df: pd.DataFrame, embeddings: np.ndarray, method: str = "centroid"
) -> pd.DataFrame:
    artists = sorted(df["artist"].unique())
    n = len(artists)
    matrix = np.zeros((n, n))

    if method == "centroid":
        # Compute one mean vector per artist
        centroids = {}
        for artist in artists:
            idx = df[df["artist"] == artist].index.tolist()
            vecs = embeddings[idx]
            centroid = vecs.mean(axis=0)
            # Re-normalize the centroid
            # norm = np.linalg.norm(centroid)
            centroids[artist] = centroid  # / norm if norm > 0 else centroid

        centroid_matrix = np.vstack([centroids[a] for a in artists])
        sim = cosine_similarity(centroid_matrix)
        matrix = sim

    elif method == "average":
        # Average all pairwise song similarities between each artist pair
        artist_indices = {
            artist: df[df["artist"] == artist].index.tolist() for artist in artists
        }
        for i, a1 in enumerate(artists):
            for j, a2 in enumerate(artists):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                if j < i:
                    # Mirror — already computed
                    matrix[i, j] = matrix[j, i]
                    continue
                vecs_a = embeddings[artist_indices[a1]]
                vecs_b = embeddings[artist_indices[a2]]
                sim = cosine_similarity(vecs_a, vecs_b).mean()
                matrix[i, j] = sim
                matrix[j, i] = sim
    else:
        raise ValueError(f"method must be 'centroid' or 'average', got '{method}'")

    return pd.DataFrame(matrix, index=artists, columns=artists).round(4)


def get_song_similarity_matrix():
    pass


def find_similar_to_song():
    pass


if __name__ == "__main__":
    pass
    # query = np.ones(384)
    # similar_songs = get_similar_songs(query, embeddings, df)

    # print(similar_songs[['artist', 'title']].head())
