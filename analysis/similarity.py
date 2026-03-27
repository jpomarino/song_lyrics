import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CORE SEARCH
# ─────────────────────────────────────────────────────────────────────────────


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


def find_similar_to_song(
    song_title: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_n: int = 10,
    artist: str | None = None,
) -> pd.DataFrame:

    mask = df["title"].str.lower() == song_title.lower()
    if artist:
        mask &= df["artist"].str.lower() == artist.lower()

    matches = df[mask]

    if matches.empty:
        raise ValueError(
            f"Song '{song_title}' not found. "
            "Try passing artist= to disambiguate, or check the title spelling."
        )

    # If multiple matches (e.g. same title, different albums) take the first
    idx = matches.index[0]
    query_vec = embeddings[idx]

    results = get_similar_songs(
        query_vec=query_vec,
        embeddings=embeddings,
        df=df,
        top_n=top_n,
        exclude_index=idx,
    )

    # Attach query metadata as DataFrame attributes for use in Streamlit
    results.attrs["query_title"] = df.iloc[idx]["title"]
    results.attrs["query_artist"] = df.iloc[idx]["artist"]

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


def get_top_similar_artists(
    artist_name: str, similarity_matrix: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:

    if artist_name not in similarity_matrix.index:
        raise ValueError(f"Artist '{artist_name}' not found in similarity matrix.")

    row = similarity_matrix.loc[artist_name].drop(labels=[artist_name])
    top = row.sort_values(ascending=False).head(top_n)

    return pd.DataFrame(
        {
            "rank": range(1, len(top) + 1),
            "artist": top.index,
            "similarity": top.values.round(4),
        }
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SONG-LEVEL SIMILARITY MATRIX (for heatmap / graph visualizations)
# ─────────────────────────────────────────────────────────────────────────────


def get_song_similarity_matrix(
    df: pd.DataFrame, embeddings: np.ndarray, artist: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if artist:
        mask = df["artist"] == artist
        sub_df = df[mask].reset_index(drop=True)
        sub_embeddings = embeddings[df[mask].index]

    else:
        sub_df = df.reset_index(drop=True)
        sub_embeddings = embeddings

    labels = (sub_df["artist"] + " - " + sub_df["title"]).to_list()
    sim_matrix = cosine_similarity(sub_embeddings).round(4)
    sim_df = pd.DataFrame(sim_matrix, index=False, columns=labels)

    return sim_df, sub_df


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY STATS
# ─────────────────────────────────────────────────────────────────────────────
def get_similarity_stats(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    For each artist, compute:
        - intra_similarity: mean cosine similarity between all songs by that artist
        - inter_similarity: mean cosine similarity between that artist's songs and all other artists' songs
        - distinctiveness: intra - inter (higher = more distinctive)

    Args:
        df (pd.DataFrame): DataFrame with all songs
        embeddings (np.ndarray): Lyric embeddings

    Returns:
        pd.DataFrame: DataFrame with columns: artist, intra_similarity, inter_similarity, distinctiveness
    """
    artists = sorted(df["artist"].unique())
    rows = []

    for artist in artists:
        artist_idx = df[df["artist"] == artist].index.tolist()
        other_idx = df[df["artist"] != artist].index.tolist()

        artist_vecs = embeddings[artist_idx]
        other_vecs = embeddings[other_idx]

        # Calculate the intra_similarity
        if len(artist_vecs) > 1:
            intra_matrix = cosine_similarity(artist_vecs)
            # Exclude the diagonal
            np.fill_diagonal(intra_matrix, np.nan)
            intra = np.nanmean(intra_matrix)
        else:
            intra = 1.0

        # Calculate the inter_similarity
        inter = cosine_similarity(artist_vecs, other_vecs).mean()

        rows.append(
            {
                "artist": artist,
                "intra_similarity": round(intra, 4),
                "inter_similarity": round(inter, 4),
                "distinctiveness": round(intra - inter, 4),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("distinctiveness", ascending=False)
        .reset_index(drop=True)
    )
