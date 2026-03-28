import sys
from pathlib import Path

# ── Make sure project root is on the path so imports work ─────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from analysis.similarity import get_artist_similarity_matrix, get_similarity_stats

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lyrics Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Tighten up the default Streamlit padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
 
    /* Sidebar artist list styling */
    .sidebar-artist { font-size: 0.85rem; color: #aaa; line-height: 1.8; }
 
    /* Metric card overrides */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADERS  (cached — run once per session)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"
# MODELS_DIR = ROOT / "models"


@st.cache_data(show_spinner="Loading song data...")
def load_data() -> pd.DataFrame:
    """Load the preprocessed lyrics DataFrame."""
    path = DATA_DIR / "final_songs.json"
    if not path.exists():
        st.error(
            f"Data file not found at {path}. Run `python pipeline/build.py` first."
        )
        st.stop()
    df = pd.read_json(path)
    df = df.reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading embeddings...")
def load_embeddings() -> np.ndarray:
    """Load the precomputed embedding matrix."""
    path = CACHE_DIR / "llm_embedding.npy"
    if not path.exists():
        st.error(
            f"Embeddings not found at {path}. Run `python pipeline/build.py` first."
        )
        st.stop()
    return np.load(str(path))


@st.cache_data(show_spinner="Loading UMAP projection...")
def load_umap() -> np.ndarray:
    """Load the precomputed 2D UMAP projection."""
    path = CACHE_DIR / "umap_embedding.npy"
    if not path.exists():
        st.warning(
            "UMAP projection not found. "
            "Run `python pipeline/build.py` to generate it. "
            "The Explore page will be unavailable until then."
        )
        return None
    return np.load(str(path))


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model() -> SentenceTransformer:
    """Load the sentence-transformers model (kept in memory across reruns)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Computing artist similarity matrix...")
def load_artist_similarity(_df, _embeddings):
    """
    Compute and cache the artist similarity matrix.
    Underscored args tell Streamlit not to hash the numpy/pandas objects
    (hashing large arrays is slow) — acceptable here since data is static.
    """
    return get_artist_similarity_matrix(_df, _embeddings, method="centroid")


@st.cache_data(show_spinner="Computing similarity stats...")
def load_similarity_stats(_df, _embeddings):
    return get_similarity_stats(_df, _embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap():
    """
    Load all shared resources into st.session_state on first run.
    Pages read from session_state rather than reloading independently,
    so everything stays in sync after dataset updates.
    """
    if "bootstrapped" in st.session_state:
        return

    with st.spinner("Initialising app — this only happens once..."):
        df = load_data()
        embeddings = load_embeddings()
        umap_2d = load_umap()
        model = load_embed_model()
        # topic_model = load_topics()

        artist_sim_matrix = load_artist_similarity(df, embeddings)
        similarity_stats = load_similarity_stats(df, embeddings)

        # topic_summary = (
        #    get_topic_summary(topic_model)
        #    if topic_model is not None else None
        # )

    st.session_state.update(
        {
            "bootstrapped": True,
            "df": df,
            "embeddings": embeddings,
            "umap_2d": umap_2d,
            "embed_model": model,
            # "topic_model":        topic_model,
            "artist_sim_matrix": artist_sim_matrix,
            "similarity_stats": similarity_stats,
            # "topic_summary":      topic_summary,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.title("🎵 Lyrics Analysis")
        st.markdown("---")

        st.markdown("### Navigation")
        st.page_link("main.py", label="🏠  Overview", icon=None)
        st.page_link("pages/01_explore.py", label="🗺️  Explore", icon=None)
        # st.page_link("app/pages/02_recommend.py",      label="🔍  Recommend",           icon=None)
        # st.page_link("app/pages/03_themes.py",         label="🎨  Themes",              icon=None)
        # st.page_link("app/pages/04_similarity.py",     label="🔗  Song Similarity",     icon=None)
        # st.page_link("app/pages/05_scrape.py",         label="➕  Add Artist",          icon=None)

        st.markdown("---")

        # Artists in the corpus
        if "df" in st.session_state:
            df = st.session_state["df"]
            st.markdown("### Artists in corpus")
            counts = df["artist"].value_counts()
            for artist, count in counts.items():
                st.markdown(
                    f"<div class='sidebar-artist'>🎤 {artist} &nbsp;·&nbsp; {count} songs</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.caption("Built with sentence-transformers, BERTopic & Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────


def render_overview():
    df = st.session_state["df"]
    embeddings = st.session_state["embeddings"]
    similarity_stats = st.session_state["similarity_stats"]
    artist_sim_matrix = st.session_state["artist_sim_matrix"]
    # topic_summary    = st.session_state["topic_summary"]

    st.title("🎵 Lyrics Analysis — Overview")
    st.markdown(
        "Explore themes, similarities, and recommendations across a "
        f"lyrics corpus of **{len(df):,} songs** by **{df['artist'].nunique()} artists**."
    )
    st.markdown("---")

    # ── Top-level metrics ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Songs", f"{len(df):,}")
    col2.metric("Artists", df["artist"].nunique())
    col3.metric("Albums", df["album"].nunique() if "album" in df.columns else "—")
    # col4.metric(
    #    "Topics Discovered",
    #    len(topic_summary) if topic_summary is not None else "—"
    # )

    st.markdown("---")

    # ── Songs per artist bar chart ────────────────────────────────────────
    st.subheader("Songs per Artist")
    counts = df["artist"].value_counts().reset_index()
    counts.columns = ["artist", "count"]
    fig_bar = px.bar(
        counts,
        x="artist",
        y="count",
        color="artist",
        labels={"artist": "Artist", "count": "Songs"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_bar.update_layout(
        showlegend=False,
        xaxis_tickangle=-30,
        height=380,
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Artist similarity heatmap ─────────────────────────────────────────
    st.subheader("Artist Similarity Heatmap")
    st.caption(
        "Cosine similarity between artist centroids in embedding space. "
        "Higher values indicate more similar lyrical content and themes."
    )
    fig_heat = px.imshow(
        artist_sim_matrix,
        color_continuous_scale="RdBu",
        zmin=0,
        zmax=1,
        aspect="auto",
        template="plotly_dark",
        labels=dict(color="Similarity"),
    )
    fig_heat.update_layout(
        height=700,  # taller to give labels more room
        margin=dict(t=20, b=120, l=120, r=20),  # extra margin for rotated labels
    )
    fig_heat.update_xaxes(
        tickangle=-45,
        tickfont=dict(size=10),
    )
    fig_heat.update_yaxes(
        tickfont=dict(size=10),
    )
    fig_heat.update_traces(
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Distinctiveness table ─────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Artist Distinctiveness")
        st.caption(
            "Intra-artist similarity minus inter-artist similarity. "
            "Higher = more stylistically cohesive and unique."
        )
        st.dataframe(
            similarity_stats.style.background_gradient(
                subset=["distinctiveness"], cmap="RdYlGn"
            ),
            use_container_width=True,
            height=380,
        )

    # ── Top topics ────────────────────────────────────────────────────────
    # with col_b:
    #    if topic_summary is not None:
    #        st.subheader("Top Discovered Topics")
    #        st.caption("Most common lyrical themes across the full corpus.")
    #        fig_topics = px.bar(
    #            topic_summary.head(15),
    #            x="song_count",
    #            y="topic_label",
    #            orientation="h",
    #            color="song_count",
    #            color_continuous_scale="Viridis",
    #            labels={"song_count": "Songs", "topic_label": "Topic"},
    #            template="plotly_dark",
    #        )
    #        fig_topics.update_layout(
    #            showlegend=False,
    #            height=380,
    #            margin=dict(t=10, b=10, l=10, r=10),
    #            yaxis=dict(autorange="reversed"),
    #            coloraxis_showscale=False,
    #        )
    #        st.plotly_chart(fig_topics, use_container_width=True)
    #    else:
    #        st.info(
    #            "Topic model not found. "
    #            "Run `python pipeline/build.py` to discover themes."
    #        )

    # st.markdown("---")

    # ── Corpus breakdown table ────────────────────────────────────────────
    st.subheader("Corpus Breakdown")

    # Build one thumbnail URL per artist (take the first non-null value)
    thumbnails = (
        df.groupby("artist")["artist_thumbnail_url"]
        .first()
        .reset_index()
        .rename(columns={"artist_thumbnail_url": "thumbnail"})
    )

    breakdown = (
        df.groupby("artist")
        .agg(
            songs=("title", "count"),
            albums=("album", lambda x: x.nunique()),
            avg_lyric_length=(
                "preprocessed_lyrics",
                lambda x: int(x.str.split().str.len().mean()),
            ),
        )
        .reset_index()
        .sort_values("songs", ascending=False)
        .merge(thumbnails, on="artist", how="left")
    )

    # Reorder so thumbnail is the first column
    breakdown = breakdown[
        ["thumbnail", "artist", "songs", "albums", "avg_lyric_length"]
    ]
    breakdown.columns = ["", "Artist", "Songs", "Albums", "Avg Words / Song"]

    st.dataframe(
        breakdown,
        use_container_width=True,
        hide_index=True,
        column_config={
            "": st.column_config.ImageColumn(
                label="",
                width="small",  # renders as a medium square thumbnail
                help="Artist photo from Genius",
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main():
    bootstrap()
    render_sidebar()
    render_overview()


if __name__ == "__main__" or True:
    main()
