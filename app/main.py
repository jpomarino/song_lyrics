import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from analysis.similarity import (
    get_artist_similarity_matrix,
    get_similarity_stats,
)


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
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .sidebar-artist  { font-size: 0.85rem; color: #aaa; line-height: 1.8; }
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
    }
    .pipeline-step {
        background: #1e1e2e;
        border-left: 3px solid #7c6af7;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #ffffff;          /* force white text regardless of Streamlit theme */
    }
    .pipeline-step strong {
        color: #ffffff;
    }
    .pipeline-step code {
        background: #2e2e42;
        color: #c8b4ff;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
    }
    .tech-badge {
        display: inline-block;
        background: #2a2a3e;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        color: #ffffff;          /* white text on dark badge */
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Colour palette used across all flat bar charts ────────────────────────────
# A single consistent colour per chart makes it clear colour carries no meaning.
BAR_COLOUR = "#a78bfa"  # soft violet — readable on both light and dark themes
WHISKER_COLOUR = "#e2d9f3"  # very light lavender — high contrast against dark bars


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"


@st.cache_data(show_spinner="Loading song data...")
def load_data() -> pd.DataFrame:
    path = DATA_DIR / "final_songs.json"
    if not path.exists():
        st.error(
            f"Data file not found at {path}. Run `python pipeline/build.py` first."
        )
        st.stop()
    df = pd.read_json(path).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading embeddings...")
def load_embeddings() -> np.ndarray:
    path = CACHE_DIR / "llm_embedding.npy"
    if not path.exists():
        st.error(
            f"Embeddings not found at {path}. Run `python pipeline/build.py` first."
        )
        st.stop()
    return np.load(str(path))


@st.cache_data(show_spinner="Loading UMAP projection...")
def load_umap() -> np.ndarray | None:
    path = CACHE_DIR / "umap_embedding.npy"
    if not path.exists():
        return None
    return np.load(str(path))


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Computing artist similarity matrix...")
def load_artist_similarity(_df, _embeddings):
    return get_artist_similarity_matrix(_df, _embeddings, method="centroid")


@st.cache_data(show_spinner="Computing similarity stats...")
def load_similarity_stats(_df, _embeddings):
    return get_similarity_stats(_df, _embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS-LEVEL EDA COMPUTATIONS (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def compute_lexical_diversity(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lexical diversity = unique words / total words per song, averaged by artist.
    Answers: which artists have the most varied vocabulary?
    """

    def diversity(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return None
        words = text.lower().split()
        return len(set(words)) / len(words) if words else None

    df = _df.copy()
    df["lex_div"] = df["preprocessed_lyrics"].apply(diversity)
    result = (
        df.groupby("artist")["lex_div"]
        .mean()
        .reset_index()
        .rename(columns={"lex_div": "lexical_diversity"})
        .sort_values("lexical_diversity", ascending=False)
    )
    result["lexical_diversity"] = result["lexical_diversity"].round(4)
    return result


@st.cache_data
def compute_song_length(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average words per song by artist.
    Answers: who writes the longest / shortest songs?
    """
    df = _df.copy()
    df["word_count"] = df["preprocessed_lyrics"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )
    result = (
        df.groupby("artist")["word_count"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "avg_words", "min": "min_words", "max": "max_words"})
        .sort_values("avg_words", ascending=False)
    )
    result["avg_words"] = result["avg_words"].round(0).astype(int)
    return result


@st.cache_data
def compute_release_timeline(_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Songs per release year per artist.
    Answers: when were these artists most prolific?
    """
    if "release_date" not in _df.columns:
        return None
    df = _df.copy()
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    result = df.groupby(["year", "artist"]).size().reset_index(name="songs")
    return result


@st.cache_data
def compute_corpus_breakdown(_df: pd.DataFrame) -> pd.DataFrame:
    """Full per-artist summary table for the corpus breakdown section."""
    df = _df.copy()
    df["word_count"] = df["preprocessed_lyrics"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )

    def lex_div(text):
        if not isinstance(text, str) or not text.strip():
            return None
        words = text.lower().split()
        return round(len(set(words)) / len(words), 3) if words else None

    df["lex_div"] = df["preprocessed_lyrics"].apply(lex_div)

    result = (
        df.groupby("artist")
        .agg(
            Songs=("title", "count"),
            Albums=("album", "nunique"),
            avg_words=("word_count", "mean"),
            avg_lex_div=("lex_div", "mean"),
        )
        .reset_index()
        .sort_values("Songs", ascending=False)
    )
    result["Avg Words / Song"] = result["avg_words"].round(0).astype(int)
    result["Lexical Diversity"] = result["avg_lex_div"].round(3)
    result = result[
        ["artist", "Songs", "Albums", "Avg Words / Song", "Lexical Diversity"]
    ]
    result.columns = [
        "Artist",
        "Songs",
        "Albums",
        "Avg Words / Song",
        "Lexical Diversity",
    ]

    # Add thumbnail if available
    if "artist_thumbnail_url" in df.columns:
        thumbs = df.groupby("artist")["artist_thumbnail_url"].first().reset_index()
        thumbs.columns = ["Artist", "thumbnail"]
        result = result.merge(thumbs, on="Artist", how="left")
        result = result[
            [
                "thumbnail",
                "Artist",
                "Songs",
                "Albums",
                "Avg Words / Song",
                "Lexical Diversity",
            ]
        ]
        result.columns = [
            "",
            "Artist",
            "Songs",
            "Albums",
            "Avg Words / Song",
            "Lexical Diversity",
        ]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap():
    if "bootstrapped" in st.session_state:
        return

    with st.spinner("Loading app — this only happens once per session..."):
        df = load_data()
        embeddings = load_embeddings()
        umap_2d = load_umap()
        model = load_embed_model()

        artist_sim_matrix = load_artist_similarity(df, embeddings)
        similarity_stats = load_similarity_stats(df, embeddings)

    st.session_state.update(
        {
            "bootstrapped": True,
            "df": df,
            "embeddings": embeddings,
            "umap_2d": umap_2d,
            "embed_model": model,
            "artist_sim_matrix": artist_sim_matrix,
            "similarity_stats": similarity_stats,
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
        st.page_link("main.py", label="🏠  Home")
        st.page_link("pages/01_explore.py", label="🗺️  Explore Embeddings")
        st.page_link("pages/02_recommend.py", label="🔍  Song Recommender")
        st.page_link("pages/03_themes.py", label="🎨  Theme Analysis")
        st.page_link("pages/04_similarity.py", label="🔗  Artist & Song Similarity")

        st.markdown("---")

        if "df" in st.session_state:
            df = st.session_state["df"]
            counts = df["artist"].value_counts()
            st.markdown("### Artists in corpus")
            for artist, count in counts.items():
                st.markdown(
                    f"<div class='sidebar-artist'>"
                    f"🎤 {artist} &nbsp;·&nbsp; {count} songs"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.caption("Built with Sentence Transformers, Ollama & Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────


def render_overview():
    df = st.session_state["df"]

    # ── Hero ──────────────────────────────────────────────────────────────
    st.title("🎵 Lyrics Analysis")
    st.markdown(
        "#### Exploring the language of music, from vocabulary and themes "
        "to lyrical similarity and song recommendations."
    )
    st.markdown("---")

    # ── Motivation ────────────────────────────────────────────────────────
    st.subheader("Why This Project?")
    st.markdown(
        """
        I have always cared deeply about song lyrics as the primary reason I
        connect with an artist, an album, or a song. Over time, I found myself
        asking questions I couldn't answer on a larger scale: *Do artists I love
        actually write about similar things, or do I just like how they sound?
        Are there songs I haven't heard yet that I'd love based on their lyrical
        content? What makes one artist's writing feel distinct from another's?*
 
        This project is an attempt to answer those questions rigorously, using NLP
        and machine learning on a corpus of lyrics I scraped from
        [Genius](https://genius.com). The result is an interactive tool that lets
        you explore lyrical similarity, discover thematic patterns, and find new
        songs based on a word, phrase, or lyric you already love.
        """
    )

    st.markdown("---")

    # ── Top-level corpus metrics ──────────────────────────────────────────
    n_songs = len(df)
    n_artists = df["artist"].nunique()
    n_albums = df["album"].nunique() if "album" in df.columns else 0
    avg_words = int(
        df["preprocessed_lyrics"]
        .apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        .mean()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Songs in corpus", f"{n_songs:,}")
    c2.metric("Artists", n_artists)
    c3.metric("Albums", f"{n_albums:,}")
    c4.metric("Avg words / song", avg_words)

    st.markdown("---")

    # ── Pipeline overview ─────────────────────────────────────────────────
    st.subheader("How It Works — Project Pipeline")
    st.markdown(
        "Each song in this app went through the following pipeline before "
        "anything was visualised or analysed."
    )

    steps = [
        (
            "🕷️  1. Scraping",
            "Lyrics, album metadata, and artist images were scraped from "
            "Genius using the `lyricsgenius` Python library. "
            "Remixes, live versions, demos, and instrumentals were excluded.",
        ),
        (
            "🧹  2. Filtering & Cleaning",
            "Duplicate songs were removed. Songs with less than 30 words were removed."
            "Non-song results were manually removed. Lyrics were lightly cleaned "
            "(section headers like `[Verse 1]` stripped, whitespace collapsed) "
            "preserving natural language for the embedding model.",
        ),
        (
            "🔢  3. Vector Embeddings",
            "Each song's lyrics were converted into a **384-dimensional vector** "
            "using `all-MiniLM-L6-v2` from Sentence Transformers. "
            "Songs that are semantically similar end up close together in this space. "
            "Long lyrics were chunked into overlapping windows and averaged.",
        ),
        (
            "🗺️  4. UMAP Projection",
            "The 384-dimensional embeddings were projected into **2D** using UMAP "
            "for visualisation. This is what you see on the Explore page — "
            "every point is a song, and proximity means lyrical similarity.",
        ),
        (
            "📐  5. Similarity Calculations",
            "**Cosine similarity** between embedding vectors powers the song "
            "recommender and the artist similarity heatmap. "
            "Songs or artists with similar language score close to 1.0.",
        ),
        (
            "🏷️  6. Theme Classification",
            "Each song was classified into 1–2 themes from a fixed 18-label "
            "taxonomy using **Llama 3.2** running locally via Ollama. "
            "This was run once offline — the app reads pre-saved results.",
        ),
    ]

    for title, body in steps:
        st.markdown(
            f"<div class='pipeline-step'><strong>{title}</strong><br>{body}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Songs per artist ──────────────────────────────────────────────────
    st.subheader("How Many Songs Does Each Artist Contribute?")
    st.caption(
        "Understanding corpus balance matters; artists with more songs have "
        "more influence on similarity calculations and theme distributions. "
        "Artists with very few songs may produce less reliable analysis."
    )

    counts_df = df["artist"].value_counts().reset_index()
    counts_df.columns = ["artist", "songs"]
    median_songs = int(counts_df["songs"].median())

    fig_songs = px.bar(
        counts_df,
        x="artist",
        y="songs",
        text="songs",
        labels={"artist": "Artist", "songs": "Songs"},
        template="plotly_dark",
    )
    fig_songs.update_traces(
        marker_color=BAR_COLOUR,
        textposition="outside",
    )
    fig_songs.add_hline(
        y=median_songs,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Median: {median_songs}",
        annotation_position="top right",
    )
    fig_songs.update_layout(
        showlegend=False,
        xaxis_tickangle=-35,
        height=420,
        margin=dict(t=20, b=80, l=10, r=40),
    )
    st.plotly_chart(fig_songs, use_container_width=True)

    st.markdown("---")

    # ── Lexical diversity ─────────────────────────────────────────────────
    st.subheader("Who Has the Most Varied Vocabulary?")
    st.caption(
        "Lexical diversity = unique words ÷ total words, averaged across all songs. "
        "A score of 1.0 means every word is used exactly once (maximally varied). "
        "A low score means the artist repeats words heavily — common in songs "
        "with long, repetitive choruses. This is a measure of vocabulary range, "
        "not lyrical quality."
    )

    lex_df = compute_lexical_diversity(df)

    fig_lex = px.bar(
        lex_df,
        x="lexical_diversity",
        y="artist",
        orientation="h",
        color="lexical_diversity",
        color_continuous_scale="Purples",
        text=lex_df["lexical_diversity"].map("{:.3f}".format),
        labels={"lexical_diversity": "Lexical Diversity", "artist": "Artist"},
        template="plotly_dark",
    )
    fig_lex.update_traces(textposition="outside")
    fig_lex.update_layout(
        height=max(380, n_artists * 32),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(range=[0, lex_df["lexical_diversity"].max() * 1.15]),
    )
    st.plotly_chart(fig_lex, use_container_width=True)

    st.markdown("---")

    # ── Song length ───────────────────────────────────────────────────────
    st.subheader("Who Writes the Longest Songs?")
    st.caption(
        "Average word count per song by artist. "
        "Artists with longer songs give the embedding model more lyrical "
        "context to work with, which generally produces more reliable similarity scores. "
        "Error bars show the min–max range across that artist's songs."
    )

    len_df = compute_song_length(df)

    fig_len = go.Figure()
    fig_len.add_trace(
        go.Bar(
            x=len_df["artist"],
            y=len_df["avg_words"],
            marker_color=BAR_COLOUR,  # uniform colour for all bars
            text=len_df["avg_words"],
            textposition="outside",
            name="Avg words",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(len_df["max_words"] - len_df["avg_words"]).tolist(),
                arrayminus=(len_df["avg_words"] - len_df["min_words"]).tolist(),
                color=WHISKER_COLOUR,  # light colour so whiskers show above bars
                thickness=2,
                width=6,  # add caps so whisker tips are visible
            ),
        )
    )
    fig_len.update_layout(
        template="plotly_dark",
        height=460,
        showlegend=False,
        xaxis_tickangle=-35,
        yaxis_title="Words per Song",
        yaxis=dict(
            # add headroom above the tallest bar+whisker so top cap isn't clipped
            range=[0, (len_df["max_words"].max()) * 1.15],
        ),
        margin=dict(t=30, b=80, l=10, r=20),
    )
    st.plotly_chart(fig_len, use_container_width=True)

    st.markdown("---")

    # ── Release timeline ──────────────────────────────────────────────────
    timeline_df = compute_release_timeline(df)
    if timeline_df is not None and not timeline_df.empty:
        st.subheader("When Were These Songs Released?")
        st.caption(
            "Number of songs released each year across the full corpus. "
            "Peaks correspond to prolific album cycles; gaps reveal quieter periods. "
            "Use the toggle below to view a single artist's release history."
        )

        all_artists = sorted(df["artist"].unique())

        view_mode = st.radio(
            "View",
            options=["All artists combined", "Single artist"],
            horizontal=True,
            key="timeline_mode",
        )

        if view_mode == "All artists combined":
            # Aggregate across all artists — one clean line
            agg = timeline_df.groupby("year")["songs"].sum().reset_index()
            fig_timeline = px.line(
                agg,
                x="year",
                y="songs",
                markers=True,
                labels={"year": "Release Year", "songs": "Songs Released"},
                template="plotly_dark",
            )
            fig_timeline.update_traces(
                line=dict(color=BAR_COLOUR, width=2.5),
                marker=dict(color=BAR_COLOUR, size=7),
            )
            fig_timeline.update_layout(
                height=380,
                xaxis=dict(dtick=1, tickangle=-45),
                margin=dict(t=10, b=60, l=10, r=10),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        else:
            selected_tl_artist = st.selectbox(
                "Select artist",
                options=all_artists,
                key="timeline_artist_select",
            )
            artist_tl = (
                timeline_df[timeline_df["artist"] == selected_tl_artist]
                .groupby("year")["songs"]
                .sum()
                .reset_index()
            )
            fig_timeline = px.line(
                artist_tl,
                x="year",
                y="songs",
                markers=True,
                labels={"year": "Release Year", "songs": "Songs Released"},
                template="plotly_dark",
                title=f"Release History — {selected_tl_artist}",
            )
            fig_timeline.update_traces(
                line=dict(color=BAR_COLOUR, width=2.5),
                marker=dict(color=BAR_COLOUR, size=8),
            )
            fig_timeline.update_layout(
                height=380,
                xaxis=dict(dtick=1, tickangle=-45),
                margin=dict(t=40, b=60, l=10, r=10),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("---")

    # ── Corpus breakdown table ────────────────────────────────────────────
    st.subheader("Full Corpus Breakdown")
    st.caption(
        "One row per artist. Lexical Diversity is averaged across all their songs. "
        "Use this table to spot imbalances before interpreting analysis results."
    )

    breakdown = compute_corpus_breakdown(df)
    has_thumbnails = breakdown.columns[0] == ""

    col_config = {}
    if has_thumbnails:
        col_config[""] = st.column_config.ImageColumn(
            label="", width="small", help="Artist image from Genius"
        )

    st.dataframe(
        breakdown,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )

    st.markdown("---")

    # ── Tech stack ────────────────────────────────────────────────────────
    st.subheader("Tech Stack")
    st.markdown(
        "This project was built as a personal data science learning exercise, "
        "touching scraping, NLP preprocessing, embedding models, unsupervised "
        "learning, LLM classification, and interactive deployment."
    )

    stack = {
        "Data collection": ["lyricsgenius", "Genius API", "requests", "BeautifulSoup"],
        "NLP & embeddings": ["sentence-transformers", "spaCy", "NLTK", "scikit-learn"],
        "Dimensionality reduction": ["UMAP"],
        "LLM theme classification": ["Ollama", "Llama 3.2", "openai (client)"],
        "Visualisation": ["Plotly", "Streamlit"],
        "Language": ["Python 3.12"],
    }

    for category, tools in stack.items():
        badges = "".join(f"<span class='tech-badge'>{t}</span>" for t in tools)
        st.markdown(
            f"**{category}** &nbsp; {badges}",
            unsafe_allow_html=True,
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
