import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.similarity import get_similar_songs

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Recommend — Lyrics Analysis", page_icon="🔍", layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────
if "bootstrapped" not in st.session_state:
    st.warning("Please start the app from 'app/main.py'.")
    st.stop()

df = st.session_state["df"]
embeddings = st.session_state["embeddings"]
umap_2d = st.session_state["umap_2d"]
embed_model = st.session_state["embed_model"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string using a shared model."""
    vec = embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[
        0
    ]
    return vec


def clean_query(text: str) -> str:
    """Light cleaning with regex."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Recommend Controls")
    st.markdown("---")

    top_n = st.slider("Number of results", min_value=3, max_value=30, value=10)

    all_artists = sorted(df["artist"].unique())
    filter_artists = st.multiselect(
        "Restrict to artists (optional)",
        options=all_artists,
        default=[],
        help="Leave empty to search full corpus.",
    )

    st.markdown("---")

    show_umap = st.checkbox("Show query on UMAP", value=True)

    st.markdown("---")
    st.caption(
        "Tip: try a lyric snippet, a mood, or an abstract concept like 'feeling like a plastic bag'."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 Song Recommendation")
st.markdown(
    "Enter a word, phrase, or lyric below. "
    "The app will find the songs in the corpus whose lyrics are most "
    "semantically similar to your input."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# QUERY INPUT
# ─────────────────────────────────────────────────────────────────────────────

query_input = st.text_area(
    label="Your query",
    placeholder=(
        "e.g.   'struggles of fame'   ·   'coffee place vibes'   ·   'crisis of faith'"
    ),
    height=100,
    help="Anything from a single word to a full lyric snippet works.",
)

search_clicked = st.button(
    "🔍 Find Similar Songs", type="primary", use_container_width=True
)

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────────────────────

if search_clicked and query_input.strip():
    cleaned_query = clean_query(query_input.strip())

    search_df = df.copy()
    search_embeddings = embeddings.copy()

    # Optionally restrict to a subset of artists
    if filter_artists:
        mask = df["artist"].isin(filter_artists)
        search_df = df[mask].reset_index(drop=True)
        search_embeddings = embeddings[df[mask].index]

    with st.spinner("Embedding query and searching . . . "):
        query_vec = embed_query(cleaned_query)
        results = get_similar_songs(
            query_vec=query_vec, embeddings=search_embeddings, df=search_df, top_n=top_n
        )

    st.markdown("---")

    # ─── Results table ──────────────────────────────────────────────────────────────────────────
    st.subheader(f'Top {top_n} songs matching: *"{cleaned_query}"*')

    results = results[
        [
            "rank",
            "album_cover_url",
            "artist",
            "title",
            "album",
            "release_date",
            "similarity",
        ]
    ]

    # Color code the similarity score column
    styled = results.style.background_gradient(
        subset=["similarity"],
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
    )  # .format({"similarity": ":.4f"})

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "album_cover_url": st.column_config.ImageColumn(
                label="album cover",
                width="small",  # renders as a medium square thumbnail
                help="Album photo from Genius",
            ),
        },
    )

    st.markdown("---")
