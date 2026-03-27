import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from analysis.similarity import (
    find_similar_to_song,
    get_top_similar_artists,
    get_song_similarity_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Similarity — Lyrics Analysis", page_icon="🔗", layout="wide"
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
artist_sim_matrix = st.session_state["artist_sim_matrix"]
similarity_stats = st.session_state["similarity_stats"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔗 Similarity Controls")
    st.markdown("---")

    top_n = st.slider("Results to show", min_value=3, max_value=30, value=10)

    st.markdown("---")

    sim_method = st.radio(
        "Artist similarity method",
        options=["Centroid", "Average"],
        index=0,
        help=(
            "Centroid: compare mean artist embeddings (faster). \n"
            "Average: mean of all pairwise song similarities (more robust, slower)."
        ),
    )

    st.markdown("---")
    st.caption(
        "Song-to-song uses cosine similarity in embedding space."
        "Scores closet to 1.0 indicate nearly identical lyric meaning."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔗 Song & Artist Similarity")
st.markdown(
    "Explore lyrical similarity between individual songs, between artists,"
    "and identify the most stylistically distinctive voices in your corpus."
)
st.markdown("---")

