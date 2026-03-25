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
    vec = embed_model.encode([text], convert_to_numpy=True, normalize_emgeddings=True)[
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
