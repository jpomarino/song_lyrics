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

    # # ── Similarity distribution chart ─────────────────────────────────────
    # st.subheader("Similarity Score Distribution")
    # st.caption("How similar the top results are relative to each other.")

    # fig_bar = px.bar(
    #     results,
    #     x="similarity",
    #     y="title",
    #     orientation="h",
    #     color="similarity",
    #     color_continuous_scale="YlGn",
    #     hover_data=["artist", "album"],
    #     labels={"similarity": "Cosine Similarity", "title": "Song"},
    #     template="plotly_dark",
    # )
    # fig_bar.update_layout(
    #     height=max(300, top_n * 30),
    #     yaxis=dict(autorange="reversed"),
    #     coloraxis_showscale=False,
    #     margin=dict(t=10, b=10, l=10, r=10),
    # )
    # st.plotly_chart(fig_bar, use_container_width=True)

    # ── UMAP overlay ──────────────────────────────────────────────────────
    if show_umap and umap_2d is not None:
        st.subheader("Query Position in Embedding Space")
        st.caption(
            "The ⭐ shows where your query lands in the UMAP projection. "
            "The highlighted points are the top results."
        )

        # Project query vector into 2D UMAP space using the fitted reducer
        # We approximate by finding the weighted centroid of top results
        top_indices_global = df[
            df["title"].isin(results["title"]) & df["artist"].isin(results["artist"])
        ].index.tolist()[:top_n]

        # Build base scatter plot
        plot_df = pd.DataFrame(
            {
                "x": umap_2d[:, 0],
                "y": umap_2d[:, 1],
                "artist": df["artist"],
                "title": df["title"],
                "is_result": [i in top_indices_global for i in range(len(df))],
            }
        )

        fig_umap = px.scatter(
            plot_df[~plot_df["is_result"]],
            x="x",
            y="y",
            color="artist",
            hover_name="title",
            template="plotly_dark",
            opacity=0.25,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_umap.update_traces(marker=dict(size=4))

        # Highlight top results
        result_points = plot_df[plot_df["is_result"]]
        fig_umap.add_trace(
            go.Scatter(
                x=result_points["x"],
                y=result_points["y"],
                mode="markers",
                marker=dict(
                    size=12,
                    color="gold",
                    line=dict(width=1.5, color="white"),
                    symbol="circle",
                ),
                text=result_points["title"],
                hovertemplate="<b>%{text}</b><extra>Top Result</extra>",
                name="Top Results",
            )
        )

        # Show query centroid as a star
        if len(top_indices_global) > 0:
            qx = umap_2d[top_indices_global, 0].mean()
            qy = umap_2d[top_indices_global, 1].mean()
            fig_umap.add_trace(
                go.Scatter(
                    x=[qx],
                    y=[qy],
                    mode="markers+text",
                    text=["⭐ Query"],
                    textposition="top center",
                    textfont=dict(color="white", size=12),
                    marker=dict(
                        size=18,
                        color="red",
                        symbol="star",
                        line=dict(width=1.5, color="white"),
                    ),
                    name="Query",
                    hovertemplate=f"<b>Query:</b> {cleaned_query}<extra></extra>",
                )
            )

        fig_umap.update_layout(
            height=550,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_umap, use_container_width=True)

    # # ── Matching themes ───────────────────────────────────────────────────
    # if show_topics and topic_model is not None:
    #     st.markdown("---")
    #     st.subheader("Matching Themes")
    #     st.caption("Topics in the corpus that are most semantically related to your query.")

    #     try:
    #         topic_matches = find_topic_for_query(
    #             query_text=cleaned_query,
    #             model=topic_model,
    #             embed_query_fn=embed_query,
    #             top_n=5,
    #         )
    #         st.dataframe(topic_matches, use_container_width=True, hide_index=True)
    #     except Exception as e:
    #         st.info(f"Could not compute topic matches: {e}")

elif search_clicked and not query_input.strip():
    st.warning("Please enter a query before searching.")
