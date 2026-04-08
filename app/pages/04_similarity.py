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
    get_artist_similarity_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Similarity — Lyrics Analysis",
    page_icon="🔗",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────

if "bootstrapped" not in st.session_state:
    st.warning("Please start the app from `app/main.py`.")
    st.stop()

df = st.session_state["df"]
embeddings = st.session_state["embeddings"]
umap_2d = st.session_state["umap_2d"]
artist_sim_matrix = st.session_state["artist_sim_matrix"]
similarity_stats = st.session_state["similarity_stats"]
all_artists = sorted(df["artist"].unique())


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔗 Similarity Controls")
    st.markdown("---")

    top_n = st.slider("Results to show", min_value=3, max_value=30, value=10)

    st.markdown("---")
    st.caption(
        "Similarity scores are cosine distances in 384-dimensional "
        "embedding space. 1.0 = identical, 0.0 = completely unrelated."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔗 Song & Artist Similarity")
st.markdown(
    """
    This page explores lyrical similarity at two scales: **song-level**
    (which individual songs are most like a given song?) and **artist-level**
    (which artists write in the most similar style to each other?).
 
    All similarity scores are **cosine similarity** computed in the 384-dimensional
    embedding space. A score of 1.0 means the lyrics encode to nearly identical
    vectors; a score near 0 means the language and themes are unrelated.
    Scores are never negative because embeddings are L2-normalised.
    """
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    [
        "🎵 Song Similarity",
        "🎤 Artist Similarity",
        "🌟 Distinctiveness",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SONG SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # ── Part A: Nearest-neighbour search ──────────────────────────────────
    st.subheader("Which songs are most lyrically similar to a chosen song?")
    st.markdown(
        """
        Select a song and the app finds its nearest neighbours in embedding
        space (the songs whose lyrics encode to the most similar vector).
        This is the same mechanism that powers the recommender page, but
        anchored to an existing song rather than a free-text query.
 
        A high similarity score (> 0.85) means the songs use strikingly
        similar language, themes, and emotional register. Scores in the
        0.6–0.8 range indicate thematic overlap but distinct styles.
        """
    )

    col_artist, col_song = st.columns([1, 2])
    with col_artist:
        query_artist = st.selectbox("Artist", options=all_artists, key="song_artist")
    with col_song:
        artist_songs = sorted(df[df["artist"] == query_artist]["title"].unique())
        query_song = st.selectbox("Song", options=artist_songs, key="song_title")

    search_btn = st.button(
        "🔍 Find Similar Songs", type="primary", use_container_width=True
    )

    if search_btn:
        with st.spinner("Computing similarities..."):
            try:
                results = find_similar_to_song(
                    song_title=query_song,
                    df=df,
                    embeddings=embeddings,
                    top_n=top_n,
                    artist=query_artist,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.markdown("---")
        st.markdown(
            f"**Top {top_n} songs most similar to "
            f"*{results.attrs.get('query_title', query_song)}* "
            f"by {results.attrs.get('query_artist', query_artist)}:**"
        )

        # Bar chart coloured by artist — colour adds information here
        # because it shows whether similar songs cluster within or across artists
        fig_bar = px.bar(
            results,
            x="similarity",
            y="title",
            color="artist",
            orientation="h",
            hover_data=["album", "similarity"],
            labels={
                "similarity": "Cosine Similarity",
                "title": "Song",
                "artist": "Artist",
            },
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bar.update_layout(
            height=max(300, top_n * 34),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(range=[0, 1.05]),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(size=10)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Colour encodes which artist each result belongs to. "
            "Results from the same artist as the query confirm the model is "
            "capturing intra-artist style. Results from other artists show "
            "cross-artist lyrical overlap."
        )

        st.download_button(
            "⬇️ Download results as CSV",
            data=results.to_csv(index=False),
            file_name=f"similar_to_{query_song[:30].replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # ── UMAP highlight ─────────────────────────────────────────────────
        if umap_2d is not None:
            st.markdown("---")
            st.subheader("Where do the most similar songs sit on the embedding map?")
            st.caption(
                "The ⭐ marks the query song. Gold points are its nearest "
                "neighbours. If they cluster tightly around the query on the "
                "UMAP, the similarity scores reflect genuine spatial proximity "
                "in embedding space — not just a numerical coincidence."
            )

            result_mask = df["title"].isin(results["title"]) & df["artist"].isin(
                results["artist"]
            )
            result_idx = df[result_mask].index.tolist()[:top_n]
            query_mask = (df["title"] == query_song) & (df["artist"] == query_artist)
            query_idx = df[query_mask].index.tolist()

            plot_df = pd.DataFrame(
                {
                    "x": umap_2d[:, 0],
                    "y": umap_2d[:, 1],
                    "artist": df["artist"],
                    "title": df["title"],
                    "is_result": [i in result_idx for i in range(len(df))],
                    "is_query": [i in query_idx for i in range(len(df))],
                }
            )

            fig_umap = px.scatter(
                plot_df[~plot_df["is_result"] & ~plot_df["is_query"]],
                x="x",
                y="y",
                color="artist",
                hover_name="title",
                template="plotly_dark",
                opacity=0.18,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_umap.update_traces(marker=dict(size=4))

            res_pts = plot_df[plot_df["is_result"]]
            fig_umap.add_trace(
                go.Scatter(
                    x=res_pts["x"],
                    y=res_pts["y"],
                    mode="markers",
                    marker=dict(
                        size=11, color="gold", line=dict(width=1.5, color="white")
                    ),
                    text=res_pts["title"],
                    hovertemplate="<b>%{text}</b><extra>Similar song</extra>",
                    name="Similar songs",
                )
            )

            q_pts = plot_df[plot_df["is_query"]]
            if not q_pts.empty:
                fig_umap.add_trace(
                    go.Scatter(
                        x=q_pts["x"],
                        y=q_pts["y"],
                        mode="markers+text",
                        text=[f"⭐ {query_song}"],
                        textposition="top center",
                        textfont=dict(color="white", size=11),
                        marker=dict(
                            size=16,
                            color="red",
                            symbol="star",
                            line=dict(width=1.5, color="white"),
                        ),
                        name="Query song",
                        hovertemplate=f"<b>{query_song}</b><extra>Query</extra>",
                    )
                )

            fig_umap.update_layout(
                height=500,
                showlegend=False,
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, title=""
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, title=""
                ),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_umap, use_container_width=True)

    # ── Part B: Full artist discography heatmap ────────────────────────────
    st.markdown("---")
    st.subheader(
        "What does the full similarity structure of an artist's discography look like?"
    )
    st.markdown(
        """
        The nearest-neighbour search above finds the single most similar song
        to a query. But a richer question is: **across an entire discography,
        which songs form coherent groups, and which are outliers?**
 
        The heatmap below shows the pairwise cosine similarity between every
        pair of songs in an artist's catalogue. Dark red blocks along the diagonal
        indicate groups of songs that are all mutually similar — these are the
        artist's thematic clusters. Pale rows or columns are outlier songs that
        don't sound like the rest of the discography.
        """
    )

    heatmap_artist = st.selectbox(
        "Select artist to explore",
        options=all_artists,
        key="heatmap_artist",
    )

    if st.button("Generate Discography Heatmap", key="gen_heatmap"):
        with st.spinner("Computing pairwise similarities..."):
            try:
                sim_matrix_df, sub_df = get_song_similarity_matrix(
                    df=df,
                    embeddings=embeddings,
                    artist=heatmap_artist,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        n_songs = len(sim_matrix_df)
        st.caption(
            f"Showing {n_songs} songs by {heatmap_artist}. "
            "Hover any cell to see the exact similarity between two songs."
        )

        fig_heat = px.imshow(
            sim_matrix_df,
            color_continuous_scale="RdBu",
            zmin=0,
            zmax=1,
            aspect="auto",
            template="plotly_dark",
            labels=dict(color="Cosine Similarity"),
        )
        fig_heat.update_layout(
            height=max(420, n_songs * 20),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(tickfont=dict(size=8), tickangle=-60),
            yaxis=dict(tickfont=dict(size=8)),
        )
        fig_heat.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br><b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>"
            )
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Red blocks = groups of mutually similar songs (thematic clusters). "
            "Blue cells = songs with little in common. "
            "Pale rows/columns = outlier songs that don't fit any cluster."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARTIST SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Which artists write most like each other?")
    st.markdown(
        """
        Each cell shows the cosine similarity between the **centroid embeddings**
        of two artists — the mean of all their songs' embedding vectors.
        A high score means the two artists' catalogues occupy overlapping
        regions of the embedding space, implying shared lyrical themes,
        vocabulary, and emotional register.
 
        This answers a question the UMAP can only suggest visually:
        *do two artists that look close on the map actually have similar
        cosine similarity scores, or is that an artefact of the 2D projection?*
        """
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
    fig_heat.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig_heat.update_yaxes(tickfont=dict(size=10))
    fig_heat.update_layout(
        height=max(500, len(all_artists) * 26),
        margin=dict(t=10, b=120, l=20, r=20),
    )
    fig_heat.update_traces(
        hovertemplate=(
            "<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>"
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Drill-down: annotate the heatmap row for one artist ───────────────
    st.markdown("---")
    st.subheader("Which artists are most similar to a chosen artist?")
    st.caption(
        "Select an artist to see their similarity scores against every other "
        "artist, ranked. This is a slice through the heatmap above — "
        "showing one row in ranked order rather than the full matrix."
    )

    drill_artist = st.selectbox(
        "Select artist",
        options=all_artists,
        key="artist_drill",
    )

    try:
        top_similar = get_top_similar_artists(
            artist_name=drill_artist,
            similarity_matrix=artist_sim_matrix,
            top_n=len(all_artists) - 1,  # show all, not just top 5
        )

        fig_drill = px.bar(
            top_similar,
            x="similarity",
            y="artist",
            orientation="h",
            text=top_similar["similarity"].map("{:.3f}".format),
            labels={
                "similarity": "Cosine Similarity to " + drill_artist,
                "artist": "Artist",
            },
            template="plotly_dark",
        )
        fig_drill.update_traces(
            marker_color="#a78bfa",
            textposition="outside",
        )
        fig_drill.update_layout(
            height=max(320, len(top_similar) * 34),
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10, l=10, r=80),
        )
        st.plotly_chart(fig_drill, use_container_width=True)
        st.caption(
            "Artists near the top are stylistically closest to the selected artist. "
            "This ranking is independent of genre labels — it reflects only the "
            "language and themes embedded in the lyrics."
        )

    except ValueError as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTINCTIVENESS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Which artists have the most unique lyrical voice?")
    st.markdown(
        """
        **Distinctiveness** combines two measurements:
 
        - **Intra-artist similarity** — how similar are an artist's songs
          *to each other*? High intra = consistent style across the discography.
        - **Inter-artist similarity** — how similar are an artist's songs
          *to everyone else's songs*? Low inter = the artist occupies a unique
          region of the embedding space.
 
        An artist is **distinctive** when their intra-similarity is high
        *and* their inter-similarity is low — they write consistently in a
        style that nobody else shares.
 
        The scatter plot below shows both dimensions simultaneously.
        Each point is an artist. **Points above the diagonal** have higher
        intra than inter similarity — their songs sound more like each other
        than like the rest of the corpus. This is the definition of a
        distinctive voice. **Points on or below the diagonal** are artists
        whose songs blend into the general corpus style.
        """
    )

    st.markdown("---")

    # ── Intra vs inter scatter — single chart, replaces table + two bar charts ──
    fig_scatter = go.Figure()

    # Reference diagonal: intra = inter (no distinctiveness)
    axis_max = (
        max(
            similarity_stats["intra_similarity"].max(),
            similarity_stats["inter_similarity"].max(),
        )
        * 1.05
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, axis_max],
            y=[0, axis_max],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1.5),
            hoverinfo="skip",
            showlegend=True,
            name="intra = inter (no distinctiveness)",
        )
    )

    # Artist points coloured by distinctiveness score
    fig_scatter.add_trace(
        go.Scatter(
            x=similarity_stats["inter_similarity"],
            y=similarity_stats["intra_similarity"],
            mode="markers+text",
            text=similarity_stats["artist"],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=14,
                color=similarity_stats["distinctiveness"],
                colorscale="RdYlGn",
                cmin=similarity_stats["distinctiveness"].min(),
                cmax=similarity_stats["distinctiveness"].max(),
                colorbar=dict(
                    title="Distinctiveness",
                    thickness=14,
                    len=0.7,
                ),
                line=dict(width=1, color="rgba(255,255,255,0.4)"),
            ),
            customdata=np.stack(
                [
                    similarity_stats["distinctiveness"],
                    similarity_stats["intra_similarity"],
                    similarity_stats["inter_similarity"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Intra-similarity: %{customdata[1]:.4f}<br>"
                "Inter-similarity: %{customdata[2]:.4f}<br>"
                "Distinctiveness: %{customdata[0]:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig_scatter.update_layout(
        template="plotly_dark",
        height=580,
        xaxis=dict(
            title="Inter-artist similarity (lower = more unique vs corpus)",
            range=[0, axis_max],
            gridcolor="rgba(255,255,255,0.07)",
        ),
        yaxis=dict(
            title="Intra-artist similarity (higher = more internally consistent)",
            range=[0, axis_max],
            gridcolor="rgba(255,255,255,0.07)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(t=20, b=60, l=60, r=20),
    )

    # Shaded region label: "More distinctive"
    fig_scatter.add_annotation(
        x=axis_max * 0.15,
        y=axis_max * 0.88,
        text="▲ More distinctive",
        showarrow=False,
        font=dict(size=11, color="rgba(100,220,100,0.7)"),
        bgcolor="rgba(0,0,0,0)",
    )
    fig_scatter.add_annotation(
        x=axis_max * 0.72,
        y=axis_max * 0.18,
        text="▼ Less distinctive",
        showarrow=False,
        font=dict(size=11, color="rgba(220,100,100,0.7)"),
        bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.caption(
        "Points above the dashed diagonal have higher intra than inter similarity "
        "— they write consistently in a style that is uniquely their own. "
        "Colour encodes the distinctiveness score (green = more distinctive). "
        "Hover any point for exact values."
    )

    # ── Ranked distinctiveness for quick reference ─────────────────────────
    st.markdown("---")
    st.subheader("Distinctiveness Ranking")
    st.caption(
        "Distinctiveness = intra-similarity − inter-similarity. "
        "A positive score means an artist's songs sound more like each other "
        "than like the rest of the corpus."
    )

    ranked = (
        similarity_stats[
            ["artist", "intra_similarity", "inter_similarity", "distinctiveness"]
        ]
        .sort_values("distinctiveness", ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))

    fig_ranked = px.bar(
        ranked,
        x="distinctiveness",
        y="artist",
        orientation="h",
        color="distinctiveness",
        color_continuous_scale="RdYlGn",
        text=ranked["distinctiveness"].map(
            lambda v: f"+{v:.3f}" if v >= 0 else f"{v:.3f}"
        ),
        labels={"distinctiveness": "Distinctiveness score", "artist": "Artist"},
        template="plotly_dark",
    )
    fig_ranked.update_traces(textposition="outside")
    fig_ranked.update_layout(
        height=max(320, len(ranked) * 34),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            zerolinewidth=2,
        ),
        margin=dict(t=10, b=10, l=10, r=80),
    )
    st.plotly_chart(fig_ranked, use_container_width=True)
    st.caption(
        "Green bars = artists with a more unique style than average. "
        "Red bars = artists whose style overlaps heavily with the corpus."
    )
