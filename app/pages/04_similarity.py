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

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    [
        "🎵 Song-to-Song",
        "🎤 Artist Matrix",
        "📊 Distinctiveness",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SONG-TO-SONG
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Find Songs Similar to a Selected Song")

    col_artist, col_song = st.columns([1, 2])

    with col_artist:
        all_artists = sorted(df["artist"].unique())
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
            f"Songs most similar to **{results.attrs.get('query_title', query_song)}** "
            f"by *{results.attrs.get('query_artist', query_artist)}*:"
        )

        # ── Results table ─────────────────────────────────────────────────
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
        styled_results = results.style.background_gradient(
            subset=["similarity"], cmap="YlGn", vmin=0.0, vmax=1.0
        ).format({"similarity": "{:.4f}"})

        st.dataframe(
            styled_results,
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

        # col_dl, _ = st.columns([1, 3])
        # with col_dl:
        #     st.download_button(
        #         "⬇️ Download CSV",
        #         data=results.to_csv(index=False),
        #         file_name=f"similar_to_{query_song[:30].replace(' ', '_')}.csv",
        #         mime="text/csv",
        #     )

        st.markdown("---")

        # ── Horizontal bar chart ──────────────────────────────────────────
        fig_bar = px.bar(
            results,
            x="similarity",
            y="title",
            color="artist",
            orientation="h",
            hover_data=["album"],
            labels={"similarity": "Cosine Similarity", "title": "Song"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bar.update_layout(
            height=max(300, top_n * 32),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── UMAP highlight ────────────────────────────────────────────────
        if umap_2d is not None:
            st.markdown("---")
            st.subheader("Embedding Space — Query & Results")

            # Indices of top results in the full df
            result_mask = df["title"].isin(results["title"]) & df["artist"].isin(
                results["artist"]
            )
            result_indices = df[result_mask].index.tolist()[:top_n]

            query_mask = (df["title"] == query_song) & (df["artist"] == query_artist)
            query_index = df[query_mask].index.tolist()

            plot_df = pd.DataFrame(
                {
                    "x": umap_2d[:, 0],
                    "y": umap_2d[:, 1],
                    "artist": df["artist"],
                    "title": df["title"],
                    "is_result": [i in result_indices for i in range(len(df))],
                    "is_query": [i in query_index for i in range(len(df))],
                }
            )

            fig_umap = px.scatter(
                plot_df[~plot_df["is_result"] & ~plot_df["is_query"]],
                x="x",
                y="y",
                color="artist",
                hover_name="title",
                template="plotly_dark",
                opacity=0.2,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_umap.update_traces(marker=dict(size=4))

            # Top results — gold
            res_points = plot_df[plot_df["is_result"]]
            fig_umap.add_trace(
                go.Scatter(
                    x=res_points["x"],
                    y=res_points["y"],
                    mode="markers",
                    marker=dict(
                        size=11, color="gold", line=dict(width=1.5, color="white")
                    ),
                    text=res_points["title"],
                    hovertemplate="<b>%{text}</b><extra>Similar Song</extra>",
                    name="Similar Songs",
                )
            )

            # Query song — red star
            q_points = plot_df[plot_df["is_query"]]
            if not q_points.empty:
                fig_umap.add_trace(
                    go.Scatter(
                        x=q_points["x"],
                        y=q_points["y"],
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
                        name="Query Song",
                        hovertemplate=f"<b>{query_song}</b><extra>Query</extra>",
                    )
                )

            fig_umap.update_layout(
                height=520,
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, title=""
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, title=""
                ),
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_umap, use_container_width=True)

    # ── Per-artist song heatmap ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Intra-Artist Song Similarity Heatmap")
    st.caption(
        "Pairwise cosine similarity between all songs by one artist. "
        "Dark diagonal = self-similarity (always 1.0). "
        "Off-diagonal clusters indicate thematically related groups of songs."
    )

    heatmap_artist = st.selectbox(
        "Artist for song heatmap",
        options=sorted(df["artist"].unique()),
        key="heatmap_artist",
    )

    if st.button("Generate Heatmap", key="gen_heatmap"):
        with st.spinner("Computing pairwise similarities..."):
            sim_matrix_df, sub_df = get_song_similarity_matrix(
                df=df,
                embeddings=embeddings,
                artist=heatmap_artist,
            )

        n_songs = len(sim_matrix_df)
        st.markdown(f"**{n_songs} songs** by {heatmap_artist}")

        fig_song_heat = px.imshow(
            sim_matrix_df,
            color_continuous_scale="RdBu",
            zmin=0,
            zmax=1,
            aspect="auto",
            template="plotly_dark",
        )
        fig_song_heat.update_layout(
            height=max(400, n_songs * 18),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(tickfont=dict(size=8), tickangle=-60),
            yaxis=dict(tickfont=dict(size=8)),
        )
        fig_song_heat.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br><b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>"
            )
        )
        st.plotly_chart(fig_song_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARTIST MATRIX
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Artist Similarity Matrix")
    st.caption(
        f"Method: **{sim_method}**. "
        "Values represent cosine similarity in embedding space — "
        "higher = more similar lyrical style."
    )

    # Use precomputed centroid matrix from session_state (fast),
    # or recompute if user selected Average
    if sim_method == "Centroid":
        display_matrix = artist_sim_matrix
    else:
        from analysis.similarity import get_artist_similarity_matrix

        with st.spinner(
            "Computing average pairwise artist similarities (this may take a moment)..."
        ):
            display_matrix = get_artist_similarity_matrix(
                df, embeddings, method="average"
            )

    fig_artist_heat = px.imshow(
        display_matrix,
        color_continuous_scale="RdBu",
        zmin=0,
        zmax=1,
        aspect="auto",
        template="plotly_dark",
        labels=dict(color="Similarity"),
    )
    fig_artist_heat.update_layout(
        height=550,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(tickangle=-40),
    )
    fig_artist_heat.update_traces(
        hovertemplate=(
            "<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>"
        )
    )
    st.plotly_chart(fig_artist_heat, use_container_width=True)

    st.markdown("---")

    # ── Top similar artists per artist ────────────────────────────────────
    st.subheader("Most Similar Artists")

    selected_for_drill = st.selectbox(
        "Find artists most similar to:",
        options=sorted(df["artist"].unique()),
        key="artist_drill",
    )

    try:
        top_similar = get_top_similar_artists(
            artist_name=selected_for_drill,
            similarity_matrix=display_matrix,
            top_n=min(5, len(df["artist"].unique()) - 1),
        )

        col_tbl, col_chart = st.columns([1, 1.5])

        with col_tbl:
            st.dataframe(
                top_similar.style.background_gradient(
                    subset=["similarity"], cmap="YlGn"
                ).format({"similarity": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_chart:
            fig_similar = px.bar(
                top_similar,
                x="similarity",
                y="artist",
                orientation="h",
                color="similarity",
                color_continuous_scale="YlGn",
                labels={"similarity": "Cosine Similarity", "artist": "Artist"},
                template="plotly_dark",
            )
            fig_similar.update_layout(
                height=280,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_similar, use_container_width=True)

    except ValueError as e:
        st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTINCTIVENESS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Artist Distinctiveness")
    st.markdown(
        "**Distinctiveness** = intra-artist similarity minus inter-artist similarity. "
        "A high score means the artist is both *internally consistent* "
        "(their songs sound like each other) and *externally different* "
        "(their songs don't sound like anyone else's). "
        "A low score means the artist's style blends into the rest of the corpus."
    )
    st.markdown("---")

    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        st.dataframe(
            similarity_stats.style.background_gradient(
                subset=["distinctiveness", "intra_similarity", "inter_similarity"],
                cmap="RdYlGn",
            ).format(
                {
                    "intra_similarity": "{:.4f}",
                    "inter_similarity": "{:.4f}",
                    "distinctiveness": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    with col_r:
        # Grouped bar chart: intra vs inter per artist
        stats_melted = similarity_stats.melt(
            id_vars="artist",
            value_vars=["intra_similarity", "inter_similarity"],
            var_name="metric",
            value_name="score",
        )
        stats_melted["metric"] = stats_melted["metric"].map(
            {
                "intra_similarity": "Intra-artist",
                "inter_similarity": "Inter-artist",
            }
        )

        fig_grouped = px.bar(
            stats_melted,
            x="artist",
            y="score",
            color="metric",
            barmode="group",
            labels={"score": "Avg Cosine Similarity", "artist": "Artist", "metric": ""},
            template="plotly_dark",
            color_discrete_map={
                "Intra-artist": "#63b3ed",
                "Inter-artist": "#fc8181",
            },
        )
        fig_grouped.update_layout(
            height=420,
            xaxis_tickangle=-30,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=60, l=10, r=10),
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    st.markdown("---")

    # Distinctiveness ranking chart
    st.subheader("Distinctiveness Ranking")
    fig_distinct = px.bar(
        similarity_stats.sort_values("distinctiveness", ascending=True),
        x="distinctiveness",
        y="artist",
        orientation="h",
        color="distinctiveness",
        color_continuous_scale="RdYlGn",
        labels={"distinctiveness": "Distinctiveness Score", "artist": "Artist"},
        template="plotly_dark",
    )
    fig_distinct.update_layout(
        height=max(300, len(similarity_stats) * 38),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig_distinct.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Baseline",
        annotation_position="top right",
    )
    st.plotly_chart(fig_distinct, use_container_width=True)

    st.caption(
        "Artists above the baseline have a more unique style than average. "
        "Artists near zero are stylistically close to the rest of the corpus."
    )
