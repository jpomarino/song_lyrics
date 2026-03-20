import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Explore — Lyrics Analysis",
    page_icon="🗺️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# GUARD — ensure bootstrap has run
# ─────────────────────────────────────────────────────────────────────────────

if "bootstrapped" not in st.session_state:
    st.warning("Please start the app from `app/main.py`.")
    st.stop()

df = st.session_state["df"]
umap_2d = st.session_state["umap_2d"]

if umap_2d is None:
    st.error(
        "UMAP projection not found. "
        "Run `python pipeline/build.py` to generate `data/cache/umap_2d.npy`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PLOT DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

plot_df = pd.DataFrame(
    {
        "x": umap_2d[:, 0],
        "y": umap_2d[:, 1],
        "artist": df["artist"],
        "title": df["title"],
        "album": df["album"].fillna("Unknown"),
    }
).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🗺️ Explore Controls")
    st.markdown("---")

    all_artists = sorted(df["artist"].unique())
    selected_artists = st.multiselect(
        "Filter by artist",
        options=all_artists,
        default=all_artists,
        help="Show only selected artists in the scatter plot.",
    )

    st.markdown("---")

    color_by = st.radio(
        "Color points by",
        options=["Artist", "Album"],
        index=0,
    )

    st.markdown("---")

    point_size = st.slider("Point size", min_value=2, max_value=12, value=5)
    point_opacity = st.slider(
        "Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.05
    )
    show_density = st.checkbox("Show density heatmap", value=False)

    st.markdown("---")
    st.caption(f"Showing {len(plot_df):,} songs")

# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────

filtered_df = plot_df[plot_df["artist"].isin(selected_artists)]

if filtered_df.empty:
    st.warning("No songs match the current filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🗺️ Lyrics Embedding Explorer")
st.markdown(
    "Each point represents a song, projected into 2D using UMAP. "
    "Songs that are close together have similar lyrical themes and style. "
    "Hover over any point to see its title, artist, and album."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────

color_col = "artist" if color_by == "Artist" else "album"

fig = px.scatter(
    filtered_df,
    x="x",
    y="y",
    color=color_col,
    hover_name="title",
    hover_data={
        "artist": True,
        "album": True,
        "x": ":.3f",
        "y": ":.3f",
    },
    template="plotly_dark",
    opacity=point_opacity,
    color_discrete_sequence=px.colors.qualitative.Pastel
    if color_by == "Artist"
    else px.colors.qualitative.Set3,
)

fig.update_traces(
    marker=dict(
        size=point_size,
        line=dict(width=0.4, color="rgba(255,255,255,0.3)"),
    )
)

fig.update_layout(
    height=620,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    legend=dict(
        title=color_by,
        itemsizing="constant",
        bgcolor="rgba(0,0,0,0.4)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
    ),
    margin=dict(t=20, b=20, l=20, r=20),
)

# Overlay density heatmap if requested
if show_density:
    density = go.Histogram2dContour(
        x=filtered_df["x"],
        y=filtered_df["y"],
        colorscale="Hot",
        reversescale=True,
        opacity=0.25,
        showscale=False,
        ncontours=20,
        hoverinfo="skip",
    )
    fig.add_trace(density)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARTIST CENTROID OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")

show_centroids = st.checkbox("Show artist centroids on map", value=True)

if show_centroids:
    centroids = (
        filtered_df.groupby("artist")[["x", "y"]]
        .mean()
        .reset_index()
        .rename(columns={"x": "cx", "y": "cy"})
    )

    fig_c = go.Figure(fig)  # clone the existing figure
    fig_c.add_trace(
        go.Scatter(
            x=centroids["cx"],
            y=centroids["cy"],
            mode="markers+text",
            text=centroids["artist"],
            textposition="top center",
            textfont=dict(size=11, color="white"),
            marker=dict(
                symbol="diamond",
                size=14,
                color="white",
                line=dict(width=1.5, color="black"),
            ),
            name="Artist Centroid",
            hovertemplate="<b>%{text}</b><br>Centroid (%{x:.2f}, %{y:.2f})<extra></extra>",
        )
    )
    fig_c.update_layout(height=620)
    st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PER-ARTIST CLUSTER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Artist Cluster Summary")
st.caption(
    "Average UMAP position per artist and intra-cluster spread (std deviation). "
    "A low spread means the artist's songs cluster tightly together in embedding space."
)

summary = (
    filtered_df.groupby("artist")[["x", "y"]]
    .agg(
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        x_std=("x", "std"),
        y_std=("y", "std"),
    )
    .reset_index()
)
summary["spread"] = ((summary["x_std"] ** 2 + summary["y_std"] ** 2) ** 0.5).round(4)
summary = summary[["artist", "x_mean", "y_mean", "spread"]].rename(
    columns={
        "artist": "Artist",
        "x_mean": "UMAP-1 (mean)",
        "y_mean": "UMAP-2 (mean)",
        "spread": "Cluster Spread ↓",
    }
)
summary = summary.sort_values("Cluster Spread ↓")

st.dataframe(
    summary.style.background_gradient(subset=["Cluster Spread ↓"], cmap="RdYlGn_r"),
    use_container_width=True,
    hide_index=True,
)

st.caption("Lower spread = more stylistically cohesive artist.")
