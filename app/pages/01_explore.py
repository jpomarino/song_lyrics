import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans

from analysis.themes_llm import load_theme_results, THEMES

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Explore — Lyrics Analysis",
    page_icon="🗺️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────

if "bootstrapped" not in st.session_state:
    st.warning("Please start the app from `app/main.py`.")
    st.stop()

df = st.session_state["df"]
umap_2d = st.session_state["umap_2d"]

if umap_2d is None:
    st.error(
        "UMAP projection not found. "
        "Run `python pipeline/build.py` to generate `data/cache/umap_embedding.npy`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD THEMES
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def load_themes_cached(_df):
    try:
        return load_theme_results(_df)
    except FileNotFoundError:
        return None


df_themes = load_themes_cached(df)
themes_available = df_themes is not None and "themes" in df_themes.columns

# ─────────────────────────────────────────────────────────────────────────────
# BUILD BASE PLOT DATAFRAME
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

if themes_available:
    plot_df["themes"] = df_themes["themes"].reset_index(drop=True)
else:
    plot_df["themes"] = [[] for _ in range(len(plot_df))]

all_artists = sorted(df["artist"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# FORMAL 2-CLUSTER KMEANS
# Clustered in full 384-d embedding space, not 2D UMAP.
# UMAP distorts distances — it is only used for visualisation.
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def compute_clusters(_embeddings: np.ndarray) -> np.ndarray:
    km = KMeans(n_clusters=2, random_state=42, n_init="auto")
    return km.fit_predict(_embeddings)


embeddings = st.session_state["embeddings"]
cluster_ids = compute_clusters(embeddings)
plot_df["cluster"] = [f"Cluster {c}" for c in cluster_ids]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🗺️ Explore Controls")
    st.markdown("---")

    selected_artists = st.multiselect(
        "Filter by artist",
        options=all_artists,
        default=all_artists,
    )

    st.markdown("---")
    point_size = st.slider("Point size", 2, 12, 5)
    point_opacity = st.slider("Opacity", 0.1, 1.0, 0.8, step=0.05)
    show_centroids = st.checkbox("Overlay artist centroids", value=False)

    st.markdown("---")
    st.caption(f"{len(plot_df):,} songs · {len(all_artists)} artists")

filtered_df = plot_df[plot_df["artist"].isin(selected_artists)].copy()

if filtered_df.empty:
    st.warning("No songs match the current filter.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🗺️ Lyrics Embedding Explorer")
st.markdown(
    """
    This page answers a core question: **do artists I love actually write
    about similar things, or do I just like how they sound?**
 
    Each point is a song. Its position comes from:
 
    1. Converting lyrics into a **384-dimensional vector** using a sentence
       embedding model — a mathematical representation of meaning.
    2. Projecting those 384 dimensions to **2D using UMAP** for visualisation.
 
    **Proximity = lyrical similarity.** Songs close together use similar language,
    themes, and emotional register. Use the tabs below to explore the map coloured
    by artist, by individual theme, or by discovered cluster.
    """
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    [
        "🎤 Artist Map",
        "🏷️ Theme Map",
        "🔬 Cluster Analysis",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARTIST MAP
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Where Do Each Artist's Songs Live?")
    st.caption(
        "Points are coloured by artist. Tight groups indicate a consistent "
        "lyrical style; spread-out artists range across many themes. "
        "Artists that overlap in the map write about similar things."
    )

    fig1 = px.scatter(
        filtered_df,
        x="x",
        y="y",
        color="artist",
        hover_name="title",
        hover_data={"artist": True, "album": True, "x": False, "y": False},
        template="plotly_dark",
        opacity=point_opacity,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig1.update_traces(
        marker=dict(
            size=point_size, line=dict(width=0.3, color="rgba(255,255,255,0.2)")
        )
    )

    n_visible = filtered_df["artist"].nunique()
    fig1.update_layout(
        height=640,
        showlegend=(n_visible <= 10),
        legend=dict(
            title="Artist",
            itemsizing="constant",
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        margin=dict(t=20, b=20, l=20, r=20),
    )

    # ── Centroid overlay — annotation boxes are theme-safe ─────────────────
    # Plotly annotations with bgcolor render an opaque box behind each label.
    # This works on both light and dark Streamlit themes without any shadows
    # or offset-text hacks that bleed through in light mode.
    if show_centroids:
        centroids = filtered_df.groupby("artist")[["x", "y"]].mean().reset_index()
        fig1.add_trace(
            go.Scatter(
                x=centroids["x"],
                y=centroids["y"],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=13,
                    color="#ffffff",
                    line=dict(width=1.5, color="#222222"),
                ),
                text=centroids["artist"],
                name="Centroid",
                hovertemplate="<b>%{text}</b><br>(%{x:.2f}, %{y:.2f})<extra></extra>",
                showlegend=True,
            )
        )
        for _, row in centroids.iterrows():
            fig1.add_annotation(
                x=row["x"],
                y=row["y"],
                text=row["artist"],
                showarrow=False,
                yshift=18,
                font=dict(size=11, color="#ffffff"),
                bgcolor="rgba(30,30,46,0.85)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1,
                borderpad=3,
            )

    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Tip: double-click an artist in the legend to isolate them. "
        "Single-click to toggle visibility."
    )

    # ── Cluster spread bar chart ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("How Tightly Do Each Artist's Songs Cluster?")
    st.caption(
        "**Cluster spread** is the standard deviation of each artist's songs "
        "in UMAP space. A low value means consistent lyrical style. "
        "A high value means the artist ranges across many themes. "
        "This is not a quality judgement — it measures stylistic range."
    )

    spread_df = (
        filtered_df.groupby("artist")[["x", "y"]]
        .agg(x_std=("x", "std"), y_std=("y", "std"))
        .reset_index()
    )
    spread_df["Cluster Spread"] = (
        (spread_df["x_std"] ** 2 + spread_df["y_std"] ** 2) ** 0.5
    ).round(4)
    spread_df = (
        spread_df[["artist", "Cluster Spread"]]
        .rename(columns={"artist": "Artist"})
        .sort_values("Cluster Spread", ascending=True)
        .reset_index(drop=True)
    )

    fig_spread = px.bar(
        spread_df,
        x="Cluster Spread",
        y="Artist",
        orientation="h",
        text=spread_df["Cluster Spread"].map("{:.4f}".format),
        labels={"Cluster Spread": "Spread (lower = more cohesive)"},
        template="plotly_dark",
    )
    fig_spread.update_traces(marker_color="#a78bfa", textposition="outside")
    fig_spread.update_layout(
        height=max(300, len(spread_df) * 32),
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(range=[0, spread_df["Cluster Spread"].max() * 1.2]),
    )
    st.plotly_chart(fig_spread, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — THEME MAP
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Where Do Specific Themes Live in Embedding Space?")
    st.markdown(
        """
        Inspired by gene expression UMAPs — where each gene is coloured by
        its expression level across cells — this view lets you select a lyrical
        theme and see which songs carry it, and *where* those songs sit on the map.
 
        **How to read this:** highlighted points are songs tagged with the
        selected theme. If a theme forms a coherent spatial region, it means
        lyrically similar songs share that theme — the embedding model has
        implicitly learned to group by theme without being trained to do so.
        If a theme is scattered randomly, it cuts across many lyrical styles.
        """
    )

    if not themes_available:
        st.info(
            "Theme data not found. "
            "Run `python analysis/themes_llm.py` to classify themes first."
        )
    else:
        theme_colours = dict(
            zip(
                THEMES,
                px.colors.qualitative.Pastel
                + px.colors.qualitative.Set2
                + px.colors.qualitative.Dark2,
            )
        )

        selected_theme = st.selectbox(
            "Select a theme to highlight",
            options=THEMES,
            key="theme_map_select",
        )
        theme_colour = theme_colours.get(selected_theme, "#a78bfa")

        filtered_df["has_theme"] = filtered_df["themes"].apply(
            lambda t: selected_theme in t if isinstance(t, list) else False
        )

        n_with = int(filtered_df["has_theme"].sum())
        n_total = len(filtered_df)
        pct = round(n_with / n_total * 100, 1) if n_total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Songs with theme", f"{n_with:,}  ({pct}%)")
        m2.metric("Songs without theme", f"{n_total - n_with:,}")
        m3.metric("Theme selected", selected_theme)

        bg_df = filtered_df[~filtered_df["has_theme"]]
        hi_df = filtered_df[filtered_df["has_theme"]]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=bg_df["x"],
                y=bg_df["y"],
                mode="markers",
                marker=dict(
                    size=point_size, color="rgba(120,120,140,0.25)", line=dict(width=0)
                ),
                text=bg_df["title"] + " — " + bg_df["artist"],
                hovertemplate="%{text}<extra>No theme</extra>",
                name="Other songs",
                showlegend=True,
            )
        )
        if not hi_df.empty:
            fig2.add_trace(
                go.Scatter(
                    x=hi_df["x"],
                    y=hi_df["y"],
                    mode="markers",
                    marker=dict(
                        size=point_size + 2,
                        color=theme_colour,
                        opacity=min(point_opacity + 0.1, 1.0),
                        line=dict(width=0.5, color="rgba(255,255,255,0.4)"),
                    ),
                    text=hi_df["title"] + " — " + hi_df["artist"],
                    hovertemplate="%{text}<extra>" + selected_theme + "</extra>",
                    name=selected_theme,
                    showlegend=True,
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=620,
            legend=dict(
                itemsizing="constant", bgcolor="rgba(0,0,0,0.4)", font=dict(size=11)
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Per-artist theme proportion ────────────────────────────────────
        st.markdown("---")
        st.subheader(f"Which Artists Sing Most About *{selected_theme}*?")
        st.caption(
            "Proportion of each artist's songs tagged with this theme. "
            "High proportions mean the theme is central to that artist's output; "
            "low proportions mean it appears occasionally or not at all."
        )

        artist_theme_counts = (
            filtered_df.groupby("artist")["has_theme"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "with_theme", "count": "total"})
        )
        artist_theme_counts["proportion"] = (
            artist_theme_counts["with_theme"] / artist_theme_counts["total"]
        ).round(4)
        artist_theme_counts = artist_theme_counts.sort_values(
            "proportion", ascending=True
        )

        fig_at = px.bar(
            artist_theme_counts,
            x="proportion",
            y="artist",
            orientation="h",
            text=artist_theme_counts["proportion"].map("{:.1%}".format),
            labels={
                "proportion": f"% songs tagged '{selected_theme}'",
                "artist": "Artist",
            },
            template="plotly_dark",
        )
        fig_at.update_traces(marker_color=theme_colour, textposition="outside")
        fig_at.update_layout(
            height=max(320, len(artist_theme_counts) * 32),
            xaxis=dict(
                tickformat=".0%",
                range=[0, artist_theme_counts["proportion"].max() * 1.2],
            ),
            margin=dict(t=10, b=10, l=10, r=80),
        )
        st.plotly_chart(fig_at, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Formal Cluster Analysis")
    st.markdown(
        """
        Visual inspection of the UMAP suggests the corpus splits into two
        broad regions. To test this formally, we applied **KMeans (k=2)** in
        the full 384-dimensional embedding space — *not* on the 2D projection,
        because UMAP distorts distances and should only be used for visualisation.
 
        Two questions:
 
        1. **Do the clusters look spatially coherent on the UMAP?**
           If the partition is meaningful, the two groups should occupy
           different regions of the map — not be randomly interspersed.
        2. **Do the clusters have statistically different theme profiles?**
           A **chi-square test of independence** tests whether theme
           distributions differ significantly between clusters, or whether
           any difference is within the noise expected by chance.
        """
    )

    # ── UMAP coloured by cluster ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("UMAP Coloured by Cluster")
    st.caption(
        "If the two clusters occupy distinct spatial regions, KMeans has found "
        "a real structural boundary in lyrical space. Heavy overlap would "
        "suggest the two-cluster assumption is too coarse for this corpus."
    )

    fig3 = px.scatter(
        filtered_df,
        x="x",
        y="y",
        color="cluster",
        hover_name="title",
        hover_data={"artist": True, "cluster": True, "x": False, "y": False},
        template="plotly_dark",
        opacity=point_opacity,
        color_discrete_map={
            "Cluster 0": "#a78bfa",
            "Cluster 1": "#34d399",
        },
    )
    fig3.update_traces(
        marker=dict(
            size=point_size, line=dict(width=0.3, color="rgba(255,255,255,0.2)")
        )
    )
    fig3.update_layout(
        height=600,
        legend=dict(title="Cluster", itemsizing="constant", bgcolor="rgba(0,0,0,0.4)"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Cluster size metrics ───────────────────────────────────────────────
    c0_n = (filtered_df["cluster"] == "Cluster 0").sum()
    c1_n = (filtered_df["cluster"] == "Cluster 1").sum()
    m1, m2 = st.columns(2)
    m1.metric(
        "Cluster 0 — songs", f"{c0_n:,}  ({round(c0_n / len(filtered_df) * 100, 1)}%)"
    )
    m2.metric(
        "Cluster 1 — songs", f"{c1_n:,}  ({round(c1_n / len(filtered_df) * 100, 1)}%)"
    )

    # ── Artist composition by cluster ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Artist Composition by Cluster")
    st.caption(
        "What fraction of each artist's songs fall into Cluster 0 vs Cluster 1? "
        "An artist concentrated in one cluster writes consistently in one lyrical "
        "register. An artist split roughly 50/50 across both clusters has a more "
        "varied output — some songs lyrical territory, some in another."
    )

    ac = filtered_df.groupby(["artist", "cluster"]).size().reset_index(name="songs")
    at = filtered_df.groupby("artist").size().reset_index(name="total")
    ac = ac.merge(at, on="artist")
    ac["proportion"] = (ac["songs"] / ac["total"]).round(4)

    fig_comp = px.bar(
        ac,
        x="proportion",
        y="artist",
        color="cluster",
        orientation="h",
        barmode="stack",
        labels={
            "proportion": "Proportion of Artist's Songs",
            "artist": "Artist",
            "cluster": "Cluster",
        },
        template="plotly_dark",
        color_discrete_map={"Cluster 0": "#a78bfa", "Cluster 1": "#34d399"},
    )
    fig_comp.update_layout(
        height=max(340, ac["artist"].nunique() * 36),
        xaxis=dict(tickformat=".0%", range=[0, 1.05]),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Theme proportions per cluster ─────────────────────────────────────
    if not themes_available:
        st.info(
            "Theme data not available — "
            "run `python analysis/themes_llm.py` to enable this section."
        )
    else:
        st.markdown("---")
        st.subheader("Theme Proportions Across Clusters")
        st.caption(
            "What fraction of each cluster's songs are tagged with each theme? "
            "Themes with large differences between the two bars are the ones "
            "that most strongly define what separates the two clusters. "
            "Themes with similar bars cut across both clusters equally."
        )

        theme_rows = []
        for clabel in ["Cluster 0", "Cluster 1"]:
            c_songs = filtered_df[filtered_df["cluster"] == clabel]
            n_c = len(c_songs)
            for theme in THEMES:
                n_t = (
                    c_songs["themes"]
                    .apply(lambda t: theme in t if isinstance(t, list) else False)
                    .sum()
                )
                theme_rows.append(
                    {
                        "cluster": clabel,
                        "theme": theme,
                        "count": int(n_t),
                        "proportion": round(n_t / n_c, 4) if n_c else 0,
                    }
                )

        tcd = pd.DataFrame(theme_rows)

        fig_tc = px.bar(
            tcd,
            x="proportion",
            y="theme",
            color="cluster",
            orientation="h",
            barmode="group",
            labels={
                "proportion": "Proportion of Cluster's Songs",
                "theme": "Theme",
                "cluster": "Cluster",
            },
            template="plotly_dark",
            color_discrete_map={"Cluster 0": "#a78bfa", "Cluster 1": "#34d399"},
        )
        fig_tc.update_layout(
            height=max(500, len(THEMES) * 36),
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
            legend=dict(bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_tc, use_container_width=True)

        # ── Chi-square test ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Are the Theme Differences Statistically Significant?")
        st.markdown(
            """
            A **chi-square test of independence** tests whether the theme
            differences between clusters could have arisen by chance.
 
            **Null hypothesis H₀:** theme assignment is independent of cluster —
            knowing which cluster a song is in tells you nothing about its themes.
 
            **How it works:** the test builds a contingency table of
            (theme × cluster) counts and measures how far the observed counts
            deviate from what we'd expect if themes were distributed identically
            across both clusters.
 
            **How to read the p-value:** below 0.05 means we reject H₀ —
            the clusters have statistically different theme profiles, providing
            formal evidence that KMeans found a meaningful thematic partition.
            """
        )

        theme_pivot = tcd.pivot(
            index="theme", columns="cluster", values="count"
        ).fillna(0)
        theme_pivot = theme_pivot[
            (theme_pivot["Cluster 0"] > 0) | (theme_pivot["Cluster 1"] > 0)
        ]

        if theme_pivot.shape[0] < 2:
            st.warning("Not enough theme data to run the chi-square test.")
        else:
            chi2, p_value, dof, expected = chi2_contingency(theme_pivot.values)

            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("χ² statistic", f"{chi2:.2f}")
            cs2.metric("Degrees of freedom", dof)
            cs3.metric("p-value", f"{p_value:.2e}")

            if p_value < 0.001:
                st.success(
                    f"**p = {p_value:.2e} < 0.001 — highly significant.** "
                    "We reject H₀: the two clusters have meaningfully different "
                    "lyrical themes. The visual separation on the UMAP reflects "
                    "a real structural difference in how these songs are written."
                )
            elif p_value < 0.05:
                st.success(
                    f"**p = {p_value:.4f} < 0.05 — statistically significant.** "
                    "The clusters show different theme profiles beyond what chance "
                    "would produce."
                )
            else:
                st.warning(
                    f"**p = {p_value:.4f} ≥ 0.05 — not statistically significant.** "
                    "We cannot rule out that the observed theme differences arose "
                    "by chance. Consider whether k=2 is the right cluster count, "
                    "or whether the theme taxonomy needs refinement."
                )

            # ── Per-theme chi-square contributions ────────────────────────
            st.markdown("---")
            st.subheader("Which Themes Drive the Cluster Separation?")
            st.caption(
                "The chi-square statistic is the sum of per-theme contributions. "
                "Themes with the largest contributions are distributed most "
                "differently between the two clusters — they are the primary "
                "drivers of the structural separation KMeans discovered."
            )

            observed = theme_pivot.values.astype(float)
            contrib = (observed - expected) ** 2 / np.where(expected > 0, expected, 1)
            per_theme = contrib.sum(axis=1)

            contrib_df = pd.DataFrame(
                {
                    "theme": theme_pivot.index,
                    "contribution": per_theme.round(4),
                }
            ).sort_values("contribution", ascending=True)

            fig_contrib = px.bar(
                contrib_df,
                x="contribution",
                y="theme",
                orientation="h",
                text=contrib_df["contribution"].map("{:.2f}".format),
                labels={"contribution": "Contribution to χ²", "theme": "Theme"},
                template="plotly_dark",
            )
            fig_contrib.update_traces(marker_color="#a78bfa", textposition="outside")
            fig_contrib.update_layout(
                height=max(400, len(contrib_df) * 32),
                margin=dict(t=10, b=10, l=10, r=80),
                xaxis=dict(range=[0, contrib_df["contribution"].max() * 1.2]),
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.caption(
                "Themes at the top of this chart are what make one cluster "
                "structurally different from the other."
            )
