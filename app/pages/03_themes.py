import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from analysis.themes_llm import (
    load_theme_results,
    load_artist_clusters,
    get_artist_theme_distribution,
    get_all_artist_theme_distributions,
    get_corpus_theme_summary,
    get_songs_by_theme,
    get_theme_overlap_matrix,
    THEMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Themes — Lyrics Analysis", page_icon="🎨", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────

if "bootstrapped" not in st.session_state:
    st.warning("Please start the app from 'app/main.py'.")
    st.stop()

df = st.session_state["df"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD THEME DATA
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading artist clusters...")
def load_themes(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load pre-classified themes from song_themes.json and merge into df
    """
    try:
        return load_theme_results(_df)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading artist clusters...")
def load_clusters() -> dict:
    return load_artist_clusters()


df_themes = load_themes(df)
cluster_data = load_clusters()

if df_themes is None:
    st.error("Theme classification not found.Run theme classification first")
    st.stop()

# Coverage check — warn if a meaningful number of songs have no themes
n_unclassified = df_themes["themes"].apply(lambda x: len(x) == 0).sum()
if n_unclassified > 0:
    pct = round(n_unclassified / len(df_themes) * 100, 1)
    st.warning(
        f"{n_unclassified} songs ({pct}%) have no theme classifications. "
        "Run `python analysis/themes_llm.py` to classify them."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE SHARED DATA
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Computing theme distributions...")
def compute_shared_data(_df_themes: pd.DataFrame):
    corpus_summary = get_corpus_theme_summary(_df_themes)
    artist_theme_wide = get_all_artist_theme_distributions(_df_themes, normalize=True)
    overlap_matrix = get_theme_overlap_matrix(_df_themes)
    return corpus_summary, artist_theme_wide, overlap_matrix


corpus_summary, artist_theme_wide, overlap_matrix = compute_shared_data(df_themes)
all_artists = sorted(df_themes["artist"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🎨 Theme Controls")
    st.markdown("---")

    selected_artist = st.selectbox(
        "Artist (for breakdown tab)",
        options=all_artists,
    )

    st.markdown("---")

    top_n_themes = st.slider(
        "Top N themes to display",
        min_value=3,
        max_value=len(THEMES),
        value=min(10, len(THEMES)),
    )

    st.markdown("---")

    # Quick stats
    classified_songs = df_themes["themes"].apply(lambda x: len(x) > 0).sum()
    st.metric("Songs with themes", f"{classified_songs:,}")
    st.metric("Theme taxonomy size", len(THEMES))

    # Most common theme
    if not corpus_summary.empty:
        top_theme = corpus_summary.iloc[0]["theme"]
        st.metric("Most common theme", top_theme)

    st.markdown("---")
    st.caption(
        f"Themes classified by Llama 3.2 (Ollama) · {len(THEMES)} theme taxonomy"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎨 Theme Analysis")
st.markdown(
    f"Exploring **{len(THEMES)} lyrical themes** across **{len(df_themes):,} songs** "
    f"by **{df_themes['artist'].nunique()} artists**, "
    "classified using Llama 3.2 running locally via Ollama."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Corpus Overview",
        "🎤 Artist Breakdown",
        "🔥 Artist × Theme Heatmap",
        "🔎 Browse by Theme",
        "🔀 Theme Overlap",
        "🗂️ Artist Clusters",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORPUS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Theme Frequency Across Full Corpus")
    st.caption(
        "Each song can have multiple themes. "
        "Proportions are computed as (songs with theme) / (total songs with any theme)."
    )

    if corpus_summary.empty:
        st.info("No theme data available.")
    else:
        display_summary = corpus_summary.head(top_n_themes).copy()

        col_chart, col_table = st.columns([1.6, 1])

        with col_chart:
            fig_corpus = px.bar(
                display_summary,
                x="song_count",
                y="theme",
                orientation="h",
                color="song_count",
                color_continuous_scale="Teal",
                text="song_count",
                labels={"song_count": "Songs", "theme": "Theme"},
                template="plotly_dark",
            )
            fig_corpus.update_traces(
                textposition="outside",
                textfont=dict(size=11),
            )
            fig_corpus.update_layout(
                height=max(380, top_n_themes * 36),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(title="Number of Songs"),
            )
            st.plotly_chart(fig_corpus, use_container_width=True)

        with col_table:
            st.dataframe(
                display_summary.style.background_gradient(
                    subset=["song_count"], cmap="YlGn"
                ).format({"proportion": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
                height=max(380, top_n_themes * 36),
            )

        st.markdown("---")

    # ── Theme proportion pie ───────────────────────────────────────────
    st.subheader("Theme Share")
    fig_pie = px.pie(
        corpus_summary.head(top_n_themes),
        names="theme",
        values="song_count",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.35,
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} songs (%{percent})<extra></extra>",
    )
    fig_pie.update_layout(
        height=460,
        showlegend=True,
        legend=dict(orientation="v", font=dict(size=10)),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ── Songs per theme count ──────────────────────────────────────────
    st.subheader("Average Themes per Song")
    avg_themes = df_themes["themes"].apply(len).mean()
    max_themes = df_themes["themes"].apply(len).max()
    songs_multi = df_themes["themes"].apply(lambda x: len(x) > 1).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg themes / song", f"{avg_themes:.2f}")
    m2.metric("Max themes on one song", max_themes)
    m3.metric("Songs with 2+ themes", f"{songs_multi:,}")

    # Distribution of theme count per song
    theme_count_dist = (
        df_themes["themes"].apply(len).value_counts().sort_index().reset_index()
    )
    theme_count_dist.columns = ["n_themes", "song_count"]

    fig_dist = px.bar(
        theme_count_dist,
        x="n_themes",
        y="song_count",
        labels={"n_themes": "Number of themes assigned", "song_count": "Songs"},
        template="plotly_dark",
        color="song_count",
        color_continuous_scale="Blues",
        text="song_count",
    )
    fig_dist.update_traces(textposition="outside")
    fig_dist.update_layout(
        height=320,
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARTIST BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader(f"Theme Breakdown — {selected_artist}")

    try:
        artist_dist = get_artist_theme_distribution(
            df_themes, selected_artist, normalize=True
        )
        artist_dist_display = artist_dist.head(top_n_themes)

        if artist_dist.empty:
            st.warning(f"No theme data found for {selected_artist}.")
        else:
            # ── Top metrics ────────────────────────────────────────────────
            top_theme_artist = artist_dist.iloc[0]["theme"]
            n_themes_used = len(artist_dist)
            artist_song_count = len(df_themes[df_themes["artist"] == selected_artist])

            m1, m2, m3 = st.columns(3)
            m1.metric("Top theme", top_theme_artist)
            m2.metric("Distinct themes", n_themes_used)
            m3.metric("Songs in corpus", artist_song_count)

            st.markdown("---")

            col_l, col_r = st.columns([1, 1.5])

            with col_l:
                st.dataframe(
                    artist_dist_display[["theme", "count", "proportion"]]
                    .style.background_gradient(subset=["proportion"], cmap="YlGn")
                    .format({"proportion": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_r:
                fig_artist_bar = px.bar(
                    artist_dist_display,
                    x="proportion",
                    y="theme",
                    orientation="h",
                    color="proportion",
                    color_continuous_scale="Teal",
                    text=artist_dist_display["proportion"].map("{:.1%}".format),
                    labels={"proportion": "% of Songs", "theme": "Theme"},
                    template="plotly_dark",
                )
                fig_artist_bar.update_traces(textposition="outside")
                fig_artist_bar.update_layout(
                    height=max(360, top_n_themes * 36),
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=80),
                )
                st.plotly_chart(fig_artist_bar, use_container_width=True)

            # ── Pie chart ──────────────────────────────────────────────────
            st.markdown("---")
            st.subheader(f"Theme Mix — {selected_artist}")

            fig_artist_pie = px.pie(
                artist_dist_display,
                names="theme",
                values="count",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.35,
            )
            fig_artist_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>%{value} songs (%{percent})<extra></extra>",
            )
            fig_artist_pie.update_layout(
                height=420,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_artist_pie, use_container_width=True)

    except ValueError as e:
        st.error(str(e))

    # ── Multi-artist stacked comparison ───────────────────────────────────
    st.markdown("---")
    st.subheader("Compare Multiple Artists")
    st.caption(
        "Each bar shows an artist's theme mix as a proportion of their songs. "
        "Hover to see exact percentages."
    )

    compare_artists = st.multiselect(
        "Select artists to compare",
        options=all_artists,
        default=all_artists[:4],
        key="compare_artists_tab2",
    )

    if compare_artists:
        rows = []
        for artist in compare_artists:
            try:
                dist = get_artist_theme_distribution(df_themes, artist, normalize=True)
                dist["artist"] = artist
                rows.append(dist.head(top_n_themes))
            except ValueError:
                continue

        if rows:
            compare_df = pd.concat(rows, ignore_index=True)

            fig_compare = px.bar(
                compare_df,
                x="proportion",
                y="artist",
                color="theme",
                orientation="h",
                barmode="stack",
                labels={"proportion": "Proportion of Songs", "artist": "Artist"},
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hover_data={"count": True, "proportion": ":.1%"},
            )
            fig_compare.update_layout(
                height=max(350, len(compare_artists) * 55),
                xaxis=dict(tickformat=".0%"),
                legend=dict(
                    title="Theme",
                    orientation="v",
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0.3)",
                ),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARTIST × THEME HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Artist × Theme Heatmap")
    st.caption(
        "Each cell shows the proportion of an artist's songs assigned to that theme. "
        "Darker = higher proportion. "
        "Useful for spotting each artist's thematic signature."
    )

    if artist_theme_wide.empty:
        st.info("Not enough theme data to build the comparison matrix.")
    else:
        # ── Artist filter ──────────────────────────────────────────────────
        filter_artists_heatmap = st.multiselect(
            "Filter artists (leave empty to show all)",
            options=all_artists,
            default=[],
            key="heatmap_artist_filter",
        )
        display_matrix = (
            artist_theme_wide.loc[filter_artists_heatmap]
            if filter_artists_heatmap
            else artist_theme_wide
        )

        # ── Theme filter ───────────────────────────────────────────────────
        filter_themes_heatmap = st.multiselect(
            "Filter themes (leave empty to show all)",
            options=THEMES,
            default=[],
            key="heatmap_theme_filter",
        )
        if filter_themes_heatmap:
            cols_to_show = [
                t for t in filter_themes_heatmap if t in display_matrix.columns
            ]
            display_matrix = display_matrix[cols_to_show]

        # Drop rows/columns that are all zeros for cleaner display
        display_matrix = display_matrix.loc[
            display_matrix.sum(axis=1) > 0,
            display_matrix.sum(axis=0) > 0,
        ]

        fig_heat = px.imshow(
            display_matrix,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            template="plotly_dark",
            labels=dict(x="Theme", y="Artist", color="Proportion"),
            zmin=0,
        )
        fig_heat.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        fig_heat.update_yaxes(tickfont=dict(size=11))
        fig_heat.update_layout(
            height=max(420, len(display_matrix) * 38),
            margin=dict(t=20, b=120, l=20, r=20),
        )
        fig_heat.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br><b>%{x}</b><br>Proportion: %{z:.1%}<extra></extra>"
            )
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # ── Dominant theme per artist table ───────────────────────────────
        st.subheader("Each Artist's Dominant Theme")
        dominant = artist_theme_wide.idxmax(axis=1).reset_index()
        dominant.columns = ["Artist", "Dominant Theme"]
        dominant["Proportion"] = artist_theme_wide.max(axis=1).values
        dominant["Proportion"] = dominant["Proportion"].map("{:.1%}".format)
        dominant = dominant.sort_values("Artist")

        st.dataframe(dominant, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BROWSE BY THEME
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Browse Songs by Theme")
    st.caption("Select a theme to see every song in the corpus assigned to it.")

    # Show theme options with song counts for context
    theme_counts = dict(zip(corpus_summary["theme"], corpus_summary["song_count"]))
    theme_options = [
        f"{t}  ({theme_counts.get(t, 0)} songs)"
        for t in THEMES
        if theme_counts.get(t, 0) > 0
    ]
    theme_lookup = {
        f"{t}  ({theme_counts.get(t, 0)} songs)": t
        for t in THEMES
        if theme_counts.get(t, 0) > 0
    }

    if not theme_options:
        st.info("No themes found. Run the classifier first.")
    else:
        selected_theme_label = st.selectbox(
            "Select theme",
            options=theme_options,
            key="browse_theme_select",
        )
        selected_theme = theme_lookup[selected_theme_label]

        try:
            theme_songs = get_songs_by_theme(df_themes, selected_theme)

            st.markdown(f"**{len(theme_songs)} songs** tagged with *{selected_theme}*")

            # Artist filter within theme
            theme_artist_options = sorted(theme_songs["artist"].unique())
            filter_theme_artist = st.multiselect(
                "Filter by artist (optional)",
                options=theme_artist_options,
                default=[],
                key="browse_theme_artist_filter",
            )

            display_songs = (
                theme_songs[theme_songs["artist"].isin(filter_theme_artist)]
                if filter_theme_artist
                else theme_songs
            )

            st.dataframe(
                display_songs[["artist", "title", "album", "themes"]],
                use_container_width=True,
                hide_index=True,
            )

            # Download
            st.download_button(
                label=f"⬇️ Download '{selected_theme}' songs as CSV",
                data=display_songs.to_csv(index=False),
                file_name=f"songs_{selected_theme.replace('/', '_').replace(' ', '_')}.csv",
                mime="text/csv",
            )

            st.markdown("---")

            # Artist distribution within the selected theme
            st.subheader(f"Artist Distribution — *{selected_theme}*")
            artist_dist_theme = theme_songs["artist"].value_counts().reset_index()
            artist_dist_theme.columns = ["artist", "count"]

            # Also compute as % of each artist's total songs
            artist_totals = df_themes["artist"].value_counts().to_dict()
            artist_dist_theme["pct_of_artist"] = (
                artist_dist_theme.apply(
                    lambda r: r["count"] / artist_totals.get(r["artist"], 1),
                    axis=1,
                )
            ).round(4)

            col_abs, col_pct = st.columns(2)

            with col_abs:
                st.markdown("**Absolute song count**")
                fig_abs = px.bar(
                    artist_dist_theme.sort_values("count", ascending=True),
                    x="count",
                    y="artist",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Teal",
                    labels={"count": "Songs", "artist": "Artist"},
                    template="plotly_dark",
                    text="count",
                )
                fig_abs.update_traces(textposition="outside")
                fig_abs.update_layout(
                    height=max(320, len(artist_dist_theme) * 32),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=40),
                )
                st.plotly_chart(fig_abs, use_container_width=True)

            with col_pct:
                st.markdown("**As % of artist's total songs**")
                fig_pct = px.bar(
                    artist_dist_theme.sort_values("pct_of_artist", ascending=True),
                    x="pct_of_artist",
                    y="artist",
                    orientation="h",
                    color="pct_of_artist",
                    color_continuous_scale="Oranges",
                    labels={"pct_of_artist": "% of Artist Songs", "artist": "Artist"},
                    template="plotly_dark",
                    text=artist_dist_theme.sort_values("pct_of_artist", ascending=True)[
                        "pct_of_artist"
                    ].map("{:.1%}".format),
                )
                fig_pct.update_traces(textposition="outside")
                fig_pct.update_layout(
                    height=max(320, len(artist_dist_theme) * 32),
                    coloraxis_showscale=False,
                    xaxis=dict(tickformat=".0%"),
                    margin=dict(t=10, b=10, l=10, r=80),
                )
                st.plotly_chart(fig_pct, use_container_width=True)

        except ValueError as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — THEME OVERLAP / CO-OCCURRENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Theme Co-occurrence Matrix")
    st.caption(
        "How often do pairs of themes appear together in the same song? "
        "Darker cells = themes that frequently co-occur. "
        "The diagonal shows how many songs have each theme at all."
    )

    if overlap_matrix.empty:
        st.info("Not enough data to compute co-occurrence.")
    else:
        # Normalise by diagonal (convert to conditional probability)
        normalise_overlap = st.checkbox(
            "Normalise (show P(theme B | theme A) instead of raw counts)",
            value=True,
        )

        display_overlap = overlap_matrix.copy().astype(float)
        if normalise_overlap:
            diag = pd.Series(
                [overlap_matrix.loc[t, t] for t in overlap_matrix.index],
                index=overlap_matrix.index,
            )
            display_overlap = display_overlap.div(diag, axis=0).round(3)
            # Diagonal becomes 1.0 after normalisation — set to NaN for clarity
            for t in display_overlap.index:
                display_overlap.loc[t, t] = float("nan")
            colorscale_label = "P(column theme | row theme)"
        else:
            colorscale_label = "Songs with both themes"

        fig_overlap = px.imshow(
            display_overlap,
            color_continuous_scale="RdPu",
            aspect="auto",
            template="plotly_dark",
            labels=dict(color=colorscale_label),
            zmin=0,
        )
        fig_overlap.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        fig_overlap.update_yaxes(tickfont=dict(size=9))
        fig_overlap.update_layout(
            height=600,
            margin=dict(t=20, b=140, l=20, r=20),
        )
        fig_overlap.update_traces(
            hovertemplate=(
                "<b>Row:</b> %{y}<br>"
                "<b>Col:</b> %{x}<br>"
                f"<b>{colorscale_label}:</b> %{{z:.3f}}<extra></extra>"
            )
        )
        st.plotly_chart(fig_overlap, use_container_width=True)

        st.markdown("---")

        # ── Top co-occurring pairs ─────────────────────────────────────────
        st.subheader("Most Frequently Co-occurring Theme Pairs")

        # Extract upper triangle as a list of (theme_a, theme_b, count) tuples
        pairs = []
        themes_in_matrix = list(overlap_matrix.index)
        for i, t1 in enumerate(themes_in_matrix):
            for j, t2 in enumerate(themes_in_matrix):
                if j <= i:
                    continue
                count = int(overlap_matrix.loc[t1, t2])
                if count > 0:
                    pairs.append(
                        {
                            "Theme A": t1,
                            "Theme B": t2,
                            "Songs": count,
                        }
                    )

        pairs_df = (
            pd.DataFrame(pairs)
            .sort_values("Songs", ascending=False)
            .reset_index(drop=True)
            .head(20)
        )
        pairs_df.insert(0, "Rank", range(1, len(pairs_df) + 1))

        st.dataframe(
            pairs_df.style.background_gradient(subset=["Songs"], cmap="RdPu"),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These pairs reveal natural thematic groupings in your corpus — "
            "e.g. if 'heartbreak' and 'longing' always co-occur, "
            "they may be redundant themes worth merging."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ARTIST CLUSTERS
# Reads from pre-saved artist_clusters.json — never calls Ollama at runtime.
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("Per-Artist Lyrical Clusters")
    st.caption(
        "Each artist's songs are grouped into clusters using KMeans in embedding "
        "space. Cluster labels were generated offline by Llama 3.2 summarising "
        "the most representative songs in each group."
    )

    if not cluster_data:
        st.info(
            "Artist clusters have not been precomputed yet. "
            "Run the following command on your local machine "
            "(with Ollama running) and redeploy:\n\n"
            "```bash\npython analysis/themes_llm.py --clusters\n```"
        )
    else:
        # ── Artist selector ────────────────────────────────────────────────
        clustered_artists = sorted(cluster_data.keys())
        selected_cluster_artist = st.selectbox(
            "Select artist",
            options=clustered_artists,
            key="cluster_artist_select",
        )

        artist_cluster = cluster_data[selected_cluster_artist]
        n_clusters = artist_cluster["n_clusters"]
        clusterable = artist_cluster["clusterable"]
        cluster_labels = artist_cluster["cluster_labels"]  # {"0": "label", ...}
        songs_list = artist_cluster["songs"]

        songs_df = pd.DataFrame(songs_list)

        # ── Summary metrics ────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Songs", len(songs_df))
        m2.metric("Clusters found", n_clusters)
        m3.metric(
            "Clustered",
            "Yes" if clusterable else f"No (< {30} songs)",
        )

        st.markdown("---")

        if not clusterable:
            st.info(
                f"{selected_cluster_artist} has too few songs for clustering. "
                "All songs are shown as a single group."
            )

        # ── Cluster label summary ──────────────────────────────────────────
        st.subheader("Cluster Labels")
        label_rows = [
            {
                "Cluster": f"Cluster {cid}",
                "Label": label,
                "Songs": int(songs_df[songs_df["cluster_id"] == int(cid)].shape[0]),
            }
            for cid, label in cluster_labels.items()
        ]
        label_df = pd.DataFrame(label_rows).sort_values("Cluster")
        st.dataframe(label_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Per-cluster song browser ───────────────────────────────────────
        st.subheader("Browse Songs by Cluster")

        cluster_options = {
            f"Cluster {cid} — {label}  ({int(songs_df[songs_df['cluster_id'] == int(cid)].shape[0])} songs)": int(
                cid
            )
            for cid, label in sorted(cluster_labels.items())
        }
        selected_cluster_label = st.selectbox(
            "Select cluster",
            options=list(cluster_options.keys()),
            key="cluster_id_select",
        )
        selected_cluster_id = cluster_options[selected_cluster_label]

        cluster_songs = (
            songs_df[songs_df["cluster_id"] == selected_cluster_id][
                ["title", "album", "cluster_label"]
            ]
            .sort_values("title")
            .reset_index(drop=True)
        )

        st.dataframe(cluster_songs, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Cluster size bar chart ─────────────────────────────────────────
        st.subheader("Cluster Size Distribution")
        size_df = (
            songs_df.groupby(["cluster_id", "cluster_label"])
            .size()
            .reset_index(name="count")
            .sort_values("cluster_id")
        )
        size_df["label"] = size_df.apply(
            lambda r: f"Cluster {r['cluster_id']}: {r['cluster_label']}", axis=1
        )

        fig_size = px.bar(
            size_df,
            x="label",
            y="count",
            color="label",
            text="count",
            labels={"label": "Cluster", "count": "Songs"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_size.update_traces(textposition="outside", showlegend=False)
        fig_size.update_layout(
            height=360,
            xaxis_tickangle=-20,
            margin=dict(t=10, b=80, l=10, r=40),
        )
        st.plotly_chart(fig_size, use_container_width=True)

        # ── Full corpus cluster overview ───────────────────────────────────
        st.markdown("---")
        st.subheader("All Artists — Cluster Summary")
        st.caption(
            "Overview of clustering results across every artist. "
            "Artists marked as not clusterable had fewer than 30 songs."
        )

        summary_rows = []
        for artist_name, data in cluster_data.items():
            labels_str = " · ".join(data["cluster_labels"].values())
            summary_rows.append(
                {
                    "Artist": artist_name,
                    "Songs": len(data["songs"]),
                    "Clusters": data["n_clusters"],
                    "Clusterable": "✅" if data["clusterable"] else "❌",
                    "Cluster Labels": labels_str,
                }
            )

        summary_df = (
            pd.DataFrame(summary_rows).sort_values("Artist").reset_index(drop=True)
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
