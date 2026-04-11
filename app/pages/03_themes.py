import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from analysis.themes_llm import (
    load_theme_results,
    load_artist_clusters,
    get_artist_theme_distribution,
    get_corpus_theme_summary,
    get_songs_by_theme,
    get_theme_overlap_matrix,
    THEMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Themes — Lyrics Analysis",
    page_icon="🎨",
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


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (all reads from pre-saved files — Ollama never called at runtime)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading theme classifications...")
def load_themes(_df):
    try:
        return load_theme_results(_df)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading artist clusters...")
def load_clusters():
    return load_artist_clusters()


df_themes = load_themes(df)
cluster_data = load_clusters()

if df_themes is None:
    st.error(
        "Theme classifications not found. Run:\n\n"
        "```bash\npython analysis/themes_llm.py\n```"
    )
    st.stop()

n_unclassified = df_themes["themes"].apply(lambda x: len(x) == 0).sum()
if n_unclassified > 0:
    pct = round(n_unclassified / len(df_themes) * 100, 1)
    st.warning(
        f"{n_unclassified} songs ({pct}%) have no theme classification. "
        "Run `python analysis/themes_llm.py` to classify them."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE SHARED DATA
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def compute_shared(_df_themes):
    corpus_summary = get_corpus_theme_summary(_df_themes)
    overlap_matrix = get_theme_overlap_matrix(_df_themes)
    return corpus_summary, overlap_matrix


corpus_summary, overlap_matrix = compute_shared(df_themes)
all_artists = sorted(df_themes["artist"].unique())


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — corpus-average theme proportions (used for divergence chart)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def compute_corpus_avg(_df_themes):
    """
    For each theme, the fraction of all theme assignments that are that theme.
    This adds to 1.0 and is the corpus-level baseline for comparison.
    """
    exploded = _df_themes.explode("themes").dropna(subset=["themes"])
    exploded = exploded[exploded["themes"] != ""]
    total = len(exploded)
    counts = exploded["themes"].value_counts()
    return (counts / total).to_dict()


corpus_avg = compute_corpus_avg(df_themes)


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
    classified = df_themes["themes"].apply(lambda x: len(x) > 0).sum()
    st.metric("Songs with themes", f"{classified:,}")
    st.metric("Theme taxonomy", f"{len(THEMES)} labels")

    if not corpus_summary.empty:
        st.metric("Most common theme", corpus_summary.iloc[0]["theme"])

    st.markdown("---")
    st.caption("Themes classified offline by Llama 3.2 via Ollama.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎨 Theme Analysis")
st.markdown(
    f"Exploring **{len(THEMES)} lyrical themes** across **{len(df_themes):,} songs** "
    f"by **{df_themes['artist'].nunique()} artists**."
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Corpus Overview",
        "🎤 Artist Themes",
        "🔎 Browse by Theme",
        "🔀 Theme Overlap",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORPUS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("What Themes Dominate This Corpus?")
    st.markdown(
        """
        Before comparing artists, it helps to understand the baseline:
        what themes are most and least common across all songs?
        This tells us whether an artist is writing about something typical
        for this corpus or genuinely unusual.
 
        **How themes were assigned:** each song's lyrics were sent (as a
        200-word snippet) to **Llama 3.2** running locally via
        [Ollama](https://ollama.com). The model was given a fixed taxonomy
        of 18 themes with one-line definitions and asked to assign 1–2 that
        best describe the song's lyrical content. The model was instructed
        to prioritize specific themes (holiday, LGBTQ+, body image) over
        general ones (longing, heartbreak) when both applied to reduce
        over-representation of vague catch-all categories.
        All classifications were run once offline and saved. The app reads
        pre-saved results at runtime and never calls the model live.
        """
    )
    st.markdown("---")

    if corpus_summary.empty:
        st.info("No theme data available.")
    else:
        col_bar, col_pie = st.columns([1.4, 1])

        with col_bar:
            st.markdown("**How many songs are tagged with each theme?**")
            st.caption(
                "Each song can carry 1–2 themes, so totals exceed the song count. "
            )
            fig_bar = px.bar(
                corpus_summary,
                x="song_count",
                y="theme",
                orientation="h",
                text="song_count",
                labels={"song_count": "Songs tagged", "theme": "Theme"},
                template="plotly_dark",
            )
            fig_bar.update_traces(
                marker_color="#a78bfa",
                textposition="outside",
            )
            fig_bar.update_layout(
                height=max(420, len(corpus_summary) * 30),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(title="Songs tagged with this theme"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            st.markdown(
                "**What share of all theme assignments does each theme account for?**"
            )
            st.caption(
                "This normalizes for the fact that songs can have multiple themes. "
            )
            fig_pie = px.pie(
                corpus_summary,
                names="theme",
                values="song_count",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.35,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "<b>%{label}</b><br>%{value} songs (%{percent})<extra></extra>"
                ),
            )
            fig_pie.update_layout(
                height=520,
                showlegend=False,
                margin=dict(t=20, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Assignment stats ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("How Many Themes Are Assigned per Song?")
        st.caption("A well-calibrated classifier should assign 1–2 themes per song. ")

        theme_counts_per_song = (
            df_themes["themes"].apply(len).value_counts().sort_index().reset_index()
        )
        theme_counts_per_song.columns = ["n_themes", "songs"]
        avg_t = df_themes["themes"].apply(len).mean()

        fig_dist = px.bar(
            theme_counts_per_song,
            x="n_themes",
            y="songs",
            text="songs",
            labels={"n_themes": "Themes assigned per song", "songs": "Number of songs"},
            template="plotly_dark",
        )
        fig_dist.update_traces(marker_color="#a78bfa", textposition="outside")
        fig_dist.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=40),
            xaxis=dict(dtick=1),
        )
        # fig_dist.add_vline(
        #     x=avg_t,
        #     line_dash="dash",
        #     line_color="rgba(255,255,255,0.4)",
        #     annotation_text=f"Avg: {avg_t:.2f}",
        #     annotation_position="top right",
        # )
        st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARTIST THEMES  (breakdown + comparison merged)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("What Does Each Artist Write About?")
    st.markdown(
        """
        Two complementary questions are answered here:
 
        1. **Per-artist breakdown**: what is the theme profile of a single
           artist, and where do their songs sit on the UMAP?
        2. **Artist comparison**: how do artists differ from each other
           *and* from the corpus average?
 
        **Note on normalisation:** theme proportions are computed as
        *fraction of all theme assignments for that artist*, so bars always
        add to 100%. This is different from "fraction of songs with this theme"
        (which does not add to 100% because songs can have multiple themes)
        and makes comparisons between artists meaningful.
        """
    )
    st.markdown("---")

    # ── A: Single-artist breakdown ─────────────────────────────────────────
    st.subheader(f"Theme Profile — {selected_artist}")

    try:
        artist_dist = get_artist_theme_distribution(
            df_themes, selected_artist, normalize=True
        )
    except ValueError as e:
        st.error(str(e))
        artist_dist = pd.DataFrame()

    if not artist_dist.empty:
        # Recompute as fraction of theme assignments (sums to 1.0)
        total_assignments = artist_dist["count"].sum()
        artist_dist["share"] = (artist_dist["count"] / total_assignments).round(4)

        top_theme = artist_dist.iloc[0]["theme"]
        n_themes = len(artist_dist)
        song_count = len(df_themes[df_themes["artist"] == selected_artist])

        m1, m2, m3 = st.columns(3)
        m1.metric("Top theme", top_theme)
        m2.metric("Themes present", n_themes)
        m3.metric("Songs in corpus", song_count)

        col_l, col_r = st.columns([1.2, 1])

        with col_l:
            st.markdown("**Share of theme assignments** — bars add to 100%")
            st.caption(
                "Each bar shows what fraction of all theme labels assigned "
                "to this artist's songs is that theme."
            )
            fig_artist = px.bar(
                artist_dist.sort_values("share"),
                x="share",
                y="theme",
                orientation="h",
                text=artist_dist.sort_values("share")["share"].map("{:.1%}".format),
                labels={"share": "Share of theme assignments", "theme": "Theme"},
                template="plotly_dark",
            )
            fig_artist.update_traces(
                marker_color="#a78bfa",
                textposition="outside",
            )
            fig_artist.update_layout(
                height=max(380, n_themes * 32),
                xaxis=dict(
                    tickformat=".0%", range=[0, artist_dist["share"].max() * 1.25]
                ),
                margin=dict(t=10, b=10, l=10, r=80),
            )
            st.plotly_chart(fig_artist, use_container_width=True)

        with col_r:
            st.markdown("**Theme mix as a proportion**")
            st.caption("Each slice = share of that artist's theme assignments.")
            fig_pie_a = px.pie(
                artist_dist,
                names="theme",
                values="count",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.35,
            )
            fig_pie_a.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "<b>%{label}</b><br>%{value} songs (%{percent})<extra></extra>"
                ),
            )
            fig_pie_a.update_layout(
                height=420,
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie_a, use_container_width=True)

        # ── Per-artist UMAP with cluster colouring ─────────────────────────
        if umap_2d is not None:
            st.markdown("---")
            st.subheader(f"Where Do {selected_artist}'s Songs Sit on the UMAP?")
            st.caption(
                "Each point is one of this artist's songs in the 2D embedding "
                "projection. Points are coloured by the artist cluster label "
                "computed offline (KMeans in 384-d embedding space). "
                "Tight groups of the same colour mean the artist has a distinct "
                "lyrical sub-style. Scattered points mean that cluster's songs "
                "span a wide thematic range."
            )

            # Build full plot_df with cluster labels
            artist_mask = df["artist"] == selected_artist
            artist_idx = df.index[artist_mask].tolist()

            artist_umap = pd.DataFrame(
                {
                    "x": umap_2d[artist_idx, 0],
                    "y": umap_2d[artist_idx, 1],
                    "title": df.loc[artist_mask, "title"].values,
                    "album": df.loc[artist_mask, "album"].fillna("Unknown").values,
                }
            ).reset_index(drop=True)

            # Attach cluster labels if available
            if cluster_data and selected_artist in cluster_data:
                artist_cluster_info = cluster_data[selected_artist]
                song_to_label = {
                    s["title"]: s["cluster_label"] for s in artist_cluster_info["songs"]
                }
                artist_umap["cluster"] = artist_umap["title"].map(
                    lambda t: song_to_label.get(t, "unassigned")
                )
            else:
                artist_umap["cluster"] = "all songs"

            # Background: all other songs, greyed out
            all_umap = pd.DataFrame(
                {
                    "x": umap_2d[:, 0],
                    "y": umap_2d[:, 1],
                }
            )
            other_mask = ~df.index.isin(artist_idx)
            other_umap = all_umap[other_mask]

            fig_umap_a = go.Figure()

            # Grey background
            fig_umap_a.add_trace(
                go.Scatter(
                    x=other_umap["x"],
                    y=other_umap["y"],
                    mode="markers",
                    marker=dict(
                        size=3, color="rgba(120,120,140,0.18)", line=dict(width=0)
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Other songs",
                )
            )

            # Artist songs coloured by cluster
            cluster_colours = px.colors.qualitative.Pastel + px.colors.qualitative.Set2
            unique_clusters = artist_umap["cluster"].unique()
            colour_map = dict(
                zip(
                    unique_clusters,
                    cluster_colours[: len(unique_clusters)],
                )
            )

            for cluster_label, colour in colour_map.items():
                sub = artist_umap[artist_umap["cluster"] == cluster_label]
                fig_umap_a.add_trace(
                    go.Scatter(
                        x=sub["x"],
                        y=sub["y"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=colour,
                            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                        ),
                        text=sub["title"] + "<br>" + sub["album"],
                        hovertemplate="%{text}<extra>" + cluster_label + "</extra>",
                        name=cluster_label,
                        showlegend=True,
                    )
                )

            fig_umap_a.update_layout(
                template="plotly_dark",
                height=480,
                legend=dict(
                    title="Cluster",
                    bgcolor="rgba(0,0,0,0.4)",
                    font=dict(size=11),
                ),
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, title=""
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, title=""
                ),
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_umap_a, use_container_width=True)

    st.markdown("---")

    # ── B: Multi-artist comparison ─────────────────────────────────────────
    st.subheader("How Do Artists Compare to Each Other and to the Corpus Average?")
    st.markdown(
        """
        Two complementary views:
 
        **Stacked bar**: each artist's full theme profile side by side,
        normalised so bars sum to 100%. This answers *what each artist writes about*.
 
        **Divergence chart**: for a selected artist, how much does each theme
        deviate from the corpus average? Bars pointing right mean the artist
        over-indexes on that theme relative to the typical artist in this corpus.
        Bars pointing left mean they under-index. This answers *what makes an
        artist's theme profile unusual*.
        """
    )

    compare_artists = st.multiselect(
        "Select artists to compare",
        options=all_artists,
        default=all_artists[:4],
        key="compare_artists",
    )

    if compare_artists:
        # Build normalised share (sums to 1.0 per artist)
        rows = []
        for artist in compare_artists:
            try:
                dist = get_artist_theme_distribution(df_themes, artist, normalize=False)
                total = dist["count"].sum()
                if total == 0:
                    continue
                dist["share"] = dist["count"] / total
                dist["artist"] = artist
                rows.append(dist)
            except ValueError:
                continue

        if rows:
            compare_df = pd.concat(rows, ignore_index=True)

            # ── Stacked bar — sums to 100% ─────────────────────────────────
            # st.markdown("**Normalised theme mix — each bar sums to 100%**")
            st.caption(
                "Wider segments = larger share of that artist's theme assignments. "
                "Artists with many small segments have diverse output; "
                "artists dominated by one or two themes have a focused style."
            )

            fig_stack = px.bar(
                compare_df,
                x="share",
                y="artist",
                color="theme",
                orientation="h",
                barmode="stack",
                labels={
                    "share": "Share of theme assignments",
                    "artist": "Artist",
                    "theme": "Theme",
                },
                template="plotly_dark",
                color_discrete_sequence=(
                    px.colors.qualitative.Pastel + px.colors.qualitative.Set2
                ),
                hover_data={"count": True, "share": ":.1%"},
            )
            fig_stack.update_layout(
                height=max(320, len(compare_artists) * 48),
                xaxis=dict(tickformat=".0%", range=[0, 1.0]),
                legend=dict(
                    title="Theme",
                    font=dict(size=9),
                    bgcolor="rgba(0,0,0,0.3)",
                ),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")

    # ── C: Divergence from corpus average ─────────────────────────────────
    st.subheader("How Does One Artist Deviate from the Corpus Average?")
    st.caption(
        "Bars pointing **right** (positive) mean the artist uses this theme "
        "more than the corpus average. Bars pointing **left** (negative) mean "
        "they use it less. The zero line is the corpus baseline. "
        "This is the clearest way to see what makes an artist's lyrical "
        "profile distinctive."
    )

    divergence_artist = st.selectbox(
        "Select artist for divergence chart",
        options=all_artists,
        key="divergence_artist",
    )

    try:
        div_dist = get_artist_theme_distribution(
            df_themes, divergence_artist, normalize=False
        )
        total_div = div_dist["count"].sum()
        if total_div > 0:
            div_dist["share"] = div_dist["count"] / total_div
            div_dist["corpus_avg"] = div_dist["theme"].map(
                lambda t: corpus_avg.get(t, 0)
            )
            div_dist["divergence"] = (div_dist["share"] - div_dist["corpus_avg"]).round(
                4
            )
            div_dist = div_dist.sort_values("divergence")

            fig_div = px.bar(
                div_dist,
                x="divergence",
                y="theme",
                orientation="h",
                text=div_dist["divergence"].map(
                    lambda v: f"+{v:.1%}" if v >= 0 else f"{v:.1%}"
                ),
                labels={
                    "divergence": "Deviation from corpus average",
                    "theme": "Theme",
                },
                template="plotly_dark",
                color="divergence",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig_div.update_traces(textposition="outside")
            fig_div.update_layout(
                height=max(420, len(div_dist) * 30),
                coloraxis_showscale=False,
                xaxis=dict(
                    tickformat=".0%",
                    zeroline=True,
                    zerolinecolor="rgba(255,255,255,0.3)",
                    zerolinewidth=2,
                ),
                margin=dict(t=10, b=10, l=10, r=100),
            )
            st.plotly_chart(fig_div, use_container_width=True)
            st.caption(
                "Red bars = above corpus average. Blue bars = below corpus average."
            )
    except ValueError as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BROWSE BY THEME
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Which Songs Belong to a Given Theme?")
    st.caption(
        "Select a theme to see every song in the corpus tagged with it. "
        "The two bar charts below answer complementary questions: "
        "the left chart asks which artists produce the most songs with this theme "
        "in absolute terms; the right chart asks which artists are most "
        "*defined* by it relative to their own output."
    )

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
        st.info("No theme data found.")
    else:
        selected_theme_label = st.selectbox(
            "Select theme",
            options=theme_options,
            key="browse_theme",
        )
        selected_theme = theme_lookup[selected_theme_label]

        try:
            theme_songs = get_songs_by_theme(df_themes, selected_theme)
            st.markdown(f"**{len(theme_songs)} songs** tagged with *{selected_theme}*")

            filter_artist = st.multiselect(
                "Filter by artist (optional)",
                options=sorted(theme_songs["artist"].unique()),
                default=[],
                key="browse_theme_artist",
            )
            display_songs = (
                theme_songs[theme_songs["artist"].isin(filter_artist)]
                if filter_artist
                else theme_songs
            )

            st.dataframe(
                display_songs[["artist", "title", "album", "themes"]],
                use_container_width=True,
                hide_index=True,
            )

            # st.download_button(
            #     label=f"⬇️ Download as CSV",
            #     data=display_songs.to_csv(index=False),
            #     file_name=(
            #         f"songs_{selected_theme.replace('/', '_').replace(' ', '_')}.csv"
            #     ),
            #     mime="text/csv",
            # )

            st.markdown("---")

            artist_dist_theme = theme_songs["artist"].value_counts().reset_index()
            artist_dist_theme.columns = ["artist", "count"]
            artist_totals = df_themes["artist"].value_counts().to_dict()
            artist_dist_theme["pct_of_artist"] = (
                artist_dist_theme.apply(
                    lambda r: r["count"] / artist_totals.get(r["artist"], 1),
                    axis=1,
                )
            ).round(4)

            col_abs, col_pct = st.columns(2)

            with col_abs:
                st.markdown(
                    "**Which artists contribute the most songs to this theme?**"
                )
                st.caption(
                    "Absolute song count — reflects both prolificacy and theme focus."
                )
                fig_abs = px.bar(
                    artist_dist_theme.sort_values("count", ascending=True),
                    x="count",
                    y="artist",
                    orientation="h",
                    text="count",
                    labels={"count": "Songs", "artist": "Artist"},
                    template="plotly_dark",
                )
                fig_abs.update_traces(marker_color="#a78bfa", textposition="outside")
                fig_abs.update_layout(
                    height=max(300, len(artist_dist_theme) * 32),
                    margin=dict(t=10, b=10, l=10, r=40),
                )
                st.plotly_chart(fig_abs, use_container_width=True)

            with col_pct:
                st.markdown(
                    "**For which artists is this theme most central to their output?**"
                )
                st.caption(
                    "% of each artist's songs with this theme — "
                    "controls for artists with larger discographies."
                )
                fig_pct = px.bar(
                    artist_dist_theme.sort_values("pct_of_artist", ascending=True),
                    x="pct_of_artist",
                    y="artist",
                    orientation="h",
                    text=artist_dist_theme.sort_values("pct_of_artist")[
                        "pct_of_artist"
                    ].map("{:.1%}".format),
                    labels={"pct_of_artist": "% of Artist's Songs", "artist": "Artist"},
                    template="plotly_dark",
                )
                fig_pct.update_traces(marker_color="#34d399", textposition="outside")
                fig_pct.update_layout(
                    height=max(300, len(artist_dist_theme) * 32),
                    xaxis=dict(
                        tickformat=".0%",
                        range=[0, artist_dist_theme["pct_of_artist"].max() * 1.2],
                    ),
                    margin=dict(t=10, b=10, l=10, r=80),
                )
                st.plotly_chart(fig_pct, use_container_width=True)

        except ValueError as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — THEME OVERLAP
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Which Themes Tend to Appear Together?")
    st.markdown(
        """
        When a song is tagged with one theme, how likely is it to also carry
        another? This co-occurrence matrix answers that question.
 
        **How to read it:** each cell shows the proportion of songs with the
        *row theme* that are *also* tagged with the *column theme*
        (conditional probability). A cell of 0.8 means 80% of songs tagged
        with the row theme are also tagged with the column theme.
 
        **Why this matters:** theme pairs with high co-occurrence may be
        redundant in the taxonomy — if "heartbreak" and "longing" almost
        always appear together, they're measuring the same thing. Theme pairs
        with near-zero co-occurrence are genuinely orthogonal dimensions of
        lyrical content.
        """
    )

    if overlap_matrix.empty:
        st.info("Not enough data to compute co-occurrence.")
    else:
        display_overlap = overlap_matrix.copy().astype(float)
        diag = pd.Series(
            [overlap_matrix.loc[t, t] for t in overlap_matrix.index],
            index=overlap_matrix.index,
        )
        # Normalise: P(col theme | row theme)
        display_overlap = display_overlap.div(diag, axis=0).round(3)
        for t in display_overlap.index:
            display_overlap.loc[t, t] = float("nan")

        fig_overlap = px.imshow(
            display_overlap,
            color_continuous_scale="RdPu",
            aspect="auto",
            template="plotly_dark",
            labels=dict(color="P(col | row)"),
            zmin=0,
            zmax=1,
        )
        fig_overlap.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        fig_overlap.update_yaxes(tickfont=dict(size=9))
        fig_overlap.update_layout(
            height=580,
            margin=dict(t=20, b=140, l=20, r=20),
        )
        fig_overlap.update_traces(
            hovertemplate=(
                "<b>Given:</b> %{y}<br>"
                "<b>Also has:</b> %{x}<br>"
                "<b>P =</b> %{z:.2f}<extra></extra>"
            )
        )
        st.plotly_chart(fig_overlap, use_container_width=True)

        # ── Top co-occurring pairs ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Most Frequently Co-occurring Theme Pairs")
        st.caption(
            "Pairs with high co-occurrence rates may represent overlapping "
            "concepts in the taxonomy. Pairs with near-zero rates are "
            "genuinely independent lyrical dimensions."
        )

        pairs = []
        themes_in_matrix = list(overlap_matrix.index)
        for i, t1 in enumerate(themes_in_matrix):
            for j, t2 in enumerate(themes_in_matrix):
                if j <= i:
                    continue
                count = int(overlap_matrix.loc[t1, t2])
                if count > 0:
                    total_t1 = int(overlap_matrix.loc[t1, t1])
                    rate = round(count / total_t1, 3) if total_t1 > 0 else 0
                    pairs.append(
                        {
                            "Theme A": t1,
                            "Theme B": t2,
                            "Co-occurrences": count,
                            "P(B | A)": rate,
                        }
                    )

        pairs_df = (
            pd.DataFrame(pairs)
            .sort_values("Co-occurrences", ascending=False)
            .reset_index(drop=True)
            .head(20)
        )
        pairs_df.insert(0, "Rank", range(1, len(pairs_df) + 1))

        st.dataframe(
            pairs_df.style.background_gradient(
                subset=["Co-occurrences", "P(B | A)"], cmap="RdPu"
            ),
            use_container_width=True,
            hide_index=True,
        )
