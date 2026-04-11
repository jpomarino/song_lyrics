# Lyrics Analysis
 
An end-to-end NLP project that scrapes song lyrics, embeds them using a sentence transformer model, and serves an interactive Streamlit app for exploring lyrical similarity, themes, and recommendations across a curated corpus of artists.
 
**Live app:** [https://jpomarino-song-lyrics-appmain-fjnwwp.streamlit.app/](https://jpomarino-song-lyrics-appmain-fjnwwp.streamlit.app/) (takes some time to load).
 
## Motivation
 
I have always cared about song lyrics more than most people I know. For me, lyrics are often the primary reason I connect with an artist. Over time I started asking questions I could not easily answer by just listening:
 
- Do artists I love actually write about similar things, or do I just like how they sound?
- Are there songs I have not heard yet that I would love based on lyrics I already know?
- What makes one artist's writing feel distinct from another's?
 
This project is an attempt to answer those questions with data. It pulls together web scraping, NLP preprocessing, vector embeddings, unsupervised clustering, LLM-based classification, and an interactive deployment. It started as a way to pick up new data science skills and turned into something I genuinely use.
 
## What the App Does
 
The app has five pages:
 
**Home** gives an overview of the song corpus including songs per artist, vocabulary diversity, average song length, and a release timeline showing when each artist was most prolific.
 
**Explore** renders all songs as a 2D UMAP scatter plot, where proximity means lyrical similarity. You can color the map by artist, highlight songs belonging to a specific theme, or run a formal cluster analysis with a chi-square test to check whether the two natural clusters in the corpus have statistically different theme profiles.
 
**Recommend** takes any word, phrase, or lyric snippet as input and returns the most semantically similar songs in the corpus. It uses the same embedding model as the rest of the pipeline, so the search is purely based on meaning rather than keyword matching.
 
**Themes** shows how songs are distributed across 18 lyrical themes classified by a local LLM. You can view the corpus-wide theme breakdown, compare artists' theme profiles side by side, see how each artist deviates from the corpus average, and explore which themes tend to co-occur.
 
**Similarity** lets you find songs most similar to a chosen song, explore an artist's full discography as a pairwise similarity heatmap, compare artists to each other, and visualize artist distinctiveness as a scatter plot of intra-artist vs inter-artist similarity.
 
## Project Structure
 
```
song_lyrics/
├── data/
│   ├── raw/
│   │   └── lyrics_raw.json           # original scraped lyrics
│   ├── processed/
│   │   ├── final_songs.json          # cleaned and preprocessed corpus
│   │   ├── song_themes.json          # LLM-classified themes per song
│   │   └── artist_clusters.json      # per-artist cluster labels
│   └── cache/
│       ├── llm_embedding.npy         # 384-dimensional song embeddings
│       ├── umap_embedding.npy        # 2D UMAP projection
│       └── theme_progress.json       # classification checkpoint
├── scraper/
│   ├── genius_scraper.py             # initial data collection script
│   └── update_dataset.py             # add new artists without re-scraping
├── pipeline/
│   ├── preprocess.py                 # text cleaning functions
│   ├── embed.py                      # chunking and embedding logic
│   └── build.py                      # end-to-end pipeline runner
├── analysis/
│   ├── similarity.py                 # cosine similarity and artist stats
│   └── themes_llm.py                 # Ollama-based theme classification
├── app/
│   ├── main.py                       # Streamlit entry point and shared state
│   └── pages/
│       ├── 01_explore.py
│       ├── 02_recommend.py
│       ├── 03_themes.py
│       └── 04_similarity.py
├── .env                              # API keys (never committed)
├── requirements.txt
└── README.md
```
 
## Pipeline Overview
 
### 1. Scraping
 
Lyrics, album metadata, release dates, and artist thumbnails are scraped from [Genius](https://genius.com) using the `lyricsgenius` Python library. Remixes, live versions, demos, and instrumentals are excluded via a keyword filter. Artist thumbnail URLs are stored separately and joined into the main DataFrame during preprocessing.
 
### 2. Filtering and Cleaning
 
Duplicate songs are removed. Lyrics go through two cleaning paths depending on downstream use: a light clean (section headers stripped, whitespace collapsed, natural language preserved) for the embedding model, and a heavier clean (stopwords removed, lemmatization applied) for any classical NLP tasks.
 
### 3. Vector Embeddings
 
Each song's lyrics are embedded using [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Sentence Transformers, producing a 384-dimensional vector per song. Songs longer than the model's token window are split into overlapping chunks, each chunk is embedded independently, and the results are averaged with weights proportional to chunk length before L2 normalization.
 
### 4. UMAP Projection
 
The 384-dimensional embeddings are projected to 2D using UMAP for visualization. Clustering and similarity calculations always use the full 384-dimensional space rather than the 2D projection, since UMAP distorts distances and should only be used for visualization.
 
### 5. Similarity Calculations
 
Cosine similarity between embedding vectors powers the song recommender, the song-to-song heatmaps, and the artist similarity matrix. Artist-level similarity is computed by comparing centroid embeddings (the mean vector across all of an artist's songs). Distinctiveness is defined as intra-artist similarity minus inter-artist similarity.
 
### 6. Theme Classification
 
Each song is classified into 1 to 2 themes from a fixed 18-label taxonomy using **Llama 3.2** running locally via [Ollama](https://ollama.com). The model receives the first 200 words of lyrics along with one-line definitions for each theme and instructions to prioritize specific themes over general ones. Classification runs once offline and results are saved to `song_themes.json`. The Streamlit app reads pre-saved results and never calls the model at runtime.
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Data collection | `lyricsgenius`, Genius API |
| NLP preprocessing | `spaCy`, `NLTK`, `re` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Dimensionality reduction | `UMAP` |
| Clustering | `scikit-learn` (KMeans) |
| LLM theme classification | Ollama, Llama 3.2, `openai` (client) |
| Statistical testing | `scipy` (chi-square) |
| Visualization | `Plotly`, `Streamlit` |
| Language | Python 3.12 |
 
## Running Locally
 
### Prerequisites
 
- Python 3.12
- [Ollama](https://ollama.com) installed and running (only needed for theme classification, not for the app itself)
- A Genius API key from [genius.com/api-clients](https://genius.com/api-clients)
 
### Setup
 
```bash
git clone https://github.com/yourusername/song-lyrics.git
cd song-lyrics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
Create a `.env` file in the project root:
 
```
GENIUS_CLIENT_ACCESS_TOKEN=your_token_here
```
 
### Running the Pipeline (first time)
 
If you are cloning this repo without the pre-built data files, you will need to run the full pipeline:
 
```bash
# 1. Scrape lyrics (this takes a while depending on corpus size)
python scraper/genius_scraper.py
 
# 2. Build embeddings and UMAP projection
python pipeline/build.py
 
# 3. Classify themes (requires Ollama running in a separate terminal)
ollama serve   # in a separate terminal
python analysis/themes_llm.py
 
# 4. Precompute artist clusters
python analysis/themes_llm.py --clusters
```
 
If you are cloning the repo with the data files already present, you can skip straight to step 4 of the app launch below.
 
### Launching the App
 
```bash
streamlit run app/main.py
```
 
### Adding New Artists
 
```bash
# Add one or more artists to the corpus
python scraper/update_dataset.py --artists "Artist Name" --rebuild
 
# Classify themes for the new artists only
python analysis/themes_llm.py --new-artists-only
 
# Recompute clusters for the new artists
python analysis/themes_llm.py --clusters --artists "Artist Name"
```
 
---
 
## Artists in the Corpus
 
The corpus covers 30+ artists I love primarily in the pop and indie pop space.
 
The corpus intentionally focuses on a single genre and demographic to make similarity and theme comparisons more meaningful. Adding artists from very different genres will shift the corpus baseline and may affect theme proportion comparisons.
 
## Limitations
 
Theme classifications are imperfect. The LLM occasionally assigns generic themes to songs that have more specific content, particularly for songs with abstract or metaphorical lyrics. The taxonomy itself reflects subjective choices about what constitutes a meaningful lyrical category.
 
UMAP projections are non-deterministic and sensitive to hyperparameter choices. The visual clusters you see depend on the `n_neighbors` and `min_dist` settings, and re-running UMAP with different random seeds will produce different layouts even with identical embeddings.
 
The corpus is intentionally narrow. Similarity scores are relative to the songs in this dataset, so a "high similarity" score means similar to other songs in this corpus, not to all music that exists.
 
## Acknowledgments
 
Lyrics data sourced from [Genius](https://genius.com) via the `lyricsgenius` library. Embeddings powered by the `all-MiniLM-L6-v2` model from the Sentence Transformers library. Theme classification powered by Llama 3.2 via Ollama.
