# Project: Semantic Search Engine (Day 2 NLP Lab — YDL 2026)

## Goal (one sentence)
Build a search engine that finds song lyrics **by meaning, not by exact words**:
type a query, get back the most similar songs ranked by cosine similarity, plus
one labelled artifact (a plot) that tells the story.

Brief: `Day2_NLP_Project_Brief.pdf` · Idea chosen: **Semantic search engine**
(combines: embeddings + cosine similarity + nearest neighbors).

## ⭐ Build it the SAME WAY as the toolbox notebook
Reference notebook: `lab_embeddings_toolbox.ipynb`
(https://github.com/cillustrisimo/ydl_2026/blob/main/day2/lab_embeddings_toolbox.ipynb)

This project must **reuse the notebook's own functions and style**, not invent new ones.
The notebook is "a reference you raid" — we copy its helpers and assemble them into a
search engine. Match its conventions exactly:

- **camelCase** function names (like the notebook: `cosineSim`, `documentVector`, …).
- **Cosine by hand with NumPy** (`a @ b / (norm·norm)`) — the notebook writes it itself;
  do NOT use `sklearn.cosine_similarity`.
- `np.random.seed(0)` and `os.makedirs("outputs", exist_ok=True)` at the top.
- Network calls wrapped in `try/except` so it degrades gracefully offline.
- Tokenize with `text.lower().split()` and **skip words not in the vocab** (OOV).

### Functions to COPY from the notebook (verbatim) and where they live
| Function | Notebook section | Role in our search engine |
|---|---|---|
| `api.load("glove-wiki-gigaword-100")` | §1 Pretrained embeddings | load GloVe (`wv`) |
| `cosineSim(a, b)` | §2 Toolkit | similarity metric |
| `nearestNeighbors(vec, vectors, words, topn=5)` | §2 Toolkit | **the ranker** — feed it song vectors instead of word vectors |
| `documentVector(text, wv)` | §5a Averaged-embedding | turn each song's lyrics → one vector |
| `plotWords(words, vectors, title)` (PCA scatter) | §3 Visualizing | base for the 2D map |
| Song2Vec colored-PCA scatter pattern | §4a X2Vec | template for the **mood-colored** map |

> Key insight: the notebook's `nearestNeighbors` already IS 90% of a search engine —
> it takes a query vector + a list of vectors + their labels and returns the top-N
> closest. We just pass **document vectors** (per song) and labels = song names.

## Requirements checklist (from the brief)
- [ ] Use at least one technique from today on text **I gathered myself** (not the demo data).
- [ ] End to end: load text → build vectors → inspect the result.
- [ ] Produce **one artifact**: a labelled plot (or interactive HTML).

## Tech stack (same libraries as the notebook)
- Python + Jupyter notebook
- `gensim` — load pretrained `glove-wiki-gigaword-100` (~128 MB, downloads once, cached).
  (Notebook uses 100-dim. A lighter `glove-wiki-gigaword-50` ~65 MB also works.)
- `numpy` — vector math + hand-written cosine
- `scikit-learn` — **only** for `PCA` / `TSNE` (and tf-idf in the stretch goal), NOT cosine
- `matplotlib` — the artifact plot
- `pandas` — load the song CSV
- `requests` / `beautifulsoup4` — only if fetching more text later (notebook §6)

## Data source — DONE ✅
**Dataset:** `Annanay/aml_song_lyrics_balanced` (Hugging Face, no Kaggle account).
A ready-made 2000-song sample is already saved at:

    song_lyrics_sample_2000.csv   (~3.5 MB)

Regenerate it any time with: `python download_song_sample.py`

**Columns:**
- `lyrics` — the song text (this is the corpus I search over)
- `mood` — happy / sad / calm / anger  ← use this to COLOR the 2D map
- `mood_cats` — numeric code for the mood
- `lyrics_filename` — "Artist___Song" (split on `___` to get a clean label)

**Sample mood balance:** happy 790, sad 783, calm 264, anger 163.

> Tip from the brief: start small, get it working, then scale up.
> (Already small — 2000 rows. Bump SAMPLE_SIZE in the script to scale.)

## Build steps (each maps to a notebook helper)
1. **Get text** — DONE: `pd.read_csv("song_lyrics_sample_2000.csv")`; corpus = `lyrics`
   column; labels = `lyrics_filename` (split on `___`); colors = `mood`.
2. **Load embeddings** — copy notebook §1: `import gensim.downloader as api;
   wv = api.load("glove-wiki-gigaword-100")` inside a `try/except`.
3. **Vectorize every song** — reuse **`documentVector(lyrics, wv)`** (§5a) over the
   `lyrics` column → `songVectors` (a list/array, one vector per song).
4. **Vectorize the query** — reuse the SAME `documentVector(query, wv)`.
5. **Search** — wrap the notebook's **`nearestNeighbors`** (§2):
   `def search(q, topn=5): return nearestNeighbors(documentVector(q, wv), songVectors, labels, topn)`
   → returns the top songs + cosine scores.
6. **Inspect** — run 2–3 queries (e.g. "heartbreak and loneliness", "party and dancing")
   and check the results make sense; expect sad-mood songs to surface for sad queries.

## Artifact (the deliverable plot) — reuse the notebook's plot code
Primary = **mood-colored 2D map** (copy the Song2Vec scatter from §4a, swap genre→mood):
- PCA the `songVectors` to 2D (notebook `plotWords` / §3 style).
- Scatter every song, **colored by `mood`** with a palette dict like the notebook's.
- Optionally mark where a query vector lands among the songs.
- Save to `outputs/` (notebook convention) with a clear title + legend.

Fallback that also satisfies "a labelled plot": bar chart of the top-5 cosine scores
for one example query.

## Write-up (3–4 sentences)
"I searched for X — the engine returned songs about Y, even though they didn't contain
the word X, which shows it matched on meaning not words. On the map, the results sit in
the [mood] cluster." Note one place it worked and one where it struggled.

## Stretch goals (only if time allows)
- Compare averaged-GloVe ranking vs tf-idf ranking (notebook §5 has the tf-idf code).
- Weight word vectors by tf-idf instead of plain averaging.
- Interactive HTML page for the search box / scatter plot.

## Notes / gotchas
- Averaging word vectors loses word order — fine for this lab; know the limitation
  (the notebook says the same in §5a).
- GloVe vocab is lowercase → lowercase + `.split()` before lookup; skip OOV tokens.
- First `api.load(...)` downloads ~128 MB once, then caches — be patient on run 1.
- `gensim`, `scikit-learn`, `matplotlib` are NOT installed yet in this env — install
  before running (`pip install gensim scikit-learn matplotlib`).
