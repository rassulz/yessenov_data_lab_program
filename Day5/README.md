# Yessenov Foundation — Grants AI Assistant

A Streamlit chatbot (on **gemma4**) that answers questions about the grants and programs
of the **Shakhmardan Yessenov Foundation** (yessenovfoundation.org). Final project of
Week 1 of Yessenov Data Lab 2026 — it puts together the whole week: **UI + LLM + data work**.

> **The one rule that drives the whole design: the bot must not invent grant facts.**
> A made-up deadline, sum, or rule is worse than "I don't know", because a person can make
> a real decision from it. So the bot answers **only** from data we collected, and when the
> data has no answer it says so honestly. The eval below measures exactly this.

---

## What it does

- **Grounded chat (RAG).** Every answer is built only from chunks retrieved from data
  scraped off the foundation site. The model is told to use the context and nothing else.
- **Honest refusal.** When the data has no answer (weather, mortgage, another university…)
  the bot says it does not know and points to the foundation — it does not guess.
- **Proof under every answer.** Source links (which page/PDF) + a **"What I found"** expander
  showing the real chunks and their similarity scores. You can check the bot, not just trust it.
- **Multilingual.** Answers in the language of the question — Russian, Kazakh, or English.
- **Names of winners.** The official winner/participant lists are part of the knowledge base, so
  the bot can answer "who got into Yessenov Data Lab 2026?" with the real names.
- **Grant recommender & comparison.** Pick who you are / field / what you want → it suggests matching
  real programs; or compare two programs side by side as a table (all grounded in the data).
- **Charts on request.** Ask "build a chart of winners by year" and the bot draws it **in the chat**
  — winners per year (`data/stats.csv`) or places per program (`data/program_stats.csv`), built from
  **real** numbers only (each verified against the source), never invented.
- **Smart email (optional).** Notices a "leave a request" intent and offers a button that
  emails a short chat summary to the administrator (only to your own inbox, only on a click).
- **Streaming** answers and **example-question buttons** for a smooth demo.

## Honesty eval

`python eval.py` runs a small test set that mixes in-data questions with out-of-data ones
(weather, football, mortgage…) and buckets each answer:

```
Не выдумал (честно): 12/12
  ✅ ответил верно:      8
  🟡 честно отказался:   4
  ❌ выдумал:            0
```

The headline number is shown as a badge in the app's sidebar. `made_up` (the bad bucket)
is **0** — the bot never invents on this set.

### Adversarial review

Beyond the fixed test set, a multi-agent adversarial pass threw **20+ trap questions** at the live
bot — false premises (a participation fee, an Astana office, a regional quota), demands for exact
numbers not in the data (IELTS score, scholarship in USD), plausible-but-absent programs (PhD at
Harvard, mortgage grants), and leading multilingual questions in Kazakh/English. The bot refused or
corrected all but one. The single miss was a **cross-program contact misattribution** (it offered a
real contact that belonged to a *different* program). Root cause: retrieval mixed chunks from several
programs and only the single best score gated the context. Fixed by **per-chunk relevance filtering**
plus an explicit "do not carry facts/contacts between programs" rule — re-tested, the bot now refuses.
Security hardening from the same review (TLS verification on all scrapes, HTML-escaped source links,
a PDF host allowlist, and a code-pinned email recipient) is in place too.

---

## How the data was collected

The site is WordPress + qTranslate, static HTML, content in Russian (`/ru/`). Approach:

1. **`scraper.py`** — a curated set of evergreen pages: about / mission / founder / boards,
   then the whole program tree (science / knowledge / resources). For each category it
   auto-discovers the leaf programs and, for cohort programs, the **latest edition** page —
   that is where the real rules live (dates, sums, who can apply). The main `<article>` is
   cleaned (nav/footer/scripts dropped) and saved as `data/docs/<slug>.json`.
2. **Rule PDFs.** Pages link to PDFs via a `kcccount=` download URL. Text-layer PDFs
   (program rules / curricula) are extracted and kept; **scanned** PDFs (e.g. `polozhenie.pdf`)
   have no text layer and are skipped automatically.
3. **Winner/participant lists** (the published name lists, pdf/xlsx) are saved as `type: list`
   docs too, so the bot can name who got into a cohort — not just the count.
4. **`build_stats.py`** — downloads the official Yessenov Data Lab winners lists (xlsx/pdf)
   and counts winners per year for the chart. Counts were also verified by hand
   (2018:21, 2019:25, 2020:20, 2023:15, 2024:20, 2025:15, 2026:20). 2021–2022 are omitted
   on purpose: there was no cohort those years (so no misleading `0`).
5. **`build_program_stats.py`** — places/grants in the latest cohort per program, taken verbatim
   from the program pages (the script asserts each number actually appears in the page text).
6. The scraper also synthesizes two helper docs from the real data: **foundation contacts**
   (phone/email/address from the site footer) and a **programs catalog** (every program grouped
   by track, ended ones marked) — so the bot can give a phone number and list all programs.
7. **`build_index.py`** — splits docs into ~paragraph chunks, embeds them with **text-1024**
   (1024-dim), and saves a small vector index under `data/index/`.

Result: **50 documents (~152k chars) → 251 chunks** in the index.

## Architecture

```
question ──> embed (text-1024) ──> cosine top-k over data/index ──> chunks
                                                                      │
   system prompt (answer ONLY from context, else refuse) + chunks ───┤
                                                                      ▼
                                              gemma4 (streamed) ──> answer + sources
```

`rag.py` holds the whole grounded core (retrieval + the anti-invention system prompt +
generation). If the best similarity is below `RELEVANCE_FLOOR` (0.40, calibrated against the
real index), the prompt nudges the model to refuse. In testing, real questions score 0.55–0.82
and out-of-data ones ≤ 0.34, so the floor separates them cleanly.

---

## Run it

```bash
# 1. environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# 2. secrets — copy the template and paste your keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   GEMMA4_API_KEY, EMBED_API_KEY  (alem.ai, OpenAI-compatible)
#   MAILERSEND_API_KEY (optional), ADMIN_EMAIL = your own inbox

# 3. build the data + index (one time; re-run to refresh)
python scraper.py              # site + PDFs + name lists -> data/docs/*.json
python build_stats.py          # YDL winners per year      -> data/stats.csv
python build_program_stats.py  # places per program        -> data/program_stats.csv
python build_index.py          # chunks + embeddings       -> data/index/

# 4. (optional) the honesty number
python eval.py             # -> data/eval_results.json (sidebar badge)

# 5. the app
streamlit run app.py
```

`data/docs/` and `data/stats.csv` are committed so the app works without re-scraping;
`data/index/` is rebuildable (`python build_index.py`) and git-ignored.

## Tech stack

| Part | Choice |
|------|--------|
| UI | Streamlit chat |
| LLM | `gemma4` (OpenAI-compatible, `https://llm.alem.ai/v1`) |
| Embeddings | `text-1024`, 1024-dim |
| Retrieval | cosine top-k over a NumPy matrix (no heavy vector DB needed) |
| Scraping | requests + BeautifulSoup; pypdf / openpyxl for PDF & xlsx |
| Email | MailerSend (`MailerSendClient` + `EmailBuilder`) |

## Project structure

```
Day5/
├── app.py              # Streamlit chat (the main deliverable)
├── rag.py              # grounded RAG core: retrieval + system prompt + generation
├── scraper.py          # collect site pages + rule PDFs  -> data/docs/
├── build_index.py      # chunk + embed                   -> data/index/
├── build_stats.py      # YDL winners per year            -> data/stats.csv
├── build_program_stats.py  # places per program          -> data/program_stats.csv
├── emailer.py          # MailerSend summary (button-only, own inbox only)
├── eval.py             # honesty test set + score        -> data/eval_results.json
├── eval_questions.json # the test set (in-data + out-of-data)
├── requirements.txt
├── data/
│   ├── docs/*.json     # scraped knowledge base incl. name lists (committed)
│   ├── stats.csv       # YDL winners per year (committed)
│   ├── program_stats.csv   # places per program (committed)
│   └── index/          # embeddings (git-ignored, rebuildable)
├── assets/             # logo + founder photos (brand UI)
└── .streamlit/
    ├── config.toml         # brand purple theme
    ├── secrets.toml        # real keys (git-ignored, NEVER committed)
    └── secrets.toml.example
```

## Notes & honesty about limits

- **Keys never live in code.** They are read from `.streamlit/secrets.toml` (git-ignored) or
  env vars. `secrets.toml.example` keeps the structure in git without secrets.
- **Email.** The feature is built and safe (one recipient = your own inbox, button-only, never
  in a loop). The shared course MailerSend key currently returns a `403 / code 1010`
  (account-level restriction on the shared key), so a live send is blocked right now; the app
  handles this gracefully — it shows the summary it *would* have sent instead of crashing.
- This is a **study project / portfolio piece**, not an official foundation product. All facts
  come from the public foundation website; the bot quotes them and refuses to go beyond them.
