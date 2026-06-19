# Day 5 — Final Project: Yessenov Foundation Grants AI Assistant

Project brief: [YDL2026_day5_project.pdf](YDL2026_day5_project.pdf) (in Russian). This file is a short summary in English.

## What this is

A **Streamlit chatbot that runs on `gemma4`**. It answers questions about the grants and the work
of the **Shakhmardan Yessenov Foundation** (yessenovfoundation.org). This is the final project of
Week 1. It puts together everything from the week: **UI + LLM + data work**.

This is a **study project and a portfolio piece**, not a real order from the foundation. The bot is
only the reason; the real goal is the skill. The project does not have to go "to production" — that
is normal.

## The most important rule: the bot must not invent grant facts

A bot that **makes up** grant details — dates, rules, money amounts, who can apply — is **not done**.
A wrong answer is worse than "I don't know", because a person can make a real decision from it.

So the bot must answer **only from the data we collected**. If the data does not have the answer, the
bot must say it does not know. Plan this from the start (system prompt, grounding, refusal), not at
the end. Test it: ask the bot something that is **not** in the data and check that it says "I don't
know" instead of making something up.

## Minimum (must have)

1. **Collect the data yourself** from yessenovfoundation.org — grant rules, programs, FAQ. How you
   scrape or parse it is your choice and part of the project (Claude can help).
2. **Build a Streamlit chat on `gemma4`** that answers questions from this data.
3. **Test on 3–4 real questions**, and on at least one question that is **not** in the data, to check
   the "I don't know" behavior above.

## Optional (nice to have)

- **RAG with the given embedding model** — do not put all the text in the prompt. Instead, find the
  right piece and send only that. Useful when there is a lot of data.
- **Persona and tone** — official consultant, friendly helper, your choice.
- **More than one topic** — grants, scholarships, this school.
- **Charts** — the bot can draw a chart (Streamlit), but **only from real numbers** in a small,
  hand-made `data/stats.csv` (for example, winners per year taken from the winners-list PDFs). Never
  chart invented numbers: a fake chart looks official and is the worst kind of made-up answer. If the
  numbers are not there, the bot says it has no data.
- **Email summary** — see below.

## Email feature (optional) — strict rules

Idea: when a chat is useful or the user leaves a request, the bot can send a short summary to the
"administrator" by email. This is the first step from a chat to an agent that **does an action**.

Rules you must follow:
- **Send only to your own email.** In this course, "administrator" means you.
- **Only on a clear action** — a button click or a clear model decision. **Never in a loop** (one bug
  can send a hundred emails, and the sending domain gets marked as spam by Gmail and others).

Email uses **MailerSend** (`from mailersend import MailerSendClient, EmailBuilder`), sender
`info@app.commit.kz`. The API key and a code example are in the PDF. **Do not put the key in the
code** — keep it in `.streamlit/secrets.toml` or an env var and read it from there.

## Decisions made

- **Data approach:** RAG with embeddings (find the right chunk, send only that).
- **Data collection:** hybrid — targeted scrape of key pages + text-layer PDFs, then a quick hand check.
- **UI style:** look like the foundation site — friendly tone, brand purple. See "UI style" below.
- **Email:** send by a button ("Send summary to me"). No model auto-send, no loops. A single test
  email to the student's own inbox is OK to check MailerSend.
- **Charts:** yes. `data/stats.csv` = YDL winners per year (2018–2026, from the winners lists);
  `data/program_stats.csv` = places/grants in the latest cohort per program (stated on the pages).
  The bot **draws a chart in the chat** when asked ("построй график…"), using only these real numbers.
- **Winner/participant lists:** also saved as `type: list` docs (not just counts), so the bot can
  answer "кто прошёл на YDL 2026?" with the real names from the official lists.
- **Environment:** local `.venv` + `requirements.txt`.
- **Project PDF** (`YDL2026_day5_project.pdf`) is git-ignored because it holds the shared MailerSend key.
- **Standout features:** build the full set in "Standout features" below (source links, honest
  refusal, grant recommender, Kazakh/multilingual, eval number, streaming, example buttons).

## Data plan (how we collect)

### Site facts (checked live)

- The site is **WordPress** with a custom Bootstrap theme and the **qTranslate** plugin for languages
  (`/ru/` and `/en/`).
- It is **static HTML**, so `requests` + `BeautifulSoup` is enough. No Playwright needed.
- Each page has **one `<article class="post">`** that holds the main text. Take that tag, drop nav,
  footer, and scripts.
- `sitemap.xml` lists ~586 URLs, but most are **news** (skip them). The useful pages are few.

### Pages to collect (Russian, use the `/ru/` prefix)

- `/ru/about-us/` and its subpages: `mission-and-reports`, `galimzhan-yessenov`,
  `the-board-of-trustees`, `the-expert-board`.
- `/ru/about-us/programs/` and the **whole programs tree**: `science/*` (graduate studies, orleu,
  research internships, research grants), `knowledge/*` (books, chess, english, lectures),
  `resources/*` (yessenov-data-lab, launch-pad, internships, it-skills).
- `/ru/sh-esenov/biografiya/` (founder).
- **Skip:** `/category/`, news slugs, feeds, media galleries.

### PDF files

Some pages link to PDFs. The real file URL is the part **after `kcccount=`** in the download link
(a `wp-content/uploads/.../*.pdf` path). There are two kinds:

- **PDF with a text layer** (for example the program and the winners/participants lists) — read it
  with `pdftotext` or `pypdf`. **Include these.**
- **Scanned image PDF** (for example the rules `polozhenie.pdf` and financial reports) — no text
  inside. To read it you need **OCR** (Tesseract). OCR is extra setup and can get numbers wrong, so
  use it **only if needed** and check the numbers by hand. For rules, prefer the text on the HTML
  program page. Skip heavy report scans.

### Steps (`scraper.py`)

1. Read `sitemap.xml`, keep only the evergreen URLs above, use the `/ru/` version.
2. For each page: fetch, take `<article>`, clean it, save `title + text + url` to
   `data/docs/<slug>.json`.
3. For text-layer PDFs on those pages: download, extract text, save the same way.
4. Look at the result by hand and drop any junk.

## UI style

The bot should look like the foundation site: a **friendly tone** and the foundation **colors**.

- **Brand color:** purple `#79378b` (from the logo). White background, dark text `#32373c`, light
  purple `#f4f1f7` for the second background.
- **Tone:** friendly helper — warm and simple, but still honest and clear (the "do not invent" rule
  still wins over being nice).
- **Theme** in `.streamlit/config.toml`:
  ```toml
  [theme]
  primaryColor = "#79378b"
  backgroundColor = "#ffffff"
  secondaryBackgroundColor = "#f4f1f7"
  textColor = "#32373c"
  font = "sans serif"
  ```
- **Assets** in `assets/` (already downloaded from the site):
  - `logo-yessenov-foundation.jpg` — foundation logo (header, page icon, bot avatar).
  - `shakhmardan-yessenov.jpg` — portrait of Shakhmardan Yessenov (about box / sidebar).
  - `shakhmardan-yessenov-banner.jpg` — wide "95 years" banner (welcome screen / header).
- Use the logo as `page_icon` in `st.set_page_config` and as the bot avatar in `st.chat_message`.

## Standout features (build these)

Goal: stand out at the demo, where bots are judged on **honesty and accuracy** ("which bot answered
right, which made it up"). So the strongest features visibly **prove the bot does not invent**.

**Trust — most important (this is what is judged):**
- **Source links** under each answer — show which page/PDF the answer came from.
- **"What I found" expander** — show the real chunks the answer is built on. Transparency = trust.
- **Friendly honest refusal** — when the data has no answer, say it kindly and point to the
  foundation (a contact or link), not a dry "I don't know".

**Useful / agentic:**
- **Grant recommender** — the bot asks 2–3 short questions (student or scientist, field, level) and
  suggests matching programs.
- **Program comparison** — answer "how is X different from Y?" as a short table.
- **Smart email** — the bot notices a "leave a request" intent and offers the send-summary button
  (still button-confirmed, never auto-send).

**Differentiator (few will do it):**
- **Multilingual** — answer in the language of the question (Russian / Kazakh / English). The site
  has RU and EN data.

**Polish (cheap, adds "wow"):**
- **Streaming answers** — print word by word.
- **Example-question buttons** on the start screen — smooth demo, shows what the bot can do.

**Quality proof (eval):**
- A small **test set** of questions (including some that are *not* in the data). A script reports:
  answered right / honestly refused / made up. This gives a **number** to show at the demo
  ("did not lie on N of 10") — strong for the portfolio too.

## Tech stack

- **UI:** Streamlit (chat)
- **LLM:** `gemma4` (given by the program; key in secrets)
- **Embeddings:** `text-1024`, 1024-dim (given by the program; key in secrets)
- **API:** OpenAI-compatible, base URL `https://llm.alem.ai/v1` (alem.ai). Use the `openai`
  client with `base_url` + key. Endpoints: `/chat/completions` and `/embeddings`.
- **Email:** `mailersend` SDK (optional)
- **Data:** scraped from yessenovfoundation.org. The content is in **Russian**, so the bot should
  answer users in Russian or Kazakh.

## Secrets and keys (IMPORTANT)

- All keys live in `.streamlit/secrets.toml`. This file is in `.gitignore` and **must NEVER be
  committed to GitHub**. Read keys in code with `st.secrets["..."]`.
- `secrets.toml.example` is a safe template (no real keys) and **is** committed, so the structure
  stays in git.
- Keys needed: `GEMMA4_API_KEY` (LLM), `EMBED_API_KEY` (text-1024 embeddings),
  `MAILERSEND_API_KEY` (the shared key is in the project PDF).
- **Email rule:** `ADMIN_EMAIL` must always be the student's own inbox
  `rassul.zeinulla@outlook.com`. Never send to any other address.

## Conventions

- Keep API keys and secrets out of the code. Use `.streamlit/secrets.toml` or env vars. Never commit them.
- Keep the scraped data in this `Day5/` folder (for example a `data/` subfolder) so the build is easy to repeat.
- So far this folder has `.streamlit/` (secrets) and `assets/` (logo + founder photos). New code
  (for example `app.py`), the scraper, and the data files go here too.

## Demo rules (context, not code)

Demos start at 15:00 sharp — **4 minutes to present + 1 minute for questions**, stopped hard. In the
demo each person must, live: ask one common question that the teacher gives at demo time, and show
one question where the bot fails. Also tell shortly: how you got the data, what approach you chose,
and what the bot can do above the minimum.
