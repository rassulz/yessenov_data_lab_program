"""
scraper.py — collect the knowledge base for the Yessenov Foundation grants bot.

Source: yessenovfoundation.org (WordPress + qTranslate, static HTML, Russian /ru/).
Strategy (hybrid, see CLAUDE.md):
  1. A curated set of evergreen INFO pages (about, mission, founder, boards).
  2. The program tree: 3 category pages (science / knowledge / resources) ->
     auto-discover leaf programs -> for cohort programs, the latest edition page
     (that is where the real rules live: dates, sums, who can apply).
  3. Text-layer rule PDFs linked from those pages (polozhenie / programma).
     Scanned PDFs have no text layer and are skipped automatically.

Every page becomes data/docs/<slug>.json = {title, text, url, section, type}.
We never invent text — only what the site actually says.

Run:  python scraper.py
"""
from __future__ import annotations
import json, re, ssl, sys, time, urllib.request, urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
import io

BASE = "https://yessenovfoundation.org"
ALLOWED_HOST = "yessenovfoundation.org"          # the knowledge base may come only from here
DOCS = Path(__file__).parent / "data" / "docs"
UA = {"User-Agent": "Mozilla/5.0 (YDL2026 study project; contact rassul.zeinulla@outlook.com)"}
_CTX = ssl.create_default_context()              # full TLS verification — this builds the KB

# ---- curated seeds -----------------------------------------------------------
INFO_PAGES = [
    ("about-us",            "/about-us/",                     "О фонде"),
    ("mission-and-reports", "/about-us/mission-and-reports/", "О фонде"),
    ("galimzhan-yessenov",  "/about-us/galimzhan-yessenov/",  "О фонде"),
    ("board-of-trustees",   "/about-us/the-board-of-trustees/","О фонде"),
    ("expert-board",        "/about-us/the-expert-board/",    "О фонде"),
    ("founder-biography",   "/sh-esenov/biografiya/",         "Основатель"),
]
CATEGORY_PAGES = [
    ("programs",  "/about-us/programs/",           "Программы"),
    ("science",   "/about-us/programs/science/",   "Наука"),
    ("knowledge", "/about-us/programs/knowledge/", "Знание"),
    ("resources", "/about-us/programs/resources/", "Ресурсы"),
]
SECTION_BY_CAT = {"science": "Наука", "knowledge": "Знание", "resources": "Ресурсы"}

# Rule / curriculum PDFs (usually a text layer) — saved as type "pdf".
PDF_KEEP = re.compile(r"(polozhenie|polozenie|programma|pravila|reglament|usloviya|condition)", re.I)
# Winners / participants name lists — saved as type "list" so the bot can answer "who got in".
LIST_KEEP = re.compile(r"(spisok|pobed|uchastnik|winner|participant)", re.I)
# Heavy scanned reports / financials — skip entirely (no Q&A value, often no text layer).
PDF_SKIP = re.compile(r"(otchet|report|finans|finance|buhgalter|balans)", re.I)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40, context=_CTX) as r:
        return r.read()


def fetch_html(path_or_url: str) -> BeautifulSoup:
    url = path_or_url if path_or_url.startswith("http") else BASE + "/ru" + path_or_url
    html = fetch(url).decode("utf-8", "ignore")
    return BeautifulSoup(html, "html.parser")


def ru_url(path_or_url: str) -> str:
    """Canonical /ru/ URL for display / fetching."""
    if path_or_url.startswith("http"):
        p = urllib.parse.urlparse(path_or_url).path
    else:
        p = path_or_url
    if not p.startswith("/ru/"):
        p = "/ru" + p
    return BASE + p


def slug_of(path_or_url: str) -> str:
    p = urllib.parse.urlparse(path_or_url).path if path_or_url.startswith("http") else path_or_url
    parts = [x for x in p.split("/") if x and x != "ru"]
    raw = urllib.parse.unquote(parts[-1]) if parts else "index"
    s = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")   # clean ASCII filename
    return s or "index"


def clean_text(article) -> str:
    """Block-by-block clean text from an <article>, de-duplicated."""
    for sel in article.select("script, style, .share, .post__views, .breadcrumbs, nav, .social"):
        sel.decompose()
    lines, seen = [], set()
    for el in article.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "blockquote"]):
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not t or t.lower().startswith("просмотр"):
            continue
        if len(t) < 2:
            continue
        key = t.lower()
        if key in seen:          # drop exact repeats (title cards repeat the heading)
            continue
        seen.add(key)
        lines.append(t)
    return "\n".join(lines)


def page_to_doc(url: str, section: str, doctype: str = "page") -> dict | None:
    soup = fetch_html(url)
    art = soup.find("article")
    if not art:
        return None
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else slug_of(url)
    text = clean_text(art)
    if len(text) < 40:           # near-empty (rare) -> not worth a doc
        return None
    return {"title": title, "text": text, "url": ru_url(url), "section": section, "type": doctype}


def discover_leaves(cat_slug: str) -> list[str]:
    """Leaf program URLs under a category, e.g. /about-us/programs/science/orleu/."""
    soup = fetch_html(f"/about-us/programs/{cat_slug}/")
    pat = re.compile(rf"^/about-us/programs/{cat_slug}/([^/]+)/$")
    out = []
    for a in soup.find_all("a", href=True):
        path = urllib.parse.urlparse(a["href"]).path
        m = pat.match(path)
        if m:
            out.append(path)
    return sorted(set(out))


def latest_edition(leaf_path: str) -> str | None:
    """For a cohort program, the most recent edition page (slug carries a 20xx year)."""
    soup = fetch_html(leaf_path)
    art = soup.find("article") or soup
    leaf = leaf_path.rstrip("/")
    best, best_year = None, -1
    for a in art.find_all("a", href=True):
        path = urllib.parse.urlparse(a["href"]).path.rstrip("/")
        if not path.startswith(leaf + "/"):
            continue
        if path.count("/") != leaf.count("/") + 1:      # exactly one level deeper
            continue
        years = re.findall(r"20\d\d", path)
        if not years:
            continue
        y = max(int(x) for x in years)
        if y > best_year:
            best_year, best = y, path + "/"
    return best


def pdf_links(url: str) -> list[tuple[str, str]]:
    """(label, real_url) for download links on a page (real URL is after kcccount=).
    Catches both PDF (rules) and XLSX (winners/participants list) downloads."""
    soup = fetch_html(url)
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "kcccount=" in href:
            real = urllib.parse.unquote(href.split("kcccount=", 1)[1])
        elif href.lower().endswith((".pdf", ".xlsx")):
            real = href
        else:
            continue
        if not real.startswith("http"):
            real = urllib.parse.urljoin(BASE, real)
        if not urllib.parse.urlparse(real).netloc.endswith(ALLOWED_HOST):
            continue                                  # never pull a file from an off-site URL
        out.append((a.get_text(strip=True), real))
    return out


def pdf_to_text(pdf_url: str) -> str:
    raw = fetch(pdf_url)
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def list_to_text(url: str) -> str:
    """Extract a winners/participants name list as readable text (PDF or XLSX)."""
    raw = fetch(url)
    if url.lower().endswith(".xlsx"):
        import openpyxl
        ws = openpyxl.load_workbook(io.BytesIO(raw), read_only=True).active
        lines = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append(" | ".join(cells))
        return "\n".join(lines).strip()
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def save(doc: dict, slug: str) -> None:
    path = DOCS / f"{slug}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    saved, pdf_saved, list_saved, skipped = [], [], [], []

    def handle(url, section, doctype="page", slug=None):
        try:
            doc = page_to_doc(url, section, doctype)
        except Exception as e:
            print(f"  ! fetch failed {url}: {type(e).__name__}", file=sys.stderr)
            return None
        if not doc:
            skipped.append(url); return None
        s = slug or slug_of(url)
        save(doc, s)
        saved.append((s, len(doc["text"]), doc["title"]))
        print(f"  + {s:<42} {len(doc['text']):>5} chars  | {doc['title'][:40]}")
        return doc

    print("== INFO pages ==")
    for slug, path, section in INFO_PAGES:
        handle(path, section, "info", slug)
        time.sleep(0.2)

    print("== Program index pages ==")
    for slug, path, section in CATEGORY_PAGES:
        handle(path, section, "index", slug)
        time.sleep(0.2)

    print("== Leaf programs + latest editions + rule PDFs ==")
    seen_pdf = set()
    for cat_slug, _, _ in CATEGORY_PAGES[1:]:                 # science / knowledge / resources
        section = SECTION_BY_CAT[cat_slug]
        for leaf in discover_leaves(cat_slug):
            handle(leaf, section, "program")
            time.sleep(0.2)
            edition = None
            try:
                edition = latest_edition(leaf)
            except Exception as e:
                print(f"  ! edition lookup failed {leaf}: {type(e).__name__}", file=sys.stderr)
            target_for_pdf = edition or leaf
            if edition:
                handle(edition, section, "edition")
                time.sleep(0.2)
            # rule PDFs + winners/participants lists from the most detailed page available
            ed_title = (edition or leaf).rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
            try:
                for label, purl in pdf_links(target_for_pdf):
                    name = purl.rsplit("/", 1)[-1]
                    if purl in seen_pdf or PDF_SKIP.search(name):
                        continue
                    base = f"{slug_of(target_for_pdf)}--{re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')}"

                    if LIST_KEEP.search(name):                 # winners / participants name list
                        seen_pdf.add(purl)
                        try:
                            text = list_to_text(purl)
                        except Exception as e:
                            print(f"  ! list failed {name}: {type(e).__name__}", file=sys.stderr); continue
                        if len(text) < 120:               # near-empty / scanned list -> drop
                            print(f"  ~ list skipped (no text layer): {name}"); continue
                        title = f"{label or 'Список'} — {ed_title}"
                        save({"title": title, "text": text, "url": purl,
                              "section": section, "type": "list"}, base)
                        list_saved.append((base, len(text)))
                        print(f"  L {base:<48} {len(text):>5} chars")
                        continue

                    if not PDF_KEEP.search(name):              # not a rule PDF and not a list
                        continue
                    seen_pdf.add(purl)
                    try:
                        text = pdf_to_text(purl)
                    except Exception as e:
                        print(f"  ! pdf failed {name}: {type(e).__name__}", file=sys.stderr); continue
                    if len(text) < 300:                       # scanned -> no text layer
                        print(f"  ~ pdf skipped (no text layer): {name}")
                        continue
                    save({"title": f"{label} (PDF)", "text": text, "url": purl,
                          "section": section, "type": "pdf"}, base)
                    pdf_saved.append((base, len(text)))
                    print(f"  P {base:<48} {len(text):>5} chars")
            except Exception as e:
                print(f"  ! pdf links failed {target_for_pdf}: {type(e).__name__}", file=sys.stderr)

    # --- Foundation contacts (from the site footer) ---
    try:
        foot = fetch_html("/").find("footer")
        ftext = re.sub(r"\s+", " ", foot.get_text(" ", strip=True)) if foot else ""
        phone = re.search(r"\+7[\s\d()\-]{9,18}\d", ftext)
        email_m = re.search(r"[\w.\-]+@[\w.\-]+\.\w{2,}", ftext)
        lines = ["Контакты научно-образовательного фонда имени академика Шахмардана Есенова:"]
        if phone:
            lines.append(f"Телефон: {phone.group().strip()}")
        if email_m:
            lines.append(f"Email: {email_m.group()}")
        if phone and email_m:
            addr = ftext[phone.end():email_m.start()].strip(" ,")
            if addr:
                lines.append(f"Адрес: {addr}")
        lines.append("Соцсети: Facebook, VK, YouTube, LinkedIn, Telegram.")
        lines.append("Официальный сайт: yessenovfoundation.org")
        if phone or email_m:
            save({"title": "Контакты фонда", "text": "\n".join(lines),
                  "url": "https://yessenovfoundation.org/ru/kontaktyi/", "section": "Контакты",
                  "type": "info"}, "kontakty")
            saved.append(("kontakty", len("\n".join(lines)), "Контакты фонда"))
            print(f"  + kontakty (phone/email/address from footer)")
    except Exception as e:
        print(f"  ! contacts failed: {type(e).__name__}", file=sys.stderr)

    # --- Programs catalog (synthesized from the scraped program pages) ---
    by_section: dict[str, list[str]] = {}
    for p in sorted(DOCS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("type") != "program":
            continue
        # mark "(завершена)" only on explicit signals — never over-label an active program
        ended = bool(re.search(r"не вед[её]тся|не реализуется|программа завершена|с\s*20\d\d\s*(?:по|–|-)\s*20\d\d\s*год", d["text"]))
        label = d["title"] + (" (завершена)" if ended else "")
        by_section.setdefault(d.get("section", "Программы"), []).append(label)
    if by_section:
        cat = ["Программы и гранты научно-образовательного фонда имени академика Шахмардана Есенова.",
               "Полный список программ по направлениям:"]
        for sec in sorted(by_section):
            cat.append(f"\nНаправление «{sec}»:")
            for t in sorted(set(by_section[sec])):
                cat.append(f"— {t}")
        save({"title": "Каталог программ фонда", "text": "\n".join(cat),
              "url": "https://yessenovfoundation.org/ru/about-us/programs/", "section": "Программы",
              "type": "info"}, "katalog-programm")
        saved.append(("katalog-programm", len("\n".join(cat)), "Каталог программ фонда"))
        print(f"  + katalog-programm ({sum(len(v) for v in by_section.values())} programs listed)")

    print(f"\nDONE. pages: {len(saved)}, rule-pdf docs: {len(pdf_saved)}, list docs: {len(list_saved)}, "
          f"skipped(empty): {len(skipped)}")
    total = sum(n for _, n, *_ in saved) + sum(n for _, n in pdf_saved) + sum(n for _, n in list_saved)
    ndocs = len(saved) + len(pdf_saved) + len(list_saved)
    print(f"Total knowledge-base size: {total:,} chars across {ndocs} docs")
    if skipped:
        print("Skipped (near-empty):", ", ".join(slug_of(u) for u in skipped))


if __name__ == "__main__":
    main()
