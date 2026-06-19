"""
build_program_stats.py -> data/program_stats.csv

Number of places/grants in the LATEST cohort of each active program, taken verbatim from the
program pages (these counts are stated in the scraped docs). The script asserts each number
actually appears in the doc text, so the chart can never show an invented figure.

Run AFTER scraper.py:  python build_program_stats.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path

BASE = Path(__file__).parent
DOCS = BASE / "data" / "docs"
OUT = BASE / "data" / "program_stats.csv"

# (doc slug, short program label, places, what they are, cohort year) — all hand-verified.
ENTRIES = [
    ("yessenov-data-lab-2026",                                  "Yessenov Data Lab",   20, "участников", 2026),
    ("az-stipendiya-im-akademika-sh-esenova-2026",              "Стипендия Есенова",   20, "стипендий",  2026),
    ("eng-orleu-program-2024",                                  "Орлеу",               10, "грантов",    2024),
    ("eng-yessenov-launch-pad-2025",                            "Yessenov Launch Pad", 10, "грантов",    2025),
    ("programma-nauchnyh-stazhirovok-v-laboratoriyah-mira-2026","Научные стажировки",  10, "грантов",    2026),
]


def main() -> None:
    rows = []
    for slug, label, places, kind, year in ENTRIES:
        d = json.loads((DOCS / f"{slug}.json").read_text(encoding="utf-8"))
        if str(places) not in d["text"]:
            raise SystemExit(f"'{label}': {places} not found in {slug}.json — refusing to chart an unverified number.")
        print(f"  {label:<22} {places:>3} {kind:<11} ({year})  <- verified in {slug}.json")
        rows.append({"program": label, "places": places, "kind": kind, "year": year, "source": d["url"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["program", "places", "kind", "year", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT} ({len(rows)} programs).")


if __name__ == "__main__":
    main()
