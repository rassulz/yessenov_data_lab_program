"""
build_stats.py — build data/stats.csv from the REAL Yessenov Data Lab winners lists.

The chart in the app must use only real, checked numbers (see CLAUDE.md). So instead
of typing numbers by hand, this script downloads the official winners lists published
on yessenovfoundation.org and counts the winners in each. Counts were also verified by
hand (2018:21, 2019:25, 2020:20, 2023:15, 2024:20, 2025:15, 2026:20).

2021 and 2022 are absent on purpose: there was no YDL cohort those years (the program
paused), so we omit them rather than write a misleading 0.

Run:  python build_stats.py
"""
from __future__ import annotations
import csv, io, re, ssl, urllib.request
from pathlib import Path
from pypdf import PdfReader
import openpyxl

OUT = Path(__file__).parent / "data" / "stats.csv"
UA = {"User-Agent": "Mozilla/5.0 (YDL2026 study project)"}
_CTX = ssl.create_default_context()              # full TLS verification — these feed the chart

# Official winners lists, one per cohort (xlsx or pdf, as published).
SOURCES = {
    2018: "https://yessenovfoundation.org/wp-content/uploads/2018/05/Pobediteli.xlsx",
    2019: "https://yessenovfoundation.org/wp-content/uploads/2019/05/YDL-spisok-pobediteli-RU.xlsx",
    2020: "https://yessenovfoundation.org/wp-content/uploads/2020/05/ydl2020-winners.pdf",
    2023: "https://yessenovfoundation.org/wp-content/uploads/2023/05/pobediteli.xlsx",
    2024: "https://yessenovfoundation.org/wp-content/uploads/2024/05/pobediteli.xlsx",
    2025: "https://yessenovfoundation.org/wp-content/uploads/2025/06/spisok-pobeditelej-upd.pdf",
    2026: "https://yessenovfoundation.org/wp-content/uploads/2026/06/spisok-pobeditelej-1.pdf",
}
# Hand-verified counts — the script asserts its automatic count matches these.
EXPECTED = {2018: 21, 2019: 25, 2020: 20, 2023: 15, 2024: 20, 2025: 15, 2026: 20}


def fetch(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60, context=_CTX).read()


def _count_from_numbers(nums: list[int]) -> int:
    """Winners = the highest row number 1..N, tolerant of a few extraction glitches."""
    if not nums:
        return 0
    top = max(nums)
    coverage = len({n for n in nums if 1 <= n <= top}) / top
    if coverage >= 0.8:          # numbering clearly runs 1..top (a couple may be glued)
        return top
    n = 0                        # otherwise fall back to the strict contiguous run
    s = set(nums)
    while n + 1 in s:
        n += 1
    return n


def count_winners(year: int, url: str) -> int:
    raw = fetch(url)
    if url.lower().endswith(".xlsx"):
        ws = openpyxl.load_workbook(io.BytesIO(raw), read_only=True).active
        nums = []
        for row in ws.iter_rows(values_only=True):
            first = (str(row[0]).strip() if row and row[0] is not None else "")
            if first.isdigit():
                nums.append(int(first))
        return _count_from_numbers(nums)
    # pdf
    txt = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages)
    nums = [int(m.group(1)) for m in re.finditer(r"(?m)^\s*(\d{1,3})\s*[А-Яа-яA-Za-z]", txt)]
    return _count_from_numbers(nums)


def main() -> None:
    rows = []
    for year, url in sorted(SOURCES.items()):
        n = count_winners(year, url)
        verified = EXPECTED.get(year)
        if verified is None:
            raise SystemExit(f"{year}: no hand-verified count in EXPECTED — refusing to chart an unverified number.")
        if n != verified:
            raise SystemExit(
                f"{year}: scraped {n} != hand-verified {verified} (source {url.rsplit('/', 1)[-1]}). "
                f"The winners list changed — re-verify by hand before writing stats.csv.")
        print(f"  {year}: {verified:>3} winners  [ok, matches source]  <- {url.rsplit('/', 1)[-1]}")
        rows.append({"year": year, "winners": verified, "source": url})   # write only verified numbers
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "winners", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT} ({len(rows)} years). 2021-2022 omitted (no cohort those years).")


if __name__ == "__main__":
    main()
