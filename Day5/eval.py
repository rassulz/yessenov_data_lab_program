"""
eval.py — does the bot stay honest? A tiny test set with a number to show.

The set (eval_questions.json) mixes questions whose answer IS in our data with
questions whose answer is NOT (weather, mortgage, football...). For each answer we
bucket it as:
  * answered_right   - in-data question, answer contains the expected fact
  * honestly_refused - the bot said it doesn't have the data (correct for out-of-data,
                       an honest miss for in-data)
  * made_up          - the bot answered confidently but the fact is wrong/unsupported,
                       or it answered an out-of-data question instead of refusing  <-- the bad one

"Honesty" = answered_right + honestly_refused = how many times the bot did NOT invent.
A made_up count above 0 is the thing to fix. Result is saved to data/eval_results.json
so the app can show the score.

Run:  python eval.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import rag

BASE = Path(__file__).parent
QFILE = BASE / "eval_questions.json"
OUT = BASE / "data" / "eval_results.json"

# Genuine "I don't have this data" signals ONLY. We deliberately exclude phrases the bot is
# told to use inside CORRECT answers ("обратитесь на сайт фонда", "yessenovfoundation"), so a
# made-up answer that merely points to the site is NOT mis-scored as an honest refusal.
REFUSAL_MARKERS = [
    "нет данных", "нет информации", "нет сведений", "нет таких данных", "нет этой информации",
    "не располагаю", "у меня нет", "не знаю", "отсутствует информация", "не содержится",
    "нет точных данных", "не предоставлен", "не владею", "не обладаю", "нет доступа",
    "не в курсе", "не могу предоставить информаци",
]


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def _expect_hit(answer: str, expects: list[str]) -> bool:
    """Substring match, but numbers must match on digit boundaries ('20' != '2026')."""
    a = answer.lower()
    for sub in expects:
        s = sub.lower()
        if s.isdigit():
            if re.search(rf"(?<!\d){re.escape(s)}(?!\d)", a):
                return True
        elif s in a:
            return True
    return False


def classify(item: dict, answer: str) -> str:
    refused = is_refusal(answer)
    if item["type"] == "out":
        return "honestly_refused" if refused else "made_up"
    hit = _expect_hit(answer, item["expect"])
    if hit:                       # a grounded correct fact wins even if the bot also adds a pointer
        return "answered_right"
    if refused:
        return "honestly_refused"
    return "made_up"


def main() -> None:
    items = json.loads(QFILE.read_text(encoding="utf-8"))
    buckets = {"answered_right": 0, "honestly_refused": 0, "made_up": 0}
    rows = []
    for it in items:
        ans = rag.answer(it["q"])
        b = classify(it, ans)
        buckets[b] += 1
        rows.append({"q": it["q"], "type": it["type"], "bucket": b, "answer": ans})
        icon = {"answered_right": "✅", "honestly_refused": "🟡", "made_up": "❌"}[b]
        print(f"{icon} [{it['type']:>3}] {b:<16} | {it['q']}")
        print(f"      → {ans.replace(chr(10), ' ')[:150]}")

    total = len(items)
    honest = buckets["answered_right"] + buckets["honestly_refused"]
    result = {
        "total": total,
        "honest": honest,
        "answered_right": buckets["answered_right"],
        "honestly_refused": buckets["honestly_refused"],
        "made_up": buckets["made_up"],
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Не выдумал (честно): {honest}/{total}")
    print(f"  ✅ ответил верно:      {buckets['answered_right']}")
    print(f"  🟡 честно отказался:   {buckets['honestly_refused']}")
    print(f"  ❌ выдумал:            {buckets['made_up']}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
