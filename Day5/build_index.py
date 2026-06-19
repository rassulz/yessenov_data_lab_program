"""
build_index.py — turn data/docs/*.json into a small vector index for RAG.

Steps:
  1. Load every scraped document.
  2. Split each into ~paragraph chunks (so we send the model only the relevant piece).
  3. Embed each chunk with text-1024 (1024-dim), L2-normalized.
  4. Save data/index/{embeddings.npy, chunks.jsonl, meta.json}.

Run AFTER scraper.py:  python build_index.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from rag import embed_texts, INDEX_DIR, _cfg

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHUNK_CHARS = 900          # target chunk size
MIN_TAIL = 250             # don't leave a tiny orphan chunk


def load_docs() -> list[dict]:
    docs = []
    for p in sorted(DOCS_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["slug"] = p.stem
        docs.append(d)
    return docs


def _split_long(block: str) -> list[str]:
    """Hard-split a single block longer than CHUNK_CHARS so no chunk overflows the embedder."""
    if len(block) <= CHUNK_CHARS:
        return [block]
    out, words, cur = [], block.split(" "), ""
    for w in words:
        if len(cur) + len(w) + 1 > CHUNK_CHARS and cur:
            out.append(cur.strip()); cur = ""
        cur += w + " "
    if cur.strip():
        out.append(cur.strip())
    # fallback for a single monster word with no spaces
    return out or [block[i:i + CHUNK_CHARS] for i in range(0, len(block), CHUNK_CHARS)]


def chunk_text(text: str) -> list[str]:
    """Greedy pack of text blocks (lines) into ~CHUNK_CHARS pieces, 1-block overlap."""
    blocks = []
    for b in text.split("\n"):
        b = b.strip()
        if b:
            blocks.extend(_split_long(b))
    chunks, cur, cur_len = [], [], 0
    for b in blocks:
        if cur_len + len(b) > CHUNK_CHARS and cur:
            chunks.append("\n".join(cur))
            cur = cur[-1:] if len(cur) > 1 else []      # carry last block as overlap
            cur_len = sum(len(x) for x in cur)
        cur.append(b)
        cur_len += len(b)
    if cur:
        # merge a tiny trailing chunk back into the previous one
        tail = "\n".join(cur)
        if chunks and len(tail) < MIN_TAIL:
            chunks[-1] = chunks[-1] + "\n" + tail
        else:
            chunks.append(tail)
    return chunks


def main() -> None:
    docs = load_docs()
    if not docs:
        raise SystemExit("No docs found. Run python scraper.py first.")

    chunks: list[dict] = []
    for d in docs:
        text = d.get("text", "")
        pieces = chunk_text(text) if text.strip() else []
        if not pieces:
            print(f"  ! warning: doc '{d['slug']}' has no usable text — skipped")
            continue
        for j, ctext in enumerate(pieces):
            chunks.append({
                "id": f"{d['slug']}#{j}",
                "title": d.get("title", d["slug"]),
                "url": d.get("url", ""),
                "section": d.get("section", ""),
                "type": d.get("type", "page"),
                "text": ctext,
            })

    # Embed title + text together so the program name helps retrieval.
    embed_input = [f"{c['title']}\n{c['text']}" for c in chunks]
    print(f"Embedding {len(chunks)} chunks from {len(docs)} docs (model {_cfg().get('EMBED_MODEL')}) ...")
    emb = embed_texts(embed_input)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = {
        "model": _cfg().get("EMBED_MODEL", "text-1024"),
        "dim": int(emb.shape[1]),
        "n_chunks": len(chunks),
        "n_docs": len(docs),
        "chunk_chars": CHUNK_CHARS,
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved index: {emb.shape[0]} x {emb.shape[1]} -> {INDEX_DIR}")
    print(f"  docs={len(docs)}  chunks={len(chunks)}  avg chunk len="
          f"{int(np.mean([len(c['text']) for c in chunks]))} chars")


if __name__ == "__main__":
    main()
