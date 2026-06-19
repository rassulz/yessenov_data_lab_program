"""
rag.py — retrieval-augmented core for the Yessenov Foundation grants bot.

Everything that must NOT invent grant facts lives here:
  * load_secrets()  - keys from env or .streamlit/secrets.toml (works in & out of Streamlit)
  * embed_texts()   - text-1024 embeddings (1024-dim), batched
  * retrieve()      - cosine top-k over the prebuilt index (build_index.py)
  * make_messages() - grounding + honest-refusal system prompt + retrieved context
  * stream_chat() / complete_chat() - gemma4 answers (streaming for the UI, full for eval)

The whole design assumes one rule: the bot answers ONLY from retrieved context and
otherwise says it does not know. That rule is enforced by the system prompt below and
by keeping the model's context limited to the chunks we actually found.
"""
from __future__ import annotations
import json, os, re
from functools import lru_cache
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "data" / "index"
SECRETS_FILE = BASE_DIR / ".streamlit" / "secrets.toml"

# Below this best-cosine score we treat the knowledge base as "no good match" and let
# the model refuse honestly. Calibrated against the real index (see README).
RELEVANCE_FLOOR = 0.40
TOP_K = 8


# --------------------------------------------------------------------------- secrets
def load_secrets() -> dict:
    """Keys from environment first, then the flat .streamlit/secrets.toml file."""
    cfg: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            # accept single OR double quotes and an optional trailing # comment
            m = re.match(r'''\s*([A-Z_0-9]+)\s*=\s*(["'])(.*?)\2\s*(#.*)?$''', line)
            if m:
                cfg[m.group(1)] = m.group(3)
    for k in ("GEMMA4_API_KEY", "GEMMA4_MODEL", "GEMMA4_BASE_URL", "EMBED_API_KEY",
              "EMBED_MODEL", "EMBED_BASE_URL", "MAILERSEND_API_KEY", "SENDER_EMAIL", "ADMIN_EMAIL"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return load_secrets()


@lru_cache(maxsize=1)
def _chat_client():
    from openai import OpenAI
    c = _cfg()
    return OpenAI(api_key=c["GEMMA4_API_KEY"], base_url=c.get("GEMMA4_BASE_URL", "https://llm.alem.ai/v1"))


@lru_cache(maxsize=1)
def _embed_client():
    from openai import OpenAI
    c = _cfg()
    return OpenAI(api_key=c["EMBED_API_KEY"], base_url=c.get("EMBED_BASE_URL", "https://llm.alem.ai/v1"))


# --------------------------------------------------------------------------- embeddings
def embed_texts(texts: list[str], batch: int = 64) -> np.ndarray:
    """L2-normalized embeddings, shape (len(texts), 1024)."""
    client = _embed_client()
    model = _cfg().get("EMBED_MODEL", "text-1024")
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


# --------------------------------------------------------------------------- index
@lru_cache(maxsize=1)
def load_index() -> tuple[np.ndarray, list[dict], dict]:
    emb_path = INDEX_DIR / "embeddings.npy"
    chunks_path = INDEX_DIR / "chunks.jsonl"
    if not emb_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Index not found. Run:  python build_index.py  (after python scraper.py)."
        )
    emb = np.load(emb_path)
    chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8")) if (INDEX_DIR / "meta.json").exists() else {}
    return emb, chunks, meta


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Top-k chunks by cosine similarity, each annotated with its score."""
    emb, chunks, _ = load_index()
    q = embed_query(query)
    scores = np.nan_to_num(emb @ q, nan=-1.0)        # degenerate embeddings -> never "relevant"
    order = np.argsort(-scores)[:k]
    hits = []
    for idx in order:
        h = dict(chunks[int(idx)])
        h["score"] = float(scores[int(idx)])
        hits.append(h)
    return hits


def condense_query(query: str, history: list[dict] | None) -> str:
    """Rewrite a follow-up ("а какие требования?") into a standalone search query using
    the recent dialog, so retrieval still finds the right program. No history -> unchanged."""
    recent = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    if not recent:
        return query
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent[-4:])
    msgs = [
        {"role": "system", "content": "Перепиши ПОСЛЕДНИЙ вопрос пользователя в один самостоятельный "
                                      "поисковый запрос на русском: подставь из диалога название программы или "
                                      "темы, если вопрос ссылается на неё местоимением или опускает её. "
                                      "Верни ТОЛЬКО запрос, одной строкой, без пояснений и кавычек."},
        {"role": "user", "content": f"Диалог:\n{convo}\n\nПоследний вопрос: {query}\n\nСамостоятельный запрос:"},
    ]
    try:
        out = complete_chat(msgs, temperature=0.0).strip().splitlines()[0]
        return out[:200] or query
    except Exception:
        return query


# --------------------------------------------------------------------------- prompt
SYSTEM_PROMPT = """\
Ты — дружелюбный ассистент научно-образовательного фонда имени академика Шахмардана Есенова \
(yessenovfoundation.org). Ты помогаешь людям разобраться в грантах, стипендиях и программах фонда.

ГЛАВНОЕ ПРАВИЛО — НЕ ВЫДУМЫВАЙ.
- Отвечай ТОЛЬКО на основе блока КОНТЕКСТ ниже. Это выдержки из реальных страниц сайта фонда.
- Никогда не придумывай факты: даты, суммы, дедлайны, требования, количество мест, контакты.
- Если в КОНТЕКСТЕ нет ответа — честно скажи, что у тебя нет этих данных, и по-доброму \
направь человека на сайт yessenovfoundation.org или к фонду напрямую. Не угадывай.
- Если программа в контексте помечена как завершённая («программа не ведётся», «реализовывалась \
с … по …») — так и скажи, не выдавай её за действующую.
- Каждый фрагмент относится к СВОЕЙ программе (она указана в заголовке фрагмента). НЕ переноси \
факты, контакты, даты или суммы из одной программы в ответ про другую. Если по нужной программе \
точных данных нет (например, нет её контакта) — так и скажи; не подставляй данные соседней программы.
- Единственный источник фактов — блок КОНТЕКСТ в этом сообщении. Не бери факты из более ранних \
реплик диалога, если их нет в КОНТЕКСТЕ; прошлые сообщения нужны только чтобы понять, о чём речь.

КАК ОТВЕЧАТЬ.
- Тон тёплый и простой, но честность важнее вежливости.
- Отвечай на языке вопроса пользователя: русский, казахский или английский. Контекст на русском — \
переведи смысл сам, если вопрос на другом языке.
- Будь конкретным и кратким. Числа и условия бери дословно из контекста.
- Старайся помочь: если по вопросу есть хотя бы частичная информация — дай её и честно отметь, \
чего не хватает. Полный отказ — только когда в контексте действительно ничего по теме нет.
- Если просят список (например, кто прошёл/победители) и он есть в контексте — перечисли имена из него.
- Если просят график, диаграмму или статистику — не говори, что не умеешь: приложение само построит \
график по реальным числам. Просто кратко прокомментируй данные.
- Не выдумывай ссылки и не вставляй URL — источники к ответу добавит приложение само."""

REFUSAL_HINT = (
    "\n\n(Примечание для ассистента: среди найденных фрагментов нет явно релевантных вопросу. "
    "Скорее всего, в собранных данных нет ответа — честно скажи об этом и направь на сайт фонда, "
    "не придумывай.)"
)


def format_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] Программа/страница: {h['title']} (раздел: {h.get('section','')})\n{h['text']}")
    return "\n\n".join(parts)


def make_messages(query: str, hits: list[dict], history: list[dict] | None = None) -> list[dict]:
    """Build the chat messages: system + grounding context + short history + question.

    Only chunks that individually clear RELEVANCE_FLOOR are shown, so the model never sees
    off-topic chunks from a neighbouring program labelled as authoritative. If nothing clears
    the floor, no chunks are sent at all and the model is told to refuse honestly."""
    relevant = [h for h in hits if h.get("score", 0.0) >= RELEVANCE_FLOOR]
    if relevant:
        grounding = "КОНТЕКСТ (выдержки со страниц фонда):\n\n" + format_context(relevant)
    else:
        grounding = ("КОНТЕКСТ: по этому вопросу в собранных данных нет релевантной информации."
                     + REFUSAL_HINT)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:        # for coreference; КОНТЕКСТ stays the only fact source
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"{grounding}\n\n---\nВОПРОС: {query}"})
    return messages


# --------------------------------------------------------------------------- generation
def stream_chat(messages: list[dict], temperature: float = 0.2):
    """Yield answer text deltas from gemma4 (for st.write_stream)."""
    client = _chat_client()
    model = _cfg().get("GEMMA4_MODEL", "gemma4")
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def complete_chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Full answer string (used by eval.py)."""
    client = _chat_client()
    model = _cfg().get("GEMMA4_MODEL", "gemma4")
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content or ""


def answer(query: str, history: list[dict] | None = None, k: int = TOP_K) -> str:
    """One-shot grounded answer (non-streaming). Convenience for scripts/eval."""
    hits = retrieve(condense_query(query, history), k)
    return complete_chat(make_messages(query, hits, history))
