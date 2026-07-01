
import time
from pathlib import Path

import numpy as np
import torch, open_clip
import streamlit as st

MODEL_NAME, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
OUTPUTS = Path("outputs")

st.set_page_config(page_title="CLIP Image Search", page_icon="\U0001F50E", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.4rem; max-width: 1180px; }
      [data-testid="stImage"] img { border-radius: 14px; height: 200px; width: 100%; object-fit: cover; }
      .hero-title { font-size: 2.15rem; font-weight: 800; margin: 0; letter-spacing:-.5px; }
      .hero-sub { color: #6b7280; font-size: 1.03rem; margin:.15rem 0 .2rem; }
      .score { text-align:center; font-size:.82rem; color:#6b7280; margin:-2px 0 16px; }
      .score b { color:#111827; }
      div.stButton > button {
          border-radius:999px; border:1px solid #e5e7eb; padding:.26rem .95rem;
          background:#f9fafb; font-size:.86rem; font-weight:500;
      }
      div.stButton > button:hover { border-color:#7c3aed; color:#7c3aed; background:#faf5ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading CLIP model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer, device

@st.cache_data(show_spinner=False)
def load_index():
    emb = np.load(OUTPUTS / "embeddings.npy")
    paths = np.load(OUTPUTS / "paths.npy", allow_pickle=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb.astype("float32"), paths

if not (OUTPUTS / "embeddings.npy").exists():
    st.error("No index found. Run clip_image_search.ipynb top to bottom first to create outputs/embeddings.npy.")
    st.stop()

model, tokenizer, device = load_model()
emb, paths = load_index()

@torch.no_grad()
def encode_query(text):
    tok = tokenizer([text]).to(device)
    v = model.encode_text(tok).float()
    v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

st.markdown('<p class="hero-title">\U0001F50E Semantic image search</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Search photos by <b>meaning</b>, not filenames \u2014 powered by CLIP.</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results", 3, 24, 12)
    n_cols = st.slider("Columns", 2, 6, 4)
    st.divider()
    st.metric("Photos in index", f"{len(paths):,}")
    st.caption(f"Model: {MODEL_NAME} / {PRETRAINED}")
    st.caption(f"Device: {device}")
    with st.expander("How it works"):
        st.write("CLIP maps your text and every image into the same 512-d space. We rank "
                 "images by cosine similarity to your query. Images are pre-embedded, so "
                 "each search is one dot product \u2014 instant.")

if "query" not in st.session_state:
    st.session_state.query = ""

st.write("Try an example:")
examples = ["a dog", "people laughing", "sunset over water", "a plate of food",
            "a person riding a bicycle", "children playing"]
for col, ex in zip(st.columns(len(examples)), examples):
    if col.button(ex, key=f"ex_{ex}", use_container_width=True):
        st.session_state.query = ex

query = st.text_input("Search", key="query", label_visibility="collapsed",
                      placeholder="e.g. a dog running on the beach")

if query:
    t0 = time.perf_counter()
    scores = emb @ encode_query(query)
    top = np.argsort(-scores)[:top_k]
    dt = (time.perf_counter() - t0) * 1000
    st.caption(f'Top {len(top)} of {len(paths):,} photos for "{query}"  \u00b7  {dt:.0f} ms')
    cols = st.columns(n_cols)
    for rank, idx in enumerate(top):
        with cols[rank % n_cols]:
            st.image(str(paths[idx]), width="stretch")
            st.markdown(f'<div class="score">#{rank+1} \u00b7 <b>{scores[idx]:.3f}</b></div>',
                        unsafe_allow_html=True)
else:
    st.info("Type a phrase above or tap an example to search the photo collection.")
