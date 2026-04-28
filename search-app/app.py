"""Streamlit search app for the Data Curator vector DB.

Three tabs:
  1. Text search       — embed text with MiniLM, search facebook_text_events
  2. Image-by-image    — embed uploaded image with CLIP, search image_classification
  3. Cross-modal       — embed text with CLIP text encoder, search image_classification
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
TEXT_COLLECTION = "facebook_text_events"
IMAGE_COLLECTION = "image_classification"

# Image paths in Qdrant were written from inside the Airflow container,
# so they look like /opt/airflow/datalake/...   The search-app mounts
# the same datalake at /data, so we swap the prefix at display time.
HOST_PREFIX_MAP = {
    "/opt/airflow/datalake/": "/data/",
}


# ── model loading (cached across reruns) ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading MiniLM…")
def load_minilm():
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Loading CLIP…")
def load_clip():
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return proc, model


@st.cache_resource
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


# ── embedding helpers ────────────────────────────────────────────────────────

def embed_minilm_text(text: str) -> list[float]:
    tok, model = load_minilm()
    enc = tok(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].tolist()


def embed_clip_image(img: Image.Image) -> list[float]:
    proc, model = load_clip()
    inputs = proc(images=[img], return_tensors="pt")
    with torch.no_grad():
        v = model.get_image_features(**inputs)
        v = v / v.norm(dim=-1, keepdim=True)
    return v[0].tolist()


def embed_clip_text(text: str) -> list[float]:
    proc, model = load_clip()
    inputs = proc(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        v = model.get_text_features(**inputs)
        v = v / v.norm(dim=-1, keepdim=True)
    return v[0].tolist()


# ── utilities ────────────────────────────────────────────────────────────────

def to_host_path(p: str) -> str:
    for src, dst in HOST_PREFIX_MAP.items():
        if p.startswith(src):
            return p.replace(src, dst, 1)
    return p


def collection_count(name: str) -> int:
    try:
        return get_qdrant().get_collection(name).points_count or 0
    except Exception:
        return 0


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Data Curator Search", page_icon="🔎", layout="wide")
st.title("🔎 Data Curator — Vector Search")

n_text = collection_count(TEXT_COLLECTION)
n_img = collection_count(IMAGE_COLLECTION)

c1, c2 = st.columns(2)
c1.metric(f"`{TEXT_COLLECTION}`", f"{n_text:,} points · 384-d")
c2.metric(f"`{IMAGE_COLLECTION}`", f"{n_img:,} points · 512-d")

tab_text, tab_image, tab_cross = st.tabs([
    "📝 Text events",
    "🖼️ Image-by-image",
    "🔀 Cross-modal (text → images)",
])

# ─────────────── Tab 1: text search ───────────────
with tab_text:
    st.caption("Find Facebook events most similar to your query, optionally filtered by label.")
    col_q, col_l, col_k = st.columns([5, 2, 1])
    q = col_q.text_input("Query", "buy cheap pills click here", key="text_q")
    label_filter = col_l.selectbox("Label", ["any", "safe", "spam", "hate", "harassment"])
    top_k = col_k.number_input("Top K", min_value=1, max_value=50, value=10)

    if st.button("Search", key="text_search", type="primary"):
        flt = None
        if label_filter != "any":
            flt = Filter(must=[FieldCondition(key="label", match=MatchValue(value=label_filter))])
        with st.spinner("Embedding & searching…"):
            hits = get_qdrant().search(
                collection_name=TEXT_COLLECTION,
                query_vector=embed_minilm_text(q),
                limit=int(top_k),
                query_filter=flt,
            )
        for h in hits:
            with st.container(border=True):
                meta = (
                    f"**Score:** `{h.score:.3f}`  ·  "
                    f"**Label:** `{h.payload.get('label', '?')}`  ·  "
                    f"**Type:** `{h.payload.get('event_type', '?')}`"
                )
                st.markdown(meta)
                st.write(h.payload.get("text", ""))
                st.caption(
                    f"by {h.payload.get('actor_name', 'unknown')}  ·  "
                    f"{h.payload.get('created_at', '')}  ·  "
                    f"event_id `{h.payload.get('event_id', '')}`"
                )

# ─────────────── Tab 2: image → image ───────────────
with tab_image:
    st.caption("Upload an image; CLIP will find the closest images in the collection.")
    upload = st.file_uploader("Image file", type=["png", "jpg", "jpeg"], key="img_upload")
    top_k_img = st.number_input("Top K", min_value=1, max_value=20, value=5, key="img_topk")

    if upload is not None:
        img = Image.open(upload).convert("RGB")
        st.image(img, caption="Query image", width=240)
        if st.button("Find similar", key="img_search", type="primary"):
            with st.spinner("Embedding & searching…"):
                hits = get_qdrant().search(
                    collection_name=IMAGE_COLLECTION,
                    query_vector=embed_clip_image(img),
                    limit=int(top_k_img),
                )
            cols = st.columns(min(len(hits), 5) or 1)
            for col, h in zip(cols, hits):
                with col:
                    path = to_host_path(h.payload.get("image_path", ""))
                    if Path(path).exists():
                        st.image(path, caption=f"{h.score:.3f} · {h.payload.get('label', '?')}")
                    else:
                        st.text(f"{h.score:.3f} · {h.payload.get('label', '?')}\n(missing: {path})")

# ─────────────── Tab 3: text → image (CLIP cross-modal) ───────────────
with tab_cross:
    st.caption("Describe what you're looking for in words; CLIP retrieves matching images.")
    q2 = st.text_input("Description", "a tall building at sunset", key="cm_q")
    top_k_cm = st.number_input("Top K", min_value=1, max_value=20, value=5, key="cm_topk")

    if st.button("Search", key="cm_search", type="primary"):
        with st.spinner("Embedding & searching…"):
            hits = get_qdrant().search(
                collection_name=IMAGE_COLLECTION,
                query_vector=embed_clip_text(q2),
                limit=int(top_k_cm),
            )
        cols = st.columns(min(len(hits), 5) or 1)
        for col, h in zip(cols, hits):
            with col:
                path = to_host_path(h.payload.get("image_path", ""))
                if Path(path).exists():
                    st.image(path, caption=f"{h.score:.3f} · {h.payload.get('label', '?')}")
                else:
                    st.text(f"{h.score:.3f} · {h.payload.get('label', '?')}\n(missing: {path})")
