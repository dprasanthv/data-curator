from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from airflow import DAG
from airflow.operators.python import PythonOperator

# sentence-transformers/all-MiniLM-L6-v2: 80 MB, 384-dim, fast on CPU
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DATALAKE_BASE = Path(os.environ.get("DATALAKE_BASE_DIR", "/opt/airflow/datalake"))
BRONZE_DIR = DATALAKE_BASE / "bronze" / "facebook_events"
OUTPUT_DIR = DATALAKE_BASE / "gold" / "text_embeddings"
BATCH_SIZE = 64


LABELS = ["safe", "spam", "hate", "harassment"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_text(record: dict) -> str:
    event_type = record.get("event_type", "")
    if event_type == "post":
        return (record.get("post") or {}).get("text", "")
    if event_type == "comment":
        return (record.get("comment") or {}).get("text", "")
    if event_type == "reaction":
        r = record.get("reaction") or {}
        return f"reacted with {r.get('reaction_type', '')} on {r.get('target_type', '')}"
    if event_type == "follow":
        return f"followed user {(record.get('follow') or {}).get('followed_user_name', '')}"
    return json.dumps(record)


def _mean_pool(token_embeddings, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


# ── Task 1: read parquet + assign random labels ───────────────────────────────

def load_and_label() -> None:
    if not BRONZE_DIR.exists():
        raise FileNotFoundError(f"Bronze dir not found: {BRONZE_DIR}")

    records = []
    parquet_files = sorted(BRONZE_DIR.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet file(s)")

    for pf in parquet_files:
        table = pq.read_table(str(pf))
        for row in table.to_pylist():
            text = _extract_text(row).strip()
            if not text:
                continue
            records.append({
                "event_id":   row.get("event_id", ""),
                "event_type": row.get("event_type", ""),
                "actor_name": (row.get("actor") or {}).get("name", ""),
                "created_at": row.get("created_at", ""),
                "text":       text,
                "label":      random.choice(LABELS),
            })

    print(f"Loaded {len(records)} records with random labels")

    cache = DATALAKE_BASE / "tmp" / "fb_labelled.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records, ensure_ascii=False))


# ── Task 2: tokenize + embed ──────────────────────────────────────────────────

def generate_embeddings() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = DATALAKE_BASE / "tmp" / "fb_labelled.json"
    if not cache.exists():
        raise FileNotFoundError("Run load_and_label first")

    records = json.loads(cache.read_text())
    if not records:
        print("No records to embed")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{MODEL_NAME}' on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    FLUSH_EVERY = 2000  # write to disk every N records to avoid OOM
    buffer: list[dict] = []
    total_written = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"embeddings_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"

    schema = pa.schema([
        pa.field("event_id",   pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("actor_name", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("text",       pa.string()),
        pa.field("label",      pa.string()),
        pa.field("embedding",  pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])

    def _flush(writer, buf):
        table = pa.table({
            "event_id":   pa.array([r["event_id"] for r in buf], pa.string()),
            "event_type": pa.array([r["event_type"] for r in buf], pa.string()),
            "actor_name": pa.array([r["actor_name"] for r in buf], pa.string()),
            "created_at": pa.array([r["created_at"] for r in buf], pa.string()),
            "text":       pa.array([r["text"] for r in buf], pa.string()),
            "label":      pa.array([r["label"] for r in buf], pa.string()),
            "embedding":  pa.array([r["embedding"] for r in buf], pa.list_(pa.float32(), EMBEDDING_DIM)),
        }, schema=schema)
        writer.write_table(table)

    with pq.ParquetWriter(str(out), schema) as writer:
        for batch_start in range(0, len(records), BATCH_SIZE):
            batch = records[batch_start : batch_start + BATCH_SIZE]
            texts = [r["text"] for r in batch]

            # ── Tokenization ─────────────────────────────────────────────────
            # Splits text into subword tokens and converts to integer IDs.
            # e.g. "followed user john_doe" → [101, 2628, 2659, 2198, 1035, 8585, 102]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            if batch_start == 0:
                ids = encoded["input_ids"][0].tolist()
                tokens = tokenizer.convert_ids_to_tokens(ids)
                print(f"\n── Tokenization sample ──")
                print(f"  text  : {texts[0][:80]}")
                print(f"  tokens: {tokens}")
                print(f"  ids   : {ids}\n")

            # ── Embedding ────────────────────────────────────────────────────
            with torch.no_grad():
                output = model(**encoded)

            # Mean-pool all token vectors → one 384-dim vector per sentence
            vecs = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)  # L2-normalise
            vecs = vecs.cpu().numpy()

            for i, rec in enumerate(batch):
                buffer.append({**rec, "embedding": vecs[i].tolist()})

            # Flush to disk periodically to free memory
            if len(buffer) >= FLUSH_EVERY:
                _flush(writer, buffer)
                total_written += len(buffer)
                print(f"  flushed {total_written}/{len(records)}")
                buffer.clear()

        if buffer:
            _flush(writer, buffer)
            total_written += len(buffer)

    print(f"\nEmbedded {total_written} records")
    print(f"Saved → {out}  ({total_written} rows × {EMBEDDING_DIM}-dim)")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="text_embeddings",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    load = PythonOperator(
        task_id="load_and_label",
        python_callable=load_and_label,
    )
    embed = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings,
    )
    load >> embed
