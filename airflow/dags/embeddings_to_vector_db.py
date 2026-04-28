"""Index gold-layer embeddings (text + image) into Qdrant for similarity search."""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from airflow import DAG
from airflow.operators.python import PythonOperator

DATALAKE_BASE = Path(os.environ.get("DATALAKE_BASE_DIR", "/opt/airflow/datalake"))
TEXT_DIR = DATALAKE_BASE / "gold" / "text_embeddings"
IMAGE_DIR = DATALAKE_BASE / "gold" / "image_embeddings"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
TEXT_COLLECTION = "facebook_text_events"
IMAGE_COLLECTION = "image_classification"
TEXT_DIM = 384
IMAGE_DIM = 512
UPSERT_BATCH = 256


def _latest_parquet(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    files = sorted(folder.glob("embeddings_*.parquet"))
    return files[-1] if files else None


def _ensure_collection(client, name: str, dim: int) -> None:
    from qdrant_client.http.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        print(f"  collection '{name}' already exists")
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"  created collection '{name}' (dim={dim}, distance=cosine)")


def _stable_uuid(seed: str) -> str:
    """Deterministic UUID from a seed string so re-runs upsert (not duplicate)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


# ── Task 1: index text embeddings ─────────────────────────────────────────────

def index_text_embeddings() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct

    pf = _latest_parquet(TEXT_DIR)
    if pf is None:
        print(f"No text embeddings found at {TEXT_DIR}")
        return
    print(f"Reading {pf}")

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    _ensure_collection(client, TEXT_COLLECTION, TEXT_DIM)

    table = pq.read_table(str(pf))
    rows = table.to_pylist()
    print(f"Indexing {len(rows)} text vectors → '{TEXT_COLLECTION}'")

    points: list = []
    total = 0
    for r in rows:
        points.append(PointStruct(
            id=_stable_uuid(f"text:{r['event_id']}"),
            vector=r["embedding"],
            payload={
                "event_id":   r["event_id"],
                "event_type": r["event_type"],
                "actor_name": r["actor_name"],
                "created_at": r["created_at"],
                "text":       r["text"],
                "label":      r["label"],
            },
        ))
        if len(points) >= UPSERT_BATCH:
            client.upsert(collection_name=TEXT_COLLECTION, points=points)
            total += len(points)
            print(f"  upserted {total}/{len(rows)}")
            points = []

    if points:
        client.upsert(collection_name=TEXT_COLLECTION, points=points)
        total += len(points)

    info = client.get_collection(TEXT_COLLECTION)
    print(f"Done. Collection '{TEXT_COLLECTION}' now has {info.points_count} points")


# ── Task 2: index image embeddings ────────────────────────────────────────────

def index_image_embeddings() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct

    pf = _latest_parquet(IMAGE_DIR)
    if pf is None:
        print(f"No image embeddings found at {IMAGE_DIR}")
        return
    print(f"Reading {pf}")

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    _ensure_collection(client, IMAGE_COLLECTION, IMAGE_DIM)

    table = pq.read_table(str(pf))
    rows = table.to_pylist()
    print(f"Indexing {len(rows)} image vectors → '{IMAGE_COLLECTION}'")

    points: list = []
    total = 0
    for r in rows:
        points.append(PointStruct(
            id=_stable_uuid(f"image:{r['task_id']}"),
            vector=r["embedding"],
            payload={
                "task_id":    r["task_id"],
                "label":      r["label"],
                "image_path": r["image_path"],
            },
        ))
        if len(points) >= UPSERT_BATCH:
            client.upsert(collection_name=IMAGE_COLLECTION, points=points)
            total += len(points)
            print(f"  upserted {total}/{len(rows)}")
            points = []

    if points:
        client.upsert(collection_name=IMAGE_COLLECTION, points=points)
        total += len(points)

    info = client.get_collection(IMAGE_COLLECTION)
    print(f"Done. Collection '{IMAGE_COLLECTION}' now has {info.points_count} points")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="embeddings_to_vector_db",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    index_text = PythonOperator(
        task_id="index_text_embeddings",
        python_callable=index_text_embeddings,
    )
    index_images = PythonOperator(
        task_id="index_image_embeddings",
        python_callable=index_image_embeddings,
    )
    # Independent — run in parallel
    [index_text, index_images]
