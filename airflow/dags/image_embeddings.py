from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
PROJECT_NAME = "Image Classification: Building / People / Car"
DATALAKE_BASE = Path(os.environ.get("DATALAKE_BASE_DIR", "/opt/airflow/datalake"))
OUTPUT_DIR = DATALAKE_BASE / "gold" / "image_embeddings"
ANNOTATIONS_CACHE = DATALAKE_BASE / "tmp" / "ls_annotations.json"
BATCH_SIZE = 32


# ── Label Studio helpers ──────────────────────────────────────────────────────

def _get_auth_token(base_url: str) -> str:
    username = os.environ.get("LABEL_STUDIO_USERNAME", "admin@datacurator.local")
    password = os.environ.get("LABEL_STUDIO_PASSWORD", "admin")
    session = requests.Session()
    login_page = session.get(f"{base_url}/user/login", timeout=10)
    login_page.raise_for_status()
    csrf = session.cookies.get("csrftoken", "")
    resp = session.post(
        f"{base_url}/user/login",
        data={"email": username, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{base_url}/user/login"},
        timeout=10,
        allow_redirects=True,
    )
    resp.raise_for_status()
    token_resp = session.get(f"{base_url}/api/current-user/token", timeout=10)
    token_resp.raise_for_status()
    return token_resp.json()["token"]


def _find_project_id(base_url: str, headers: dict) -> int:
    resp = requests.get(f"{base_url}/api/projects/", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    projects = data.get("results") if isinstance(data, dict) else data
    for proj in projects:
        if proj["title"] == PROJECT_NAME:
            return proj["id"]
    raise ValueError(f"Project '{PROJECT_NAME}' not found in Label Studio")


# ── Task 1: export annotations ────────────────────────────────────────────────

def export_annotations() -> None:
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://label-studio:8080")
    token = _get_auth_token(base_url)
    headers = {"Authorization": f"Token {token}"}

    project_id = _find_project_id(base_url, headers)

    resp = requests.get(
        f"{base_url}/api/projects/{project_id}/export",
        headers=headers,
        params={"exportType": "JSON"},
        timeout=120,
    )
    resp.raise_for_status()
    annotations = resp.json()

    annotated = [t for t in annotations if t.get("annotations")]
    print(f"Total tasks: {len(annotations)} | Annotated: {len(annotated)}")

    ANNOTATIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_CACHE.write_text(json.dumps(annotated, ensure_ascii=False))


# ── Task 2: generate CLIP embeddings ─────────────────────────────────────────

def _parse_label(task: dict) -> str | None:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "choices":
                choices = result["value"].get("choices", [])
                if choices:
                    return choices[0]
    return None


def _parse_image_path(task: dict) -> Path | None:
    image_url = task["data"].get("image", "")
    if "?d=" not in image_url:
        return None
    rel = image_url.split("?d=", 1)[-1].lstrip("/")
    # strip accidental double prefix e.g. data/images/...
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return DATALAKE_BASE / rel


def generate_embeddings() -> None:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not ANNOTATIONS_CACHE.exists():
        raise FileNotFoundError("Run export_annotations first")

    annotations = json.loads(ANNOTATIONS_CACHE.read_text())
    if not annotations:
        print("No annotated tasks to embed")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{MODEL_NAME}' on {device} ...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    records: list[dict] = []
    skipped = 0

    # Process in batches
    for batch_start in range(0, len(annotations), BATCH_SIZE):
        batch = annotations[batch_start : batch_start + BATCH_SIZE]
        images, meta = [], []

        for task in batch:
            img_path = _parse_image_path(task)
            if img_path is None or not img_path.exists():
                skipped += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"  skip {img_path.name}: {exc}")
                skipped += 1
                continue
            images.append(img)
            meta.append({"task_id": task["id"], "label": _parse_label(task), "path": str(img_path)})

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalise
            feats = feats.cpu().numpy()

        for i, m in enumerate(meta):
            records.append({**m, "embedding": feats[i].tolist()})

        done = batch_start + len(batch)
        print(f"  embedded {min(done, len(annotations))}/{len(annotations)} tasks")

    print(f"Embedded {len(records)} images | skipped {skipped}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "task_id":    pa.array([r["task_id"] for r in records], pa.int64()),
            "label":      pa.array([r["label"] for r in records], pa.string()),
            "image_path": pa.array([r["path"] for r in records], pa.string()),
            "embedding":  pa.array(
                [r["embedding"] for r in records],
                pa.list_(pa.float32(), EMBEDDING_DIM),
            ),
        }
    )
    out = OUTPUT_DIR / f"embeddings_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"
    pq.write_table(table, out)
    print(f"Saved → {out}  ({len(records)} rows × {EMBEDDING_DIM}-dim)")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="image_embeddings",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    export = PythonOperator(
        task_id="export_annotations",
        python_callable=export_annotations,
    )
    embed = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings,
    )
    export >> embed
