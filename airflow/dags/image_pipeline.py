from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

CATEGORIES: dict[str, tuple[str, int]] = {
    "building": ("building,architecture,skyscraper", 3),
    "people": ("people,person,crowd", 3),
    "car": ("car,automobile,vehicle", 3),
}

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Choices name="label" toName="image" choice="single" showInLine="true">
    <Choice value="building" background="#0080FF"/>
    <Choice value="people" background="#00C853"/>
    <Choice value="car" background="#FF5252"/>
  </Choices>
</View>"""

PROJECT_NAME = "Image Classification: Building / People / Car"


def _image_dir() -> Path:
    return Path(os.environ.get("DATALAKE_BASE_DIR", "/opt/airflow/datalake")) / "images"


def download_images() -> None:
    for category, (keyword, count) in CATEGORIES.items():
        cat_dir = _image_dir() / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        downloaded = skipped = failed = 0
        for i in range(count):
            out_path = cat_dir / f"{category}_{i:04d}.jpg"
            if out_path.exists():
                skipped += 1
                continue

            url = f"https://loremflickr.com/800/600/{keyword}?lock={i}"
            try:
                resp = requests.get(url, timeout=20, allow_redirects=True)
                content_type = resp.headers.get("content-type", "")
                if resp.status_code == 200 and "image" in content_type and len(resp.content) > 1000:
                    out_path.write_bytes(resp.content)
                    downloaded += 1
                else:
                    print(f"  skip {out_path.name}: status={resp.status_code} ct={content_type} size={len(resp.content)}")
                    failed += 1
            except Exception as exc:
                print(f"  error {out_path.name}: {exc}")
                failed += 1

            time.sleep(0.3)

        print(
            f"[{category}] downloaded={downloaded} skipped={skipped} "
            f"failed={failed} total_on_disk={skipped + downloaded}/{count}"
        )


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


def _get_or_create_project(base_url: str, headers: dict) -> int:
    resp = requests.get(f"{base_url}/api/projects/", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    projects = data.get("results") if isinstance(data, dict) else data
    for proj in projects:
        if proj["title"] == PROJECT_NAME:
            return proj["id"]

    create = requests.post(
        f"{base_url}/api/projects/",
        headers=headers,
        json={"title": PROJECT_NAME, "label_config": LABEL_CONFIG},
        timeout=10,
    )
    if not create.ok:
        raise RuntimeError(f"Failed to create project [{create.status_code}]: {create.text[:400]}")
    return create.json()["id"]


def _ensure_local_storage(base_url: str, headers: dict, project_id: int, storage_path: str) -> None:
    """Create a LocalFilesImportStorage record if one doesn't exist for this path.

    Label Studio's localfiles_data view requires a matching storage record to
    authorise file serving — without it the view always returns 404.
    """
    resp = requests.get(
        f"{base_url}/api/storages/localfiles",
        headers=headers,
        params={"project": project_id},
        timeout=10,
    )
    resp.raise_for_status()
    for storage in resp.json():
        if storage.get("path") == storage_path:
            print(f"Local storage already exists (id={storage['id']})")
            return

    create = requests.post(
        f"{base_url}/api/storages/localfiles",
        headers=headers,
        json={
            "project": project_id,
            "path": storage_path,
            "use_blob_urls": False,
            "title": "Datalake Images",
            "regex_filter": r".*\.jpg",
        },
        timeout=10,
    )
    if not create.ok:
        raise RuntimeError(f"Failed to create storage [{create.status_code}]: {create.text[:400]}")
    print(f"Created local storage id={create.json()['id']} path={storage_path}")


def import_images_to_label_studio() -> None:
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://label-studio:8080")
    image_dir = _image_dir()

    token = _get_auth_token(base_url)
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    project_id = _get_or_create_project(base_url, headers)

    # Register the storage path so Label Studio's security check allows file serving
    _ensure_local_storage(base_url, headers, project_id, "/data/images")

    tasks: list[dict] = []
    for category in CATEGORIES:
        for img_path in sorted((image_dir / category).glob("*.jpg")):
            # relative to datalake root (/data in LS); e.g. images/building/building_0000.jpg
            rel = img_path.relative_to(image_dir.parent)
            tasks.append({"data": {"image": f"/data/local-files/?d={rel}", "category_hint": category}})

    if not tasks:
        print("No images found — run download_images first")
        return

    total_imported = 0
    batch_size = 100
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        resp = requests.post(
            f"{base_url}/api/projects/{project_id}/import",
            headers=headers,
            data=json.dumps(batch, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Import failed [{resp.status_code}]: {resp.text[:500]}")
        total_imported += resp.json().get("task_count", 0)
        print(f"  batch {i // batch_size + 1}: imported {resp.json().get('task_count', 0)} tasks")

    print(f"Done — {total_imported} image tasks in Label Studio project '{PROJECT_NAME}' (id={project_id})")


with DAG(
    dag_id="image_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    download = PythonOperator(
        task_id="download_images",
        python_callable=download_images,
        execution_timeout=timedelta(hours=2),
    )
    push = PythonOperator(
        task_id="import_images_to_label_studio",
        python_callable=import_images_to_label_studio,
    )
    download >> push
