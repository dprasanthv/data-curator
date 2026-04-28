from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_label_studio_task(record: dict) -> dict:
    event_type = record.get("event_type", "unknown")
    actor = record.get("actor") or {}

    text_map = {
        "post": (record.get("post") or {}).get("text", ""),
        "comment": (record.get("comment") or {}).get("text", ""),
        "reaction": (
            f"reacted with {(record.get('reaction') or {}).get('reaction_type', '')} "
            f"on {(record.get('reaction') or {}).get('target_type', '')}"
        ),
        "follow": f"followed user {(record.get('follow') or {}).get('followed_user_name', '')}",
        "user": f"user registered: {(record.get('user') or {}).get('name', '')}",
    }

    return {
        "data": {
            "text": text_map.get(event_type, json.dumps(record)),
            "event_type": event_type,
            "event_id": record.get("event_id"),
            "actor_name": actor.get("name"),
            "actor_user_id": actor.get("user_id"),
            "created_at": record.get("created_at"),
            "raw": json.dumps(record, ensure_ascii=False),
        }
    }


def parquet_to_label_studio() -> None:
    bronze_dir = Path(
        os.environ.get("DATALAKE_BASE_DIR", "/opt/airflow/datalake")
    ) / "bronze" / os.environ.get("DATALAKE_DATASET", "facebook_events")

    ls_tasks_dir = Path(
        os.environ.get("LABEL_STUDIO_TASKS_DIR", "/opt/airflow/datalake/label-studio/tasks")
    )
    max_tasks = int(os.environ.get("LS_MAX_TASKS_PER_RUN", "200"))
    lookback_hours = int(os.environ.get("LS_LOOKBACK_HOURS", "2"))

    now = _utc_now()
    tasks: list[dict] = []

    cutoff = now.timestamp() - lookback_hours * 3600
    for parquet_file in sorted(bronze_dir.rglob("*.parquet")):
        if parquet_file.stat().st_mtime < cutoff:
            continue
        table = pq.read_table(str(parquet_file))
        for record in table.to_pylist():
            tasks.append(_to_label_studio_task(record))
            if len(tasks) >= max_tasks:
                break
        if len(tasks) >= max_tasks:
            break

    if not tasks:
        return

    out_file = (
        ls_tasks_dir / f"tasks-{now.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


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


def _ls_headers(base_url: str) -> dict[str, str]:
    token = _get_auth_token(base_url)
    return {"Authorization": f"Token {token}"}


def _get_or_create_project(base_url: str, project_name: str) -> int:
    headers = _ls_headers(base_url)
    resp = requests.get(f"{base_url}/api/projects/", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    projects = data.get("results") if isinstance(data, dict) else data
    for proj in projects:
        if proj["title"] == project_name:
            return proj["id"]

    create = requests.post(
        f"{base_url}/api/projects/",
        headers=headers,
        json={
            "title": project_name,
            "label_config": (
                "<View>"
                "<Text name='text' value='$text'/>"
                "<Choices name='label' toName='text' choice='single'>"
                "<Choice value='safe'/>"
                "<Choice value='spam'/>"
                "<Choice value='hate'/>"
                "<Choice value='harassment'/>"
                "</Choices>"
                "</View>"
            ),
        },
        timeout=10,
    )
    create.raise_for_status()
    return create.json()["id"]


def import_to_label_studio(**context: dict) -> None:
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://label-studio:8080")
    project_name = os.environ.get("LS_PROJECT_NAME", "Facebook Events")
    ls_tasks_dir = Path(
        os.environ.get("LABEL_STUDIO_TASKS_DIR", "/opt/airflow/datalake/label-studio/tasks")
    )

    project_id = _get_or_create_project(base_url, project_name)

    headers = _ls_headers(base_url)
    headers["Content-Type"] = "application/json"
    imported = 0
    for jsonl_file in sorted(ls_tasks_dir.glob("*.jsonl")):
        tasks_batch = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks_batch.append(json.loads(line))

        if not tasks_batch:
            jsonl_file.rename(jsonl_file.with_suffix(".jsonl.imported"))
            continue

        resp = requests.post(
            f"{base_url}/api/projects/{project_id}/import",
            headers=headers,
            data=json.dumps(tasks_batch, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Label Studio import failed [{resp.status_code}]: {resp.text[:500]}"
            )
        imported += resp.json().get("task_count", 0)
        jsonl_file.rename(jsonl_file.with_suffix(".jsonl.imported"))

    print(f"Imported {imported} tasks into Label Studio project '{project_name}' (id={project_id})")


with DAG(
    dag_id="parquet_to_label_studio",
    start_date=datetime(2024, 1, 1),
    schedule=timedelta(minutes=5),
    catchup=False,
) as dag:
    convert = PythonOperator(
        task_id="convert_parquet_to_ls_tasks",
        python_callable=parquet_to_label_studio,
    )
    push = PythonOperator(
        task_id="import_to_label_studio",
        python_callable=import_to_label_studio,
    )
    convert >> push
