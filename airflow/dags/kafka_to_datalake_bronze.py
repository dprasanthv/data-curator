from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from airflow import DAG
from airflow.operators.python import PythonOperator
from kafka import KafkaConsumer


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(path), compression="snappy")


def kafka_to_bronze() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    topic = os.environ.get("KAFKA_TOPIC", "facebook.events")

    base_dir = Path(os.environ.get("DATALAKE_BASE_DIR", "/opt/airflow/datalake"))
    dataset = os.environ.get("DATALAKE_DATASET", "facebook_events")
    layer = os.environ.get("DATALAKE_LAYER", "bronze")

    max_messages = int(os.environ.get("MAX_MESSAGES_PER_RUN", "500"))
    max_seconds = int(os.environ.get("MAX_SECONDS_PER_RUN", "20"))

    now = _utc_now()
    dt = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")

    out_dir = base_dir / layer / dataset / f"dt={dt}" / f"hour={hour}"
    out_file = out_dir / f"part-{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=os.environ.get("KAFKA_GROUP_ID", "airflow_datalake_bronze"),
        enable_auto_commit=True,
        auto_offset_reset=os.environ.get("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=1000,
    )

    start = time.time()
    records: list[dict[str, Any]] = []

    for msg in consumer:
        event = dict(msg.value) if isinstance(msg.value, dict) else msg.value
        if isinstance(event, dict):
            event.setdefault("_ingested_at", now.isoformat())
            event.setdefault("_kafka", {})
            event["_kafka"].update(
                {
                    "topic": msg.topic,
                    "partition": msg.partition,
                    "offset": msg.offset,
                    "key": msg.key.decode("utf-8") if msg.key else None,
                    "timestamp_ms": msg.timestamp,
                }
            )
            records.append(event)

        if len(records) >= max_messages:
            break
        if time.time() - start >= max_seconds:
            break

    consumer.close()

    if records:
        _write_parquet(out_file, records)


with DAG(
    dag_id="kafka_to_datalake_bronze",
    start_date=datetime(2024, 1, 1),
    schedule=timedelta(minutes=1),
    catchup=False,
) as dag:
    PythonOperator(task_id="kafka_to_bronze", python_callable=kafka_to_bronze)
