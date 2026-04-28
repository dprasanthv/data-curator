from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from kafka.admin import KafkaAdminClient


def _check_kafka() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    topic = os.environ.get("KAFKA_TOPIC", "facebook.events")

    client = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="airflow-healthcheck")
    try:
        topics = client.list_topics()
        if topic not in topics:
            raise RuntimeError(f"Topic '{topic}' not found in Kafka. Available: {topics}")
    finally:
        client.close()


with DAG(
    dag_id="facebook_events_healthcheck",
    start_date=datetime(2024, 1, 1),
    schedule=timedelta(minutes=1),
    catchup=False,
) as dag:
    PythonOperator(task_id="check_kafka", python_callable=_check_kafka)
