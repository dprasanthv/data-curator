# Data Curator

An end-to-end data pipeline that streams synthetic social media events through Kafka, lands them in a Parquet datalake, and turns them into ML-ready vector embeddings — orchestrated by Airflow with a Label Studio annotation layer in the middle.

## What this project does

```
┌───────────────┐   Kafka    ┌──────────────┐  Airflow   ┌─────────────────┐
│   Producer    │──────────▶ │    Bronze    │──────────▶ │  Label Studio   │
│ (Faker-based  │  facebook. │   datalake   │  parquet → │ (human-in-loop  │
│  social sim)  │   events   │  (Parquet)   │   tasks    │  annotation)    │
└───────────────┘            └──────────────┘            └─────────────────┘
                                    │                            │
                                    ▼                            ▼
                            ┌──────────────────────────────────────┐
                            │       Gold layer (Parquet)           │
                            │  • text embeddings  (MiniLM, 384-d)  │
                            │  • image embeddings (CLIP,  512-d)   │
                            └──────────────────────────────────────┘
```

The pipeline covers two parallel tracks:

**Text track** — A producer fakes Facebook-style events (posts, comments, reactions, follows, user registrations) and pushes them to Kafka. An Airflow DAG consumes them into a partitioned Parquet datalake. Records are then assigned classification labels (`safe / spam / hate / harassment`), tokenized via a BERT WordPiece tokenizer, and embedded as 384-dim vectors using `sentence-transformers/all-MiniLM-L6-v2`.

**Image track** — Another DAG downloads images across three categories (`building / people / car`), imports them into Label Studio for human annotation, then generates 512-dim CLIP embeddings (`openai/clip-vit-base-patch32`) from the annotated set.

Both tracks output Parquet files in the gold layer ready for downstream ML use (similarity search, classifier training, clustering).

## Stack

| Component | Purpose |
|---|---|
| **Kafka** (KRaft, single node) | Event streaming backbone |
| **Faker producer** | Synthetic Facebook-style event generator |
| **Airflow** (LocalExecutor + Postgres) | DAG orchestration |
| **PyArrow** | Parquet read/write for the datalake |
| **Label Studio** | Manual annotation UI (text + images) |
| **HuggingFace Transformers** | CLIP & MiniLM models |
| **PyTorch** | Inference engine (CPU) |
| **Kafbat UI** | Kafka topic browser |

## Project layout

```
data-curator/
├── docker-compose.yml          # all services
├── producer/                   # Faker → Kafka producer
│   ├── producer.py
│   ├── Dockerfile
│   └── requirements.txt
├── airflow/
│   ├── Dockerfile              # bakes CLIP weights into image
│   ├── requirements.txt        # torch, transformers, pyarrow, ...
│   └── dags/
│       ├── kafka_to_datalake_bronze.py    # Kafka → Parquet (bronze)
│       ├── parquet_to_label_studio.py     # bronze → Label Studio tasks
│       ├── image_pipeline.py              # download imgs → Label Studio
│       ├── image_embeddings.py            # annotated imgs → CLIP vectors
│       ├── text_embeddings.py             # bronze events → MiniLM vectors
│       └── facebook_events_healthcheck.py
├── datalake/                   # mounted into Airflow + Label Studio
│   ├── bronze/facebook_events/dt=YYYY-MM-DD/hour=HH/*.parquet
│   ├── images/{building,people,car}/
│   ├── label-studio/tasks/     # JSONL tasks pending import
│   └── gold/
│       ├── text_embeddings/    # MiniLM 384-d Parquet
│       └── image_embeddings/   # CLIP 512-d Parquet
└── label-studio-data/          # LS internal state (sqlite, etc.)
```

## Prerequisites

- Docker Desktop (4 GB RAM minimum, 8 GB recommended — model loading is heavy)
- ~2 GB free disk for Docker images (CLIP weights are baked in at build time)

## Running it

### 1. Build & start everything

```bash
docker-compose up -d --build
```

First build takes 5–10 minutes — it installs PyTorch and pre-downloads the CLIP model weights into the Airflow image (this avoids macOS Docker volume rename bugs at runtime).

### 2. Open the UIs

| Service | URL | Credentials |
|---|---|---|
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **Label Studio** | http://localhost:8081 | `admin@datacurator.local` / `admin` |
| **Kafbat (Kafka UI)** | http://localhost:8082 | none |

### 3. Watch data flow in

The producer starts emitting events immediately (one every 250 ms). Within a few minutes you should see:

- Events flowing in Kafbat under topic `facebook.events`
- The `kafka_to_datalake_bronze` DAG running every few minutes, dropping Parquet files into `datalake/bronze/facebook_events/`
- The `parquet_to_label_studio` DAG converting those into JSONL and importing them into a "Facebook Events" project in Label Studio

### 4. Generate text embeddings

This DAG reads directly from the bronze Parquet, assigns random labels, tokenizes, and embeds — no Label Studio interaction required:

```bash
docker-compose exec airflow-scheduler airflow dags test text_embeddings 2024-01-01
```

Output → `datalake/gold/text_embeddings/embeddings_<timestamp>.parquet`

### 5. Generate image embeddings

Trigger the image pipeline first (downloads images and imports them into Label Studio):

```bash
docker-compose exec airflow-scheduler airflow dags test image_pipeline 2024-01-01
```

Annotate at least a handful of images in Label Studio, then run:

```bash
docker-compose exec airflow-scheduler airflow dags test image_embeddings 2024-01-01
```

Output → `datalake/gold/image_embeddings/embeddings_<timestamp>.parquet`

## Output schema

### `gold/text_embeddings/*.parquet`

| column | type | description |
|---|---|---|
| `event_id` | string | original Kafka event id |
| `event_type` | string | `post / comment / reaction / follow / user` |
| `actor_name` | string | who triggered the event |
| `created_at` | string | ISO timestamp |
| `text` | string | the text that was embedded |
| `label` | string | `safe / spam / hate / harassment` |
| `embedding` | `list<float32>[384]` | L2-normalised MiniLM vector |

### `gold/image_embeddings/*.parquet`

| column | type | description |
|---|---|---|
| `task_id` | int64 | Label Studio task id |
| `label` | string | `building / people / car` |
| `image_path` | string | absolute path on disk |
| `embedding` | `list<float32>[512]` | L2-normalised CLIP vector |

Vectors are L2-normalised, so dot product == cosine similarity — ready for nearest-neighbour search.

## Stopping & cleaning up

```bash
docker-compose down                # stop containers, keep data
docker-compose down -v             # also delete Kafka + Postgres volumes
rm -rf datalake/bronze datalake/gold datalake/label-studio  # wipe datalake
```

## Troubleshooting

- **OOM during embeddings** — `text_embeddings.py` flushes to Parquet every 2,000 records via `pq.ParquetWriter` to keep memory bounded. If you still hit OOM, lower `BATCH_SIZE` in the DAG or give Docker more memory.
- **Label Studio 404 when serving local files** — check `LOCAL_FILES_SERVING_ENABLED=true` and `LOCAL_FILES_DOCUMENT_ROOT=/data` in `docker-compose.yml`, and ensure a `LocalFilesImportStorage` record exists in the project's source storage.
- **CLIP model download failures on macOS** — already handled by baking the weights into the Airflow image at build time. If you change models, update `airflow/Dockerfile`.
- **DAG parse errors for `torch` / `transformers`** — these are imported lazily inside task functions on purpose; do not move them to the top of the DAG file.
