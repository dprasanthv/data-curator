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
                            └──────────────────┬───────────────────┘
                                               │
                                               ▼
                            ┌──────────────────────────────────────┐
                            │      Qdrant vector DB (cosine)       │
                            │  • facebook_text_events  (384-d)     │
                            │  • image_classification  (512-d)     │
                            └──────────────────┬───────────────────┘
                                               │
                                               ▼
                            ┌──────────────────────────────────────┐
                            │   Streamlit search app (port 8501)   │
                            │  • text  → MiniLM → text events     │
                            │  • image → CLIP   → similar images   │
                            │  • text  → CLIP   → cross-modal      │
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
| **Qdrant** | Vector DB for similarity search |
| **Streamlit** | Search UI (text / image / cross-modal) |
| **Kafbat UI** | Kafka topic browser |

## Project layout

```
data-curator/
├── docker-compose.yml          # all services
├── producer/                   # Faker → Kafka producer
│   ├── producer.py
│   ├── Dockerfile
│   └── requirements.txt
├── search-app/                 # Streamlit vector search UI
│   ├── app.py                  # 3 tabs: text / image / cross-modal
│   ├── Dockerfile              # CPU-only torch + bakes in MiniLM + CLIP
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
│       ├── embeddings_to_vector_db.py     # gold Parquet → Qdrant
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

## How data moves between components

This section traces a single event end-to-end so you can see what actually happens at each hop and what shape the data takes.

### Hop 1 — Producer → Kafka

The `producer` container generates a fake event with `Faker` and publishes it as JSON to the `facebook.events` Kafka topic.

```json
{
  "event_id": "8f3c…",
  "event_type": "post",
  "created_at": "2026-04-27T20:51:11Z",
  "actor": { "user_id": "u-42", "name": "Alice Smith" },
  "post": { "post_id": "p-99", "text": "Just deployed my first DAG!" }
}
```

**Transport:** Kafka (KRaft single-node, port 9092 internally, 9092 published to host).
**Inspect:** Kafbat UI → topic `facebook.events` → Messages.

### Hop 2 — Kafka → Bronze (Parquet on disk)

DAG `kafka_to_datalake_bronze` runs every few minutes. It opens a `KafkaConsumer`, drains a batch of messages, and writes them as a single Snappy-compressed Parquet file partitioned by date and hour.

```
datalake/bronze/facebook_events/
└── dt=2026-04-27/
    └── hour=20/
        └── events-20260427T205111-<uuid>.parquet
```

Each row preserves the original JSON structure (nested `actor`, `post`, etc.). The Parquet schema is inferred by PyArrow from the dict.

**Why Parquet?** Columnar, compressed, fast to scan from downstream tools without re-decoding JSON.

### Hop 3a — Bronze → Label Studio (text track, optional)

DAG `parquet_to_label_studio` reads recent bronze files and converts each event into a Label Studio task. The `text` field is built per `event_type`:

| `event_type` | `text` value |
|---|---|
| `post` | `post.text` |
| `comment` | `comment.text` |
| `reaction` | `"reacted with <reaction_type> on <target_type>"` |
| `follow` | `"followed user <followed_user_name>"` |
| `user` | `"user registered: <name>"` |

Tasks are written first as JSONL (one task per line) into `datalake/label-studio/tasks/`, then POSTed to Label Studio's `/api/projects/{id}/import` endpoint.

```jsonl
{"data": {"text": "Just deployed my first DAG!", "event_type": "post", "event_id": "8f3c…", ...}}
```

A human then opens Label Studio → "Facebook Events" project and clicks `safe` / `spam` / `hate` / `harassment` for each task.

### Hop 3b — Bronze → Gold text embeddings (no human in the loop)

DAG `text_embeddings` reads bronze Parquet **directly** (skipping Label Studio entirely), assigns random labels, and embeds. Two tasks:

1. **`load_and_label`** — globs all `bronze/facebook_events/**/*.parquet`, extracts the `text` field per `event_type`, attaches a random label, and caches the records as JSON in `datalake/tmp/`.
2. **`generate_embeddings`** — for each batch of 64 texts:
   - **Tokenize:** WordPiece → `input_ids` of shape `[64, ≤128]`
   - **Forward pass:** `AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` → token embeddings `[64, seq_len, 384]`
   - **Mean-pool** across the sequence dimension (ignoring `[PAD]`) → `[64, 384]`
   - **L2-normalise** → unit vectors
   - **Append** to a Parquet file via `pq.ParquetWriter`, flushing every 2,000 rows to keep memory bounded.

Final output:

```
datalake/gold/text_embeddings/embeddings_<UTC-timestamp>.parquet
```

### Hop 4 — Image track (LoremFlickr → Label Studio → CLIP)

DAG `image_pipeline`:

1. Downloads images from LoremFlickr into `datalake/images/{building,people,car}/`.
2. Creates a Label Studio project with image classification config.
3. Imports each image as a task whose `data.image` URL points at Label Studio's local files endpoint:
   ```
   /data/local-files/?d=images/building/building_0001.jpg
   ```
   Label Studio resolves `?d=…` against `LOCAL_FILES_DOCUMENT_ROOT=/data`, which is the bind-mounted `./datalake` folder.

A human annotates the images. Then DAG `image_embeddings`:

1. **`export_annotations`** — calls `/api/projects/{id}/export?exportType=JSON`, keeps only annotated tasks, caches them in `datalake/tmp/ls_annotations.json`.
2. **`generate_embeddings`** — for each batch:
   - Reverse the `?d=…` URL back into a real disk path.
   - Open with PIL, convert to RGB.
   - `CLIPProcessor` → resize to 224×224 + normalise.
   - `CLIPModel.get_image_features()` → `[batch, 512]`.
   - L2-normalise.
   - Write a single Parquet file with `task_id`, `label`, `image_path`, `embedding`.

### Hop 5 — Gold Parquet → Qdrant (vector DB)

DAG `embeddings_to_vector_db` reads the **latest** Parquet file from each gold folder and upserts the vectors into Qdrant. Two parallel tasks:

| Task | Source | Qdrant collection | Vector dim |
|---|---|---|---|
| `index_text_embeddings`  | `gold/text_embeddings/embeddings_*.parquet`  | `facebook_text_events`  | 384 |
| `index_image_embeddings` | `gold/image_embeddings/embeddings_*.parquet` | `image_classification`  | 512 |

For each row a Qdrant `PointStruct` is built with:

- **`id`** — deterministic UUIDv5 derived from `event_id` / `task_id` (so re-runs upsert instead of duplicating)
- **`vector`** — the embedding
- **`payload`** — full row metadata (label, text/image_path, actor, etc.) so you can filter at query time

Points are upserted in batches of 256. Both collections use **cosine distance** (matches the L2-normalised vectors).

**Example query** — find the 5 events most similar to a given embedding, filtered to `label=spam`:

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

client = QdrantClient(url="http://localhost:6333")
hits = client.search(
    collection_name="facebook_text_events",
    query_vector=my_embedding,           # 384-d list/np.array
    limit=5,
    query_filter=Filter(must=[
        FieldCondition(key="label", match=MatchValue(value="spam"))
    ]),
)
for h in hits:
    print(h.score, h.payload["text"])
```

### Volume mounts that make all of this work

The same `./datalake` folder on your host is mounted into multiple containers:

| Mount target | Container | Why |
|---|---|---|
| `/opt/airflow/datalake` | `airflow-webserver`, `airflow-scheduler`, `airflow-init` | DAGs read/write bronze + gold Parquet here |
| `/data` | `label-studio` | Label Studio serves images from here via `LOCAL_FILES_DOCUMENT_ROOT` |

Because both containers see the same physical files, an image written by Airflow's `image_pipeline` is immediately visible to Label Studio without any copy step.

### Hop 6 — Qdrant → Streamlit search app

The `search-app` container is a self-contained Streamlit UI on port **8501**. On startup it loads two HuggingFace models that were baked into its image at build time:

- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) for text → 384-d
- **CLIP** (`openai/clip-vit-base-patch32`) for image → 512-d *and* text → 512-d

It offers three search modes:

| Tab | Input | Encoder | Collection queried |
|---|---|---|---|
| Text events | text query | MiniLM | `facebook_text_events` |
| Image-by-image | image upload | CLIP image encoder | `image_classification` |
| Cross-modal | text query | CLIP **text** encoder | `image_classification` |

The cross-modal tab is the most magical bit: because CLIP's text and image encoders share the same 512-d space, you can type *"a tall building at sunset"* and Qdrant returns visually matching images — no labels required.

The datalake is mounted into the search app at `/data` (read-only) so the UI can render the actual JPG files referenced by `image_path` in the Qdrant payload.

## Prerequisites

- Docker Desktop (4 GB RAM minimum, 8 GB recommended — model loading is heavy)
- ~2 GB free disk for Docker images (CLIP weights are baked in at build time)

## Running it — full data creation walkthrough

This is the complete sequence to go from a fresh clone to populated bronze + gold layers.

### Step 1 — Build & start everything

```bash
docker-compose up -d --build
```

First build takes 5–10 minutes — it installs PyTorch and pre-downloads the CLIP model weights into the Airflow image (this avoids macOS Docker volume rename bugs at runtime).

Wait for all containers to be healthy:

```bash
docker-compose ps
```

You should see `kafka`, `producer`, `airflow-postgres`, `airflow-init` (exited 0), `airflow-webserver`, `airflow-scheduler`, `kafbat`, `label-studio` all running.

### Step 2 — Open the UIs

| Service | URL | Credentials |
|---|---|---|
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **Label Studio** | http://localhost:8081 | `admin@datacurator.local` / `admin` |
| **Kafbat (Kafka UI)** | http://localhost:8082 | none |
| **Qdrant Web UI** | http://localhost:6333/dashboard | none |
| **Search app (Streamlit)** | http://localhost:8501 | none |

### Step 3 — Generate Kafka events (automatic)

The `producer` container starts immediately and emits one synthetic Facebook event every 250 ms to topic `facebook.events`. Each event is one of:

- **`post`** — user creates a post with text
- **`comment`** — user comments on a post
- **`reaction`** — user reacts (like / love / haha / ...) to a post or comment
- **`follow`** — user follows another user
- **`user`** — new user registration

**Verify:** open Kafbat → topic `facebook.events` → Messages tab. You should see events arriving in real time.

### Step 4 — Land events into the bronze datalake

The `kafka_to_datalake_bronze` DAG runs on a schedule, consuming from Kafka and writing partitioned Parquet:

```
datalake/bronze/facebook_events/dt=YYYY-MM-DD/hour=HH/events-*.parquet
```

It's enabled by default and runs every few minutes. To force-run it once:

```bash
docker-compose exec airflow-scheduler airflow dags test kafka_to_datalake_bronze 2024-01-01
```

**Verify:**

```bash
ls datalake/bronze/facebook_events/dt=*/hour=*/ | head
```

After 5 minutes of producer runtime you should have ~1,000+ events spread across multiple Parquet files.

### Step 5 — Push events into Label Studio (optional, for manual annotation)

If you want to label events in the UI instead of randomly via script, run:

```bash
docker-compose exec airflow-scheduler airflow dags test parquet_to_label_studio 2024-01-01
```

This DAG:
1. Reads recent bronze Parquet files
2. Converts each event to a Label Studio task JSONL line
3. Creates (or finds) the "Facebook Events" project with `safe / spam / hate / harassment` labels
4. POSTs the tasks to Label Studio

**Verify:** open Label Studio → "Facebook Events" project. You should see tasks queued for annotation.

### Step 6 — Download images for the image track

```bash
docker-compose exec airflow-scheduler airflow dags test image_pipeline 2024-01-01
```

This DAG:
1. Downloads images from LoremFlickr into `datalake/images/{building, people, car}/`
2. Creates a Label Studio project "Image Classification: Building / People / Car"
3. Imports each image as a task pointing to its local file URL

**Verify:**

```bash
ls datalake/images/building | head
ls datalake/images/people | head
ls datalake/images/car | head
```

Then open Label Studio and **annotate at least a few images** (the image embedding DAG only embeds annotated ones).

### Step 7 — Generate text embeddings (gold layer)

This DAG bypasses Label Studio entirely — it reads bronze Parquet directly, assigns random labels, tokenizes (BERT WordPiece), and embeds with MiniLM:

```bash
docker-compose exec airflow-scheduler airflow dags test text_embeddings 2024-01-01
```

For ~25k events on CPU this takes ~10 minutes. Output:

```
datalake/gold/text_embeddings/embeddings_<UTC-timestamp>.parquet
```

**Verify:**

```bash
ls -lh datalake/gold/text_embeddings/
```

### Step 8 — Generate image embeddings (gold layer)

This DAG exports annotations from Label Studio and runs CLIP on each annotated image:

```bash
docker-compose exec airflow-scheduler airflow dags test image_embeddings 2024-01-01
```

Output:

```
datalake/gold/image_embeddings/embeddings_<UTC-timestamp>.parquet
```

**Verify:**

```bash
ls -lh datalake/gold/image_embeddings/
```

### Step 9 — Index embeddings into Qdrant

Upserts the latest gold Parquet for both tracks into the Qdrant vector DB:

```bash
docker-compose exec airflow-scheduler airflow dags test embeddings_to_vector_db 2024-01-01
```

**Verify:** open http://localhost:6333/dashboard and you should see two collections:

- `facebook_text_events` — N points × 384 dim
- `image_classification` — N points × 512 dim

Or from the CLI:

```bash
curl -s http://localhost:6333/collections | python -m json.tool
```

### Step 10 — Query the vectors via the search UI

Open http://localhost:8501 in your browser. Three tabs:

- **Text events** — type a query like `"buy cheap pills"`, optionally filter by label
- **Image-by-image** — upload a photo, get the most visually similar images
- **Cross-modal** — type a description (e.g. `"a tall building at sunset"`) and CLIP retrieves matching images even though the index has no captions

If you don't see results, make sure Step 9 ran successfully and that Qdrant has points in both collections.

### Quick recap — DAG dependency map

```
producer  ──► kafka_to_datalake_bronze  ──► text_embeddings  ──┐
                          │                                    │
                          └──────────────► parquet_to_label_studio
                                                               ▼
image_pipeline  ──► [manual annotation]  ──► image_embeddings ──► embeddings_to_vector_db  ──►  Qdrant
```

You can run the text track end-to-end without any human labelling. The image track requires you to click through at least a few images in Label Studio.

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
docker-compose down -v             # also delete Kafka + Postgres + Qdrant volumes
rm -rf datalake/bronze datalake/gold datalake/label-studio  # wipe datalake
```

## Troubleshooting

- **OOM during embeddings** — `text_embeddings.py` flushes to Parquet every 2,000 records via `pq.ParquetWriter` to keep memory bounded. If you still hit OOM, lower `BATCH_SIZE` in the DAG or give Docker more memory.
- **Label Studio 404 when serving local files** — check `LOCAL_FILES_SERVING_ENABLED=true` and `LOCAL_FILES_DOCUMENT_ROOT=/data` in `docker-compose.yml`, and ensure a `LocalFilesImportStorage` record exists in the project's source storage.
- **CLIP model download failures on macOS** — already handled by baking the weights into the Airflow image at build time. If you change models, update `airflow/Dockerfile`.
- **DAG parse errors for `torch` / `transformers`** — these are imported lazily inside task functions on purpose; do not move them to the top of the DAG file.
