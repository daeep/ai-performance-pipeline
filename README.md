# AI Performance Pipeline

End-to-end AI pipeline: web scraping → PostgreSQL + pgvector → LLM embeddings → semantic search.

Built to process NVIDIA Developer Forum posts (DGX Spark category) into a searchable knowledge base.

---

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Discourse   │────▶│  PostgreSQL     │────▶│  Embedding API   │
│  Forum       │     │  + pgvector     │     │  Qwen3-0.6B      │
│  (Source)    │     │  (Storage)      │     │  (Vectorize)     │
└──────────────┘     └────────┬────────┘     └────────┬─────────┘
                              │                        │
                              ▼                        ▼
                     ┌────────────────────────────────────┐
                     │         SEMANTIC SEARCH            │
                     │  "How to optimize vLLM on DGX?"    │
                     │  → cosine similarity via pgvector  │
                     │  → ranked relevant posts           │
                     └────────────────────────────────────┘
```

---

## Pipeline Stages

### 1. Scraping (Chronicon)
- **Tool:** [Chronicon](https://github.com/19-84/chronicon) — incremental Discourse archiver
- **Source:** NVIDIA Developer Forum, DGX Spark category
- **Method:** Category-scoped, incremental (only new/updated posts)
- **Schedule:** Daily CronJob in Kubernetes

### 2. Storage (PostgreSQL + pgvector)
- **Database:** CNPG PostgreSQL 16
- **Extension:** pgvector v0.8.0
- **Schema:**

```sql
-- Posts table (from Chronicon)
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id),
    post_number INTEGER,
    raw_text TEXT,
    cooked_html TEXT,
    created_at TIMESTAMP,
    username VARCHAR(255)
);

-- Embeddings table
CREATE TABLE post_embeddings (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id),
    embedding VECTOR(1024),
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast similarity search
CREATE INDEX ON post_embeddings USING ivfflat (embedding vector_cosine_ops);
```

### 3. Embedding Generation
- **Model:** Qwen3-Embedding-0.6B (1024-dimensional vectors)
- **API:** OpenAI-compatible, self-hosted on GMKtec Evo X2
- **Process:**
  1. Fetch unprocessed posts (batch of 500)
  2. Generate summary + embedding for each post
  3. Insert into `post_embeddings` with ON CONFLICT skip
  4. Track progress with `last_processed_id`

### 4. Semantic Search
- **Query:** Natural language question
- **Method:** Cosine similarity via pgvector `<=>` operator
- **Result:** Top-K most relevant posts with summaries

---

## Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nvidia-forum-embeddings
spec:
  schedule: "0 6 * * *"  # Daily at 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: embedder
            image: python:3.11-slim
            command: ["python", "/scripts/embed.py"]
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: nvidia-forum-cnpg-secret
                  key: database_url
            - name: EMBEDDING_API_URL
              value: "<embedding-api-endpoint>"
            - name: EMBEDDING_MODEL
              value: "qwen3-embedding-0.6b"
            - name: BATCH_SIZE
              value: "500"
```

---

## Metrics

| Metric | Value |
|--------|-------|
| **Topics scraped** | 1,551 |
| **Posts scraped** | 18,611 |
| **Embeddings generated** | 19,133 |
| **Embedding model** | Qwen3-Embedding-0.6B |
| **Vector dimension** | 1024 |
| **Processing schedule** | Daily at 6:00 AM |
| **Batch size** | 500 posts/run |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Scraper | Chronicon (Go) — Discourse API |
| Database | PostgreSQL 16 + pgvector 0.8.0 |
| Embeddings | Qwen3-Embedding-0.6B (self-hosted) |
| Orchestration | Kubernetes CronJob |
| GitOps | Flux CD |
| Secrets | SOPS + Age encryption |

---

## Related Repos

- [homelab-ai-platform](https://github.com/daeep/homelab-ai-platform) — infrastructure running this pipeline
- [hermes-ai-agents](https://github.com/daeep/hermes-ai-agents) — agents that built and maintain this pipeline
