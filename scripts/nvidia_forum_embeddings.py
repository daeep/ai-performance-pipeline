#!/usr/bin/env python3
"""
nvidia-forum-embeddings: Generate pgvector embeddings for NVIDIA Forum posts.

Process:
1. Reads posts from `nvidia_forum` DB that don't have embeddings yet
2. Strips HTML from post content, combines with topic title
3. Sends batches to an OpenAI-compatible embedding endpoint (Spark's vLLM)
4. Stores vectors in `post_embeddings` table

Runs as a K8s CronJob. Idempotent — skips already-embedded posts.
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
import ssl
from datetime import datetime, timezone

# ── Configuration ────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "")
EMBEDDING_API_URL = os.environ.get(
    "EMBEDDING_API_URL",
    "http://<internal-ip>:19701/v1/embeddings",
)
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))  # 5 min between runs
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

# ── Helpers ──────────────────────────────────────────────────────────────────

def strip_html(html: str) -> str:
    """Remove HTML tags, decode common entities, collapse whitespace."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]  # cap at 2000 chars per chunk


def embed_batch(texts: list[str]) -> list[list[float]] | None:
    """Send a batch of texts to the embedding API. Returns list of vectors."""
    payload = {
        "input": texts,
        "model": EMBEDDING_MODEL,
    }
    req = urllib.request.Request(
        EMBEDDING_API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Disable SSL verification for internal endpoint
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                result = json.loads(resp.read().decode())
                data = result.get("data", [])
                # Sort by index to preserve order
                data.sort(key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in data]
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, KeyError, OSError) as exc:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  API call failed (attempt {attempt}/{MAX_RETRIES}): "
                      f"{exc}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  API call failed after {MAX_RETRIES} attempts: "
                      f"{exc}", flush=True)
                return None
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print(f"[{datetime.now(timezone.utc).isoformat()}] "
          f"NVIDIA Forum Embedding Pipeline starting", flush=True)

    if not DATABASE_URL:
        print("FATAL: DATABASE_URL not set", flush=True)
        sys.exit(1)

    if DRY_RUN:
        print("DRY RUN — no writes will be made", flush=True)

    # ── Connect ──────────────────────────────────────────────────────────
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("FATAL: psycopg2 not installed. "
              "Install with: pip install psycopg2-binary", flush=True)
        sys.exit(1)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        print(f"  DB connected", flush=True)
    except Exception as exc:
        print(f"FATAL: cannot connect to DB: {exc}", flush=True)
        sys.exit(1)

    # ── Check embedding API health ───────────────────────────────────────
    try:
        # Quick check — does the endpoint respond?
        req = urllib.request.Request(
            EMBEDDING_API_URL.replace("/embeddings", "/models"),
            method="GET",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            models = json.loads(resp.read().decode())
            available = [m["id"] for m in models.get("data", [])]
            print(f"  Embedding API ready. Available models: "
                  f"{len(available)}", flush=True)
    except Exception as exc:
        print(f"WARN: Embedding API not reachable ({exc}). "
              f"Posts will be queued until API is available.", flush=True)

    # ── Find posts without embeddings ────────────────────────────────────
    cur.execute("""
        SELECT p.id, p.topic_id, p.post_number, p.cooked, p.raw,
               t.title AS topic_title, p.created_at
        FROM posts p
        JOIN topics t ON t.id = p.topic_id
        LEFT JOIN post_embeddings pe
            ON pe.post_id = p.id
            AND pe.model = %s
        WHERE pe.id IS NULL
        ORDER BY p.created_at ASC
        LIMIT 500
    """, (EMBEDDING_MODEL,))

    rows = cur.fetchall()
    if not rows:
        print("  No pending posts found — all up to date.", flush=True)
        cur.close()
        conn.close()
        print(f"  Done in {time.time() - start:.1f}s", flush=True)
        return

    total = len(rows)
    print(f"  Found {total} posts to embed", flush=True)
    embedded = 0
    errors = 0

    # ── Process in batches ───────────────────────────────────────────────
    for offset in range(0, total, BATCH_SIZE):
        batch = rows[offset:offset + BATCH_SIZE]
        texts = []
        post_ids = []

        for row in batch:
            # Build text from topic title + post content
            title = row["topic_title"] or ""
            content = strip_html(row["cooked"] or row["raw"] or "")
            if not content:
                continue
            text = f"{title}\n\n{content}" if title else content
            texts.append(text)
            post_ids.append(row["id"])

        if not texts:
            continue

        print(f"  Batch {offset // BATCH_SIZE + 1}/"
              f"{(total + BATCH_SIZE - 1) // BATCH_SIZE}: "
              f"{len(texts)} posts...", flush=True)

        vectors = embed_batch(texts)
        if vectors is None:
            errors += len(texts)
            print(f"    SKIPPED — API unavailable, will retry next run",
                  flush=True)
            continue

        if len(vectors) != len(post_ids):
            print(f"    MISMATCH: got {len(vectors)} vectors for "
                  f"{len(post_ids)} texts. Skipping batch.", flush=True)
            errors += len(post_ids)
            continue

        if DRY_RUN:
            print(f"    DRY RUN — would insert {len(vectors)} embeddings",
                  flush=True)
            embedded += len(vectors)
            continue

        # Insert embeddings in bulk
        values = []
        params = []
        for pid, vec in zip(post_ids, vectors):
            values.append(
                "(%s, %s::vector, %s, NOW())"
            )
            params.extend([pid, json.dumps(vec), EMBEDDING_MODEL])

        if values:
            sql = (
                "INSERT INTO post_embeddings "
                "(post_id, embedding, model, created_at) VALUES "
                + ", ".join(values) +
                " ON CONFLICT (post_id, model) DO NOTHING"
            )
            try:
                cur.execute(sql, params)
                embedded += len(vectors)
                print(f"    ✓ {len(vectors)} embeddings stored", flush=True)
            except Exception as exc:
                print(f"    DB INSERT FAILED: {exc}", flush=True)
                errors += len(vectors)

        # Brief pause between batches
        if offset + BATCH_SIZE < total:
            time.sleep(1)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n=== Summary ===", flush=True)
    print(f"  Processed: {total} posts", flush=True)
    print(f"  Embedded:  {embedded}", flush=True)
    print(f"  Errors:    {errors}", flush=True)
    print(f"  Duration:  {elapsed:.1f}s", flush=True)

    cur.close()
    conn.close()

    if errors == total:
        print("WARN: All batches failed — embedding API may be down", flush=True)
        sys.exit(0)  # Don't alert, next run will retry


if __name__ == "__main__":
    main()
