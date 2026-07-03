-- NVIDIA Forum Database Schema
-- PostgreSQL 16 + pgvector 0.8.0

CREATE EXTENSION IF NOT EXISTS vector;

-- Site metadata
CREATE TABLE IF NOT EXISTS site_metadata (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Forum categories
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255),
    description TEXT,
    parent_category_id INTEGER,
    topic_count INTEGER DEFAULT 0,
    post_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Forum topics
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    title TEXT NOT NULL,
    slug VARCHAR(255),
    post_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    last_posted_at TIMESTAMP,
    created_by VARCHAR(255),
    tags TEXT[] DEFAULT '{}'
);

-- Forum posts
CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id),
    post_number INTEGER,
    username VARCHAR(255),
    raw TEXT,
    cooked TEXT,
    like_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Post embeddings (pgvector)
CREATE TABLE IF NOT EXISTS post_embeddings (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id),
    embedding VECTOR(1024),
    model VARCHAR(255) DEFAULT 'qwen3-embedding-0.6b',
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(post_id, model)
);

-- Index for cosine similarity search
CREATE INDEX IF NOT EXISTS idx_post_embeddings_vector
    ON post_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for filtering by model
CREATE INDEX IF NOT EXISTS idx_post_embeddings_model
    ON post_embeddings(model);
