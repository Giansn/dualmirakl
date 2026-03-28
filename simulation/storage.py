"""
DuckDB storage layer for dualmirakl.

Provides persistent, queryable storage for:
  - GraphRAG entities and relations (Enhancement 1)
  - Agent memories across simulation runs (Enhancement 2)
  - Generated personas cache (Enhancement 3)
  - Post-sim analysis reports (Enhancement 4)

DuckDB is embedded (no server), supports native FLOAT[384] arrays for
e5-small-v2 embeddings, and runs on CPU alongside the simulation.

Usage:
    from simulation.storage import get_db, ensure_schema

    db = get_db()                    # singleton connection
    ensure_schema(db)                # create tables if missing
    db.execute("INSERT INTO ...", [...])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get(
    "DUALMIRAKL_DB",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dualmirakl.duckdb"),
)

_connection = None


def get_db(path: Optional[str] = None):
    """
    Get or create a singleton DuckDB connection.

    The database file lives in data/dualmirakl.duckdb by default.
    Set DUALMIRAKL_DB env var to override.
    """
    global _connection
    if _connection is not None:
        return _connection

    import duckdb

    db_path = path or _DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _connection = duckdb.connect(db_path)
    ensure_schema(_connection)
    logger.info("DuckDB connected: %s", db_path)
    return _connection


def close_db() -> None:
    """Close the singleton connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def get_memory_db():
    """Get an in-memory DuckDB connection (for tests)."""
    import duckdb
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def ensure_schema(conn) -> None:
    """Create all tables if they don't exist."""

    # ── GraphRAG entities ─────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id          VARCHAR PRIMARY KEY,
            name        VARCHAR NOT NULL,
            type        VARCHAR NOT NULL,
            properties  JSON,
            embedding   FLOAT[384],
            source_doc  VARCHAR,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ── GraphRAG relations ────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id          VARCHAR PRIMARY KEY,
            src_entity  VARCHAR NOT NULL,
            tgt_entity  VARCHAR NOT NULL,
            rel_type    VARCHAR NOT NULL,
            weight      FLOAT DEFAULT 1.0,
            context     VARCHAR,
            embedding   FLOAT[384],
            source_doc  VARCHAR,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ── Agent memories (persistent across runs) ───────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_memories (
            id          VARCHAR PRIMARY KEY,
            run_id      VARCHAR NOT NULL,
            agent_id    VARCHAR NOT NULL,
            tick        INTEGER NOT NULL,
            memory_type VARCHAR DEFAULT 'episodic',
            title       VARCHAR NOT NULL,
            content     VARCHAR NOT NULL,
            tags        VARCHAR[],
            embedding   FLOAT[384],
            importance  FLOAT DEFAULT 1.0,
            decay_rate  FLOAT DEFAULT 0.05,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ── Generated personas cache ──────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generated_personas (
            id              VARCHAR PRIMARY KEY,
            scenario_name   VARCHAR NOT NULL,
            source_doc_hash VARCHAR,
            archetype_id    VARCHAR,
            identity        VARCHAR NOT NULL,
            behavior_rules  VARCHAR NOT NULL,
            emotional_range VARCHAR NOT NULL,
            knowledge_bounds VARCHAR NOT NULL,
            consistency_rules VARCHAR NOT NULL,
            hard_limits     VARCHAR NOT NULL,
            properties      JSON,
            created_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ── Analysis reports ──────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_reports (
            id          VARCHAR PRIMARY KEY,
            run_id      VARCHAR NOT NULL,
            report      JSON NOT NULL,
            questions   JSON,
            n_steps     INTEGER,
            tools_used  JSON,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # ── Experiment tracking (Phase A) ────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id  VARCHAR PRIMARY KEY,
            name           VARCHAR NOT NULL,
            description    VARCHAR DEFAULT '',
            config         JSON,
            model_version  VARCHAR,
            git_hash       VARCHAR,
            llm_model      VARCHAR,
            llm_version    VARCHAR,
            created_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id            VARCHAR PRIMARY KEY,
            experiment_id     VARCHAR,
            parameters        JSON,
            sim_seed          INTEGER,
            llm_seed          INTEGER,
            temperature       FLOAT DEFAULT 0.7,
            status            VARCHAR DEFAULT 'running',
            wall_time_seconds FLOAT,
            created_at        TIMESTAMP DEFAULT current_timestamp
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tick_data (
            run_id      VARCHAR NOT NULL,
            step        INTEGER NOT NULL,
            metric_name VARCHAR NOT NULL,
            value       FLOAT NOT NULL,
            agent_id    VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_summaries (
            experiment_id  VARCHAR NOT NULL,
            step           INTEGER NOT NULL,
            metric_name    VARCHAR NOT NULL,
            n_runs         INTEGER NOT NULL,
            mean           FLOAT,
            std            FLOAT,
            p5             FLOAT,
            p25            FLOAT,
            median         FLOAT,
            p75            FLOAT,
            p95            FLOAT,
            var_aleatory   FLOAT,
            var_epistemic  FLOAT,
            var_llm        FLOAT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            prompt_hash    VARCHAR PRIMARY KEY,
            model_id       VARCHAR NOT NULL,
            temperature    FLOAT NOT NULL,
            seed           INTEGER,
            response_text  VARCHAR NOT NULL,
            created_at     TIMESTAMP DEFAULT current_timestamp,
            hit_count      INTEGER DEFAULT 0
        )
    """)

    # ── Prompt versioning (Phase B) ──────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_versions (
            prompt_hash  VARCHAR PRIMARY KEY,
            prompt_text  VARCHAR NOT NULL,
            agent_type   VARCHAR,
            first_seen   TIMESTAMP DEFAULT current_timestamp,
            run_id       VARCHAR
        )
    """)

    logger.debug("DuckDB schema ensured")


# ── Vector utilities ──────────────────────────────────────────────────────

def cosine_similarity_sql(embedding_col: str, query_embedding: list[float]) -> str:
    """
    Build a DuckDB SQL expression for cosine similarity.

    DuckDB supports list_dot_product and list_cosine_similarity natively
    since v0.10. We use list_cosine_similarity for cleaner code.
    """
    # DuckDB's list_cosine_similarity returns 1 - cosine_distance
    # so higher = more similar (range 0 to 1 for normalized vectors)
    vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
    return f"list_cosine_similarity({embedding_col}, {vec_str}::FLOAT[384])"


def vector_search(
    conn,
    table: str,
    embedding_col: str,
    query_embedding: list[float],
    top_k: int = 5,
    threshold: float = 0.5,
    where_clause: str = "",
) -> list[dict]:
    """
    Perform a vector similarity search on a DuckDB table.

    Returns list of dicts with all columns plus a 'similarity' column,
    ordered by descending similarity, filtered by threshold.
    """
    sim_expr = cosine_similarity_sql(embedding_col, query_embedding)
    where = f"WHERE {where_clause} AND" if where_clause else "WHERE"

    sql = f"""
        SELECT *, {sim_expr} AS similarity
        FROM {table}
        {where} {sim_expr} >= {threshold}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """
    result = conn.execute(sql).fetchdf()
    return result.to_dict("records") if len(result) > 0 else []
