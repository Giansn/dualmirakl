"""
LLM response cache and prompt versioning for dualmirakl.

SHA-256 hash of (messages, model_id, temperature, seed) → stored response.
DuckDB-backed with module-level singleton. Enables deterministic replay
and massive cost savings on ensemble runs.

Also provides `compute_prompt_hash()` for prompt versioning — tracks
prompt text identity in event payloads for drift detection.

Toggle: CACHE_ENABLED env var (default "1").

Usage:
    from simulation.response_cache import cache, compute_prompt_hash

    hit = cache.lookup(prompt_text, "swarm", 0.7, seed=42)
    if hit is None:
        response = await call_llm(...)
        cache.store(prompt_text, "swarm", 0.7, 42, response)

    # Prompt versioning
    h = compute_prompt_hash("system prompt + user message")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "1") == "1"


def compute_prompt_hash(prompt_text: str) -> str:
    """SHA-256 of prompt text for versioning (not model/temp/seed — just the prompt).

    Used to tag event payloads with a fingerprint of the prompt that generated
    each LLM response. Enables prompt drift detection across code versions.
    """
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


class ResponseCache:
    """SHA-256 prompt+seed hash cache for LLM responses. DuckDB-backed."""

    def __init__(self, enabled: bool | None = None, db=None):
        self._enabled = enabled if enabled is not None else CACHE_ENABLED
        self._db = db
        self._hits = 0
        self._misses = 0

    @property
    def db(self):
        """Lazy DuckDB connection (same pattern as DuckDBMemoryBackend)."""
        if self._db is None:
            from simulation.storage.db import get_db
            self._db = get_db()
        return self._db

    @staticmethod
    def _hash_key(
        prompt_text: str,
        model_id: str,
        temperature: float,
        seed: int | None,
    ) -> str:
        """SHA-256 of (prompt_text, model_id, temperature, seed)."""
        key_data = json.dumps(
            {"prompt": prompt_text, "model": model_id,
             "temperature": temperature, "seed": seed},
            sort_keys=True, ensure_ascii=True,
        )
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def lookup(
        self,
        prompt_text: str,
        model_id: str,
        temperature: float,
        seed: int | None = None,
    ) -> str | None:
        """Check cache. Returns response_text or None. Increments hit_count on hit."""
        if not self._enabled:
            self._misses += 1
            return None
        h = self._hash_key(prompt_text, model_id, temperature, seed)
        try:
            row = self.db.execute(
                "SELECT response_text FROM response_cache WHERE prompt_hash = ?",
                [h],
            ).fetchone()
            if row is not None:
                self.db.execute(
                    "UPDATE response_cache SET hit_count = hit_count + 1 WHERE prompt_hash = ?",
                    [h],
                )
                self._hits += 1
                return row[0]
        except Exception as e:
            logger.debug("Cache lookup failed: %s", e)
        self._misses += 1
        return None

    def store(
        self,
        prompt_text: str,
        model_id: str,
        temperature: float,
        seed: int | None,
        response_text: str,
    ) -> None:
        """Store a response in cache."""
        if not self._enabled:
            return
        h = self._hash_key(prompt_text, model_id, temperature, seed)
        try:
            self.db.execute(
                """INSERT OR REPLACE INTO response_cache
                   (prompt_hash, model_id, temperature, seed, response_text)
                   VALUES (?, ?, ?, ?, ?)""",
                [h, model_id, temperature, seed, response_text],
            )
        except Exception as e:
            logger.debug("Cache store failed: %s", e)

    @property
    def hit_rate(self) -> float:
        """Current session hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Return {hits, misses, hit_rate, total_cached}."""
        total_cached = 0
        try:
            row = self.db.execute("SELECT COUNT(*) FROM response_cache").fetchone()
            total_cached = row[0] if row else 0
        except Exception:
            pass
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "total_cached": total_cached,
        }

    def clear(self) -> int:
        """Clear all cached responses. Returns count deleted."""
        try:
            row = self.db.execute("SELECT COUNT(*) FROM response_cache").fetchone()
            count = row[0] if row else 0
            self.db.execute("DELETE FROM response_cache")
            self._hits = 0
            self._misses = 0
            return count
        except Exception as e:
            logger.debug("Cache clear failed: %s", e)
            return 0


# Module-level singleton
cache = ResponseCache()
