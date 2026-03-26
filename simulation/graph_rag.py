"""
GraphRAG — Document → Knowledge Graph → Context Injection for dualmirakl.

Pipeline:
  1. Document chunks (from gateway POST /v1/documents) →
  2. Authority-slot entity/relation extraction (batch, JSON-mode) →
  3. DuckDB storage (entities + relations with embeddings) →
  4. Context injection into sim_loop (graph triples filtered by similarity)
  5. Seed into graph_memory.py before tick 0

Usage:
    from simulation.graph_rag import extract_graph, query_graph_context

    # During document upload (gateway.py)
    entities, relations = await extract_graph(chunks, embed_fn, doc_name)

    # Before simulation start (sim_loop.py)
    context = query_graph_context(scenario_context, embed_fn, top_k=20)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

# Maximum chunks to batch per LLM call for entity extraction
EXTRACTION_BATCH_SIZE = 5


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Entity:
    id: str
    name: str
    type: str
    properties: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
        }


@dataclass
class Relation:
    id: str
    source: str      # entity name
    target: str      # entity name
    rel_type: str
    context: str = ""
    weight: float = 1.0
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "rel_type": self.rel_type,
            "context": self.context,
            "weight": self.weight,
        }


# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = (
    "You are a knowledge extraction specialist. Given text chunks from a domain "
    "document, extract all meaningful entities (people, concepts, organizations, "
    "processes, metrics, conditions) and the relationships between them.\n\n"
    "You MUST output ONLY valid JSON — no markdown fences, no explanation."
)

EXTRACTION_USER_TEMPLATE = """Extract entities and relationships from these text chunks.

OUTPUT FORMAT — a single JSON object:
{{
  "entities": [
    {{"name": "<canonical name>", "type": "<person|concept|organization|process|metric|condition|other>", "properties": {{}}}},
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>", "type": "<verb or relationship>", "context": "<1-sentence context>"}},
  ]
}}

RULES:
- Entity names must be canonical (e.g., "Social Work" not "social work" or "SW").
- Deduplicate entities — if the same concept appears multiple times, list it once.
- Relations must reference entity names that exist in your entities list.
- Extract at least 3 entities and 2 relations per batch (if the text supports it).
- Relation types should be descriptive verbs (e.g., "influences", "requires", "leads_to", "part_of").
- Properties can include any relevant metadata from the text.

TEXT CHUNKS:
---
{chunks_text}
---

Output ONLY the JSON object:"""


def _build_extraction_prompt(chunks: list[str]) -> tuple[str, str]:
    """Build system + user prompt for entity extraction."""
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}] {chunk}" for i, chunk in enumerate(chunks)
    )
    user = EXTRACTION_USER_TEMPLATE.format(chunks_text=chunks_text)
    return EXTRACTION_SYSTEM, user


def _parse_extraction_output(raw: str) -> dict:
    """Parse and validate LLM extraction output."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in the text
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse extraction output as JSON: {e}")
        else:
            raise ValueError(f"No JSON object found in extraction output: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # Ensure required keys
    data.setdefault("entities", [])
    data.setdefault("relations", [])

    return data


# ── Main extraction pipeline ─────────────────────────────────────────────────

async def extract_graph(
    chunks: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    doc_name: str = "unnamed",
    db=None,
) -> tuple[list[Entity], list[Relation]]:
    """
    Extract entities and relations from document chunks using the authority LLM.

    Args:
        chunks: Text chunks from document upload.
        embed_fn: Batch embedding function (e.g., e5-small-v2 model.encode).
        doc_name: Source document name for provenance.
        db: Optional DuckDB connection (if None, uses singleton).

    Returns:
        Tuple of (entities, relations) with embeddings computed.
    """
    import orchestrator

    if db is None:
        from simulation.storage import get_db
        db = get_db()

    all_entities: dict[str, Entity] = {}
    all_relations: list[Relation] = []

    # Process chunks in batches
    for batch_start in range(0, len(chunks), EXTRACTION_BATCH_SIZE):
        batch = chunks[batch_start:batch_start + EXTRACTION_BATCH_SIZE]
        system, user = _build_extraction_prompt(batch)

        try:
            raw_output = await orchestrator.agent_turn(
                agent_id="graph_rag_extractor",
                backend="authority",
                system_prompt=system,
                user_message=user,
                max_tokens=2048,
            )
        except Exception as e:
            logger.warning(
                "GraphRAG extraction failed for batch %d-%d: %s",
                batch_start, batch_start + len(batch), e,
            )
            continue

        try:
            data = _parse_extraction_output(raw_output)
        except ValueError as e:
            logger.warning("GraphRAG parse failed for batch %d: %s", batch_start, e)
            continue

        # Process entities
        for ent_data in data.get("entities", []):
            name = ent_data.get("name", "").strip()
            if not name:
                continue
            ent_type = ent_data.get("type", "other")
            props = ent_data.get("properties", {})

            if name not in all_entities:
                all_entities[name] = Entity(
                    id=str(uuid4()),
                    name=name,
                    type=ent_type,
                    properties=props,
                )
            else:
                # Merge properties
                all_entities[name].properties.update(props)

        # Process relations
        for rel_data in data.get("relations", []):
            source = rel_data.get("source", "").strip()
            target = rel_data.get("target", "").strip()
            rel_type = rel_data.get("type", "related_to")
            context = rel_data.get("context", "")

            if not source or not target:
                continue
            if source not in all_entities or target not in all_entities:
                continue

            all_relations.append(Relation(
                id=str(uuid4()),
                source=source,
                target=target,
                rel_type=rel_type,
                context=context,
            ))

    entities = list(all_entities.values())
    if not entities:
        logger.info("GraphRAG: no entities extracted from '%s'", doc_name)
        return [], []

    # Compute embeddings for entities and relations
    entity_texts = [f"{e.name}: {e.type}" for e in entities]
    entity_vecs = embed_fn(entity_texts)
    for ent, vec in zip(entities, entity_vecs):
        ent.embedding = vec

    if all_relations:
        rel_texts = [f"{r.source} {r.rel_type} {r.target}" for r in all_relations]
        rel_vecs = embed_fn(rel_texts)
        for rel, vec in zip(all_relations, rel_vecs):
            rel.embedding = vec

    # Persist to DuckDB
    _persist_to_db(db, entities, all_relations, doc_name)

    logger.info(
        "GraphRAG: extracted %d entities, %d relations from '%s'",
        len(entities), len(all_relations), doc_name,
    )

    return entities, all_relations


def _persist_to_db(
    db,
    entities: list[Entity],
    relations: list[Relation],
    doc_name: str,
) -> None:
    """Batch insert entities and relations into DuckDB."""
    for ent in entities:
        embedding_list = ent.embedding.tolist() if ent.embedding is not None else None
        db.execute(
            """INSERT OR REPLACE INTO entities (id, name, type, properties, embedding, source_doc)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [ent.id, ent.name, ent.type,
             json.dumps(ent.properties), embedding_list, doc_name],
        )

    for rel in relations:
        embedding_list = rel.embedding.tolist() if rel.embedding is not None else None
        db.execute(
            """INSERT INTO relations (id, src_entity, tgt_entity, rel_type, weight, context, embedding, source_doc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [rel.id, rel.source, rel.target, rel.rel_type,
             rel.weight, rel.context, embedding_list, doc_name],
        )


# ── Context retrieval ────────────────────────────────────────────────────────

def query_graph_context(
    scenario_context: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    top_k: int = 20,
    threshold: float = 0.4,
    db=None,
) -> str:
    """
    Retrieve relevant graph triples for injection into agent prompts.

    Embeds the scenario context, queries DuckDB for similar entities and
    relations, and formats them as compact triples.

    Args:
        scenario_context: The scenario description or world context summary.
        embed_fn: Batch embedding function.
        top_k: Maximum triples to return.
        threshold: Minimum cosine similarity.
        db: Optional DuckDB connection.

    Returns:
        Formatted graph context string for prompt injection, or empty string.
    """
    if db is None:
        from simulation.storage import get_db
        try:
            db = get_db()
        except Exception:
            return ""

    # Check if we have any entities
    count = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    if count == 0:
        return ""

    # Embed the scenario context
    q_vec = embed_fn([scenario_context])[0]
    q_list = q_vec.tolist()

    # Query similar relations
    from simulation.storage import cosine_similarity_sql
    sim_expr = cosine_similarity_sql("embedding", q_list)

    try:
        results = db.execute(f"""
            SELECT src_entity, rel_type, tgt_entity, context,
                   {sim_expr} AS similarity
            FROM relations
            WHERE embedding IS NOT NULL
              AND {sim_expr} >= {threshold}
            ORDER BY similarity DESC
            LIMIT {top_k}
        """).fetchall()
    except Exception as e:
        logger.warning("GraphRAG context query failed: %s", e)
        return ""

    if not results:
        # Fall back to entity listing
        try:
            ent_results = db.execute(f"""
                SELECT name, type,
                       {cosine_similarity_sql("embedding", q_list)} AS similarity
                FROM entities
                WHERE embedding IS NOT NULL
                  AND {cosine_similarity_sql("embedding", q_list)} >= {threshold}
                ORDER BY similarity DESC
                LIMIT {top_k}
            """).fetchall()
        except Exception:
            return ""

        if not ent_results:
            return ""

        lines = ["[GRAPH CONTEXT — key entities from domain documents:]"]
        for name, etype, sim in ent_results:
            lines.append(f"  - {name} ({etype})")
        return "\n".join(lines) + "\n"

    # Format as triples
    lines = ["[GRAPH CONTEXT — knowledge from domain documents:]"]
    for src, rel, tgt, ctx, sim in results:
        triple = f"  {src} → {rel} → {tgt}"
        if ctx:
            triple += f"  ({ctx[:60]})"
        lines.append(triple)

    return "\n".join(lines) + "\n"


def get_graph_entities(db=None) -> list[dict]:
    """Get all entities from the graph for inspection."""
    if db is None:
        from simulation.storage import get_db
        db = get_db()

    rows = db.execute(
        "SELECT id, name, type, properties, source_doc FROM entities ORDER BY name"
    ).fetchall()

    return [
        {"id": r[0], "name": r[1], "type": r[2],
         "properties": json.loads(r[3]) if r[3] else {}, "source_doc": r[4]}
        for r in rows
    ]


def get_graph_relations(db=None) -> list[dict]:
    """Get all relations from the graph for inspection."""
    if db is None:
        from simulation.storage import get_db
        db = get_db()

    rows = db.execute(
        "SELECT id, src_entity, tgt_entity, rel_type, weight, context, source_doc "
        "FROM relations ORDER BY src_entity"
    ).fetchall()

    return [
        {"id": r[0], "source": r[1], "target": r[2], "type": r[3],
         "weight": r[4], "context": r[5], "source_doc": r[6]}
        for r in rows
    ]


def clear_graph(db=None) -> None:
    """Clear all entities and relations."""
    if db is None:
        from simulation.storage import get_db
        db = get_db()
    db.execute("DELETE FROM entities")
    db.execute("DELETE FROM relations")
    logger.info("GraphRAG: cleared all entities and relations")
