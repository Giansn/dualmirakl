"""
Ontology Generator — LLM-driven archetype and transition rule generation.

Instead of hand-authoring archetypes and transition rules in scenario.yaml,
this module uses an LLM (via the authority backend) to auto-generate them
from a domain document.  Inspired by MiroFish's approach to deriving
simulation ontologies from unstructured text.

Usage:
    from simulation.ontology_generator import generate_ontology

    result = await generate_ontology(
        document_text="<domain document>",
        scenario_name="my-scenario",
        n_profiles=4,
    )
    # result contains 'archetypes' and 'transitions' dicts ready to merge
    # into a scenario.yaml structure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Transition functions that are registered as built-ins in transitions.py
REGISTERED_FUNCTIONS = frozenset({
    "escalation_sustained",
    "recovery_sustained",
    "threshold_cross",
    "oscillation_detect",
})


def build_prompt(document_text: str, scenario_name: str, n_profiles: int) -> str:
    """
    Construct the system+user prompt that instructs the LLM to produce
    archetype profiles, distribution, and transition rules as valid JSON.
    """
    system = (
        "You are an expert simulation designer. Given a domain document, "
        "you extract the key agent archetypes and behavioural transition "
        "rules for a multi-agent simulation.\n\n"
        "You MUST output ONLY valid JSON — no markdown fences, no explanation, "
        "no commentary before or after the JSON object."
    )

    user = f"""Analyse the following domain document and generate an ontology for a simulation called "{scenario_name}".

Generate exactly {n_profiles} archetype profiles and appropriate transition rules.

OUTPUT FORMAT — a single JSON object with this exact structure:

{{
  "archetypes": {{
    "profiles": [
      {{
        "id": "<short_id like K1, K2, ...>",
        "label": "<human-readable label>",
        "description": "<1-2 sentence description of this archetype>",
        "properties": {{
          "susceptibility": "<low|medium|high>",
          "resilience": "<low|medium|high>"
        }}
      }}
    ],
    "distribution": {{
      "<id>": <float fraction, all must sum to 1.0>
    }}
  }},
  "transitions": [
    {{
      "from": "<profile id>",
      "to": "<profile id>",
      "function": "<one of: escalation_sustained, recovery_sustained, threshold_cross, oscillation_detect>",
      "params": {{<function-specific params>}}
    }}
  ]
}}

RULES:
- Profile IDs must be short identifiers (K1, K2, K3, ...).
- Distribution fractions MUST sum to exactly 1.0.
- Every transition must reference valid profile IDs from the profiles list.
- Transition functions must be one of: escalation_sustained, recovery_sustained, threshold_cross, oscillation_detect.
- escalation_sustained params: threshold (float 0-1), consecutive_ticks (int).
- recovery_sustained params: threshold (float 0-1), consecutive_ticks (int).
- threshold_cross params: threshold (float 0-1), direction ("up" or "down").
- oscillation_detect params: window (int), amplitude (float 0-1).
- Generate at least 2 transition rules.
- Properties must include "susceptibility" and "resilience" keys.

DOMAIN DOCUMENT:
---
{document_text}
---

Output ONLY the JSON object:"""

    return json.dumps({
        "system": system,
        "user": user,
    })


def parse_llm_output(raw_output: str) -> dict:
    """
    Parse and validate the raw LLM output string into a structured dict.
    Handles common LLM quirks like markdown code fences.

    Raises ValueError if the output is not valid JSON or fails structural checks.
    """
    text = raw_output.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}")

    # Structural checks
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    if "archetypes" not in data:
        raise ValueError("Missing 'archetypes' key in LLM output")
    if "transitions" not in data:
        raise ValueError("Missing 'transitions' key in LLM output")

    archetypes = data["archetypes"]
    if "profiles" not in archetypes or not isinstance(archetypes["profiles"], list):
        raise ValueError("archetypes.profiles must be a list")
    if "distribution" not in archetypes or not isinstance(archetypes["distribution"], dict):
        raise ValueError("archetypes.distribution must be a dict")

    return data


def validate_ontology(data: dict) -> list[str]:
    """
    Validate the parsed ontology dict for semantic correctness.
    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    archetypes = data.get("archetypes", {})
    profiles = archetypes.get("profiles", [])
    distribution = archetypes.get("distribution", {})
    transitions = data.get("transitions", [])

    # Collect profile IDs
    profile_ids = set()
    for p in profiles:
        pid = p.get("id", "")
        if not pid:
            errors.append("Profile missing 'id'")
        elif pid in profile_ids:
            errors.append(f"Duplicate profile id: {pid}")
        profile_ids.add(pid)

        if not p.get("label"):
            errors.append(f"Profile '{pid}' missing 'label'")

    # Distribution references valid profiles
    for dist_id in distribution:
        if dist_id not in profile_ids:
            errors.append(
                f"Distribution references undefined profile '{dist_id}'"
            )

    # Distribution sums to 1.0
    if distribution:
        total = sum(distribution.values())
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"Distribution fractions sum to {total:.3f}, expected 1.0"
            )

    # All profiles have distribution entries
    for pid in profile_ids:
        if pid and pid not in distribution:
            errors.append(f"Profile '{pid}' has no distribution entry")

    # Transition rules
    for i, rule in enumerate(transitions):
        from_p = rule.get("from", "")
        to_p = rule.get("to", "")
        fn = rule.get("function", "")

        if from_p not in profile_ids:
            errors.append(
                f"Transition[{i}] 'from' references undefined profile '{from_p}'"
            )
        if to_p not in profile_ids:
            errors.append(
                f"Transition[{i}] 'to' references undefined profile '{to_p}'"
            )
        if fn not in REGISTERED_FUNCTIONS:
            errors.append(
                f"Transition[{i}] uses unregistered function '{fn}'. "
                f"Must be one of: {sorted(REGISTERED_FUNCTIONS)}"
            )

    return errors


async def generate_ontology(
    document_text: str,
    scenario_name: str,
    n_profiles: int = 4,
) -> dict:
    """
    Use the authority LLM backend to analyse a domain document and generate
    archetype profiles, distribution, and transition rules.

    Args:
        document_text: The raw text of the domain document to analyse.
        scenario_name: Name for the scenario (used in prompt context).
        n_profiles: Number of archetype profiles to generate (default 4).

    Returns:
        A dict with 'archetypes' and 'transitions' keys, ready to be merged
        into a scenario.yaml structure.

    Raises:
        ValueError: If the LLM output cannot be parsed or fails validation.
        RuntimeError: If the LLM call itself fails.
    """
    # We import here to avoid import-time side effects (httpx client creation)
    # when the module is imported in test environments.
    import orchestrator

    prompt_data = json.loads(build_prompt(document_text, scenario_name, n_profiles))

    system_prompt = prompt_data["system"]
    user_message = prompt_data["user"]

    try:
        raw_output = await orchestrator.agent_turn(
            agent_id="ontology_generator",
            backend="authority",
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=2048,
        )
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    # Parse the raw output
    data = parse_llm_output(raw_output)

    # Validate
    errors = validate_ontology(data)
    if errors:
        raise ValueError(
            f"Ontology validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        "Generated ontology for '%s': %d profiles, %d transitions",
        scenario_name,
        len(data["archetypes"]["profiles"]),
        len(data["transitions"]),
    )

    return data


# ── Persona generation from Knowledge Graph (Enhancement 3) ──────────────────

from dataclasses import dataclass, field as dataclass_field
from typing import Optional


@dataclass
class PersonaSpec:
    """
    Generated persona specification matching agent_rolesv3.py's
    6-component PARTICIPANT_TEMPLATE structure.
    """
    id: str
    archetype_id: str
    identity: str
    behavior_rules: str
    emotional_range: str
    knowledge_bounds: str
    consistency_rules: str
    hard_limits: str
    properties: dict = dataclass_field(default_factory=dict)

    def to_system_prompt(self) -> str:
        """Convert to a complete participant system prompt."""
        return (
            f"{self.identity}\n\n"
            f"BEHAVIOR RULES:\n{self.behavior_rules}\n\n"
            f"EMOTIONAL RANGE:\n{self.emotional_range}\n\n"
            f"KNOWLEDGE BOUNDS:\n{self.knowledge_bounds}\n\n"
            f"CONSISTENCY RULES:\n{self.consistency_rules}\n\n"
            f"HARD LIMITS:\n{self.hard_limits}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "archetype_id": self.archetype_id,
            "identity": self.identity,
            "behavior_rules": self.behavior_rules,
            "emotional_range": self.emotional_range,
            "knowledge_bounds": self.knowledge_bounds,
            "consistency_rules": self.consistency_rules,
            "hard_limits": self.hard_limits,
            "properties": self.properties,
        }


PERSONA_SYSTEM = (
    "You are an expert simulation designer specializing in creating diverse, "
    "realistic agent personas for multi-agent simulations. You generate "
    "heterogeneous personalities grounded in domain knowledge.\n\n"
    "You MUST output ONLY valid JSON — no markdown fences, no explanation."
)

PERSONA_USER_TEMPLATE = """Generate {n_personas} diverse agent personas for a simulation called "{scenario_name}".

DOMAIN CONTEXT (entities and relationships extracted from domain documents):
{graph_context}

ARCHETYPE DEFINITIONS (each persona must belong to one archetype):
{archetype_context}

OUTPUT FORMAT — a JSON array of persona objects:
[
  {{
    "archetype_id": "<which archetype this persona belongs to>",
    "identity": "<2-3 sentences: who this person is, their background, their role in the domain>",
    "behavior_rules": "<2-3 rules: how they typically act, their default patterns, their tendencies>",
    "emotional_range": "<their emotional spectrum: what triggers strong reactions, what calms them, their baseline mood>",
    "knowledge_bounds": "<what they know and don't know: expertise areas, blind spots, information sources>",
    "consistency_rules": "<what stays constant: core values, non-negotiable beliefs, persistent traits>",
    "hard_limits": "<what they will NEVER do: ethical boundaries, absolute refusals, breaking points>"
  }}
]

RULES:
- Each persona must be distinct — different backgrounds, motivations, knowledge levels.
- Distribute personas across archetypes according to the provided distribution.
- Ground personas in the domain context — reference specific entities when appropriate.
- Include resistance and hesitancy in behavior rules to counteract LLM agreeability bias.
- Knowledge bounds should vary: some personas know a lot about the domain, others know little.
- Hard limits should feel realistic and persona-specific, not generic.
- Do NOT mention AI, simulation, scores, or technical framing.

Output ONLY the JSON array:"""


def _build_persona_prompt(
    graph_context: str,
    archetypes: list[dict],
    distribution: dict[str, float],
    scenario_name: str,
    n_personas: int,
) -> tuple[str, str]:
    """Build the prompt for persona generation."""
    archetype_lines = []
    for arch in archetypes:
        dist_pct = distribution.get(arch["id"], 0) * 100
        archetype_lines.append(
            f"  {arch['id']} ({arch['label']}): {arch.get('description', '')} "
            f"[target: {dist_pct:.0f}% of personas]"
        )
    archetype_context = "\n".join(archetype_lines)

    user = PERSONA_USER_TEMPLATE.format(
        n_personas=n_personas,
        scenario_name=scenario_name,
        graph_context=graph_context or "No domain documents uploaded.",
        archetype_context=archetype_context,
    )
    return PERSONA_SYSTEM, user


def _parse_persona_output(raw: str, n_personas: int) -> list[dict]:
    """Parse LLM persona generation output."""
    text = raw.strip()

    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError("Could not parse persona output as JSON array")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    return data


async def generate_personas(
    scenario_name: str,
    archetypes: list[dict],
    distribution: dict[str, float],
    n_personas: int = 4,
    graph_context: str = "",
    db=None,
) -> list[PersonaSpec]:
    """
    Generate heterogeneous agent personas from domain knowledge.

    Uses the authority LLM to create diverse personas grounded in the
    knowledge graph and distributed across archetypes.

    Args:
        scenario_name: Name of the scenario.
        archetypes: List of archetype dicts (id, label, description, properties).
        distribution: Archetype distribution fractions.
        n_personas: Number of personas to generate.
        graph_context: Formatted graph context from GraphRAG.
        db: Optional DuckDB connection for caching.

    Returns:
        List of PersonaSpec objects ready for sim_loop integration.
    """
    import orchestrator

    # Check cache first
    if db is not None:
        cached = _load_cached_personas(db, scenario_name, n_personas)
        if cached:
            logger.info("Loaded %d cached personas for '%s'", len(cached), scenario_name)
            return cached

    system, user = _build_persona_prompt(
        graph_context, archetypes, distribution, scenario_name, n_personas,
    )

    try:
        raw_output = await orchestrator.agent_turn(
            agent_id="persona_generator",
            backend="authority",
            system_prompt=system,
            user_message=user,
            max_tokens=4096,
        )
    except Exception as e:
        raise RuntimeError(f"Persona generation LLM call failed: {e}") from e

    persona_data = _parse_persona_output(raw_output, n_personas)

    personas = []
    for i, pd in enumerate(persona_data):
        spec = PersonaSpec(
            id=f"persona_{i}",
            archetype_id=pd.get("archetype_id", "K2"),
            identity=pd.get("identity", ""),
            behavior_rules=pd.get("behavior_rules", ""),
            emotional_range=pd.get("emotional_range", ""),
            knowledge_bounds=pd.get("knowledge_bounds", ""),
            consistency_rules=pd.get("consistency_rules", ""),
            hard_limits=pd.get("hard_limits", ""),
            properties={
                "knowledge_depth": 0.3 + (hash(pd.get("identity", "")) % 60) / 100,
            },
        )
        personas.append(spec)

    # Cache in DuckDB
    if db is not None:
        _cache_personas(db, scenario_name, personas)

    logger.info(
        "Generated %d personas for '%s' (%d archetypes)",
        len(personas), scenario_name, len(set(p.archetype_id for p in personas)),
    )

    return personas


def _cache_personas(db, scenario_name: str, personas: list[PersonaSpec]) -> None:
    """Cache generated personas in DuckDB."""
    import hashlib
    for spec in personas:
        cache_id = hashlib.sha256(
            f"{scenario_name}:{spec.id}:{spec.identity[:50]}".encode()
        ).hexdigest()[:16]
        db.execute(
            """INSERT OR REPLACE INTO generated_personas
               (id, scenario_name, archetype_id, identity, behavior_rules,
                emotional_range, knowledge_bounds, consistency_rules, hard_limits, properties)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [cache_id, scenario_name, spec.archetype_id,
             spec.identity, spec.behavior_rules, spec.emotional_range,
             spec.knowledge_bounds, spec.consistency_rules, spec.hard_limits,
             json.dumps(spec.properties)],
        )


def _load_cached_personas(
    db, scenario_name: str, n_personas: int,
) -> Optional[list[PersonaSpec]]:
    """Load cached personas if available and count matches."""
    rows = db.execute(
        "SELECT id, archetype_id, identity, behavior_rules, emotional_range, "
        "knowledge_bounds, consistency_rules, hard_limits, properties "
        "FROM generated_personas WHERE scenario_name = ? ORDER BY id",
        [scenario_name],
    ).fetchall()

    if len(rows) != n_personas:
        return None

    personas = []
    for i, row in enumerate(rows):
        personas.append(PersonaSpec(
            id=f"persona_{i}",
            archetype_id=row[1],
            identity=row[2],
            behavior_rules=row[3],
            emotional_range=row[4],
            knowledge_bounds=row[5],
            consistency_rules=row[6],
            hard_limits=row[7],
            properties=json.loads(row[8]) if row[8] else {},
        ))

    return personas
