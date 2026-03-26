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
