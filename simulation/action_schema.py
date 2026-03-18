"""
Structured action schemas for dualmirakl agent outputs.

Inspired by Manus's tools.json: typed JSON function-calling definitions with
required/optional params. Replaces free-form text output + cosine codebook
matching with deterministic structured extraction.

Usage modes:
  1. Schema injected into system prompts → LLM returns JSON
  2. parse_action() validates and extracts structured fields
  3. Cosine codebook matching remains as FALLBACK for non-JSON responses

Each action schema follows the Manus/OpenAI function-calling format:
  {name, description, parameters: {type, properties, required}}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Participant action schemas ────────────────────────────────────────────────

PARTICIPANT_ACTIONS = {
    "respond": {
        "description": "Express thoughts, emotions, and behavioral response to the stimulus.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "What the participant does in response to the stimulus.",
                },
                "emotion": {
                    "type": "string",
                    "description": "Current emotional state (e.g. curious, anxious, frustrated, calm).",
                },
                "thought": {
                    "type": "string",
                    "description": "Internal reasoning or reflection.",
                },
                "intensity": {
                    "type": "number",
                    "description": "Self-assessed engagement intensity from 0.0 (disengaged) to 1.0 (fully absorbed).",
                },
                "narrative": {
                    "type": "string",
                    "description": "Free-form first-person narrative of the response (preserves behavioral richness).",
                },
            },
            "required": ["action", "emotion", "narrative"],
        },
    },
    "disengage": {
        "description": "Participant steps away from the activity.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the participant is disengaging.",
                },
                "duration": {
                    "type": "string",
                    "enum": ["brief", "extended", "permanent"],
                    "description": "How long the disengagement lasts.",
                },
                "emotion": {
                    "type": "string",
                    "description": "Emotional state during disengagement.",
                },
                "narrative": {
                    "type": "string",
                    "description": "Free-form first-person narrative.",
                },
            },
            "required": ["reason", "narrative"],
        },
    },
    "escalate": {
        "description": "Participant increases engagement beyond previous level.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "What the participant does to escalate.",
                },
                "trigger": {
                    "type": "string",
                    "description": "What prompted the escalation.",
                },
                "emotion": {
                    "type": "string",
                    "description": "Emotional state during escalation.",
                },
                "narrative": {
                    "type": "string",
                    "description": "Free-form first-person narrative.",
                },
            },
            "required": ["action", "narrative"],
        },
    },
}


# ── Observer action schemas ───────────────────────────────────────────────────

OBSERVER_A_ACTIONS = {
    "analyse": {
        "description": "Report on population dynamics. Do NOT propose interventions.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step analysis of the data before reaching conclusions.",
                },
                "trajectory_summary": {
                    "type": "string",
                    "description": "Summary of individual participant trajectories.",
                },
                "clustering": {
                    "type": "string",
                    "enum": ["converging", "diverging", "stable", "chaotic"],
                    "description": "Overall population dynamics pattern.",
                },
                "concern_level": {
                    "type": "string",
                    "enum": ["none", "low", "moderate", "high"],
                    "description": "Level of concern about current trajectories.",
                },
                "flagged_participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participant IDs showing concerning patterns.",
                },
            },
            "required": ["reasoning", "trajectory_summary", "clustering", "concern_level"],
        },
    },
}

OBSERVER_B_ACTIONS = {
    "no_intervention": {
        "description": "No intervention is needed at this time.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "Why no intervention is warranted.",
                },
            },
            "required": ["rationale"],
        },
    },
    "intervene": {
        "description": "Propose a specific intervention from the codebook.",
        "parameters": {
            "type": "object",
            "properties": {
                "intervention_type": {
                    "type": "string",
                    "enum": ["pause_prompt", "boundary_warning",
                             "pacing_adjustment", "dynamics_dampening"],
                    "description": "Type of intervention to apply.",
                },
                "target": {
                    "type": "string",
                    "description": "Target participant_id, or 'all' for population-wide.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this intervention is needed.",
                },
            },
            "required": ["intervention_type", "target", "rationale"],
        },
    },
}


# ── Schema → prompt injection ─────────────────────────────────────────────────

def schema_to_prompt(actions: dict, role_label: str) -> str:
    """
    Convert action schemas into a prompt section that instructs the LLM
    to respond with structured JSON.
    """
    lines = [
        f"\n\nOUTPUT FORMAT — You MUST respond with a JSON object.",
        f"Choose ONE of the following actions:\n",
    ]
    for name, schema in actions.items():
        desc = schema["description"]
        params = schema["parameters"]["properties"]
        required = set(schema["parameters"].get("required", []))
        lines.append(f'Action "{name}": {desc}')
        lines.append("  Fields:")
        for pname, pdef in params.items():
            req = " (REQUIRED)" if pname in required else " (optional)"
            ptype = pdef.get("type", "string")
            pdesc = pdef.get("description", "")
            enum_vals = pdef.get("enum")
            enum_note = f' — one of: {enum_vals}' if enum_vals else ""
            lines.append(f'    "{pname}": {ptype}{req} — {pdesc}{enum_note}')
        lines.append("")

    lines.append(
        'Your response must be valid JSON with an "action" field set to the action name, '
        "plus the required fields for that action. Example:"
    )
    # Provide a minimal example from the first action
    first_name = next(iter(actions))
    first_req = actions[first_name]["parameters"].get("required", [])
    example = {"action": first_name}
    for f in first_req[:3]:
        example[f] = "..."
    lines.append(f"  {json.dumps(example)}")
    lines.append("")

    return "\n".join(lines)


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_action(response: str, actions: dict) -> Optional[dict]:
    """
    Parse a structured JSON response against the action schema.

    Returns parsed dict with validated fields, or None if parsing fails.
    The caller should fall back to free-text processing on None.
    """
    # Try to extract JSON from the response
    text = response.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(parsed, dict):
        return None

    # Validate action field
    action_name = parsed.get("action")
    if action_name not in actions:
        # If no explicit action field, try to infer from keys
        for name, schema in actions.items():
            required = set(schema["parameters"].get("required", []))
            if required.issubset(parsed.keys()):
                parsed["action"] = name
                action_name = name
                break
        if action_name not in actions:
            return None

    # Validate required fields
    schema = actions[action_name]
    required = set(schema["parameters"].get("required", []))
    missing = required - set(parsed.keys())
    if missing:
        logger.debug(f"Structured output missing required fields: {missing}")
        return None

    # Validate enum fields
    properties = schema["parameters"]["properties"]
    for field_name, field_def in properties.items():
        if field_name in parsed and "enum" in field_def:
            if parsed[field_name] not in field_def["enum"]:
                logger.debug(
                    f"Invalid enum value for {field_name}: {parsed[field_name]} "
                    f"(valid: {field_def['enum']})"
                )
                # Don't fail — just log. The value is still useful.

    # Coerce intensity to float if present
    if "intensity" in parsed:
        try:
            parsed["intensity"] = float(parsed["intensity"])
            parsed["intensity"] = max(0.0, min(1.0, parsed["intensity"]))
        except (ValueError, TypeError):
            parsed.pop("intensity", None)

    return parsed


def extract_narrative(parsed: Optional[dict], raw_response: str) -> str:
    """
    Get the narrative text from a parsed action, falling back to raw response.
    This ensures embed_score_batch always has text to score against anchors.
    """
    if parsed and parsed.get("narrative"):
        return parsed["narrative"]
    if parsed and parsed.get("action"):
        # Build narrative from structured fields
        parts = []
        if parsed.get("action") and parsed["action"] not in PARTICIPANT_ACTIONS:
            parts.append(parsed.get("action", ""))
        if parsed.get("thought"):
            parts.append(parsed["thought"])
        if parsed.get("reason"):
            parts.append(parsed["reason"])
        if parts:
            return " ".join(parts)
    return raw_response
