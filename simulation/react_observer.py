"""
ReACT-style observer for dualmirakl Phase D1.

Inspired by MiroFish's ReportAgent: instead of a single-pass analysis,
the observer iteratively reasons, calls tools, and observes results
before producing a final structured analysis.

ReACT loop (Yao et al. 2023):
  1. Reason: what do I need to know?
  2. Act: call a tool
  3. Observe: receive tool result
  4. Repeat until ready → final_answer

Tools available:
  - query_scores:       score trajectories for participants
  - query_events:       search the event stream by type/agent/tick
  - check_interventions: list active and historical interventions
  - query_memory:       search agent memories semantically
  - interview_agent:    ask a participant a direct question (swarm GPU)

Drop-in replacement for ObserverAgent.analyse() — same signature, richer output.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from simulation.action_schema import (
    OBSERVER_A_ACTIONS,
    parse_action,
    schema_to_prompt,
)
from simulation.safety import (
    ObserverMode,
    SafetyTier,
    validate_observer_output,
)

logger = logging.getLogger(__name__)

# Maximum characters of tool result text injected per step
_MAX_TOOL_RESULT_CHARS = 2000


# ── Observer tool definitions ────────────────────────────────────────────────

OBSERVER_TOOLS = {
    "query_scores": {
        "description": "Get score trajectories for one or more participants.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Participant IDs to query. Omit or empty for all.",
                },
                "last_n_ticks": {
                    "type": "integer",
                    "description": "Number of recent ticks to include (default: all).",
                },
            },
            "required": [],
        },
    },
    "query_events": {
        "description": "Search the simulation event stream.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "enum": [
                        "stimulus", "response", "score", "observation",
                        "intervention", "compliance", "flame_snapshot",
                    ],
                    "description": "Filter by event type.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Filter by agent ID.",
                },
                "since_tick": {
                    "type": "integer",
                    "description": "Only events from this tick onward.",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Return only the N most recent matches.",
                },
            },
            "required": [],
        },
    },
    "check_interventions": {
        "description": "List active and recent interventions with their status.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_expired": {
                    "type": "boolean",
                    "description": "Include expired interventions (default: false).",
                },
            },
            "required": [],
        },
    },
    "query_memory": {
        "description": "Search an agent's memories by semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent whose memories to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Semantic search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default: 3).",
                },
            },
            "required": ["agent_id", "query"],
        },
    },
    "interview_agent": {
        "description": (
            "Ask a participant agent a direct question. The agent responds "
            "in-character. Uses the swarm GPU — use sparingly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Participant to interview.",
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask the agent.",
                },
            },
            "required": ["agent_id", "question"],
        },
    },
    "query_graph": {
        "description": (
            "Query the shared knowledge graph for entity relationships, "
            "behavioral patterns, and temporal connections between agents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Specific node to query (e.g. agent ID). Omit for overview.",
                },
                "edge_type": {
                    "type": "string",
                    "enum": [
                        "scored_at", "responded_to", "intervention_on",
                        "escalated", "disengaged", "flagged_by",
                        "clustered_with", "received_stimulus",
                    ],
                    "description": "Filter by relationship type.",
                },
            },
            "required": [],
        },
    },
}

# Safety tiers for tools
TOOL_SAFETY: dict[str, SafetyTier] = {
    "query_scores": SafetyTier.AUTO,
    "query_events": SafetyTier.AUTO,
    "check_interventions": SafetyTier.AUTO,
    "query_memory": SafetyTier.AUTO,
    "interview_agent": SafetyTier.REVIEW,
    "query_graph": SafetyTier.AUTO,
}


# ── Tool execution ───────────────────────────────────────────────────────────

async def _exec_query_scores(
    args: dict,
    world_state,
    participants: list,
    tick: int,
) -> str:
    """Execute query_scores tool against event stream."""
    agent_ids = args.get("agent_ids") or [p.agent_id for p in participants]
    last_n = args.get("last_n_ticks")

    lines = []
    for aid in agent_ids:
        trajectory = world_state.stream.score_trajectory(aid)
        if last_n and trajectory:
            trajectory = trajectory[-last_n:]
        if trajectory:
            formatted = ", ".join(f"{s:.3f}" for s in trajectory)
            lines.append(f"{aid}: [{formatted}] (latest={trajectory[-1]:.3f})")
        else:
            lines.append(f"{aid}: no scores yet")
    return "\n".join(lines) if lines else "No score data available."


async def _exec_query_events(
    args: dict,
    world_state,
    participants: list,
    tick: int,
) -> str:
    """Execute query_events tool against event stream."""
    events = world_state.stream.query(
        agent_id=args.get("agent_id"),
        event_type=args.get("event_type"),
        since_tick=args.get("since_tick"),
        last_n=args.get("last_n", 10),
    )
    if not events:
        return "No matching events found."

    lines = []
    for e in events:
        # Compact summary per event
        payload_summary = {}
        for k, v in e.payload.items():
            if isinstance(v, str) and len(v) > 80:
                payload_summary[k] = v[:77] + "..."
            else:
                payload_summary[k] = v
        lines.append(
            f"[T{e.tick}/{e.phase}] {e.event_type} "
            f"agent={e.agent_id} {json.dumps(payload_summary, default=str)}"
        )
    return "\n".join(lines)


async def _exec_check_interventions(
    args: dict,
    world_state,
    participants: list,
    tick: int,
) -> str:
    """Execute check_interventions tool."""
    include_expired = args.get("include_expired", False)

    # Active interventions
    active = []
    for iv in world_state.active_interventions:
        remaining = "permanent" if iv.duration == -1 else f"{iv.duration} ticks"
        active.append(
            f"  [{iv.type}] {iv.description} "
            f"(activated T{iv.activated_at}, remaining: {remaining}, source: {iv.source})"
        )

    lines = [f"Active interventions ({len(active)}):"]
    lines.extend(active if active else ["  none"])

    if include_expired:
        # Historical from event stream
        all_ivs = world_state.stream.query(event_type="intervention")
        expired = [
            e for e in all_ivs
            if e.payload.get("duration", -1) != -1
            and e.payload.get("activated_at", e.tick) + e.payload.get("duration", 0) <= tick
        ]
        lines.append(f"\nExpired interventions ({len(expired)}):")
        for e in expired[-10:]:  # last 10
            lines.append(
                f"  [T{e.tick}] {e.payload.get('type', '?')}: "
                f"{e.payload.get('description', '')[:60]}"
            )

    return "\n".join(lines)


async def _exec_query_memory(
    args: dict,
    world_state,
    participants: list,
    tick: int,
) -> str:
    """Execute query_memory tool against agent memory store."""
    if world_state.memory is None:
        return "Memory store not enabled for this simulation."

    agent_id = args["agent_id"]
    query = args["query"]
    top_k = args.get("top_k", 3)

    memories = world_state.memory.retrieve(agent_id, query, tick, top_k)
    if not memories:
        return f"No memories found for {agent_id} matching '{query}'."

    lines = [f"Memories for {agent_id} (top {len(memories)}):"]
    for mem in memories:
        tag_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
        lines.append(f"  - {mem.title}{tag_str} (T{mem.tick_created}): {mem.content}")
    return "\n".join(lines)


async def _exec_interview_agent(
    args: dict,
    world_state,
    participants: list,
    tick: int,
) -> str:
    """Execute interview_agent tool — sends a question to a participant via swarm GPU."""
    target_id = args["agent_id"]
    question = args["question"]

    # Find the participant
    target = None
    for p in participants:
        if p.agent_id == target_id:
            target = p
            break
    if target is None:
        return f"Agent '{target_id}' not found. Available: {[p.agent_id for p in participants]}"

    # Import here to avoid circular dependency
    from simulation.sim_loop import _resilient_agent_turn

    response = await _resilient_agent_turn(
        agent_id=f"{target_id}_interview",
        backend=target.cfg["backend"],
        system_prompt=target._build_system_prompt(),
        user_message=(
            f"[Interview at tick {tick}] An observer is asking you: {question}\n"
            f"Answer honestly and in character."
        ),
        history=target.history[-4:],
        max_tokens=128,
    )
    return f"{target_id} responds: {response}"


async def _exec_query_graph(
    args: dict,
    world_state,
    participants: list,
    tick: int,
) -> str:
    """Execute query_graph tool against the shared knowledge graph."""
    if world_state.graph is None:
        return "Graph memory not enabled for this simulation."

    node_id = args.get("node_id")
    return world_state.graph.query_text(
        node_id=node_id,
        active_only=True,
        include_properties=True,
    )


# Tool name → executor mapping
_TOOL_EXECUTORS = {
    "query_scores": _exec_query_scores,
    "query_events": _exec_query_events,
    "check_interventions": _exec_check_interventions,
    "query_memory": _exec_query_memory,
    "interview_agent": _exec_interview_agent,
    "query_graph": _exec_query_graph,
}


# ── ReACT prompt construction ────────────────────────────────────────────────

def _tools_to_prompt() -> str:
    """Convert OBSERVER_TOOLS into a human-readable prompt section."""
    lines = [
        "\n\nAVAILABLE TOOLS — call one per turn by responding with JSON:",
    ]
    for name, schema in OBSERVER_TOOLS.items():
        desc = schema["description"]
        params = schema["parameters"]["properties"]
        required = set(schema["parameters"].get("required", []))
        param_parts = []
        for pname, pdef in params.items():
            req = " (REQUIRED)" if pname in required else ""
            param_parts.append(f"{pname}: {pdef.get('description', '')}{req}")
        params_str = "; ".join(param_parts) if param_parts else "none"
        lines.append(f'  "{name}": {desc}  Params: {params_str}')

    lines.append("")
    lines.append(
        'TOOL CALL FORMAT: {{"action": "tool_name", ...params}}'
    )
    lines.append(
        'FINAL ANSWER: when ready, call {{"action": "final_answer", ...analysis fields}}'
    )
    lines.append(
        "You MUST call final_answer to conclude. "
        "Use at least 1 tool before your final answer."
    )
    lines.append("")
    return "\n".join(lines)


def _build_react_prompt(
    tick: int,
    window: str,
    stats_note: str,
    active: str,
    observations: list[dict],
    step: int,
    max_steps: int,
) -> str:
    """Build the user message for each ReACT step."""
    lines = [
        f"[Tick {tick}] Observation window:\n{window}\n{stats_note}\n",
        f"Active interventions: {active}\n",
    ]

    if observations:
        lines.append(f"TOOL RESULTS SO FAR ({len(observations)} calls):")
        for obs in observations:
            result_text = obs["result"]
            if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "...[truncated]"
            lines.append(f"  [{obs['tool']}] {result_text}")
        lines.append("")

    remaining = max_steps - step
    if remaining <= 1:
        lines.append(
            "You MUST call final_answer now — no more tool calls allowed."
        )
    else:
        lines.append(
            f"Step {step + 1}/{max_steps}. "
            f"Call a tool to gather information, or call final_answer if ready."
        )

    return "\n".join(lines)


# ── ReactObserver ────────────────────────────────────────────────────────────

class ReactObserver:
    """
    ReACT-style observer for Phase D1 — drop-in replacement for ObserverAgent.analyse().

    Instead of a single LLM call, iteratively reasons and calls tools to build
    a richer analysis before producing the final OBSERVER_A_ACTIONS output.

    Args:
        agent_id: observer ID (typically "observer_a")
        role: key into AGENT_ROLES dict
        max_steps: maximum ReACT iterations (default 5)
        history_window: LLM conversation history window
        max_tokens: max tokens per LLM call
        world_context: optional world context string
        enabled_tools: subset of tool names to enable (default: all)
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        max_steps: int = 5,
        history_window: int = 4,
        max_tokens: int = 512,
        world_context: Optional[str] = None,
        enabled_tools: Optional[list[str]] = None,
    ):
        from simulation.agent_rolesv3 import AGENT_ROLES

        self.agent_id = agent_id
        self.cfg = AGENT_ROLES[role]
        self.max_steps = max(2, max_steps)  # at least 1 tool + final_answer
        self.history: list[dict] = []
        self.history_window = history_window
        self.max_tokens = max_tokens
        self.analyses: list[str] = []
        self.world_context = world_context
        self._last_parsed: Optional[dict] = None
        self.participants: list = []

        # Filter tools
        if enabled_tools is not None:
            self._tools = {
                k: v for k, v in OBSERVER_TOOLS.items()
                if k in enabled_tools
            }
        else:
            self._tools = dict(OBSERVER_TOOLS)

    def set_participants(self, participants: list) -> None:
        """Set participant references for interview tool."""
        self.participants = participants

    def _system_prompt(self) -> str:
        base = self.cfg["system"]
        if self.world_context:
            return f"[World Context]\n{self.world_context}\n\n{base}"
        return base

    async def analyse(
        self,
        tick: int,
        world_state,
        n_participants: int,
    ) -> str:
        """
        ReACT analysis loop — same signature as ObserverAgent.analyse().

        Iteratively calls tools then produces a final analysis conforming
        to OBSERVER_A_ACTIONS schema.
        """
        from simulation.sim_loop import _resilient_agent_turn, _format_stats
        from simulation.event_stream import TOOL_USE, OBSERVATION, COMPLIANCE
        from simulation.agent_rolesv3 import check_compliance

        window = world_state.observer_prompt_window(tick, n_participants)
        active = (
            ", ".join(iv.description for iv in world_state.active_interventions)
            or "none"
        )
        stats = world_state.compute_score_statistics(tick)
        stats_note = _format_stats(stats)

        # Build system prompt with tool descriptions + final answer schema
        system = self._system_prompt()
        system += _tools_to_prompt()
        system += schema_to_prompt(
            {"final_answer": OBSERVER_A_ACTIONS["analyse"]},
            "observer_a",
        )

        observations: list[dict] = []
        final_response = None

        for step in range(self.max_steps):
            prompt = _build_react_prompt(
                tick, window, stats_note, active,
                observations, step, self.max_steps,
            )

            response = await _resilient_agent_turn(
                agent_id=self.agent_id,
                backend=self.cfg["backend"],
                system_prompt=system,
                user_message=prompt,
                history=self.history[-self.history_window:],
                max_tokens=self.max_tokens,
            )

            # Try to parse as JSON
            parsed = _parse_react_response(response, self._tools)

            if parsed is None:
                # Unparseable — treat as final answer attempt on last step
                if step == self.max_steps - 1:
                    final_response = response
                    break
                # Otherwise inject error and retry
                observations.append({
                    "tool": "system",
                    "result": "Could not parse your response as JSON. Please respond with a valid JSON tool call.",
                })
                continue

            action = parsed.get("action", "")

            # Final answer
            if action == "final_answer":
                # Re-parse as OBSERVER_A_ACTIONS
                parsed["action"] = "analyse"
                self._last_parsed = parsed
                final_response = response
                break

            # Tool call
            if action not in _TOOL_EXECUTORS:
                observations.append({
                    "tool": "system",
                    "result": f"Unknown tool '{action}'. Available: {list(self._tools.keys())} or 'final_answer'.",
                })
                continue

            # Safety gate check
            tool_tier = TOOL_SAFETY.get(action, SafetyTier.APPROVE)
            if tool_tier == SafetyTier.APPROVE:
                observations.append({
                    "tool": "system",
                    "result": f"Tool '{action}' requires APPROVE tier — blocked.",
                })
                world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                    "safety_tier": tool_tier.value,
                    "action": f"tool.{action}",
                    "status": "blocked",
                })
                continue

            # Execute tool
            try:
                result = await _TOOL_EXECUTORS[action](
                    parsed, world_state, self.participants, tick,
                )
            except Exception as e:
                result = f"Tool error: {e}"
                logger.warning(f"[{self.agent_id}] tool '{action}' failed: {e}")

            observations.append({"tool": action, "result": result})

            # Emit tool use to event stream
            world_state.stream.emit(tick, "D", TOOL_USE, self.agent_id, {
                "tool": action,
                "args": {k: v for k, v in parsed.items() if k != "action"},
                "result_length": len(result),
                "step": step,
            })

            # Log REVIEW tier tool calls
            if tool_tier == SafetyTier.REVIEW:
                world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                    "safety_tier": "review",
                    "action": f"tool.{action}",
                    "status": "executed",
                })

            logger.debug(
                f"[{self.agent_id}] step {step}: tool={action} result_len={len(result)}"
            )

        # Fallback: force parse final response if not already parsed
        if final_response is None:
            final_response = response
        if self._last_parsed is None:
            self._last_parsed = parse_action(final_response, OBSERVER_A_ACTIONS)

        # Annotate with tool usage metadata
        if self._last_parsed:
            self._last_parsed["_react_steps"] = len(observations)
            self._last_parsed["_tools_used"] = [o["tool"] for o in observations]

        # Mode enforcement: ANALYSE mode compliance gate
        mode_violations = validate_observer_output(
            final_response, ObserverMode.ANALYSE, self._last_parsed,
        )
        if mode_violations:
            logger.debug(f"[MODE] {self.agent_id} ANALYSE violations: {mode_violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "observer_a", "violations": mode_violations,
            })
            world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                "mode": ObserverMode.ANALYSE.value,
                "violations": mode_violations,
            })

        # Compliance check
        violations = check_compliance(final_response, "observer_a")
        if violations:
            logger.debug(f"[COMPLIANCE] {self.agent_id} used intervention keywords: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "observer_a", "violations": violations,
            })

        self.history.append({"role": "assistant", "content": final_response})
        self.analyses.append(final_response)

        logger.info(
            f"[{self.agent_id}] ReACT analysis complete: "
            f"{len(observations)} tool calls in {step + 1} steps"
        )

        return final_response


# ── Response parsing ─────────────────────────────────────────────────────────

def _parse_react_response(
    response: str,
    tools: dict,
) -> Optional[dict]:
    """
    Parse a ReACT step response. Accepts:
      - Tool call: {"action": "tool_name", ...params}
      - Final answer: {"action": "final_answer", ...analysis fields}
    Returns parsed dict or None.
    """
    import re

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

    action = parsed.get("action")
    if action is None:
        # Try to infer: if it has OBSERVER_A required fields, it's final_answer
        obs_required = {"reasoning", "trajectory_summary", "clustering", "concern_level"}
        if obs_required.issubset(parsed.keys()):
            parsed["action"] = "final_answer"
            return parsed
        return None

    # Valid if it's a known tool or final_answer
    if action in tools or action == "final_answer":
        return parsed

    return None
