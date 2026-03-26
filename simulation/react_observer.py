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

# ── Post-Sim ReACT Analysis (Enhancement 4) ──────────────────────────────────

POST_SIM_TOOLS = {
    "query_trajectories": {
        "description": "Get score trajectories from a completed simulation run.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Specific agent to query. Omit for all.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["score_log", "susceptibility", "resilience", "initial_score", "final_score"],
                    "description": "Which metric to retrieve (default: score_log).",
                },
            },
            "required": [],
        },
    },
    "query_dynamics": {
        "description": (
            "Get dynamics analysis results: Lyapunov exponents, bifurcation, "
            "transfer entropy, emergence index, Sobol indices, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "enum": ["ode", "bifurcation", "lyapunov", "sobol",
                             "transfer_entropy", "emergence", "attractors",
                             "stochastic_resonance", "all"],
                    "description": "Which analysis module to query (default: all).",
                },
            },
            "required": [],
        },
    },
    "compare_agents": {
        "description": "Compare two agents across one or more metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_a": {"type": "string", "description": "First agent ID."},
                "agent_b": {"type": "string", "description": "Second agent ID."},
                "metric": {
                    "type": "string",
                    "enum": ["score_log", "susceptibility", "resilience"],
                    "description": "Metric to compare.",
                },
            },
            "required": ["agent_a", "agent_b"],
        },
    },
    "query_events": {
        "description": "Search the simulation event stream by type, agent, or tick.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "enum": ["stimulus", "response", "score", "observation",
                             "intervention", "compliance"],
                    "description": "Filter by event type.",
                },
                "agent_id": {"type": "string", "description": "Filter by agent ID."},
                "last_n": {"type": "integer", "description": "Return last N matches."},
            },
            "required": [],
        },
    },
    "query_graph": {
        "description": "Query the knowledge graph for entity relationships and patterns.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Specific node to query."},
            },
            "required": [],
        },
    },
    "interview_memory": {
        "description": (
            "Reconstruct an agent's perspective from their persisted memories "
            "and ask them a question post-simulation via the swarm GPU."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent to interview."},
                "question": {"type": "string", "description": "Question to ask."},
            },
            "required": ["agent_id", "question"],
        },
    },
    "statistical_test": {
        "description": (
            "Run a statistical test on simulation data. "
            "Supports: mann_whitney, kolmogorov_smirnov, correlation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "enum": ["mann_whitney", "kolmogorov_smirnov", "correlation"],
                    "description": "Statistical test to run.",
                },
                "group_a": {"type": "string", "description": "First agent or group."},
                "group_b": {"type": "string", "description": "Second agent or group."},
            },
            "required": ["test", "group_a", "group_b"],
        },
    },
}

POST_SIM_TOOL_SAFETY: dict[str, SafetyTier] = {
    "query_trajectories": SafetyTier.AUTO,
    "query_dynamics": SafetyTier.AUTO,
    "compare_agents": SafetyTier.AUTO,
    "query_events": SafetyTier.AUTO,
    "query_graph": SafetyTier.AUTO,
    "interview_memory": SafetyTier.REVIEW,
    "statistical_test": SafetyTier.AUTO,
}


class PostSimData:
    """Container for loaded post-simulation data."""

    def __init__(self, run_dir: str):
        from pathlib import Path
        self.run_dir = Path(run_dir)
        self._cache: dict[str, dict] = {}

    def _load(self, name: str) -> dict:
        if name not in self._cache:
            fpath = self.run_dir / f"{name}.json"
            if fpath.exists():
                self._cache[name] = json.loads(fpath.read_text(encoding="utf-8"))
            else:
                self._cache[name] = {}
        return self._cache[name]

    @property
    def config(self) -> dict:
        return self._load("config")

    @property
    def trajectories(self) -> dict:
        return self._load("trajectories")

    @property
    def observations(self) -> list:
        data = self._load("observations")
        return data if isinstance(data, list) else []

    @property
    def event_stream(self) -> list:
        data = self._load("event_stream")
        return data if isinstance(data, list) else []

    @property
    def dynamics(self) -> dict:
        return self._load("dynamics_analysis")

    @property
    def interventions(self) -> list:
        data = self._load("interventions")
        return data if isinstance(data, list) else []

    @property
    def compliance(self) -> list:
        data = self._load("compliance")
        return data if isinstance(data, list) else []

    @property
    def agent_memories(self) -> list:
        data = self._load("agent_memories")
        return data if isinstance(data, list) else []

    @property
    def graph(self) -> dict:
        return self._load("graph_memory")


# ── Post-sim tool executors ───────────────────────────────────────────────────

def _postsim_query_trajectories(args: dict, data: PostSimData) -> str:
    agent_id = args.get("agent_id")
    metric = args.get("metric", "score_log")
    trajs = data.trajectories

    if agent_id and agent_id in trajs:
        agent_data = trajs[agent_id]
        if metric in agent_data:
            val = agent_data[metric]
            if isinstance(val, list):
                formatted = ", ".join(f"{v:.3f}" for v in val)
                return f"{agent_id} {metric}: [{formatted}]"
            return f"{agent_id} {metric}: {val}"
        return f"{agent_id}: metric '{metric}' not found. Available: {list(agent_data.keys())}"

    # All agents
    lines = []
    for aid, agent_data in sorted(trajs.items()):
        if metric in agent_data:
            val = agent_data[metric]
            if isinstance(val, list):
                summary = f"[{val[0]:.3f}...{val[-1]:.3f}] ({len(val)} ticks)"
            else:
                summary = f"{val}"
            lines.append(f"{aid}: {metric}={summary}")
    return "\n".join(lines) if lines else "No trajectory data."


def _postsim_query_dynamics(args: dict, data: PostSimData) -> str:
    module = args.get("module", "all")
    dynamics = data.dynamics

    if not dynamics:
        return "No dynamics analysis available for this run."

    if module == "all":
        lines = [f"Dynamics analysis modules: {list(dynamics.keys())}"]
        for key, val in dynamics.items():
            if isinstance(val, dict):
                summary = json.dumps(val, default=str)[:200]
            else:
                summary = str(val)[:200]
            lines.append(f"  {key}: {summary}")
        return "\n".join(lines)

    if module in dynamics:
        return json.dumps(dynamics[module], indent=2, default=str)[:_MAX_TOOL_RESULT_CHARS]

    return f"Module '{module}' not found. Available: {list(dynamics.keys())}"


def _postsim_compare_agents(args: dict, data: PostSimData) -> str:
    agent_a = args.get("agent_a", "")
    agent_b = args.get("agent_b", "")
    metric = args.get("metric", "score_log")
    trajs = data.trajectories

    if agent_a not in trajs:
        return f"Agent '{agent_a}' not found. Available: {list(trajs.keys())}"
    if agent_b not in trajs:
        return f"Agent '{agent_b}' not found. Available: {list(trajs.keys())}"

    a_data = trajs[agent_a]
    b_data = trajs[agent_b]

    lines = [f"Comparison: {agent_a} vs {agent_b} on '{metric}'"]

    if metric == "score_log" and "score_log" in a_data and "score_log" in b_data:
        a_scores = a_data["score_log"]
        b_scores = b_data["score_log"]
        import numpy as _np
        lines.append(f"  {agent_a}: mean={_np.mean(a_scores):.3f}, std={_np.std(a_scores):.3f}, final={a_scores[-1]:.3f}")
        lines.append(f"  {agent_b}: mean={_np.mean(b_scores):.3f}, std={_np.std(b_scores):.3f}, final={b_scores[-1]:.3f}")
        # Correlation
        if len(a_scores) == len(b_scores):
            corr = float(_np.corrcoef(a_scores, b_scores)[0, 1])
            lines.append(f"  Pearson correlation: {corr:.3f}")
    else:
        for aid, adata in [(agent_a, a_data), (agent_b, b_data)]:
            val = adata.get(metric, "N/A")
            lines.append(f"  {aid}: {metric}={val}")

    return "\n".join(lines)


def _postsim_query_events(args: dict, data: PostSimData) -> str:
    event_type = args.get("event_type")
    agent_id = args.get("agent_id")
    last_n = args.get("last_n", 10)

    events = data.event_stream
    if not events:
        return "No event stream data."

    filtered = events
    if event_type:
        filtered = [e for e in filtered if e.get("event_type") == event_type]
    if agent_id:
        filtered = [e for e in filtered if e.get("agent_id") == agent_id]

    filtered = filtered[-last_n:]
    if not filtered:
        return "No matching events."

    lines = []
    for e in filtered:
        payload_str = json.dumps(e.get("payload", {}), default=str)[:100]
        lines.append(f"[T{e.get('tick')}/{e.get('phase')}] {e.get('event_type')} agent={e.get('agent_id')} {payload_str}")
    return "\n".join(lines)


def _postsim_query_graph(args: dict, data: PostSimData) -> str:
    graph = data.graph
    if not graph:
        return "No graph memory data for this run."

    node_id = args.get("node_id")
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    if node_id:
        if node_id in nodes:
            node = nodes[node_id]
            related = [e for e in edges if e.get("src") == node_id or e.get("dst") == node_id]
            lines = [f"{node.get('label', node_id)} ({node.get('type')})"]
            for e in related[:10]:
                other = e["dst"] if e["src"] == node_id else e["src"]
                lines.append(f"  -> {other} [{e.get('edge_type')}]")
            return "\n".join(lines)
        return f"Node '{node_id}' not found."

    return f"Graph: {len(nodes)} nodes, {len(edges)} edges. Node types: {set(n.get('type') for n in nodes.values())}"


async def _postsim_interview_memory(args: dict, data: PostSimData) -> str:
    agent_id = args.get("agent_id", "")
    question = args.get("question", "")

    memories = [m for m in data.agent_memories if m.get("agent_id") == agent_id]
    if not memories:
        return f"No memories found for '{agent_id}'."

    memory_context = "\n".join(
        f"- {m.get('title', '')}: {m.get('content', '')}"
        for m in memories[:5]
    )

    import orchestrator
    try:
        response = await orchestrator.agent_turn(
            agent_id=f"{agent_id}_postsim_interview",
            backend="swarm",
            system_prompt=(
                f"You are {agent_id} from a completed simulation. "
                f"Based on your memories, answer the question in character.\n\n"
                f"YOUR MEMORIES:\n{memory_context}"
            ),
            user_message=f"Question: {question}\nAnswer honestly and in character.",
            max_tokens=192,
        )
        return f"{agent_id} responds: {response}"
    except Exception as e:
        return f"Interview failed: {e}"


def _postsim_statistical_test(args: dict, data: PostSimData) -> str:
    test_name = args.get("test", "")
    group_a = args.get("group_a", "")
    group_b = args.get("group_b", "")
    trajs = data.trajectories

    if group_a not in trajs or group_b not in trajs:
        available = list(trajs.keys())
        return f"Agent(s) not found. Available: {available}"

    a_scores = trajs[group_a].get("score_log", [])
    b_scores = trajs[group_b].get("score_log", [])

    if not a_scores or not b_scores:
        return "Insufficient data for statistical test."

    import numpy as _np
    from scipy import stats as _stats

    if test_name == "mann_whitney":
        stat, pval = _stats.mannwhitneyu(a_scores, b_scores, alternative="two-sided")
        return (
            f"Mann-Whitney U test: {group_a} vs {group_b}\n"
            f"  U={stat:.2f}, p={pval:.4f}\n"
            f"  {'Significant' if pval < 0.05 else 'Not significant'} at alpha=0.05"
        )
    elif test_name == "kolmogorov_smirnov":
        stat, pval = _stats.ks_2samp(a_scores, b_scores)
        return (
            f"Kolmogorov-Smirnov test: {group_a} vs {group_b}\n"
            f"  D={stat:.4f}, p={pval:.4f}\n"
            f"  {'Distributions differ' if pval < 0.05 else 'Distributions similar'} at alpha=0.05"
        )
    elif test_name == "correlation":
        if len(a_scores) != len(b_scores):
            return "Score logs have different lengths — cannot compute correlation."
        corr, pval = _stats.pearsonr(a_scores, b_scores)
        return (
            f"Pearson correlation: {group_a} vs {group_b}\n"
            f"  r={corr:.4f}, p={pval:.4f}\n"
            f"  {'Significant' if pval < 0.05 else 'Not significant'} at alpha=0.05"
        )
    else:
        return f"Unknown test '{test_name}'. Available: mann_whitney, kolmogorov_smirnov, correlation"


_POST_SIM_EXECUTORS = {
    "query_trajectories": lambda args, data: _postsim_query_trajectories(args, data),
    "query_dynamics": lambda args, data: _postsim_query_dynamics(args, data),
    "compare_agents": lambda args, data: _postsim_compare_agents(args, data),
    "query_events": lambda args, data: _postsim_query_events(args, data),
    "query_graph": lambda args, data: _postsim_query_graph(args, data),
    "interview_memory": lambda args, data: _postsim_interview_memory(args, data),
    "statistical_test": lambda args, data: _postsim_statistical_test(args, data),
}


class PostSimAnalyser:
    """
    Post-simulation ReACT analyser using the authority slot.

    Loads data from a completed run directory, iteratively queries it
    using the ReACT pattern, and produces a structured analysis report.

    Usage:
        analyser = PostSimAnalyser(run_dir="data/run_20260325_120000_s42")
        report = await analyser.analyse(questions=["What drove the divergence?"])
    """

    def __init__(
        self,
        run_dir: str,
        max_steps: int = 8,
        max_tokens: int = 768,
    ):
        self.data = PostSimData(run_dir)
        self.max_steps = max(2, max_steps)
        self.max_tokens = max_tokens
        self.tools = dict(POST_SIM_TOOLS)

    def _system_prompt(self, questions: list[str]) -> str:
        config = self.data.config
        run_meta = config.get("config", config)

        questions_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))

        base = (
            "You are an expert simulation analyst reviewing the results of a "
            "completed multi-agent simulation. Use the available tools to "
            "investigate the data, then produce a structured analysis report.\n\n"
            f"RUN METADATA:\n"
            f"  Ticks: {run_meta.get('n_ticks', '?')}, "
            f"Participants: {run_meta.get('n_participants', '?')}, "
            f"Score mode: {run_meta.get('score_mode', '?')}\n\n"
            f"RESEARCH QUESTIONS:\n{questions_str}\n\n"
            "Your final_answer MUST be a JSON object with this structure:\n"
            '{"executive_summary": "...", '
            '"key_findings": [{"finding": "...", "evidence": "...", "confidence": "high|medium|low"}], '
            '"agent_spotlights": [{"agent_id": "...", "notable_behavior": "..."}], '
            '"recommendations": "...", '
            '"limitations": "..."}'
        )
        return base

    def _tools_prompt(self) -> str:
        lines = ["\n\nAVAILABLE TOOLS — call one per turn:"]
        for name, schema in self.tools.items():
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
        lines.append('TOOL CALL: {"action": "tool_name", ...params}')
        lines.append('FINAL ANSWER: {"action": "final_answer", ...report fields}')
        lines.append("Use at least 2 tools before final_answer.")
        return "\n".join(lines)

    async def analyse(
        self,
        questions: list[str],
        on_step: Optional[callable] = None,
    ) -> dict:
        """
        Run the post-sim ReACT analysis loop.

        Args:
            questions: Research questions to investigate.
            on_step: Optional callback(step, tool, result_preview).

        Returns:
            Structured report dict with executive_summary, key_findings, etc.
        """
        import orchestrator

        system = self._system_prompt(questions) + self._tools_prompt()
        observations: list[dict] = []
        report = None

        for step in range(self.max_steps):
            # Build user message
            lines = []
            if observations:
                lines.append(f"TOOL RESULTS ({len(observations)} calls):")
                for obs in observations:
                    result_text = obs["result"]
                    if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                        result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "...[truncated]"
                    lines.append(f"  [{obs['tool']}] {result_text}")

            remaining = self.max_steps - step
            if remaining <= 1:
                lines.append("You MUST call final_answer now.")
            else:
                lines.append(f"Step {step + 1}/{self.max_steps}. Call a tool or final_answer.")

            prompt = "\n".join(lines) if lines else "Begin your analysis. Call a tool."

            try:
                response = await orchestrator.agent_turn(
                    agent_id="post_sim_analyser",
                    backend="authority",
                    system_prompt=system,
                    user_message=prompt,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                logger.error("Post-sim analysis LLM call failed: %s", e)
                break

            parsed = _parse_react_response(response, self.tools)

            if parsed is None:
                if step == self.max_steps - 1:
                    report = {"executive_summary": response, "raw": True}
                    break
                observations.append({
                    "tool": "system",
                    "result": "Could not parse as JSON. Use valid JSON.",
                })
                continue

            action = parsed.get("action", "")

            if action == "final_answer":
                parsed.pop("action", None)
                report = parsed
                break

            if action not in _POST_SIM_EXECUTORS:
                observations.append({
                    "tool": "system",
                    "result": f"Unknown tool '{action}'.",
                })
                continue

            # Execute tool
            try:
                executor = _POST_SIM_EXECUTORS[action]
                if action == "interview_memory":
                    result = await executor(parsed, self.data)
                else:
                    result = executor(parsed, self.data)
            except Exception as e:
                result = f"Tool error: {e}"

            observations.append({"tool": action, "result": result})

            if on_step:
                on_step(step, action, result[:100])

            logger.debug("[post-sim] step %d: tool=%s", step, action)

        if report is None:
            report = {"executive_summary": "Analysis incomplete — max steps reached."}

        report["_meta"] = {
            "n_steps": len(observations),
            "tools_used": [o["tool"] for o in observations],
            "run_dir": str(self.data.run_dir),
        }

        # Persist report to DuckDB
        self._persist_report(report, questions)

        return report

    def _persist_report(self, report: dict, questions: list[str]) -> None:
        """Save analysis report to DuckDB."""
        try:
            from simulation.storage import get_db
            db = get_db()
            run_id = self.data.config.get("run_id", str(self.data.run_dir.name))
            meta = report.get("_meta", {})
            db.execute(
                """INSERT INTO analysis_reports
                   (id, run_id, report, questions, n_steps, tools_used)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [str(__import__("uuid").uuid4()), run_id,
                 json.dumps(report, default=str),
                 json.dumps(questions),
                 meta.get("n_steps", 0),
                 json.dumps(meta.get("tools_used", []))],
            )
        except Exception as e:
            logger.warning("Failed to persist analysis report: %s", e)


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
