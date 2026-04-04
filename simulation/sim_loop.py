"""Backward-compat shim -- import from simulation.core submodules instead.

Original 2066-line module decomposed into:
  simulation.core.state       -- ObsEntry, WorldState, _format_stats, re-exports
  simulation.core.agents_impl -- EnvironmentAgent, ParticipantAgent, ObserverAgent
  simulation.core.tick        -- _batch_phase_b, run_tick
  simulation.core.runner      -- run_simulation
  simulation.core.cli         -- CLI entry point (_prompt, __main__)
"""
from simulation.core.state import *  # noqa: F401,F403
from simulation.core.agents_impl import *  # noqa: F401,F403
from simulation.core.tick import *  # noqa: F401,F403
from simulation.core.runner import *  # noqa: F401,F403
from simulation.core.cli import *  # noqa: F401,F403

# Names the original sim_loop.py imported at top-level (not in any core __all__)
from simulation.config.legacy_roles import (  # noqa: F401
    AGENT_ROLES, INTERVENTION_CODEBOOK, ENGAGEMENT_ANCHORS,
    INTERVENTION_THRESHOLD, PERSONA_SUMMARY_INTERVAL,
    PERSONA_SUMMARY_TEMPLATE, EMBED_BATCH_SIZE, check_compliance,
)
from simulation.core.event_stream import (  # noqa: F401
    EventStream, SimEvent,
    STIMULUS, RESPONSE, SCORE, OBSERVATION, INTERVENTION as EV_INTERVENTION,
    COMPLIANCE, FLAME_SNAPSHOT, PERSONA, CONTEXT, TOOL_USE, GRAPH_UPDATE,
)
from simulation.config.action_schema import (  # noqa: F401
    PARTICIPANT_ACTIONS, OBSERVER_A_ACTIONS, OBSERVER_B_ACTIONS,
    schema_to_prompt, parse_action, extract_narrative,
)
from simulation.core.safety import (  # noqa: F401
    ObserverMode, SafetyTier, SafetyGate,
    validate_observer_output, ACTION_SAFETY,
)
from simulation.knowledge.agent_memory import AgentMemoryStore, DuckDBMemoryBackend  # noqa: F401
from simulation.knowledge.graph_memory import GraphMemory  # noqa: F401
from simulation.core.topology import TopologyManager, combine_stimuli  # noqa: F401
from simulation.storage.tracking import tracker  # noqa: F401
from orchestrator import agent_turn, close_client  # noqa: F401

# Expose all underscore-prefixed names too
import simulation.core.state as _s  # noqa: F811
import simulation.core.agents_impl as _a  # noqa: F811
import simulation.core.tick as _t  # noqa: F811
import simulation.core.runner as _r  # noqa: F811
globals().update({
    k: v for mod in (_s, _a, _t, _r)
    for k, v in mod.__dict__.items()
    if k.startswith("_") and not k.startswith("__")
})
