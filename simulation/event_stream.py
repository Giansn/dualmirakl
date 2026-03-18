"""
Unified event stream for the dualmirakl simulation loop.

Inspired by Manus's event-stream architecture: all simulation state flows
through a single chronological log with typed events. Replaces scattered
state (score_logs, history lists, active_interventions, compliance logs)
with one queryable stream.

Event types:
  stimulus       — Phase A: environment → participant stimulus
  response       — Phase B: participant reply
  score          — Phase C: embedding signal + score update
  observation    — Phase D1: observer_a analysis
  intervention   — Phase D2: observer_b intervention
  compliance     — compliance violation detected
  flame_snapshot — Phase F: FLAME population stats
  persona        — Pre-phase: persona summary refresh
  context        — system context (world_context, config, etc.)
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Event types (string constants) ────────────────────────────────────────────

STIMULUS = "stimulus"
RESPONSE = "response"
SCORE = "score"
OBSERVATION = "observation"
INTERVENTION = "intervention"
COMPLIANCE = "compliance"
FLAME_SNAPSHOT = "flame_snapshot"
PERSONA = "persona"
CONTEXT = "context"

ALL_EVENT_TYPES = frozenset({
    STIMULUS, RESPONSE, SCORE, OBSERVATION, INTERVENTION,
    COMPLIANCE, FLAME_SNAPSHOT, PERSONA, CONTEXT,
})


# ── SimEvent ──────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SimEvent:
    """Single event in the simulation stream."""
    tick: int
    phase: str              # "Pre", "A", "B", "C", "D", "F", or "system"
    event_type: str         # one of ALL_EVENT_TYPES
    agent_id: str           # "environment", "participant_0", "observer_a", "flame", "system"
    payload: dict           # type-specific data
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        """Serializable representation for JSON export."""
        return {
            "tick": self.tick,
            "phase": self.phase,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "payload": self.payload,
        }


# ── EventStream ───────────────────────────────────────────────────────────────

class EventStream:
    """
    Chronological, append-only event log for the simulation.

    Design principles (from Manus architecture):
      - Single source of truth: all phases emit to one stream
      - Queryable: filter by agent_id, event_type, tick range
      - Append-only: events are never mutated or deleted
      - Exportable: full stream → JSON for replay/audit

    Coexists with existing WorldState — both can be used during migration.
    """

    def __init__(self):
        self._events: list[SimEvent] = []
        self._by_tick: dict[int, list[SimEvent]] = {}
        self._by_agent: dict[str, list[SimEvent]] = {}
        self._by_type: dict[str, list[SimEvent]] = {}

    # ── Emit ──────────────────────────────────────────────────────────────

    def emit(
        self,
        tick: int,
        phase: str,
        event_type: str,
        agent_id: str,
        payload: dict,
    ) -> SimEvent:
        """Append a new event to the stream. Returns the created event."""
        event = SimEvent(
            tick=tick,
            phase=phase,
            event_type=event_type,
            agent_id=agent_id,
            payload=payload,
        )
        self._events.append(event)
        self._by_tick.setdefault(tick, []).append(event)
        self._by_agent.setdefault(agent_id, []).append(event)
        self._by_type.setdefault(event_type, []).append(event)
        return event

    # ── Batch emit ─────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def batch(self):
        """
        Context manager for batched event emission (Mesa-inspired optimization).

        Defers index updates until the batch completes. Events are appended to
        _events immediately but _by_tick/_by_agent/_by_type are built once on exit.

        Usage:
            with stream.batch() as b:
                b.emit(tick, "A", STIMULUS, "p_0", {...})
                b.emit(tick, "A", STIMULUS, "p_1", {...})
            # indices updated here
        """
        pending: list[SimEvent] = []

        class BatchEmitter:
            def emit(_, tick, phase, event_type, agent_id, payload):
                event = SimEvent(tick=tick, phase=phase, event_type=event_type,
                                 agent_id=agent_id, payload=payload)
                pending.append(event)
                return event

        yield BatchEmitter()

        # Flush: append all and index in one pass
        for event in pending:
            self._events.append(event)
            self._by_tick.setdefault(event.tick, []).append(event)
            self._by_agent.setdefault(event.agent_id, []).append(event)
            self._by_type.setdefault(event.event_type, []).append(event)

    # ── Query ─────────────────────────────────────────────────────────────

    def query(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since_tick: Optional[int] = None,
        until_tick: Optional[int] = None,
        last_n: Optional[int] = None,
    ) -> list[SimEvent]:
        """
        Filter events by agent_id, event_type, and/or tick range.
        Returns newest-last ordering. Use last_n to limit results.
        """
        # Start from the narrowest index available
        if event_type is not None and agent_id is None and since_tick is None:
            candidates = self._by_type.get(event_type, [])
        elif agent_id is not None and event_type is None and since_tick is None:
            candidates = self._by_agent.get(agent_id, [])
        elif since_tick is not None and until_tick is not None and agent_id is None and event_type is None:
            candidates = []
            for t in range(since_tick, until_tick + 1):
                candidates.extend(self._by_tick.get(t, []))
        else:
            candidates = self._events

        result = candidates
        # Apply remaining filters
        if agent_id is not None and candidates is not self._by_agent.get(agent_id):
            result = [e for e in result if e.agent_id == agent_id]
        if event_type is not None and candidates is not self._by_type.get(event_type):
            result = [e for e in result if e.event_type == event_type]
        if since_tick is not None:
            result = [e for e in result if e.tick >= since_tick]
        if until_tick is not None:
            result = [e for e in result if e.tick <= until_tick]
        if last_n is not None:
            result = result[-last_n:]
        return result

    def window(self, agent_id: str, tick: int, window_size: int) -> list[SimEvent]:
        """
        Get events for a specific agent within the last `window_size` ticks.
        Equivalent to the old history[-SIM_HISTORY_WINDOW:] pattern.
        """
        since = max(1, tick - window_size + 1)
        return self.query(agent_id=agent_id, since_tick=since, until_tick=tick)

    def latest(self, event_type: str, agent_id: Optional[str] = None) -> Optional[SimEvent]:
        """Get the most recent event of a given type (optionally for a specific agent)."""
        events = self.query(agent_id=agent_id, event_type=event_type, last_n=1)
        return events[0] if events else None

    def tick_events(self, tick: int) -> list[SimEvent]:
        """All events for a specific tick, in emission order."""
        return list(self._by_tick.get(tick, []))

    # ── Aggregate helpers ─────────────────────────────────────────────────

    def score_trajectory(self, agent_id: str) -> list[float]:
        """Extract score series for a participant — replaces ParticipantAgent.score_log."""
        return [
            e.payload["score_after"]
            for e in self._by_agent.get(agent_id, [])
            if e.event_type == SCORE
        ]

    def active_interventions(self, tick: int) -> list[dict]:
        """
        Get interventions that are still active at the given tick.
        An intervention is active if:
          - duration == -1 (permanent), or
          - activated_at + duration > tick
        """
        result = []
        for e in self._by_type.get(INTERVENTION, []):
            p = e.payload
            activated = p.get("activated_at", e.tick)
            duration = p.get("duration", -1)
            if duration == -1 or activated + duration > tick:
                result.append(p)
        return result

    def response_texts(self, tick: int) -> dict[str, str]:
        """Get all participant responses for a tick — {agent_id: response_text}."""
        return {
            e.agent_id: e.payload.get("content", "")
            for e in self._by_tick.get(tick, [])
            if e.event_type == RESPONSE
        }

    # ── Introspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._events)

    @property
    def ticks(self) -> list[int]:
        """Sorted list of ticks that have events."""
        return sorted(self._by_tick.keys())

    @property
    def agents(self) -> list[str]:
        """List of agent IDs that have emitted events."""
        return list(self._by_agent.keys())

    # ── Export ────────────────────────────────────────────────────────────

    def export(self) -> list[dict]:
        """Full stream as list of dicts for JSON serialization."""
        return [e.to_dict() for e in self._events]

    def export_by_tick(self) -> dict[int, list[dict]]:
        """Stream grouped by tick — matches the old _log_by_tick pattern."""
        return {
            tick: [e.to_dict() for e in events]
            for tick, events in self._by_tick.items()
        }
