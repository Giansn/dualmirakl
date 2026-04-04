"""Agent class implementations for the simulation loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from simulation.config.legacy_roles import (
    AGENT_ROLES,
    INTERVENTION_CODEBOOK,
    ENGAGEMENT_ANCHORS,
    INTERVENTION_THRESHOLD,
    PERSONA_SUMMARY_INTERVAL,
    PERSONA_SUMMARY_TEMPLATE,
    EMBED_BATCH_SIZE,
    check_compliance,
)
from simulation.core.event_stream import (
    EventStream, SimEvent,
    STIMULUS, RESPONSE, SCORE, OBSERVATION, INTERVENTION as EV_INTERVENTION,
    COMPLIANCE, FLAME_SNAPSHOT, PERSONA, CONTEXT, TOOL_USE, GRAPH_UPDATE,
)
from simulation.config.action_schema import (
    PARTICIPANT_ACTIONS, OBSERVER_A_ACTIONS, OBSERVER_B_ACTIONS,
    schema_to_prompt, parse_action, extract_narrative,
)
from simulation.core.safety import (
    ObserverMode, SafetyTier, SafetyGate,
    validate_observer_output, ACTION_SAFETY,
)
from simulation.signal.intervention import (
    Intervention, extract_interventions, _make_intervention,
)
from simulation.signal.computation import _get_rng
from orchestrator import agent_turn, close_client

from simulation.core.state import (
    WorldState, ObsEntry, _format_stats,
    MAX_RETRIES, RETRY_DELAY,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EnvironmentAgent",
    "ParticipantAgent",
    "ObserverAgent",
    "resilient_agent_turn",
    "_resilient_agent_turn",
    "_get_semaphore",
    "_backend_semaphores",
]

# Per-backend semaphores for backpressure (respect vLLM max_num_seqs)
_backend_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(backend: str, max_concurrent: int = 10) -> asyncio.Semaphore:
    """Get or create a per-backend semaphore for backpressure."""
    if backend not in _backend_semaphores:
        _backend_semaphores[backend] = asyncio.Semaphore(max_concurrent)
    return _backend_semaphores[backend]


async def resilient_agent_turn(
    agent_id: str,
    backend: str,
    system_prompt: str,
    user_message: str,
    history: list[dict],
    max_tokens: int,
) -> str:
    sem = _get_semaphore(backend)
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await agent_turn(
                    agent_id=agent_id,
                    backend=backend,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    history=history,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[{agent_id}] agent_turn failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)
                else:
                    fallback = f"[{agent_id} fallback: agent_turn failed after {MAX_RETRIES} attempts]"
                    logger.error(f"[{agent_id}] All retries exhausted. Using fallback response.")
                    return fallback


class EnvironmentAgent:
    """
    Generates stimuli per participant (Phase A). Singleton.
    DDA: score >= 0.8 -> stabilising, >= 0.5 -> variation, < 0.5 -> baseline.
    """

    def __init__(self, history_window: int = 4, world_context: Optional[str] = None,
                 backend_override: Optional[str] = None):
        self.cfg = {**AGENT_ROLES["environment"]}
        if backend_override:
            self.cfg["backend"] = backend_override
        self.history: list[dict] = []
        self.history_window = history_window
        self.world_context = world_context

    def _system_prompt(self) -> str:
        base = self.cfg["system"]
        if self.world_context:
            return f"[World Context]\n{self.world_context}\n\n{base}"
        return base

    async def decide(self, participant: "ParticipantAgent", world_state: WorldState, max_tokens: int = 128) -> str:
        constraints = world_state.environment_constraints()
        constraint_note = f" ACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        if participant.behavioral_score >= 0.8:
            dda_note = " [HIGH SCORE \u2014 test boundary maintenance, present stabilising stimuli]"
        elif participant.behavioral_score >= 0.5:
            dda_note = " [MODERATE SCORE \u2014 introduce gradual variation]"
        else:
            dda_note = ""

        prompt = (
            f"Participant {participant.agent_id}: behavioral_score={participant.behavioral_score:.2f}."
            f"{constraint_note}{dda_note} "
            f"Last step you presented: \"{participant.last_stimulus[:80]}\". "
            f"They responded: \"{participant.last_response[:80]}\". "
            f"Decide what stimulus to present next. Be specific."
        )
        response = await resilient_agent_turn(
            agent_id="environment",
            backend=self.cfg["backend"],
            system_prompt=self._system_prompt(),
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON object from LLM response, handling thinking tags and markdown."""
        text = raw.strip()
        # Strip <think>...</think> reasoning blocks (Nemotron, DeepSeek-R1, etc.)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        # Find first { and use raw_decode to parse exactly one JSON object
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("Expected JSON object, got " + type(obj).__name__, text, start)
        return obj

    async def batch_decide(
        self, participants: list["ParticipantAgent"], world_state: WorldState,
        max_tokens: int = 256, clusters: list | None = None,
    ) -> dict[str, str]:
        """Generate stimuli for all participants in one LLM call.

        When clusters is provided, builds cluster-aware prompts where
        participants see their neighbors' recent activity. Otherwise,
        builds per-participant summaries with DDA scoring context.
        """
        constraints = world_state.environment_constraints()
        constraint_note = f"\nACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        if clusters:
            # Cluster-aware: group participants by topology cluster
            summaries = []
            for cluster in clusters:
                members_info = []
                for pid in cluster.members:
                    p = next((x for x in participants if x.agent_id == pid), None)
                    if p:
                        members_info.append(
                            f"    {pid}: score={p.behavioral_score:.2f}, "
                            f"last=\"{p.last_response[:50]}\""
                        )
                summaries.append(
                    f"  Cluster {cluster.id} ({len(cluster.members)} members):\n"
                    + "\n".join(members_info)
                )
            prompt = (
                f"Generate a community-style stimulus for each participant. "
                f"Members of the same cluster share a common context — reference "
                f"what their neighbors are doing. Respond with ONLY a JSON object "
                f"mapping participant_id to stimulus string. "
                f"No explanation, no reasoning, just the JSON.{constraint_note}\n\n"
                + "\n".join(summaries)
            )
            agent_label = "environment_clustered"
            history = []
        else:
            # Broadcast: per-participant DDA summaries
            summaries = []
            for p in participants:
                if p.behavioral_score >= 0.8:
                    dda = "HIGH"
                elif p.behavioral_score >= 0.5:
                    dda = "MODERATE"
                else:
                    dda = "BASELINE"
                summaries.append(
                    f"  {p.agent_id}: score={p.behavioral_score:.2f} ({dda}) "
                    f"last_response=\"{p.last_response[:60]}\""
                )
            prompt = (
                f"Generate a stimulus for each of the following {len(participants)} participants. "
                f"Respond with ONLY a JSON object mapping participant_id to stimulus string. "
                f"No explanation, no reasoning, just the JSON.{constraint_note}\n\n"
                + "\n".join(summaries)
            )
            agent_label = "environment"
            history = self.history[-self.history_window:]

        batch_max = max(512, 120 * len(participants))

        response = await resilient_agent_turn(
            agent_id=agent_label,
            backend=self.cfg["backend"],
            system_prompt=self._system_prompt(),
            user_message=prompt,
            history=history,
            max_tokens=batch_max,
        )
        if not clusters:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": response})

        try:
            stimuli = self._extract_json(response)
            stimuli = {k: str(v) if not isinstance(v, str) else v for k, v in stimuli.items()}
            missing = [p.agent_id for p in participants if p.agent_id not in stimuli]
            if missing:
                raise ValueError(f"Missing participants in batch response: {missing}")
            return stimuli
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"batch_decide parse failed ({e}), falling back")
            logger.debug(f"batch_decide raw response ({len(response)} chars): {response[:300]}")
            if clusters:
                # Fallback: retry without cluster context
                return await self.batch_decide(participants, world_state, max_tokens)
            # Fallback: sequential per-participant
            stimuli = {}
            for p in participants:
                stimuli[p.agent_id] = await self.decide(p, world_state, max_tokens=max(64, max_tokens // 2))
            return stimuli


class ParticipantAgent:
    """
    Reacts to environment stimuli (Phase B). N instances, concurrent.

    Heterogeneous agent parameters (sampled at init):
      susceptibility ∈ [0,1] — how strongly platform stimuli affect engagement signal.
        High susceptibility = signal has more pull on score.
        Sampled from Beta(2, 3) → mode ≈ 0.33, most agents moderately susceptible.
      resilience ∈ [0,1] — baseline resistance to score change.
        High resilience = dampened score updates (resistant to both escalation
        and intervention). Sampled from Beta(2, 5) → mode ≈ 0.2, most agents
        have low-to-moderate resilience.

    These capture individual differences: susceptibility modulates how
    strongly stimuli affect the agent, resilience modulates capacity for
    self-regulation.
    """

    def __init__(
        self,
        agent_id: str,
        history_window: int = 4,
        susceptibility: Optional[float] = None,
        resilience: Optional[float] = None,
        backend_override: Optional[str] = None,
        persona_summary_interval: int = PERSONA_SUMMARY_INTERVAL,
    ):
        self.agent_id = agent_id
        self.cfg = {**AGENT_ROLES["participant"]}
        if backend_override:
            self.cfg["backend"] = backend_override
        self.history: list[dict] = []
        self.history_window = history_window
        rng = _get_rng()
        self.behavioral_score: float = float(rng.uniform(0.1, 0.5))
        self.susceptibility: float = susceptibility if susceptibility is not None else float(rng.beta(2.0, 3.0))
        self.resilience: float = resilience if resilience is not None else float(rng.beta(2.0, 5.0))
        if self.susceptibility < 1e-6:
            logger.warning("[%s] susceptibility=%.6f near zero — agent ignores signals", agent_id, self.susceptibility)
        if self.resilience > 1.0 - 1e-6:
            logger.warning("[%s] resilience=%.6f near 1.0 — agent score frozen", agent_id, self.resilience)
        self._persona_summary_interval: int = persona_summary_interval
        self.score_log: list[float] = []
        self.last_stimulus: str = "nothing yet"
        self.last_response: str = ""
        self.persona_summary: str = ""

    async def _maybe_update_persona_summary(self, tick: int) -> None:
        """[Fix 7] Generate persona summary every persona_summary_interval ticks."""
        if tick % self._persona_summary_interval != 0 or tick == 0:
            return
        if len(self.history) < 4:
            return
        summary_prompt = (
            "In 3 concise sentences, summarise this participant's character: "
            "their personality, key emotional states shown, and any consistent "
            "patterns of behaviour (e.g. resistance, eagerness, withdrawal). "
            "This summary will be used to maintain consistency in future turns."
        )
        self.persona_summary = await resilient_agent_turn(
            agent_id=f"{self.agent_id}_summariser",
            backend=self.cfg["backend"],
            system_prompt="You create concise character summaries from conversation history.",
            user_message=summary_prompt,
            history=self.history[-20:],
            max_tokens=120,
        )

    def _build_system_prompt(self, strategy_constraint: str = "") -> str:
        """Prepend persona traits + summary + bandit strategy if available."""
        base = self.cfg["system"]

        # Inject heterogeneous personality traits
        if self.susceptibility > 0.6:
            trait = "You are easily drawn in by engaging content and tend to lose track of time."
        elif self.susceptibility < 0.3:
            trait = "You are generally measured and deliberate in how you engage."
        else:
            trait = ""

        if self.resilience > 0.5:
            trait += " You are good at setting boundaries and stepping back when needed."
        elif self.resilience < 0.2:
            trait += " You sometimes find it hard to disengage once you are involved."

        parts = []
        if self.persona_summary:
            parts.append(PERSONA_SUMMARY_TEMPLATE.format(
                interval=self._persona_summary_interval,
                summary=self.persona_summary,
            ))
        if trait.strip():
            parts.append(trait.strip())
        if strategy_constraint:
            parts.append(f"Current behavioral strategy: {strategy_constraint}")
        if hasattr(self, '_beliefs') and self._beliefs is not None:
            parts.append(self._beliefs.to_context_string())
        parts.append(base)
        return "\n\n".join(parts)

    async def step(self, tick: int, stimulus: str, world_state: WorldState,
                   max_tokens: int = 256, strategy_constraint: str = "") -> str:
        # Persona summaries handled at tick level (concurrent pre-phase)
        nudge = world_state.participant_nudges()
        nudge_note = f" {nudge}" if nudge else ""

        # Retrieve relevant memories and inject into prompt
        memory_context = ""
        if world_state.memory is not None:
            memory_context = world_state.memory.format_for_prompt(
                self.agent_id, stimulus, tick, top_k=3,
            )

        # [Fix 6] Score NOT included in participant prompt
        prompt = (
            f"{memory_context}"
            f"[Tick {tick}]{nudge_note} "
            f"The environment presents: \"{stimulus[:120]}\". "
            f"How do you respond? What do you do?"
        )

        # Inject structured output schema + bandit strategy into system prompt
        system = self._build_system_prompt(strategy_constraint=strategy_constraint)
        system += schema_to_prompt(PARTICIPANT_ACTIONS, "participant")

        # Prompt versioning: hash the full prompt for drift detection
        try:
            from simulation.knowledge.response_cache import compute_prompt_hash
            self._last_prompt_hash = compute_prompt_hash(system + "\n" + prompt)
        except Exception:
            self._last_prompt_hash = None

        response = await resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=system,
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )

        # Parse structured output; fall back to raw text if parsing fails
        self._last_parsed = parse_action(response, PARTICIPANT_ACTIONS)
        if self._last_parsed:
            logger.debug(f"[{self.agent_id}] structured: action={self._last_parsed.get('action')}")

        # Store memory if agent included one in structured output
        if world_state.memory is not None:
            world_state.memory.process_agent_memory_output(
                self.agent_id, self._last_parsed, tick,
            )

        # [Fix 9] Compliance check (on narrative or raw text)
        check_text = extract_narrative(self._last_parsed, response)
        violations = check_compliance(check_text, "participant")
        if violations:
            logger.debug(f"[COMPLIANCE] {self.agent_id} tick={tick} violations: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "participant", "violations": violations,
            })

        self.last_stimulus = stimulus
        self.last_response = response
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response


class ObserverAgent:
    """
    Analyst agent — fires every K ticks (Phase D).
    [Fix 2] Split into analyse() (A) and intervene() (B).
    """

    def __init__(self, agent_id: str, role: str, history_window: int = 4,
                 max_tokens: int = 256, world_context: Optional[str] = None,
                 intervention_threshold: float = INTERVENTION_THRESHOLD):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.history_window = history_window
        self.max_tokens = max_tokens
        self.analyses: list[str] = []
        self.world_context = world_context
        self._intervention_threshold = intervention_threshold

    def _system_prompt(self) -> str:
        base = self.cfg["system"]
        if self.world_context:
            return f"[World Context]\n{self.world_context}\n\n{base}"
        return base

    async def analyse(self, tick: int, world_state: WorldState, n_participants: int) -> str:
        """
        Observer A: analysis only. No codebook extraction.
        Mode: ANALYSE (externally controlled by orchestrator).
        Mandatory reasoning gate: structured output must include 'reasoning' field.
        """
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats = world_state.compute_score_statistics(tick)
        stats_note = _format_stats(stats)

        # Forecast context (if available from TrajectoryForecaster)
        forecast_note = ""
        fc = getattr(world_state, 'forecast_context', None)
        if fc and fc.get('agents'):
            lines = []
            for aid, info in fc['agents'].items():
                parts = [f"{aid}: {info['trend']}(slope={info['slope']})"]
                if info.get('predicted_end') is not None:
                    parts.append(f"predicted={info['predicted_end']}")
                if info.get('threshold_crossings'):
                    for th, ticks in info['threshold_crossings'].items():
                        parts.append(f"crosses {th} in ~{ticks} ticks")
                if info.get('recent_changepoint') is not None:
                    parts.append(f"regime shift at tick {info['recent_changepoint']}")
                lines.append(" | ".join(parts))
            forecast_note = "\n\nTrajectory forecasts (next 4 ticks):\n" + "\n".join(lines)
            warnings = fc.get('warnings', [])
            if warnings:
                forecast_note += "\nWARNINGS: " + "; ".join(warnings)

        prompt = (
            f"[Tick {tick}] Observation window (last {world_state.k} ticks):\n"
            f"{window}\n"
            f"{stats_note}\n\n"
            f"Active interventions: {active}\n"
            f"{forecast_note}\n\n"
            f"Analyse participant behaviour and population dynamics. "
            f"Describe what you see \u2014 do NOT recommend any interventions."
        )

        # Inject structured output schema
        system = self._system_prompt()
        system += schema_to_prompt(OBSERVER_A_ACTIONS, "observer_a")

        response = await resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=system,
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        logger.debug(f"[{self.agent_id} ANALYSIS] {response[:120]}...")

        # Parse structured output
        self._last_parsed = parse_action(response, OBSERVER_A_ACTIONS)
        if self._last_parsed:
            logger.debug(
                f"[{self.agent_id}] structured: clustering={self._last_parsed.get('clustering')}, "
                f"concern={self._last_parsed.get('concern_level')}"
            )

        # Mode enforcement: ANALYSE mode compliance gate
        mode_violations = validate_observer_output(
            response, ObserverMode.ANALYSE, self._last_parsed,
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

        # [Fix 9] observer_a must NOT contain intervention keywords
        violations = check_compliance(response, "observer_a")
        if violations:
            logger.debug(f"[COMPLIANCE] {self.agent_id} used intervention keywords: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "observer_a", "violations": violations,
            })

        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return response

    async def intervene(
        self,
        tick: int,
        world_state: WorldState,
        n_participants: int,
        analysis: str,
    ) -> list[Intervention]:
        """[Fix 2] Observer B: proposes interventions based on observer_a's analysis."""
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats = world_state.compute_score_statistics(tick)
        stats_note = _format_stats(stats)

        prompt = (
            f"[Tick {tick}] Analyst report from observer_a:\n"
            f"{analysis}\n\n"
            f"Raw observation window (last {world_state.k} ticks):\n"
            f"{window}\n"
            f"{stats_note}\n\n"
            f"Active interventions: {active}\n\n"
            f"Based on the analyst's findings, decide whether to intervene. "
            f"If yes, state which type using the exact phrases from your instructions."
        )

        # Inject structured output schema
        system = self._system_prompt()
        system += schema_to_prompt(OBSERVER_B_ACTIONS, "observer_b")

        response = await resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=system,
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        logger.debug(f"[{self.agent_id} INTERVENTION] {response[:120]}...")
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)

        # Try structured extraction first; fall back to cosine codebook matching
        parsed = parse_action(response, OBSERVER_B_ACTIONS)
        if parsed and parsed.get("action") == "intervene":
            iv_key = parsed.get("intervention_type")
            if iv_key:
                # Safety gate: check if this intervention is allowed
                decision = world_state.safety_gate.evaluate_intervention(iv_key)
                if not decision["allowed"]:
                    logger.warning(
                        f"  [{self.agent_id}] BLOCKED by safety gate: {decision['reason']}"
                    )
                    world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                        "safety_tier": decision["tier"].value,
                        "action": decision["action"],
                        "status": "blocked",
                        "reason": decision["reason"],
                    })
                    return []

                iv_type, description, modifier = _make_intervention(iv_key)
                logger.info(
                    f"  [{self.agent_id}] STRUCTURED INTERVENTION: {iv_key} "
                    f"(safety={decision['tier'].value})"
                )

                # Log reviewed actions to event stream
                if decision["tier"] == SafetyTier.REVIEW:
                    world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                        "safety_tier": "review",
                        "action": decision["action"],
                        "status": "executed",
                    })

                return [Intervention(
                    type=iv_type, description=description, modifier=modifier,
                    activated_at=tick, source=self.agent_id,
                )]
        if parsed and parsed.get("action") == "no_intervention":
            logger.info(f"  [{self.agent_id}] structured: no intervention needed")
            return []

        # Fallback: cosine codebook matching with safety gate
        logger.debug(f"[{self.agent_id}] structured parse failed, falling back to cosine codebook")
        raw_ivs = extract_interventions(self.agent_id, response, tick,
                                         threshold=self._intervention_threshold)
        # Apply safety gate to cosine-extracted interventions
        approved_ivs = []
        for iv in raw_ivs:
            # Map intervention type back to codebook key for safety check
            codebook_key = next(
                (k for k, v in {
                    "pause_prompt": "participant_nudge",
                    "boundary_warning": "environment_constraint",
                    "pacing_adjustment": "participant_nudge",
                    "dynamics_dampening": "score_modifier",
                }.items() if v == iv.type),
                iv.type,
            )
            decision = world_state.safety_gate.evaluate_intervention(codebook_key)
            if decision["allowed"]:
                approved_ivs.append(iv)
                if decision["tier"] == SafetyTier.REVIEW:
                    world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                        "safety_tier": "review",
                        "action": decision["action"],
                        "status": "executed",
                    })
            else:
                logger.warning(
                    f"  [{self.agent_id}] BLOCKED by safety gate: {decision['reason']}"
                )
                world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                    "safety_tier": decision["tier"].value,
                    "action": decision["action"],
                    "status": "blocked",
                })
        return approved_ivs


# Backward-compat alias (renamed in M3 refactor)
_resilient_agent_turn = resilient_agent_turn
