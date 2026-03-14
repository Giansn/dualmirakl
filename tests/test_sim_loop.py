"""
Tests for simulation loop — validates imports, math, data structures, and
provides a duration estimator for planning GPU time.

Run: python -m pytest tests/test_sim_loop.py -v
      (from dualmirakl/ root)

All tests use mocks — no live vLLM servers needed.
"""

import asyncio
import math
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Import tests ──────────────────────────────────────────────────────────────

class TestImports:
    def test_agent_rolesv3_imports(self):
        from simulation.agent_rolesv3 import (
            AGENT_ROLES, INTERVENTION_CODEBOOK, ENGAGEMENT_ANCHORS,
            INTERVENTION_THRESHOLD, PERSONA_SUMMARY_INTERVAL,
            PERSONA_SUMMARY_TEMPLATE, EMBED_BATCH_SIZE, check_compliance,
        )
        assert set(AGENT_ROLES.keys()) == {"observer_a", "observer_b", "participant", "environment"}
        assert len(ENGAGEMENT_ANCHORS["high"]) == 10
        assert len(ENGAGEMENT_ANCHORS["low"]) == 10
        assert 0.0 < INTERVENTION_THRESHOLD < 1.0

    def test_sim_loop_imports(self):
        from simulation.sim_loop import (
            WorldState, ObsEntry, Intervention,
            EnvironmentAgent, ParticipantAgent, ObserverAgent,
            update_score, run_tick, run_simulation,
            embed_score_batch, morris_screening, sobol_first_order,
        )

    def test_orchestrator_imports(self):
        from orchestrator import agent_turn, chat, close_client, health_check


# ── WorldState tests ──────────────────────────────────────────────────────────

class TestWorldState:
    def test_log_and_retrieve(self):
        from simulation.sim_loop import WorldState, ObsEntry
        ws = WorldState(k=3)
        entry = ObsEntry(tick=1, participant_id="p_0", score_before=0.3,
                         score_after=0.35, stimulus="hi", response="hello",
                         signal=0.5, signal_se=0.02)
        ws.log(entry)
        assert len(ws.full_log()) == 1
        assert ws.full_log()[0].participant_id == "p_0"

    def test_observer_prompt_window_small(self):
        from simulation.sim_loop import WorldState, ObsEntry
        ws = WorldState(k=2)
        for t in range(1, 4):
            ws.log(ObsEntry(tick=t, participant_id="p_0",
                            score_before=0.3, score_after=0.35,
                            stimulus="s", response="r",
                            signal=0.5, signal_se=0.01))
        window = ws.observer_prompt_window(3, n_participants=4)
        assert "p_0" in window
        # Should only include ticks > 3-2 = 1, so ticks 2 and 3
        assert "[T1]" not in window
        assert "[T2]" in window

    def test_observer_prompt_window_large_cohort(self):
        from simulation.sim_loop import WorldState, ObsEntry
        ws = WorldState(k=2)
        for t in range(1, 3):
            for i in range(20):
                ws.log(ObsEntry(tick=t, participant_id=f"p_{i}",
                                score_before=0.3, score_after=0.35,
                                stimulus="s", response="r",
                                signal=0.5, signal_se=0.01))
        window = ws.observer_prompt_window(2, n_participants=20)
        # Large cohort -> aggregate stats, not per-participant
        assert "mean score" in window
        assert "p_0" not in window

    def test_score_dampening(self):
        from simulation.sim_loop import WorldState, Intervention
        ws = WorldState(k=3)
        ws.active_interventions.append(Intervention(
            type="score_modifier", description="test",
            modifier={"dampening": 0.6}, activated_at=1,
        ))
        assert abs(ws.score_dampening() - 0.6) < 1e-6

    def test_apply_interventions_expiry(self):
        from simulation.sim_loop import WorldState, Intervention
        ws = WorldState(k=3)
        ws.active_interventions.append(Intervention(
            type="participant_nudge", description="nudge",
            modifier={}, activated_at=1, duration=1,
        ))
        ws.active_interventions.append(Intervention(
            type="participant_nudge", description="permanent",
            modifier={}, activated_at=1, duration=-1,
        ))
        ws.apply_interventions()
        # duration=1 decremented to 0, stays this tick
        assert len(ws.active_interventions) == 2
        ws.apply_interventions()
        # duration=0 now removed
        assert len(ws.active_interventions) == 1
        assert ws.active_interventions[0].description == "permanent"

    def test_compute_score_statistics(self):
        from simulation.sim_loop import WorldState, ObsEntry
        ws = WorldState(k=3)
        scores = [0.3, 0.5, 0.7, 0.9]
        for i, s in enumerate(scores):
            ws.log(ObsEntry(tick=1, participant_id=f"p_{i}",
                            score_before=s - 0.05, score_after=s,
                            stimulus="s", response="r",
                            signal=0.5, signal_se=0.01))
        stats = ws.compute_score_statistics(1)
        assert abs(stats["mean"] - np.mean(scores)) < 1e-6
        assert stats["n_above_threshold"] == 2  # 0.7 and 0.9
        assert stats["n_total"] == 4

    def test_compliance_log(self):
        from simulation.sim_loop import WorldState
        ws = WorldState(k=3)
        ws._compliance_log.append({"tick": 1, "agent": "p_0", "violations": ["test"]})
        assert len(ws.compliance_report()) == 1


# ── Score math tests ──────────────────────────────────────────────────────────

class TestScoreMath:
    def test_update_score_basic(self):
        from simulation.sim_loop import update_score
        # Score 0.3, signal 0.7, alpha 0.2, no dampening
        new = update_score(0.3, 0.7, dampening=1.0, alpha=0.2)
        expected = 0.3 + 0.2 * (0.7 - 0.3)
        assert abs(new - expected) < 1e-6

    def test_update_score_dampened(self):
        from simulation.sim_loop import update_score
        new = update_score(0.3, 0.7, dampening=0.5, alpha=0.2)
        expected = 0.3 + 0.5 * 0.2 * (0.7 - 0.3)
        assert abs(new - expected) < 1e-6

    def test_update_score_clamps(self):
        from simulation.sim_loop import update_score
        assert update_score(0.99, 1.0, alpha=0.5) <= 1.0
        assert update_score(0.01, 0.0, alpha=0.5) >= 0.0

    def test_cosine_similarity(self):
        from simulation.sim_loop import _cosine
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine(a, b) - 1.0) < 1e-6
        c = np.array([0.0, 1.0, 0.0])
        assert abs(_cosine(a, c)) < 1e-6


# ── Compliance checker tests ─────────────────────────────────────────────────

class TestCompliance:
    def test_participant_violation(self):
        from simulation.agent_rolesv3 import check_compliance
        violations = check_compliance("I am an AI and I cannot help", "participant")
        assert len(violations) >= 1

    def test_participant_clean(self):
        from simulation.agent_rolesv3 import check_compliance
        violations = check_compliance("I feel overwhelmed by this situation", "participant")
        assert len(violations) == 0

    def test_observer_a_violation(self):
        from simulation.agent_rolesv3 import check_compliance
        violations = check_compliance("I suggest we take a pause and dampen the dynamics", "observer_a")
        assert len(violations) >= 1

    def test_unknown_role(self):
        from simulation.agent_rolesv3 import check_compliance
        violations = check_compliance("anything", "nonexistent_role")
        assert violations == []


# ── Agent class tests (mocked) ───────────────────────────────────────────────

class TestAgentClasses:
    def test_participant_init(self):
        from simulation.sim_loop import ParticipantAgent, set_seed
        set_seed(42)
        p = ParticipantAgent("p_0", history_window=4)
        assert p.agent_id == "p_0"
        assert 0.1 <= p.behavioral_score <= 0.5
        assert p.history == []
        assert p.persona_summary == ""

    def test_environment_init(self):
        from simulation.sim_loop import EnvironmentAgent
        e = EnvironmentAgent(history_window=4)
        assert e.history == []

    def test_observer_init(self):
        from simulation.sim_loop import ObserverAgent
        o = ObserverAgent("observer_a", "observer_a", history_window=4, max_tokens=256)
        assert o.agent_id == "observer_a"
        assert o.analyses == []

    def test_participant_build_system_prompt_no_summary(self):
        from simulation.sim_loop import ParticipantAgent, set_seed
        set_seed(42)
        p = ParticipantAgent("p_0")
        prompt = p._build_system_prompt()
        assert "PERSONA CONTINUITY" not in prompt
        assert "You are playing the role" in prompt

    def test_participant_build_system_prompt_with_summary(self):
        from simulation.sim_loop import ParticipantAgent, set_seed
        set_seed(42)
        p = ParticipantAgent("p_0")
        p.persona_summary = "A cautious individual who resists pressure."
        prompt = p._build_system_prompt()
        assert "PERSONA CONTINUITY" in prompt
        assert "cautious individual" in prompt


# ── ObsEntry tests ────────────────────────────────────────────────────────────

class TestObsEntry:
    def test_to_str(self):
        from simulation.sim_loop import ObsEntry
        e = ObsEntry(tick=1, participant_id="p_0",
                     score_before=0.30, score_after=0.35,
                     stimulus="test stimulus", response="test response",
                     signal=0.5, signal_se=0.02)
        s = e.to_str()
        assert "[T1]" in s
        assert "p_0" in s
        assert "0.30" in s
        assert "0.35" in s


# ── Sensitivity analysis tests ────────────────────────────────────────────────

class TestSensitivityAnalysis:
    def test_morris_screening(self):
        from simulation.sim_loop import morris_screening
        # Simple quadratic: f(x) = x[0]^2 + x[1]
        def f(x):
            return x[0] ** 2 + x[1]
        results = morris_screening(f, bounds=[(0, 1), (0, 1)], r=5)
        assert 0 in results and 1 in results
        assert "mu_star" in results[0]

    def test_sobol_first_order(self):
        from simulation.sim_loop import sobol_first_order
        def f(x):
            return x[0] * 3 + x[1] * 0.1
        results = sobol_first_order(f, bounds=[(0, 1), (0, 1)], n_samples=128)
        # x[0] should dominate
        assert results[0] > results[1]


# ── Duration estimator ────────────────────────────────────────────────────────

def estimate_duration(
    n_ticks: int = 12,
    n_participants: int = 4,
    k: int = 3,
    avg_vllm_latency_s: float = 2.0,
    avg_embed_latency_s: float = 0.1,
) -> dict:
    """
    Estimate simulation wall-clock duration based on parameters.

    Per tick:
      Phase A (sequential):  n_participants * avg_vllm_latency
      Phase B (concurrent):  1 * avg_vllm_latency (asyncio.gather)
      Phase C (batch embed): 1 * avg_embed_latency
      Phase D (every K):     2 * avg_vllm_latency (sequential A→B observers)
      Persona summary:       every PERSONA_SUMMARY_INTERVAL ticks, n_participants * latency

    Returns dict with per-phase and total estimates.
    """
    persona_interval = 10  # default PERSONA_SUMMARY_INTERVAL

    phase_a_per_tick = n_participants * avg_vllm_latency_s
    phase_b_per_tick = avg_vllm_latency_s  # concurrent
    phase_c_per_tick = avg_embed_latency_s  # batch
    phase_d_per_tick = 2 * avg_vllm_latency_s  # sequential A→B

    observer_ticks = n_ticks // k
    persona_ticks = max(0, n_ticks // persona_interval - 1)  # skip tick 0

    total_a = phase_a_per_tick * n_ticks
    total_b = phase_b_per_tick * n_ticks
    total_c = phase_c_per_tick * n_ticks
    total_d = phase_d_per_tick * observer_ticks
    total_persona = persona_ticks * n_participants * avg_vllm_latency_s
    total = total_a + total_b + total_c + total_d + total_persona

    return {
        "n_ticks": n_ticks,
        "n_participants": n_participants,
        "k": k,
        "phase_a_total_s": round(total_a, 1),
        "phase_b_total_s": round(total_b, 1),
        "phase_c_total_s": round(total_c, 1),
        "phase_d_total_s": round(total_d, 1),
        "persona_summary_s": round(total_persona, 1),
        "estimated_total_s": round(total, 1),
        "estimated_total_min": round(total / 60, 1),
        "vllm_calls": (
            n_ticks * n_participants      # Phase A
            + n_ticks                     # Phase B (gather, but still N calls)
            + observer_ticks * 2          # Phase D
            + persona_ticks * n_participants  # summaries
        ),
    }


class TestDurationEstimator:
    def test_basic_estimate(self):
        est = estimate_duration(n_ticks=12, n_participants=4, k=3)
        assert est["estimated_total_s"] > 0
        assert est["vllm_calls"] > 0
        assert est["phase_d_total_s"] > 0

    def test_scales_with_participants(self):
        small = estimate_duration(n_participants=2)
        large = estimate_duration(n_participants=10)
        assert large["estimated_total_s"] > small["estimated_total_s"]

    def test_no_observer_when_k_exceeds_ticks(self):
        est = estimate_duration(n_ticks=5, k=100)
        assert est["phase_d_total_s"] == 0


# ── Print estimate when run directly ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Simulation Duration Estimates ──\n")
    configs = [
        {"n_ticks": 12, "n_participants": 4, "k": 3, "avg_vllm_latency_s": 2.0},
        {"n_ticks": 24, "n_participants": 4, "k": 3, "avg_vllm_latency_s": 2.0},
        {"n_ticks": 12, "n_participants": 10, "k": 3, "avg_vllm_latency_s": 2.0},
        {"n_ticks": 48, "n_participants": 8, "k": 6, "avg_vllm_latency_s": 1.5},
    ]
    for cfg in configs:
        est = estimate_duration(**cfg)
        print(f"  {cfg}")
        print(f"    Total: {est['estimated_total_min']} min ({est['estimated_total_s']}s)")
        print(f"    vLLM calls: {est['vllm_calls']}")
        print(f"    Phase A: {est['phase_a_total_s']}s | B: {est['phase_b_total_s']}s | "
              f"C: {est['phase_c_total_s']}s | D: {est['phase_d_total_s']}s")
        print()
