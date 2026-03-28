"""
Tests for Phase B: prompt versioning, variance decomposition,
nested ensemble, and archetype batching.

Run: python -m pytest tests/test_phase_b.py -v
"""

import sys
import os
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Prompt versioning tests ──────────────────────────────────────────────────

class TestPromptVersioning:

    def test_compute_prompt_hash_deterministic(self):
        from simulation.response_cache import compute_prompt_hash
        h1 = compute_prompt_hash("You are a test agent.\n[Tick 1] Hello")
        h2 = compute_prompt_hash("You are a test agent.\n[Tick 1] Hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_prompt_hash_differs(self):
        from simulation.response_cache import compute_prompt_hash
        h1 = compute_prompt_hash("prompt version A")
        h2 = compute_prompt_hash("prompt version B")
        assert h1 != h2

    def test_compute_prompt_hash_empty(self):
        from simulation.response_cache import compute_prompt_hash
        h = compute_prompt_hash("")
        assert len(h) == 64

    def test_prompt_versions_table_created(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        tables = db.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "prompt_versions" in table_names

    def test_record_prompt_version_insert(self):
        from simulation.storage import get_memory_db
        from simulation.experiment_db import ExperimentDB
        db = get_memory_db()
        exp_db = ExperimentDB(db=db)
        exp_db.record_prompt_version("abc123", "test prompt", "participant", "run_test")
        row = db.execute(
            "SELECT prompt_text, agent_type FROM prompt_versions WHERE prompt_hash = 'abc123'"
        ).fetchone()
        assert row[0] == "test prompt"
        assert row[1] == "participant"

    def test_record_prompt_version_idempotent(self):
        from simulation.storage import get_memory_db
        from simulation.experiment_db import ExperimentDB
        db = get_memory_db()
        exp_db = ExperimentDB(db=db)
        exp_db.record_prompt_version("abc123", "first insert", "participant", "run_1")
        exp_db.record_prompt_version("abc123", "second insert", "participant", "run_2")
        count = db.execute("SELECT COUNT(*) FROM prompt_versions WHERE prompt_hash = 'abc123'").fetchone()
        assert count[0] == 1  # second insert ignored


# ── Variance decomposition tests ─────────────────────────────────────────────

class TestDecomposeVariance:

    def test_single_group(self):
        """Single param set: all variance is within, none epistemic."""
        from stats.validation import decompose_variance
        result = decompose_variance(
            group_means=[0.5],
            group_variances=[0.01],
            group_sizes=[10],
        )
        assert result["var_epistemic"] == 0.0
        assert abs(result["var_within"] - 0.01) < 1e-8
        assert result["pct_epistemic"] == 0.0
        assert result["pct_within"] == 1.0

    def test_all_epistemic(self):
        """Groups with zero within-variance: all variance is epistemic."""
        from stats.validation import decompose_variance
        result = decompose_variance(
            group_means=[0.2, 0.8],
            group_variances=[0.0, 0.0],
            group_sizes=[5, 5],
        )
        assert result["var_within"] == 0.0
        assert result["var_epistemic"] > 0
        assert result["pct_epistemic"] == 1.0

    def test_balanced_decomposition(self):
        """Mixed case with both epistemic and within-group variance."""
        from stats.validation import decompose_variance
        result = decompose_variance(
            group_means=[0.3, 0.5, 0.7],
            group_variances=[0.01, 0.02, 0.01],
            group_sizes=[5, 5, 5],
        )
        assert result["var_epistemic"] > 0
        assert result["var_within"] > 0
        assert abs(result["var_total"] - result["var_epistemic"] - result["var_within"]) < 1e-8

    def test_variance_sums(self):
        """var_total == var_epistemic + var_within always."""
        from stats.validation import decompose_variance
        for _ in range(10):
            rng = np.random.default_rng(42)
            n_groups = rng.integers(2, 6)
            means = rng.uniform(0.1, 0.9, n_groups).tolist()
            variances = rng.uniform(0.001, 0.05, n_groups).tolist()
            sizes = rng.integers(3, 20, n_groups).tolist()
            result = decompose_variance(means, variances, sizes)
            assert abs(result["var_total"] - result["var_epistemic"] - result["var_within"]) < 1e-6

    def test_weighted_means(self):
        """Larger groups have more weight."""
        from stats.validation import decompose_variance
        # Group 1: mean=0.2, n=100. Group 2: mean=0.8, n=1.
        # Grand mean should be near 0.2, not 0.5
        result = decompose_variance(
            group_means=[0.2, 0.8],
            group_variances=[0.01, 0.01],
            group_sizes=[100, 1],
        )
        # var_epistemic should be small since grand mean is dominated by group 1
        assert result["var_epistemic"] < 0.01


# ── Nested ensemble tests ────────────────────────────────────────────────────

class TestNestedEnsemble:

    @staticmethod
    def _mock_participants(n=4, seed=42):
        rng = np.random.RandomState(seed)
        participants = []
        for i in range(n):
            p = MagicMock()
            p.agent_id = f"participant_{i}"
            p.behavioral_score = rng.uniform(0.3, 0.7)
            p.score_log = [rng.uniform(0.2, 0.8) for _ in range(12)]
            p.susceptibility = rng.beta(2, 3)
            p.resilience = rng.beta(2, 5)
            participants.append(p)
        return participants

    @staticmethod
    def _mock_world_state():
        ws = MagicMock()
        ws.stream = MagicMock()
        ws.stream.export.return_value = []
        return ws

    def test_nested_ensemble_basic(self):
        from simulation.ensemble import run_nested_ensemble

        call_count = 0

        async def mock_sim(**kwargs):
            nonlocal call_count
            call_count += 1
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_nested_ensemble(
                    parameter_grid=[{"scoring_alpha": 0.1}, {"scoring_alpha": 0.2}],
                    n_inner=3,
                    base_seed=42,
                )

        result = asyncio.run(_run())
        assert call_count == 6  # 2 param sets * 3 inner
        assert result.total_runs == 6
        assert len(result.param_sets) == 2

    def test_nested_ensemble_variance_decomposition(self):
        from simulation.ensemble import run_nested_ensemble

        # Make param sets produce clearly different means
        async def mock_sim(**kwargs):
            seed = kwargs.get("seed", 42)
            # param set 0 (seeds 42-44): low scores
            # param set 1 (seeds 45-47): high scores
            if seed < 45:
                ps = self._mock_participants(seed=seed)
                for p in ps:
                    p.behavioral_score = 0.3 + np.random.RandomState(seed).uniform(-0.05, 0.05)
            else:
                ps = self._mock_participants(seed=seed)
                for p in ps:
                    p.behavioral_score = 0.7 + np.random.RandomState(seed).uniform(-0.05, 0.05)
            return ps, self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_nested_ensemble(
                    parameter_grid=[{"scoring_alpha": 0.1}, {"scoring_alpha": 0.3}],
                    n_inner=3,
                    base_seed=42,
                )

        result = asyncio.run(_run())
        vd = result.variance_decomposition
        assert "var_epistemic" in vd
        assert "var_within" in vd
        assert "var_total" in vd
        # With clearly different means, epistemic should dominate
        assert vd["var_epistemic"] > 0
        assert vd["pct_epistemic"] > 0.5

    def test_nested_ensemble_sequential_seeds(self):
        from simulation.ensemble import run_nested_ensemble
        seeds_used = []

        async def mock_sim(**kwargs):
            seeds_used.append(kwargs["seed"])
            return self._mock_participants(seed=kwargs["seed"]), self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                await run_nested_ensemble(
                    parameter_grid=[{}, {}],
                    n_inner=2,
                    base_seed=100,
                )

        asyncio.run(_run())
        # param set 0: 100, 101. param set 1: 102, 103
        assert seeds_used == [100, 101, 102, 103]

    def test_nested_ensemble_failed_run(self):
        from simulation.ensemble import run_nested_ensemble
        call_count = 0

        async def mock_sim(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fail")
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_nested_ensemble(
                    parameter_grid=[{}],
                    n_inner=3,
                    base_seed=42,
                )

        result = asyncio.run(_run())
        assert result.total_completed == 2  # 1 failed out of 3
        assert result.total_runs == 3

    def test_nested_result_to_dict(self):
        from simulation.ensemble import NestedEnsembleResult
        result = NestedEnsembleResult(
            experiment_id="exp_001",
            param_sets=[{"alpha": 0.1}],
            variance_decomposition={"var_epistemic": 0.01, "var_within": 0.02},
        )
        d = result.to_dict()
        assert d["experiment_id"] == "exp_001"
        assert d["n_param_sets"] == 1
        json.dumps(d)  # must be JSON-serializable


# ── Archetype batching tests ─────────────────────────────────────────────────

class TestBatchingConfig:

    def test_default_disabled(self):
        from simulation.scenario import BatchingConfig
        bc = BatchingConfig()
        assert bc.enabled is False
        assert bc.mode == "representative"
        assert bc.min_group_size == 4

    def test_custom_values(self):
        from simulation.scenario import BatchingConfig
        bc = BatchingConfig(enabled=True, mode="cache", min_group_size=2)
        assert bc.enabled is True
        assert bc.mode == "cache"

    def test_invalid_mode_rejected(self):
        from simulation.scenario import BatchingConfig
        with pytest.raises(Exception):
            BatchingConfig(mode="invalid")

    def test_scenario_config_has_batching(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "description": "test"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
        })
        assert hasattr(config, "batching")
        assert config.batching.enabled is False

    def test_scenario_with_batching_enabled(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "batch_test", "description": "test"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 8},
            ]},
            "batching": {"enabled": True, "mode": "representative", "min_group_size": 2},
        })
        assert config.batching.enabled is True
        assert config.batching.min_group_size == 2


class TestBatchPhaseB:

    def test_batch_reduces_calls(self):
        """Representative batching should make K calls, not N."""
        from simulation.sim_loop import _batch_phase_b

        call_count = 0
        original_step = None

        # Create 6 mock participants in 2 archetype groups
        participants = []
        for i in range(6):
            p = MagicMock()
            p.agent_id = f"participant_{i}"
            p.history = []
            p._last_parsed = None
            p._last_prompt_hash = f"hash_{i}"

            async def mock_step(tick, stim, ws, max_tokens=256, strategy_constraint="", _p=p):
                nonlocal call_count
                call_count += 1
                _p.history.append({"role": "assistant", "content": f"response from {_p.agent_id}"})
                _p._last_parsed = {"action": "respond"}
                return f"response from {_p.agent_id}"

            p.step = mock_step
            participants.append(p)

        archetype_groups = {
            "aggressive": ["participant_0", "participant_1", "participant_2", "participant_3"],
            "cautious": ["participant_4", "participant_5"],
        }
        stimuli = {f"participant_{i}": f"stimulus {i}" for i in range(6)}
        bandit_selections = {f"participant_{i}": "" for i in range(6)}
        world_state = MagicMock()

        responses = asyncio.run(_batch_phase_b(
            tick=1, participants=participants, world_state=world_state,
            stimuli=stimuli, archetype_groups=archetype_groups,
            min_group_size=3, max_tokens=256,
            bandit_selections=bandit_selections,
        ))

        # "aggressive" group (4 agents, >= 3): 1 call for representative
        # "cautious" group (2 agents, < 3): 2 normal calls
        # Total: 3 calls, not 6
        assert call_count == 3
        assert len(responses) == 6

    def test_batch_tags_batched_from(self):
        """Agents that received a copied response should have _batched_from set."""
        from simulation.sim_loop import _batch_phase_b

        participants = []
        for i in range(4):
            p = MagicMock()
            p.agent_id = f"participant_{i}"
            p.history = []
            p._last_parsed = None
            p._last_prompt_hash = None

            async def mock_step(tick, stim, ws, max_tokens=256, strategy_constraint="", _p=p):
                _p.history.append({"role": "assistant", "content": "shared"})
                _p._last_parsed = {"action": "respond"}
                _p._last_prompt_hash = "hash_rep"
                return "shared"

            p.step = mock_step
            participants.append(p)

        archetype_groups = {"profile_a": ["participant_0", "participant_1", "participant_2", "participant_3"]}
        stimuli = {f"participant_{i}": "stim" for i in range(4)}

        asyncio.run(_batch_phase_b(
            tick=1, participants=participants, world_state=MagicMock(),
            stimuli=stimuli, archetype_groups=archetype_groups,
            min_group_size=2, max_tokens=256,
            bandit_selections={f"participant_{i}": "" for i in range(4)},
        ))

        # participant_0 is the representative — no batched_from
        assert participants[0]._batched_from is None
        # participants 1-3 got copied response
        assert participants[1]._batched_from == "participant_0"
        assert participants[2]._batched_from == "participant_0"
        assert participants[3]._batched_from == "participant_0"
