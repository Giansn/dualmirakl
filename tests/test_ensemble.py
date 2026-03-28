"""
Tests for Phase A: DuckDB schema expansion, response cache, experiment
tracking, EnsembleConfig, and ensemble orchestration.

Run: python -m pytest tests/test_ensemble.py -v
"""

import sys
import os
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Schema tests ─────────────────────────────────────────────────────────────

class TestSchema:
    """Verify the 5 new tables are created by ensure_schema."""

    def test_new_tables_created(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        tables = db.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "experiments" in table_names
        assert "runs" in table_names
        assert "tick_data" in table_names
        assert "ensemble_summaries" in table_names
        assert "response_cache" in table_names

    def test_existing_tables_still_present(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        tables = db.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "entities" in table_names
        assert "relations" in table_names
        assert "agent_memories" in table_names
        assert "generated_personas" in table_names
        assert "analysis_reports" in table_names

    def test_idempotent_schema(self):
        from simulation.storage import get_memory_db, ensure_schema
        db = get_memory_db()
        # Second call should not fail
        ensure_schema(db)
        tables = db.execute("SHOW TABLES").fetchall()
        assert len(tables) >= 10  # 5 original + 5 new

    def test_experiments_insert(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        db.execute(
            "INSERT INTO experiments (experiment_id, name, config) VALUES (?, ?, ?)",
            ["exp_001", "test_exp", '{"key": "value"}'],
        )
        row = db.execute("SELECT * FROM experiments WHERE experiment_id = 'exp_001'").fetchone()
        assert row is not None
        assert row[1] == "test_exp"  # name column

    def test_runs_insert(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        db.execute(
            "INSERT INTO runs (run_id, sim_seed, status) VALUES (?, ?, ?)",
            ["run_test_s42", 42, "running"],
        )
        row = db.execute("SELECT status FROM runs WHERE run_id = 'run_test_s42'").fetchone()
        assert row[0] == "running"

    def test_tick_data_insert(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        db.execute(
            "INSERT INTO tick_data (run_id, step, metric_name, value) VALUES (?, ?, ?, ?)",
            ["run_test", 1, "mean_score", 0.5],
        )
        rows = db.execute("SELECT * FROM tick_data WHERE run_id = 'run_test'").fetchall()
        assert len(rows) == 1

    def test_ensemble_summaries_insert(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        db.execute(
            """INSERT INTO ensemble_summaries
               (experiment_id, step, metric_name, n_runs, mean, std, p5, p25, median, p75, p95)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ["exp_001", 1, "mean_score", 10, 0.5, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
        )
        row = db.execute("SELECT n_runs FROM ensemble_summaries WHERE experiment_id = 'exp_001'").fetchone()
        assert row[0] == 10

    def test_response_cache_insert(self):
        from simulation.storage import get_memory_db
        db = get_memory_db()
        db.execute(
            """INSERT INTO response_cache
               (prompt_hash, model_id, temperature, response_text)
               VALUES (?, ?, ?, ?)""",
            ["abc123", "swarm", 0.7, "hello world"],
        )
        row = db.execute("SELECT response_text FROM response_cache WHERE prompt_hash = 'abc123'").fetchone()
        assert row[0] == "hello world"


# ── Response cache tests ─────────────────────────────────────────────────────

class TestResponseCache:
    """Verify the response cache module."""

    def _make_cache(self):
        from simulation.storage import get_memory_db
        from simulation.response_cache import ResponseCache
        db = get_memory_db()
        return ResponseCache(enabled=True, db=db)

    def test_cache_miss_returns_none(self):
        cache = self._make_cache()
        assert cache.lookup("hello", "swarm", 0.7) is None

    def test_cache_store_and_hit(self):
        cache = self._make_cache()
        cache.store("hello", "swarm", 0.7, None, "world")
        assert cache.lookup("hello", "swarm", 0.7) == "world"

    def test_cache_disabled_always_miss(self):
        from simulation.storage import get_memory_db
        from simulation.response_cache import ResponseCache
        db = get_memory_db()
        cache = ResponseCache(enabled=False, db=db)
        cache.store("hello", "swarm", 0.7, None, "world")
        assert cache.lookup("hello", "swarm", 0.7) is None

    def test_hit_count_increments(self):
        cache = self._make_cache()
        cache.store("prompt1", "authority", 0.5, None, "response1")
        cache.lookup("prompt1", "authority", 0.5)
        cache.lookup("prompt1", "authority", 0.5)
        row = cache.db.execute(
            "SELECT hit_count FROM response_cache LIMIT 1"
        ).fetchone()
        assert row[0] == 2

    def test_different_temperatures_different_entries(self):
        cache = self._make_cache()
        cache.store("prompt", "swarm", 0.0, None, "greedy_response")
        cache.store("prompt", "swarm", 0.7, None, "stochastic_response")
        assert cache.lookup("prompt", "swarm", 0.0) == "greedy_response"
        assert cache.lookup("prompt", "swarm", 0.7) == "stochastic_response"

    def test_different_seeds_different_entries(self):
        cache = self._make_cache()
        cache.store("prompt", "swarm", 0.7, 42, "response_42")
        cache.store("prompt", "swarm", 0.7, 99, "response_99")
        assert cache.lookup("prompt", "swarm", 0.7, 42) == "response_42"
        assert cache.lookup("prompt", "swarm", 0.7, 99) == "response_99"

    def test_hash_deterministic(self):
        from simulation.response_cache import ResponseCache
        h1 = ResponseCache._hash_key("test", "model", 0.7, 42)
        h2 = ResponseCache._hash_key("test", "model", 0.7, 42)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_differs_for_different_inputs(self):
        from simulation.response_cache import ResponseCache
        h1 = ResponseCache._hash_key("test_a", "model", 0.7, 42)
        h2 = ResponseCache._hash_key("test_b", "model", 0.7, 42)
        assert h1 != h2

    def test_clear(self):
        cache = self._make_cache()
        cache.store("p1", "swarm", 0.7, None, "r1")
        cache.store("p2", "swarm", 0.7, None, "r2")
        count = cache.clear()
        assert count == 2
        assert cache.lookup("p1", "swarm", 0.7) is None

    def test_stats(self):
        cache = self._make_cache()
        cache.store("prompt", "swarm", 0.7, None, "resp")
        cache.lookup("prompt", "swarm", 0.7)  # hit
        cache.lookup("other", "swarm", 0.7)   # miss
        s = cache.stats
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["total_cached"] == 1


# ── EnsembleConfig tests ─────────────────────────────────────────────────────

class TestEnsembleConfig:
    """Verify EnsembleConfig integration into ScenarioConfig."""

    def test_default_values(self):
        from simulation.scenario import EnsembleConfig
        ec = EnsembleConfig()
        assert ec.enabled is False
        assert ec.n_runs == 10
        assert ec.cv_threshold == 0.05
        assert ec.base_seed == 42

    def test_custom_values(self):
        from simulation.scenario import EnsembleConfig
        ec = EnsembleConfig(enabled=True, n_runs=20, cv_threshold=0.03)
        assert ec.enabled is True
        assert ec.n_runs == 20
        assert ec.cv_threshold == 0.03

    def test_scenario_config_has_ensemble(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "description": "test"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
        })
        assert hasattr(config, "ensemble")
        assert config.ensemble.enabled is False
        assert config.ensemble.n_runs == 10

    def test_existing_yaml_loads_without_ensemble(self):
        """Loading an existing scenario that lacks ensemble: block still works."""
        from simulation.scenario import ScenarioConfig
        # Minimal config with no ensemble key
        config = ScenarioConfig.from_dict({
            "meta": {"name": "legacy", "description": "no ensemble"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 1},
            ]},
        })
        assert config.ensemble.enabled is False

    def test_scenario_with_ensemble_enabled(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "ensemble_test", "description": "test"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
            "ensemble": {
                "enabled": True,
                "n_runs": 5,
                "cv_threshold": 0.10,
                "base_seed": 100,
            },
        })
        assert config.ensemble.enabled is True
        assert config.ensemble.n_runs == 5
        assert config.ensemble.cv_threshold == 0.10
        assert config.ensemble.base_seed == 100


# ── Experiment DB tests ──────────────────────────────────────────────────────

class TestExperimentDB:
    """Verify the experiment tracking helpers."""

    def _make_db(self):
        from simulation.storage import get_memory_db
        from simulation.experiment_db import ExperimentDB
        db = get_memory_db()
        return ExperimentDB(db=db), db

    def test_create_experiment(self):
        exp_db, db = self._make_db()
        exp_id = exp_db.create_experiment("test_exp", config={"n": 10})
        assert exp_id.startswith("exp_")
        row = db.execute(
            "SELECT name FROM experiments WHERE experiment_id = ?", [exp_id]
        ).fetchone()
        assert row[0] == "test_exp"

    def test_register_run(self):
        exp_db, db = self._make_db()
        exp_db.register_run("run_test_s42", sim_seed=42)
        row = db.execute(
            "SELECT status, sim_seed FROM runs WHERE run_id = 'run_test_s42'"
        ).fetchone()
        assert row[0] == "running"
        assert row[1] == 42

    def test_complete_run(self):
        exp_db, db = self._make_db()
        exp_db.register_run("run_test_s42", sim_seed=42)
        exp_db.complete_run("run_test_s42", wall_time_seconds=5.0)
        row = db.execute(
            "SELECT status, wall_time_seconds FROM runs WHERE run_id = 'run_test_s42'"
        ).fetchone()
        assert row[0] == "completed"
        assert row[1] == 5.0

    def test_fail_run(self):
        exp_db, db = self._make_db()
        exp_db.register_run("run_fail", sim_seed=1)
        exp_db.fail_run("run_fail", wall_time_seconds=1.0)
        row = db.execute(
            "SELECT status FROM runs WHERE run_id = 'run_fail'"
        ).fetchone()
        assert row[0] == "failed"

    def test_tick_data_buffered_flush(self):
        exp_db, db = self._make_db()
        exp_db.record_tick("run_test", 1, {"mean_score": 0.5, "std_score": 0.1})
        # Not yet in DB
        rows = db.execute("SELECT * FROM tick_data").fetchall()
        assert len(rows) == 0
        # Flush
        count = exp_db.flush_ticks()
        assert count == 2
        rows = db.execute("SELECT * FROM tick_data").fetchall()
        assert len(rows) == 2

    def test_record_tick_with_agent_id(self):
        exp_db, db = self._make_db()
        exp_db.record_tick("run_test", 1, {"score": 0.6}, agent_id="participant_0")
        exp_db.flush_ticks()
        row = db.execute(
            "SELECT agent_id, value FROM tick_data WHERE agent_id = 'participant_0'"
        ).fetchone()
        assert row[0] == "participant_0"
        assert abs(row[1] - 0.6) < 1e-5

    def test_write_ensemble_summary(self):
        exp_db, db = self._make_db()
        stats = {
            "n_runs": 5, "mean": 0.5, "std": 0.1,
            "p5": 0.3, "p25": 0.4, "median": 0.5, "p75": 0.6, "p95": 0.7,
        }
        exp_db.write_ensemble_summary("exp_001", 1, "mean_score", stats)
        row = db.execute(
            "SELECT n_runs, mean, p95 FROM ensemble_summaries WHERE experiment_id = 'exp_001'"
        ).fetchone()
        assert row[0] == 5
        assert abs(row[1] - 0.5) < 1e-5
        assert abs(row[2] - 0.7) < 1e-5

    def test_get_experiment_runs(self):
        exp_db, db = self._make_db()
        exp_id = exp_db.create_experiment("multi_run")
        exp_db.register_run("run_1", experiment_id=exp_id, sim_seed=1)
        exp_db.register_run("run_2", experiment_id=exp_id, sim_seed=2)
        runs = exp_db.get_experiment_runs(exp_id)
        assert len(runs) == 2

    def test_get_tick_data(self):
        exp_db, db = self._make_db()
        exp_db.record_tick("run_test", 1, {"mean_score": 0.5})
        exp_db.record_tick("run_test", 2, {"mean_score": 0.6})
        exp_db.flush_ticks()
        data = exp_db.get_tick_data("run_test", "mean_score")
        assert len(data) == 2
        assert abs(data[0]["value"] - 0.5) < 1e-5
        assert abs(data[1]["value"] - 0.6) < 1e-5


# ── Ensemble loop tests ──────────────────────────────────────────────────────

class TestEnsembleHelpers:
    """Test ensemble helper functions."""

    def test_check_convergence_converged(self):
        from simulation.ensemble import _check_convergence
        # Very tight cluster → low CV
        values = [0.50, 0.51, 0.49, 0.50, 0.50]
        result = _check_convergence(values, cv_threshold=0.05)
        assert result["converged"] is True
        assert result["cv"] < 0.05

    def test_check_convergence_not_converged(self):
        from simulation.ensemble import _check_convergence
        # Wide spread → high CV
        values = [0.1, 0.5, 0.9, 0.2, 0.8]
        result = _check_convergence(values, cv_threshold=0.05)
        assert result["converged"] is False
        assert result["cv"] > 0.05

    def test_compute_percentile_bands(self):
        from simulation.ensemble import _compute_percentile_bands
        # 3 runs, 2 agents each, 4 ticks
        all_score_logs = [
            [[0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7]],  # run 1
            [[0.35, 0.45, 0.55, 0.65], [0.45, 0.55, 0.65, 0.75]],  # run 2
            [[0.32, 0.42, 0.52, 0.62], [0.42, 0.52, 0.62, 0.72]],  # run 3
        ]
        bands = _compute_percentile_bands(all_score_logs, 4)
        assert len(bands) == 4
        assert 1 in bands
        assert "mean" in bands[1]
        assert "p5" in bands[1]
        assert "p95" in bands[1]
        assert bands[1]["n_runs"] == 3
        # Step 1: means of agents per run are [0.35, 0.4, 0.37]
        assert 0.3 < bands[1]["mean"] < 0.5

    def test_compute_percentile_bands_single_run(self):
        from simulation.ensemble import _compute_percentile_bands
        all_score_logs = [
            [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6]],
        ]
        bands = _compute_percentile_bands(all_score_logs, 3)
        assert len(bands) == 3
        assert bands[1]["n_runs"] == 1

    def test_ensemble_result_to_dict(self):
        from simulation.ensemble import EnsembleResult
        result = EnsembleResult(
            experiment_id="exp_001",
            runs=[{"status": "completed"}, {"status": "failed"}],
            ensemble_summary={"n_completed": 1},
            convergence={"achieved": False},
        )
        d = result.to_dict()
        assert d["experiment_id"] == "exp_001"
        assert d["n_runs_completed"] == 1
        assert d["n_runs_failed"] == 1
        # Verify JSON-serializable
        json.dumps(d)


class TestRunEnsemble:
    """Test the run_ensemble orchestrator with mocked run_simulation."""

    @staticmethod
    def _mock_participants(n=4, seed=42):
        """Create mock participant objects."""
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

    def test_basic_ensemble(self):
        import asyncio
        from simulation.ensemble import run_ensemble

        call_count = 0

        async def mock_sim(**kwargs):
            nonlocal call_count
            call_count += 1
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        async def _run():
            nonlocal call_count
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_ensemble(n_runs=3, base_seed=10)

        result = asyncio.run(_run())
        assert call_count == 3
        assert len(result.runs) == 3
        assert all(r["status"] == "completed" for r in result.runs)

    def test_sequential_seeds(self):
        import asyncio
        from simulation.ensemble import run_ensemble
        seeds_used = []

        async def mock_sim(**kwargs):
            seeds_used.append(kwargs["seed"])
            return self._mock_participants(seed=kwargs["seed"]), self._mock_world_state()

        async def _run():
            # Use very tight cv_threshold=0.001 to prevent early stopping
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                await run_ensemble(n_runs=4, base_seed=100, cv_threshold=0.001)

        asyncio.run(_run())
        assert seeds_used == [100, 101, 102, 103]

    def test_convergence_early_stop(self):
        import asyncio
        from simulation.ensemble import run_ensemble

        async def mock_sim(**kwargs):
            ps = self._mock_participants(seed=42)  # same seed = same scores
            return ps, self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_ensemble(n_runs=20, cv_threshold=0.5, base_seed=42)

        result = asyncio.run(_run())
        # Should stop well before 20 since all runs are identical
        assert len(result.runs) < 20
        assert result.convergence["achieved"] is True

    def test_failed_run_skipped(self):
        import asyncio
        from simulation.ensemble import run_ensemble

        call_count = 0

        async def mock_sim(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("GPU OOM")
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_ensemble(n_runs=4, base_seed=42)

        result = asyncio.run(_run())
        assert call_count == 4
        completed = [r for r in result.runs if r["status"] == "completed"]
        failed = [r for r in result.runs if r["status"] == "failed"]
        assert len(completed) == 3
        assert len(failed) == 1
        assert "GPU OOM" in failed[0]["error"]

    def test_on_run_callback(self):
        import asyncio
        from simulation.ensemble import run_ensemble

        progress_reports = []

        async def mock_sim(**kwargs):
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        def on_run(info):
            progress_reports.append(info)

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                await run_ensemble(n_runs=3, on_run=on_run)

        asyncio.run(_run())
        assert len(progress_reports) == 3
        assert progress_reports[0]["completed_runs"] == 1
        assert progress_reports[2]["completed_runs"] == 3

    def test_percentile_bands_in_summary(self):
        import asyncio
        from simulation.ensemble import run_ensemble

        async def mock_sim(**kwargs):
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_ensemble(n_runs=5, base_seed=42)

        result = asyncio.run(_run())
        assert "percentile_bands" in result.ensemble_summary
        bands = result.ensemble_summary["percentile_bands"]
        assert len(bands) > 0
        first_step = bands[list(bands.keys())[0]]
        assert "mean" in first_step
        assert "p5" in first_step
        assert "p95" in first_step

    def test_insufficient_runs_convergence(self):
        """With only 2 successful runs and 1 failure, convergence still reported."""
        import asyncio
        from simulation.ensemble import run_ensemble

        call_count = 0

        async def mock_sim(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fail")
            return self._mock_participants(seed=kwargs.get("seed", 42)), self._mock_world_state()

        async def _run():
            with patch("simulation.sim_loop.run_simulation", side_effect=mock_sim):
                return await run_ensemble(n_runs=3, base_seed=42)

        result = asyncio.run(_run())
        assert result.convergence is not None
