"""
Experiment tracking layer for dualmirakl.

Write-behind DuckDB persistence for experiments, runs, and per-tick metrics.
Non-blocking: tick data is buffered in-memory and batch-flushed at boundaries.

Usage:
    from simulation.experiment_db import ExperimentDB

    db = ExperimentDB()
    exp_id = db.create_experiment("my_experiment", config={...})
    db.register_run("run_20260327_s42", experiment_id=exp_id, sim_seed=42)
    db.record_tick("run_20260327_s42", 1, {"mean_score": 0.5})
    db.flush_ticks()
    db.complete_run("run_20260327_s42", wall_time_seconds=12.3)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def get_git_hash() -> str | None:
    """Return short git hash of current HEAD, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class ExperimentDB:
    """Write-behind experiment tracking layer. Non-blocking writes via batch flush."""

    def __init__(self, db=None):
        self._db = db
        self._tick_buffer: list[tuple] = []

    @property
    def db(self):
        """Lazy DuckDB connection (same pattern as DuckDBMemoryBackend)."""
        if self._db is None:
            from simulation.storage import get_db
            self._db = get_db()
        return self._db

    def create_experiment(
        self,
        name: str,
        description: str = "",
        config: dict | None = None,
        git_hash: str | None = None,
        llm_model: str | None = None,
    ) -> str:
        """INSERT into experiments. Returns experiment_id."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rnd = os.urandom(2).hex()
        experiment_id = f"exp_{ts}_{rnd}"
        self.db.execute(
            """INSERT INTO experiments
               (experiment_id, name, description, config, git_hash, llm_model)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [experiment_id, name, description,
             json.dumps(config) if config else None,
             git_hash or get_git_hash(), llm_model],
        )
        logger.info("Created experiment %s: %s", experiment_id, name)
        return experiment_id

    def register_run(
        self,
        run_id: str,
        experiment_id: str | None = None,
        parameters: dict | None = None,
        sim_seed: int = 42,
        temperature: float = 0.7,
    ) -> None:
        """INSERT into runs with status='running'."""
        self.db.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, experiment_id, parameters, sim_seed, temperature, status)
               VALUES (?, ?, ?, ?, ?, 'running')""",
            [run_id, experiment_id,
             json.dumps(parameters) if parameters else None,
             sim_seed, temperature],
        )

    def record_tick(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
        agent_id: str | None = None,
    ) -> None:
        """Buffer tick_data rows. Does NOT write immediately (non-blocking)."""
        for metric_name, value in metrics.items():
            self._tick_buffer.append((run_id, step, metric_name, value, agent_id))

    def flush_ticks(self) -> int:
        """Batch INSERT buffered tick_data. Returns count flushed."""
        if not self._tick_buffer:
            return 0
        count = len(self._tick_buffer)
        self.db.executemany(
            "INSERT INTO tick_data (run_id, step, metric_name, value, agent_id) VALUES (?, ?, ?, ?, ?)",
            self._tick_buffer,
        )
        self._tick_buffer.clear()
        return count

    def complete_run(
        self,
        run_id: str,
        wall_time_seconds: float,
        status: str = "completed",
    ) -> None:
        """UPDATE runs SET status, wall_time_seconds."""
        self.db.execute(
            "UPDATE runs SET status = ?, wall_time_seconds = ? WHERE run_id = ?",
            [status, wall_time_seconds, run_id],
        )

    def fail_run(self, run_id: str, wall_time_seconds: float) -> None:
        """UPDATE runs SET status='failed'."""
        self.complete_run(run_id, wall_time_seconds, status="failed")

    def write_ensemble_summary(
        self,
        experiment_id: str,
        step: int,
        metric_name: str,
        stats: dict,
    ) -> None:
        """INSERT into ensemble_summaries."""
        self.db.execute(
            """INSERT INTO ensemble_summaries
               (experiment_id, step, metric_name, n_runs, mean, std,
                p5, p25, median, p75, p95,
                var_aleatory, var_epistemic, var_llm)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [experiment_id, step, metric_name,
             stats.get("n_runs", 0),
             stats.get("mean"), stats.get("std"),
             stats.get("p5"), stats.get("p25"), stats.get("median"),
             stats.get("p75"), stats.get("p95"),
             stats.get("var_aleatory"), stats.get("var_epistemic"),
             stats.get("var_llm")],
        )

    def get_experiment_runs(self, experiment_id: str) -> list[dict]:
        """Query runs for an experiment."""
        rows = self.db.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY created_at",
            [experiment_id],
        ).fetchdf()
        return rows.to_dict("records") if len(rows) > 0 else []

    def get_tick_data(
        self,
        run_id: str,
        metric_name: str | None = None,
    ) -> list[dict]:
        """Query tick_data for a run."""
        if metric_name:
            rows = self.db.execute(
                "SELECT * FROM tick_data WHERE run_id = ? AND metric_name = ? ORDER BY step",
                [run_id, metric_name],
            ).fetchdf()
        else:
            rows = self.db.execute(
                "SELECT * FROM tick_data WHERE run_id = ? ORDER BY step, metric_name",
                [run_id],
            ).fetchdf()
        return rows.to_dict("records") if len(rows) > 0 else []
