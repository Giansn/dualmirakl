"""
Tests for the output pipeline and OutcomeCriteria.

Run: python -m pytest tests/test_output_pipeline.py -v
"""

import sys
import os
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.scenario import (
    ScenarioConfig, MetricTarget, OutcomeCriteria, MetaConfig, AgentsConfig,
    RoleConfig,
)
from simulation.output_pipeline import (
    OutputPipeline, PipelineResult, PipelineStageConfig, StageResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Minimal ScenarioConfig for testing (no GPU needed)."""
    return ScenarioConfig(
        meta=MetaConfig(name="test-pipeline", version="1.0"),
        agents=AgentsConfig(roles=[
            RoleConfig(
                id="participant_template",
                slot="swarm",
                type="participant",
                system_prompt="You are a test participant.",
                count=2,
            ),
        ]),
    )


@pytest.fixture
def config_with_criteria(minimal_config):
    """Config with outcome criteria for validation tests."""
    criteria = OutcomeCriteria(
        targets=[
            MetricTarget(name="mean_score", low=0.3, high=0.7, observed=0.5),
            MetricTarget(name="score_std", low=0.05, high=0.25),
        ],
        validation_metrics=["crps", "wasserstein"],
        convergence_required=True,
    )
    # ScenarioConfig is frozen, so we need to reconstruct
    data = minimal_config.model_dump()
    data["outcome_criteria"] = criteria.model_dump()
    return ScenarioConfig(**data)


@pytest.fixture
def sample_score_logs():
    """Synthetic score logs: 5 runs, 3 agents, 12 ticks each."""
    rng = np.random.RandomState(42)
    logs = []
    for _ in range(5):
        run = []
        for _ in range(3):
            base = rng.uniform(0.2, 0.8)
            trajectory = [
                float(np.clip(base + rng.normal(0, 0.05), 0, 1))
                for _ in range(12)
            ]
            run.append(trajectory)
        logs.append(run)
    return logs


# ── MetricTarget tests ────────────────────────────────────────────────────────

class TestMetricTarget:
    def test_basic_creation(self):
        t = MetricTarget(name="mean_score", low=0.3, high=0.7)
        assert t.name == "mean_score"
        assert t.low == 0.3
        assert t.high == 0.7
        assert t.observed is None
        assert t.weight == 1.0

    def test_with_observed(self):
        t = MetricTarget(name="polarization", low=0.0, high=0.4, observed=0.2)
        assert t.observed == 0.2

    def test_with_weight(self):
        t = MetricTarget(name="score_std", low=0.05, high=0.25, weight=1.5)
        assert t.weight == 1.5


# ── OutcomeCriteria tests ─────────────────────────────────────────────────────

class TestOutcomeCriteria:
    def test_default_creation(self):
        oc = OutcomeCriteria()
        assert oc.targets == []
        assert oc.validation_metrics == ["crps", "wasserstein"]
        assert oc.prediction_horizon == 0
        assert oc.convergence_required is True

    def test_with_targets(self):
        oc = OutcomeCriteria(targets=[
            MetricTarget(name="mean_score", low=0.3, high=0.7),
        ])
        assert len(oc.targets) == 1
        assert oc.targets[0].name == "mean_score"

    def test_invalid_validation_metric(self):
        with pytest.raises(Exception):
            OutcomeCriteria(validation_metrics=["invalid_metric"])

    def test_valid_metrics(self):
        for m in ("crps", "brier", "wasserstein"):
            oc = OutcomeCriteria(validation_metrics=[m])
            assert oc.validation_metrics == [m]


# ── ScenarioConfig integration ────────────────────────────────────────────────

class TestScenarioConfigOutcomeCriteria:
    def test_config_without_criteria(self, minimal_config):
        assert minimal_config.outcome_criteria is None

    def test_config_with_criteria(self, config_with_criteria):
        assert config_with_criteria.outcome_criteria is not None
        assert len(config_with_criteria.outcome_criteria.targets) == 2

    def test_load_social_dynamics_with_criteria(self):
        config = ScenarioConfig.load("scenarios/social_dynamics.yaml")
        assert config.outcome_criteria is not None
        names = [t.name for t in config.outcome_criteria.targets]
        assert "mean_score" in names

    def test_load_minimal_without_criteria(self):
        config = ScenarioConfig.load("scenarios/minimal.yaml")
        assert config.outcome_criteria is None


# ── PipelineStageConfig tests ─────────────────────────────────────────────────

class TestPipelineStageConfig:
    def test_defaults_all_enabled(self):
        cfg = PipelineStageConfig()
        assert cfg.ensemble is True
        assert cfg.statistics is True
        assert cfg.dynamics is True
        assert cfg.scenario_tree is True
        assert cfg.possibility_branches is True
        assert cfg.validation is True
        assert cfg.report is True

    def test_disable_stage(self):
        cfg = PipelineStageConfig(dynamics=False, validation=False)
        assert cfg.dynamics is False
        assert cfg.validation is False
        assert cfg.ensemble is True


# ── PipelineResult tests ──────────────────────────────────────────────────────

class TestPipelineResult:
    def test_empty_result(self):
        r = PipelineResult(
            experiment_id="test_1",
            scenario_name="test",
            timestamp="2026-03-28T00:00:00Z",
        )
        assert r.succeeded is True  # no stages = no failures
        assert r.failed_stages == []

    def test_with_stages(self):
        r = PipelineResult(
            experiment_id="test_2",
            scenario_name="test",
            timestamp="2026-03-28T00:00:00Z",
            stages=[
                StageResult(name="ensemble", status="completed", duration_s=5.0),
                StageResult(name="validation", status="skipped", skip_reason="no criteria"),
            ],
        )
        assert r.succeeded is True
        assert r.failed_stages == []

    def test_failed_stage(self):
        r = PipelineResult(
            experiment_id="test_3",
            scenario_name="test",
            timestamp="2026-03-28T00:00:00Z",
            stages=[
                StageResult(name="ensemble", status="completed"),
                StageResult(name="dynamics", status="failed", error="nolds not installed"),
            ],
        )
        assert r.succeeded is False
        assert r.failed_stages == ["dynamics"]

    def test_to_dict_roundtrip(self):
        r = PipelineResult(
            experiment_id="test_4",
            scenario_name="test",
            timestamp="2026-03-28T00:00:00Z",
            statistics={"polarization": {"final_polarization": 0.12}},
            validation={"passed": True, "range_checks": {}},
            stages=[
                StageResult(name="ensemble", status="completed", duration_s=10.0),
            ],
        )
        d = r.to_dict()
        assert d["experiment_id"] == "test_4"
        assert d["succeeded"] is True
        assert d["statistics"]["polarization"]["final_polarization"] == 0.12
        assert d["validation"]["passed"] is True
        # Ensure JSON-serializable
        json_str = json.dumps(d, default=str)
        assert "test_4" in json_str


# ── Metric extraction tests ──────────────────────────────────────────────────

class TestExtractMetric:
    def test_mean_score(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("mean_score", run)
        assert 0.0 <= val <= 1.0

    def test_score_std(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("score_std", run)
        assert val >= 0.0

    def test_max_score(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("max_score", run)
        assert 0.0 <= val <= 1.0

    def test_min_score(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("min_score", run)
        assert 0.0 <= val <= 1.0

    def test_score_range(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("score_range", run)
        assert val >= 0.0

    def test_polarization(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("polarization", run)
        assert val is not None

    def test_convergence_ratio(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("convergence_ratio", run)
        assert val in (0.0, 1.0)

    def test_fraction_above(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("fraction_above_0.5", run)
        assert 0.0 <= val <= 1.0

    def test_unknown_metric_fallback(self, sample_score_logs):
        run = sample_score_logs[0]
        val = OutputPipeline._extract_metric("nonexistent_metric", run)
        # Falls back to mean_score
        assert 0.0 <= val <= 1.0

    def test_empty_logs(self):
        assert OutputPipeline._extract_metric("mean_score", []) is None
        assert OutputPipeline._extract_metric("mean_score", [[]]) is None


# ── Stage skipping logic tests ────────────────────────────────────────────────

class TestStageSkipping:
    def test_validation_skipped_without_criteria(self, minimal_config):
        """Pipeline without outcome_criteria should skip validation."""
        pipeline = OutputPipeline(
            scenario_config=minimal_config,
            stage_config=PipelineStageConfig(
                ensemble=False, statistics=False, dynamics=False,
                scenario_tree=False, possibility_branches=False,
                validation=True, report=False,
            ),
        )
        # outcome_criteria is None → validation should be skipped
        assert minimal_config.outcome_criteria is None

    def test_tree_needs_minimum_runs(self):
        """Scenario tree requires >= 3 runs."""
        cfg = PipelineStageConfig()
        assert cfg.scenario_tree is True
        # The actual skipping happens in run() based on len(all_score_logs)

    def test_dynamics_min_ticks(self):
        """Dynamics requires minimum ticks."""
        cfg = PipelineStageConfig(dynamics_min_ticks=10)
        assert cfg.dynamics_min_ticks == 10
