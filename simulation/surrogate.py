"""
Neural / GP surrogate for parameter space exploration.

Generates synthetic training data from the fast surrogate objective,
trains both GP and MLP models, benchmarks, and picks the best.
Provides predict() for Optuna acceleration and explain() for SHAP
sensitivity analysis.

Usage:
    from simulation.surrogate import SurrogateModel

    sm = SurrogateModel()
    sm.build(n_samples=5000, include_flame=True)
    mean, std = sm.predict(X_new)
    importance = sm.explain(X_new)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Comparison of GP vs MLP surrogate quality."""
    gp_r2: float
    mlp_r2: float
    gp_rmse: float
    mlp_rmse: float
    gp_fit_ms: float
    mlp_fit_ms: float
    gp_predict_ms: float  # per 1000 predictions
    mlp_predict_ms: float
    winner: str
    n_train: int
    n_test: int
    n_params: int


TARGETS = ["mean_final", "std_final", "stability", "loss"]


class SurrogateModel:
    """
    Dual GP/MLP surrogate with automatic model selection.

    Trains on synthetic data from _surrogate_objective, benchmarks
    GP vs MLP on held-out test set, uses the winner for predictions.
    """

    def __init__(self):
        self._model = None
        self._model_type: str = ""
        self._gp = None
        self._mlp = None
        self._scaler_X = None
        self._scaler_y = None
        self._param_names: list[str] = []
        self._target_names: list[str] = TARGETS
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._benchmark: Optional[BenchmarkResult] = None

    def build(
        self,
        n_samples: int = 5000,
        include_flame: bool = False,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> BenchmarkResult:
        """
        Generate training data, train GP + MLP, benchmark, pick winner.

        Args:
            n_samples: Total samples to generate (fast, ~1ms each).
            include_flame: Include FLAME parameters in the search space.
            test_fraction: Fraction held out for testing.
            seed: RNG seed for reproducibility.

        Returns:
            BenchmarkResult comparing GP vs MLP.
        """
        logger.info("Surrogate: generating %d training samples...", n_samples)
        X, y, param_names = self._generate_data(n_samples, include_flame, seed)
        self._param_names = param_names

        # Train/test split
        rng = np.random.default_rng(seed)
        n_test = int(n_samples * test_fraction)
        indices = rng.permutation(n_samples)
        test_idx, train_idx = indices[:n_test], indices[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        self._X_train = X_train
        self._y_train = y_train

        # Standardize
        self._scaler_X = (X_train.mean(axis=0), X_train.std(axis=0) + 1e-10)
        self._scaler_y = (y_train.mean(axis=0), y_train.std(axis=0) + 1e-10)

        X_train_s = (X_train - self._scaler_X[0]) / self._scaler_X[1]
        X_test_s = (X_test - self._scaler_X[0]) / self._scaler_X[1]

        # Pass raw y to both — each model handles its own normalization
        # GP uses normalize_y=True internally; MLP standardizes internally
        gp_r2, gp_rmse, gp_fit_ms, gp_pred_ms = self._train_gp(
            X_train_s, y_train[:, -1], X_test_s, y_test[:, -1]
        )

        mlp_r2, mlp_rmse, mlp_fit_ms, mlp_pred_ms = self._train_mlp(
            X_train_s, y_train, X_test_s, y_test
        )

        # Pick winner by R² on test set
        winner = "mlp" if mlp_r2 >= gp_r2 else "gp"
        self._model_type = winner
        self._model = self._mlp if winner == "mlp" else self._gp

        self._benchmark = BenchmarkResult(
            gp_r2=gp_r2, mlp_r2=mlp_r2,
            gp_rmse=gp_rmse, mlp_rmse=mlp_rmse,
            gp_fit_ms=gp_fit_ms, mlp_fit_ms=mlp_fit_ms,
            gp_predict_ms=gp_pred_ms, mlp_predict_ms=mlp_pred_ms,
            winner=winner,
            n_train=len(train_idx), n_test=n_test,
            n_params=X.shape[1],
        )

        logger.info(
            "Surrogate: winner=%s  GP(R²=%.4f, %.0fms)  MLP(R²=%.4f, %.0fms)  "
            "n_train=%d n_test=%d n_params=%d",
            winner, gp_r2, gp_fit_ms, mlp_r2, mlp_fit_ms,
            len(train_idx), n_test, X.shape[1],
        )
        return self._benchmark

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict metrics for parameter vectors.

        Args:
            X: (n_points, n_params) parameter vectors.

        Returns:
            (mean, std) — predicted metrics. For MLP, std is estimated
            via dropout or returns zeros. For GP, std is native.
        """
        if self._model is None:
            raise RuntimeError("Call build() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_s = (X - self._scaler_X[0]) / self._scaler_X[1]

        if self._model_type == "gp":
            # GP trains on raw y with normalize_y=True — output is raw scale
            mean, std = self._gp.predict(X_s, return_std=True)
            return mean, std
        else:
            # MLP trains on standardized y — unscale output
            y_s = self._mlp.predict(X_s)
            mean = y_s * self._mlp_y_std + self._mlp_y_mean
            return mean, np.zeros_like(mean)

    def predict_loss(self, params: dict) -> float:
        """Predict loss for a parameter dict (Optuna integration)."""
        X = self._params_to_vector(params)
        mean, _ = self.predict(X)
        if mean.ndim == 1:
            return float(mean[-1])  # loss is last column
        return float(mean[0, -1])

    def explain(self, X: Optional[np.ndarray] = None, n_background: int = 100) -> dict:
        """
        SHAP-based parameter importance analysis.

        Args:
            X: Points to explain. If None, uses training data sample.
            n_background: Background samples for SHAP.

        Returns:
            {param_name: mean_abs_shap_value} dict, sorted by importance.
        """
        try:
            import shap
        except ImportError:
            # Fallback: permutation importance
            return self._permutation_importance(X)

        if self._model is None:
            raise RuntimeError("Call build() first.")

        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(self._X_train), min(n_background, len(self._X_train)), replace=False)
        X_bg = (self._X_train[bg_idx] - self._scaler_X[0]) / self._scaler_X[1]

        if X is None:
            X_explain = X_bg[:50]
        else:
            X_explain = (np.asarray(X) - self._scaler_X[0]) / self._scaler_X[1]

        if self._model_type == "mlp":
            explainer = shap.KernelExplainer(self._mlp.predict, X_bg)
        else:
            explainer = shap.KernelExplainer(
                lambda x: self._gp.predict(x), X_bg
            )

        shap_values = explainer.shap_values(X_explain, silent=True)

        # Mean absolute SHAP per feature
        if isinstance(shap_values, list):
            # Multi-output: average across outputs
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        importance = np.mean(sv, axis=0)
        result = {
            name: float(imp)
            for name, imp in sorted(
                zip(self._param_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        }
        return result

    def _permutation_importance(self, X: Optional[np.ndarray] = None) -> dict:
        """Fallback importance via permutation (no SHAP dependency)."""
        if self._X_train is None or self._y_train is None:
            return {}

        rng = np.random.default_rng(42)
        n = min(500, len(self._X_train))
        idx = rng.choice(len(self._X_train), n, replace=False)
        X_test = self._X_train[idx]
        y_test = self._y_train[idx, -1]  # loss column

        baseline_pred, _ = self.predict(X_test)
        if baseline_pred.ndim == 2:
            baseline_pred = baseline_pred[:, -1]
        baseline_mse = float(np.mean((y_test - baseline_pred) ** 2))

        importance = {}
        for i, name in enumerate(self._param_names):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_pred, _ = self.predict(X_perm)
            if perm_pred.ndim == 2:
                perm_pred = perm_pred[:, -1]
            perm_mse = float(np.mean((y_test - perm_pred) ** 2))
            importance[name] = max(0, perm_mse - baseline_mse)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ── Data generation ──────────────────────────────────────────────────

    def _generate_data(
        self, n_samples: int, include_flame: bool, seed: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Generate (X, y) from the fast surrogate objective."""
        from simulation.optimize import (
            DUALMIRAKL_PARAMS, FLAME_PARAMS, _surrogate_objective,
        )

        params_spec = dict(DUALMIRAKL_PARAMS)
        if include_flame:
            params_spec.update(FLAME_PARAMS)

        param_names = list(params_spec.keys())
        n_params = len(param_names)
        rng = np.random.default_rng(seed)

        X = np.zeros((n_samples, n_params))
        y = np.zeros((n_samples, len(TARGETS)))

        # Latin Hypercube Sampling for better coverage
        for d, name in enumerate(param_names):
            spec = params_spec[name]
            lo, hi = spec["low"], spec["high"]
            cuts = np.linspace(lo, hi, n_samples + 1)
            X[:, d] = rng.uniform(cuts[:-1], cuts[1:])
            rng.shuffle(X[:, d])

        # Evaluate each sample through the fast surrogate
        class _FakeTrial:
            """Minimal trial interface for _surrogate_objective."""
            def __init__(self, params: dict):
                self._params = params
                self.attrs = {}
                self.number = 0
            def suggest_float(self, name, low, high):
                return self._params[name]
            def suggest_int(self, name, low, high):
                return int(round(self._params[name]))
            def set_user_attr(self, key, val):
                self.attrs[key] = val

        t0 = time.monotonic()
        for i in range(n_samples):
            params = {name: X[i, d] for d, name in enumerate(param_names)}
            trial = _FakeTrial(params)
            loss = _surrogate_objective(
                trial, n_ticks=12, n_agents=4,
                include_flame=include_flame,
            )
            y[i] = [
                trial.attrs.get("mean_final", 0.5),
                trial.attrs.get("std_final", 0.1),
                trial.attrs.get("stability", 0.05),
                loss,
            ]

        elapsed = time.monotonic() - t0
        logger.info("Surrogate: generated %d samples in %.1fs (%.1fms/sample)",
                     n_samples, elapsed, elapsed / n_samples * 1000)
        return X, y, param_names

    def _params_to_vector(self, params: dict) -> np.ndarray:
        """Convert param dict to feature vector."""
        return np.array([params.get(n, 0.0) for n in self._param_names]).reshape(1, -1)

    # ── GP training ──────────────────────────────────────────────────────

    def _train_gp(self, X_train, y_train, X_test, y_test):
        """Train GP on single output (loss). Returns (r2, rmse, fit_ms, pred_ms)."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        except ImportError:
            return 0.0, float("inf"), 0.0, 0.0

        # GP scales O(n³) — cap training set for reasonable fit time
        max_gp = min(2000, len(X_train))
        X_gp = X_train[:max_gp]
        y_gp = y_train[:max_gp]

        kernel = Matern(nu=2.5) + WhiteKernel()
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3,
            normalize_y=True, random_state=42,
        )

        t0 = time.monotonic()
        gp.fit(X_gp, y_gp)
        fit_ms = (time.monotonic() - t0) * 1000

        # Test predictions
        t0 = time.monotonic()
        y_pred = gp.predict(X_test)
        pred_ms = (time.monotonic() - t0) * 1000

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

        self._gp = gp
        return r2, rmse, fit_ms, pred_ms

    # ── MLP training ─────────────────────────────────────────────────────

    def _train_mlp(self, X_train, y_train, X_test, y_test):
        """Train MLP on multi-output (raw y). Returns (r2, rmse, fit_ms, pred_ms)."""
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError:
            return 0.0, float("inf"), 0.0, 0.0

        # Standardize y internally for MLP training
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0) + 1e-10
        y_train_s = (y_train - y_mean) / y_std

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            batch_size=min(256, len(X_train)),
        )

        t0 = time.monotonic()
        mlp.fit(X_train, y_train_s)
        fit_ms = (time.monotonic() - t0) * 1000

        # Test predictions (unscale)
        t0 = time.monotonic()
        y_pred_s = mlp.predict(X_test)
        pred_ms = (time.monotonic() - t0) * 1000
        y_pred = y_pred_s * y_std + y_mean

        # Store scaler for predict()
        self._mlp_y_mean = y_mean
        self._mlp_y_std = y_std

        # R² on loss column (primary metric)
        loss_pred = y_pred[:, -1] if y_pred.ndim == 2 else y_pred
        loss_true = y_test[:, -1] if y_test.ndim == 2 else y_test
        ss_res = np.sum((loss_true - loss_pred) ** 2)
        ss_tot = np.sum((loss_true - np.mean(loss_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((loss_true - loss_pred) ** 2)))

        self._mlp = mlp
        return r2, rmse, fit_ms, pred_ms
