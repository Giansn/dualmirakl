"""
Gaussian Process emulator for dualmirakl parameter space exploration.

Trains a GP surrogate on (parameter_vector, output_metric) pairs from
simulation runs. Predicts output + uncertainty for new parameter vectors
without running the full simulation. Used to accelerate history matching
and active learning for parameter space exploration.

Uses sklearn.gaussian_process if available, falls back to scipy RBF.

Usage:
    from simulation.gp_emulator import GPEmulator

    gp = GPEmulator()
    gp.fit(X_train, y_train)      # (n_samples, n_params), (n_samples,)
    mean, std = gp.predict(X_new)
    suggestions = gp.suggest_next(bounds, n_suggestions=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HAS_SKLEARN = False
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error
    _HAS_SKLEARN = True
except ImportError:
    logger.debug("scikit-learn not available; GP emulator will use scipy RBF fallback")


@dataclass
class EmulatorResult:
    """Training result with diagnostics."""
    mean: np.ndarray
    std: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    r2_score: float
    n_train: int
    backend: str  # "sklearn_gp" or "scipy_rbf"


class GPEmulator:
    """
    Gaussian Process emulator for simulation surrogate modeling.

    Trains on (parameter_vectors, output_metric) pairs, predicts
    output + uncertainty for new parameter vectors.
    """

    def __init__(self, kernel: str = "rbf", normalize: bool = True):
        """
        Args:
            kernel: "rbf" or "matern" (sklearn only).
            normalize: Standardize inputs to [0,1].
        """
        self._kernel_name = kernel
        self._normalize = normalize
        self._gp = None
        self._rbf = None  # scipy fallback
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_min: Optional[np.ndarray] = None
        self._X_range: Optional[np.ndarray] = None
        self._backend = "sklearn_gp" if _HAS_SKLEARN else "scipy_rbf"

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features to [0,1]."""
        if not self._normalize or self._X_min is None:
            return X
        return (X - self._X_min) / np.maximum(self._X_range, 1e-10)

    def fit(self, X: np.ndarray, y: np.ndarray) -> EmulatorResult:
        """
        Train the emulator on simulation data.

        Args:
            X: Parameter vectors, shape (n_samples, n_params).
            y: Output metric values, shape (n_samples,).

        Returns:
            EmulatorResult with training diagnostics.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._X_train = X.copy()
        self._y_train = y.copy()

        # Compute normalization
        if self._normalize:
            self._X_min = X.min(axis=0)
            self._X_range = X.max(axis=0) - X.min(axis=0)
        X_norm = self._normalize_X(X)

        if _HAS_SKLEARN:
            # Build kernel
            if self._kernel_name == "matern":
                kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel()
            else:
                kernel = RBF(length_scale_bounds=(1e-3, 1e2)) + WhiteKernel()

            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            )
            self._gp.fit(X_norm, y)

            # Training predictions for diagnostics
            y_pred = self._gp.predict(X_norm)
            r2 = float(r2_score(y, y_pred))
        else:
            # Scipy RBF fallback
            from scipy.interpolate import RBFInterpolator
            self._rbf = RBFInterpolator(X_norm, y, kernel="thin_plate_spline")
            y_pred = self._rbf(X_norm)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        logger.info("GPEmulator fit: n=%d, R²=%.4f, backend=%s", len(y), r2, self._backend)

        return EmulatorResult(
            mean=y_pred,
            std=np.zeros_like(y_pred),  # training std not meaningful
            X_train=X,
            y_train=y,
            r2_score=r2,
            n_train=len(y),
            backend=self._backend,
        )

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict output metric and uncertainty for new parameter vectors.

        Args:
            X_new: Parameter vectors, shape (n_points, n_params).

        Returns:
            (mean, std) arrays of shape (n_points,).
        """
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        X_norm = self._normalize_X(X_new)

        if self._gp is not None:
            mean, std = self._gp.predict(X_norm, return_std=True)
            return mean, std
        elif self._rbf is not None:
            mean = self._rbf(X_norm)
            # RBF doesn't provide uncertainty — return zeros
            return mean, np.zeros_like(mean)
        else:
            raise RuntimeError("Emulator not fitted. Call fit() first.")

    def validate(self, n_folds: int = 5) -> dict:
        """
        K-fold cross-validation of the emulator.

        Returns:
            {r2, rmse, max_error, n_folds}
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Emulator not fitted. Call fit() first.")

        X_norm = self._normalize_X(self._X_train)
        y = self._y_train
        n = len(y)
        n_folds = min(n_folds, n)

        if n < 3:
            return {"r2": 0.0, "rmse": float("inf"), "max_error": float("inf"), "n_folds": 0}

        if _HAS_SKLEARN and self._gp is not None:
            # Refit with CV
            y_pred = cross_val_predict(self._gp, X_norm, y, cv=min(n_folds, n))
            r2 = float(r2_score(y, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            max_err = float(np.max(np.abs(y - y_pred)))
        else:
            # Manual leave-one-out for RBF
            from scipy.interpolate import RBFInterpolator
            errors = []
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                rbf_i = RBFInterpolator(X_norm[mask], y[mask], kernel="thin_plate_spline")
                pred_i = rbf_i(X_norm[i:i+1])
                errors.append(y[i] - pred_i[0])
            errors = np.array(errors)
            y_pred = y - errors
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            max_err = float(np.max(np.abs(errors)))

        return {
            "r2": round(r2, 4),
            "rmse": round(rmse, 6),
            "max_error": round(max_err, 6),
            "n_folds": n_folds,
        }

    def suggest_next(
        self,
        bounds: list[tuple[float, float]],
        n_suggestions: int = 5,
        strategy: str = "uncertainty",
        n_candidates: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Suggest next parameter points to evaluate (active learning).

        Args:
            bounds: [(low, high), ...] for each parameter dimension.
            n_suggestions: Number of points to suggest.
            strategy: "uncertainty" (max std) or "expected_improvement".
            n_candidates: Number of random candidates to evaluate.
            seed: RNG seed.

        Returns:
            (n_suggestions, n_params) array of suggested parameter vectors.
        """
        rng = np.random.default_rng(seed)
        n_params = len(bounds)

        # Latin Hypercube candidates
        candidates = np.zeros((n_candidates, n_params))
        for d in range(n_params):
            lo, hi = bounds[d]
            cuts = np.linspace(lo, hi, n_candidates + 1)
            candidates[:, d] = rng.uniform(cuts[:-1], cuts[1:])
            rng.shuffle(candidates[:, d])

        mean, std = self.predict(candidates)

        if strategy == "uncertainty":
            # Select points with highest predicted uncertainty
            scores = std
        elif strategy == "expected_improvement":
            # EI = E[max(0, y_best - f(x))]
            if self._y_train is not None:
                y_best = float(np.min(self._y_train))
                z = (y_best - mean) / np.maximum(std, 1e-10)
                from scipy.stats import norm
                scores = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
                scores[std < 1e-10] = 0.0
            else:
                scores = std
        else:
            # Random
            scores = rng.random(n_candidates)

        top_indices = np.argsort(scores)[-n_suggestions:][::-1]
        return candidates[top_indices]
