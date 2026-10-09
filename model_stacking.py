"""
model_stacking.py
=================
Weighted model stacking used by the player trainer (hitterspitchers_train.py)
and the game trainer (train_game_model.py).

Two stacks are built from the base models' VALIDATION-window predictions:

  stack_even  every base model gets the same weight (1/k)
  stack_opt   non-negative weights that sum to 1, chosen to minimise the
              validation loss (RMSE for counts, log loss for win probability)

Honest selection: the optimised stack is scored for model selection with
CROSS-FITTED predictions - the validation window is split chronologically in
two, weights are fit on one half and applied to the other (and vice versa) -
so it is never graded on the rows its weights were fit on. The weights that
ship are then fit on the whole validation window. The test window is never
touched.

The wrapper classes live in this importable module so the pickled bundles
load in hitterspitchers_today.py / daily_mlb_model_runner.py.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _loss(kind: str):
    if kind == "rmse":
        return lambda y, p: float(np.sqrt(np.mean((p - y) ** 2)))
    if kind == "logloss":
        def ll(y, p):
            p = np.clip(p, EPS, 1 - EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return ll
    if kind == "auc":                       # minimised, so negative AUC
        from sklearn.metrics import roc_auc_score
        return lambda y, p: -float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0
    raise ValueError(kind)


def even_weights(k: int) -> np.ndarray:
    return np.full(k, 1.0 / k)


def fit_weights(P: np.ndarray, y: np.ndarray, kind: str = "rmse") -> np.ndarray:
    """Weights on the probability simplex (w >= 0, sum w = 1) minimising the loss
    of P @ w against y. Falls back to even weights if the optimiser fails."""
    from scipy.optimize import minimize
    P = np.asarray(P, float); y = np.asarray(y, float)
    k = P.shape[1]
    if k == 1:
        return np.ones(1)
    f = _loss(kind)
    if kind == "auc":
        # AUC is rank-based and flat almost everywhere, so a gradient optimiser
        # cannot move it. Seeded random search over the simplex (Dirichlet
        # draws + each single model + even weights) is exact enough for k <= 6.
        rng = np.random.default_rng(42)
        cands = np.vstack([np.eye(k), even_weights(k)[None, :], rng.dirichlet(np.ones(k), 3000)])
        scores = [f(y, P @ w) for w in cands]
        return cands[int(np.argmin(scores))]
    res = minimize(lambda w: f(y, P @ w), even_weights(k), method="SLSQP",
                   bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None) if res.success else even_weights(k)
    return w / w.sum() if w.sum() > 0 else even_weights(k)


def crossfit_predictions(P: np.ndarray, y: np.ndarray, kind: str = "rmse") -> np.ndarray:
    """Out-of-fold stacked predictions over a chronologically ordered window:
    weights fit on the first half score the second half and vice versa."""
    P = np.asarray(P, float); y = np.asarray(y, float)
    n = len(y)
    if n < 40:
        return P @ fit_weights(P, y, kind)
    h = n // 2
    out = np.empty(n)
    out[h:] = P[h:] @ fit_weights(P[:h], y[:h], kind)
    out[:h] = P[:h] @ fit_weights(P[h:], y[h:], kind)
    return out


def describe(names, weights) -> str:
    return ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights))


class StackedRegressor:
    """Weighted average of fitted regressors (each may be a CalibratedRegressor)."""

    def __init__(self, models, weights, names, floor: float | None = 0.0):
        self.models = list(models)
        self.weights = np.asarray(weights, float)
        self.names = list(names)
        self.floor = floor

    def predict(self, X):
        P = np.column_stack([np.asarray(m.predict(X), float) for m in self.models])
        out = P @ self.weights
        return np.clip(out, self.floor, None) if self.floor is not None else out

    def __repr__(self):
        return f"StackedRegressor({describe(self.names, self.weights)})"


class StackedClassifier:
    """Weighted average of fitted classifiers' P(class 1)."""

    def __init__(self, models, weights, names):
        self.models = list(models)
        self.weights = np.asarray(weights, float)
        self.names = list(names)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        P = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        p1 = np.clip(P @ self.weights, 0.0, 1.0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def __repr__(self):
        return f"StackedClassifier({describe(self.names, self.weights)})"
