"""
model_calibration.py
=====================
A tiny, picklable wrapper that bakes a post-hoc linear recalibration into a
fitted regressor.

Why this lives in its own module
--------------------------------
The saved pitcher_*.pkl / hitter_*.pkl bundles are unpickled inside
hitterspitchers_today.py (and the props/NRFI code) at scoring time. A wrapper
class defined inside the training *script* would pickle as
`__main__.CalibratedRegressor` and fail to load anywhere else — that's the
exact AttributeError this project hit before. Defining it in an importable
module means the pickle references `model_calibration.CalibratedRegressor`,
which Python re-imports automatically on load, so the calibrated model scores
correctly with NO changes to the inference code.

What it does
------------
Given a fitted base estimator (a sklearn Pipeline or TransformedTargetRegressor)
and two scalars (a, b) learned on a held-out calibration slice, `.predict`
returns:

        y_cal = clip(a + b * base.predict(X), floor, None)

When (a, b) == (0.0, 1.0) this is the identity, so an uncalibrated model
behaves exactly as before.
"""

from __future__ import annotations

import numpy as np


class CalibratedRegressor:
    """Wrap a fitted regressor and apply a post-hoc calibration to its output.

    Two modes:
      - Linear (default):  y_cal = a + b * y_raw
      - Isotonic (when `iso` is set): y_cal = iso(y_raw), a monotone
        non-parametric mapping that can correct curved bias the linear
        a+b form can't (e.g. "model under-predicts the top quartile and
        over-predicts the bottom"). a / b are ignored in this mode.

    Output is clipped to `floor` (default 0) so a calibrated count never goes
    negative. Pickles cleanly because the class lives in its own importable
    module, not __main__.
    """

    def __init__(self, base, a: float = 0.0, b: float = 1.0,
                 floor: float | None = 0.0, iso=None):
        self.base = base
        self.a = float(a)
        self.b = float(b)
        self.floor = floor
        # `iso` is either None (linear mode) or a fitted IsotonicRegression
        # from sklearn.isotonic. We store the fitted object directly so it
        # pickles with the bundle.
        self.iso = iso

    def predict(self, X):
        raw = np.asarray(self.base.predict(X), dtype=float)
        if self.iso is not None:
            # IsotonicRegression's .predict expects 1-D input and returns 1-D
            out = np.asarray(self.iso.predict(raw), dtype=float)
        else:
            out = self.a + self.b * raw
        if self.floor is not None:
            out = np.clip(out, self.floor, None)
        return out

    # ── convenience passthroughs so existing helpers keep working ──
    @property
    def named_steps(self):
        return getattr(self.base, "named_steps", {})

    @property
    def regressor_(self):
        # lets feature_importance() unwrap to the underlying pipeline
        return getattr(self.base, "regressor_", self.base)

    def __getattr__(self, item):
        # Delegate unknown attributes to the wrapped estimator. Must raise
        # AttributeError (never KeyError) for dunder probes / before `base`
        # is restored, otherwise pickle.load() breaks.
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(item)
        return getattr(base, item)


def fit_linear_calibration(y_true, y_pred):
    """Return (a, b) for y_cal = a + b * y_pred, fit by least squares.

    Falls back to the identity (0.0, 1.0) when there's too little data or the
    fit is degenerate (negative / explosive slope), so calibration can never
    make a model worse than uncalibrated by construction-gone-wrong.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) < 50 or np.nanstd(yp[mask]) < 1e-9:
        return 0.0, 1.0
    try:
        b, a = np.polyfit(yp[mask], yt[mask], 1)  # slope, intercept
    except Exception:
        return 0.0, 1.0
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0.2 or b > 3.0:
        return 0.0, 1.0
    return float(a), float(b)


def fit_isotonic_calibration(y_true, y_pred):
    """Fit a monotone non-parametric calibration y_cal = iso(y_pred).

    Use when residuals show curvature the linear a+b form can't capture
    (typically: model under-predicts the top quintile of a count target
    like HR or H). Returns a fitted sklearn IsotonicRegression, or None if
    there's too little data — in which case the caller should fall back to
    the linear fit (or identity).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) < 100 or np.nanstd(yp[mask]) < 1e-9:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None
    try:
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(yp[mask], yt[mask])
    except Exception:
        return None
    return iso
