"""Unit checks for the statistics used by audit_models.py (no data files needed)."""
import numpy as np
import pandas as pd

import audit_models as A


def test_calibration_slope_and_intercept_recover_truth():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 40000)
    p_true = A.sigmoid(0.2 + 0.8 * z)
    y = (rng.random(len(z)) < p_true).astype(float)
    p_model = A.sigmoid(z)                                  # model is over-confident: true slope 0.8
    a, b = A.logistic_recal(y, p_model)
    assert abs(b - 0.8) < 0.05 and abs(a - 0.2) < 0.05
    a2, b2 = A.logistic_recal(y, p_true)                    # a calibrated model gives ~(0, 1)
    assert abs(b2 - 1) < 0.05 and abs(a2) < 0.05
    assert abs(A.cal_in_the_large(y, p_true)) < 0.03


def test_regression_slope_detects_compression_and_spread():
    rng = np.random.default_rng(1)
    mu = rng.gamma(4, 0.5, 20000)
    y = rng.poisson(mu).astype(float)
    assert abs(A.cal_slope(y, mu) - 1) < 0.05
    shrunk = mu.mean() + 0.5 * (mu - mu.mean())              # compressed toward the mean -> slope ~2
    assert A.cal_slope(y, shrunk) > 1.8
    spread = mu.mean() + 2.0 * (mu - mu.mean())              # over-dispersed -> slope ~0.5
    assert A.cal_slope(y, spread) < 0.6


def test_poisson_deviance_zero_at_perfect_and_handles_zero_counts():
    y = np.array([0.0, 1.0, 3.0])
    assert np.allclose(A.pois_dev_rows(y, np.clip(y, 1e-9, None)), 0, atol=1e-5)
    assert np.isfinite(A.pois_dev_rows(np.array([0.0]), np.array([0.0]))).all()


def test_wilson_interval_contains_rate():
    lo, hi = A.wilson(8, 10)
    assert lo < 0.8 < hi and 0 <= lo and hi <= 1


def test_block_bootstrap_by_date_brackets_point():
    dates = np.repeat(pd.date_range("2026-07-01", periods=60), 20)
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1, len(dates)); b = rng.normal(1.2, 1, len(dates))
    pt, lo, hi = A.block_boot(dates, {"a": a, "b": b}, lambda s: s["a"] / s["cnt"] - s["b"] / s["cnt"], n=300)
    assert lo < pt < hi and hi < 0


def test_feature_taxonomy():
    assert A.family("K_last10") == "rolling"
    assert A.family("K_std") == "expanding"
    assert A.family("IP_last5") == "opportunity"
    assert A.family("pitcher_k_rate_vs_hand_last10_R") == "handedness"
    assert A.family("team_k_rate_vs_hand_last10") == "team/opponent"
    assert A.family("p_tt_k") == "true_talent" and A.family("matchup_k") == "matchup"
    assert A.family("lineup_spot") == "lineup" and A.family("barrel_proxy_last5") == "batted_ball"
    assert A.family("avg_velocity_last7") == "velocity/pitch_mix"
    assert A.construct("K_last7") == A.construct("K_std") == A.construct("k_rate_last10") == "strikeout skill"
    assert A.construct("HR_last10") == "home-run skill" and A.construct("H_std") == "hit skill"
    assert A.stem("pitcher_k_rate_vs_hand_last10_R") == "pitcher_k_rate_vs_hand"
