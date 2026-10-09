"""Model stacking: weights stay on the simplex, the optimised stack is scored
out-of-fold, and the wrappers predict like the models they combine."""
import numpy as np

import model_stacking as stk


def _data(n=400, seed=0):
    r = np.random.default_rng(seed)
    y = r.poisson(3, n).astype(float)
    P = np.column_stack([y + r.normal(0, s, n) for s in (0.5, 1.0, 3.0)])
    return P, y


def test_weights_on_simplex_and_favour_best_model():
    P, y = _data()
    w = stk.fit_weights(P, y, "rmse")
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-8
    assert w[0] == w.max() and w[2] < 0.2


def test_crossfit_uses_only_the_other_half():
    P, y = _data()
    cf = stk.crossfit_predictions(P, y, "rmse")
    h = len(y) // 2
    w_first = stk.fit_weights(P[:h], y[:h], "rmse")
    assert np.allclose(cf[h:], P[h:] @ w_first)          # second half scored with first-half weights
    y2 = y.copy(); y2[h:] += 100                           # change second-half outcomes ...
    cf2 = stk.crossfit_predictions(P, y2, "rmse")
    assert np.allclose(cf2[h:], cf[h:])                    # ... their own predictions do not move


def test_logloss_weights_and_classifier_wrapper():
    r = np.random.default_rng(1)
    y = (r.random(500) < 0.55).astype(float)
    good = np.clip(0.55 + 0.2 * (y - 0.5) + r.normal(0, 0.05, 500), 0.01, 0.99)
    P = np.column_stack([good, np.full(500, 0.5)])
    w = stk.fit_weights(P, y, "logloss")
    assert w[0] > w[1]

    class M:
        def __init__(self, p): self.p = p
        def predict_proba(self, X): return np.column_stack([1 - self.p, self.p])
    c = stk.StackedClassifier([M(P[:, 0]), M(P[:, 1])], [0.5, 0.5], ["a", "b"])
    pr = c.predict_proba(None)
    assert np.allclose(pr.sum(1), 1) and np.allclose(pr[:, 1], P.mean(1))


def test_regressor_wrapper_is_weighted_mean_and_non_negative():
    class M:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.full(len(X), self.v)
    s = stk.StackedRegressor([M(2.0), M(-10.0)], [0.75, 0.25], ["a", "b"])
    assert np.allclose(s.predict(np.zeros((3, 1))), 0.0)          # 1.5 - 2.5 clipped at 0
    s2 = stk.StackedRegressor([M(2.0), M(4.0)], [0.75, 0.25], ["a", "b"])
    assert np.allclose(s2.predict(np.zeros((3, 1))), 2.5)


def test_auc_weights_pick_the_ranking_model():
    r = np.random.default_rng(3)
    y = (r.random(800) < 0.5).astype(float)
    good = np.clip(0.5 + 0.15 * (y - 0.5) + r.normal(0, 0.1, 800), 0.01, 0.99)   # ranks games
    noise = r.random(800)                                                       # does not
    w = stk.fit_weights(np.column_stack([good, noise]), y, "auc")
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-8 and w[0] > 0.8
