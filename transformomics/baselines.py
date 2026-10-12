"""Non-transformer baselines that use the *same* inputs, splits and targets.

'own'          - linear model on the spot's own expression (the v1 logistic-regression baseline).
'own_nbr_mean' - linear model on own expression + mean expression of the k spatial neighbors.
                 This is the key baseline: if the transformer cannot beat it, the attention
                 machinery is not adding anything beyond simple neighborhood smoothing.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV


def features(X, nbr_idx, kind: str):
    if kind == "own":
        return X
    if kind == "own_nbr_mean":
        return np.hstack([X, X[nbr_idx].mean(1)])
    raise ValueError(kind)


def fit_predict_baseline(X, nbr_idx, y, tr, te, task: str, kind: str):
    F = features(X, nbr_idx, kind)
    if task == "program":
        mu, sd = y[tr].mean(0), y[tr].std(0)
        sd[sd == 0] = 1.0
        m = RidgeCV(alphas=[10, 100, 1e3, 1e4, 1e5]).fit(F[tr], (y[tr] - mu) / sd)
        return m.predict(F[te])
    m = LogisticRegression(max_iter=1000, C=0.1).fit(F[tr], y[tr])
    proba = m.predict_proba(F[te])
    full = np.zeros((len(te), int(y.max()) + 1))
    full[:, m.classes_] = proba
    return full
