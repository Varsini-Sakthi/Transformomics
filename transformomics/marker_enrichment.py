"""Do gene-attribution scores rank known marker genes higher than chance?

Replaces "saliency was validated against canonical markers" (a hand-picked look at plots) with a
test that has a null distribution:

  statistic : rank-AUC = P(score of a marker gene > score of a non-marker gene)   (0.5 = chance)
  null      : random gene sets of the same size, each member drawn from the same mean-expression
              bin as the corresponding marker (so 'abundant genes score high' is not rewarded)
  baselines : the same test applied to logistic-regression coefficients and one-vs-rest DE AUC,
              so the model's attribution is compared with what simple methods already give.

Rules for honest use:
  * take marker sets from a source independent of the model and of the labels' derivation;
  * report every class, not selected ones; markers absent from the model's input genes are
    reported as not testable (``n_in_input``), never silently dropped;
  * sets of 1-3 genes have almost no power - say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_auc(scores, set_idx) -> float:
    s = np.asarray(scores, dtype=float)
    n, k = len(s), len(set_idx)
    if k == 0 or k >= n:
        return np.nan
    r = rankdata(s)
    return float((r[set_idx].sum() - k * (k + 1) / 2) / (k * (n - k)))


def matched_null_sets(mean_expr, set_idx, n_null: int, rng, n_bins: int = 20) -> np.ndarray:
    """[n_null, k] random gene sets matched to ``set_idx`` on mean-expression bin."""
    me = np.asarray(mean_expr, dtype=float)
    n = len(me)
    bins = np.empty(n, dtype=int)
    bins[np.argsort(me, kind="stable")] = np.arange(n) * n_bins // n
    pools = {b: np.where(bins == b)[0] for b in range(n_bins)}
    in_set = np.zeros(n, dtype=bool)
    in_set[set_idx] = True
    out = np.empty((n_null, len(set_idx)), dtype=int)
    for i in range(n_null):
        chosen = set()
        for j, g in enumerate(set_idx):
            pool = pools[bins[g]]
            pool = pool[~in_set[pool]]
            pool = pool[[p not in chosen for p in pool]] if len(chosen) else pool
            if len(pool) == 0:                         # extremely small bin: fall back to any non-set gene
                pool = np.where(~in_set)[0]
            pick = int(rng.choice(pool))
            chosen.add(pick)
            out[i, j] = pick
    return out


def enrichment(scores, set_idx, mean_expr, n_null: int = 1000, seed: int = 0) -> dict:
    obs = rank_auc(scores, set_idx)
    if np.isnan(obs):
        return {"auc": np.nan, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "p": np.nan}
    rng = np.random.default_rng(seed)
    sets = matched_null_sets(mean_expr, set_idx, n_null, rng)
    null = np.array([rank_auc(scores, s) for s in sets])
    sd = null.std()
    return {"auc": obs, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
            "p": float((1 + (null >= obs - 1e-12).sum()) / (n_null + 1))}


def _bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        tmp = np.empty(len(pv))
        tmp[order] = np.minimum(ranked, 1.0)
        q[ok] = tmp
    return q


def enrichment_table(score_mats: dict, class_names, genes, markers: dict, mean_expr,
                     n_null: int = 1000, seed: int = 0, min_markers: int = 1) -> pd.DataFrame:
    """One row per (method, class). ``score_mats[method]`` is [C, G] aligned with ``genes``.

    ``markers`` maps class name -> list of gene symbols. Higher score must mean 'more evidence
    for the class' (signed attribution, coefficient, or DE AUC), NOT an absolute value.
    """
    gidx = {g: i for i, g in enumerate(np.asarray(genes).astype(str))}
    rows = []
    for method, M in score_mats.items():
        for ci, cname in enumerate(class_names):
            if cname not in markers:
                continue
            listed = list(dict.fromkeys(markers[cname]))
            idx = np.array([gidx[g] for g in listed if g in gidx], dtype=int)
            row = {"method": method, "class": cname, "n_listed": len(listed), "n_in_input": len(idx)}
            if len(idx) < min_markers or np.all(np.isnan(M[ci])):
                row.update({"auc": np.nan, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "p": np.nan})
            else:
                s = np.nan_to_num(M[ci], nan=np.nanmin(M[ci]))
                row.update(enrichment(s, idx, mean_expr, n_null, seed + ci))
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh"] = np.nan
        for m, d in df.groupby("method"):
            df.loc[d.index, "q_bh"] = _bh(d["p"].to_numpy())
    return df


# ------------------------------------------------------------------ simple baselines
def _subsample(n, max_n, seed):
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def logreg_scores(X, y, n_classes: int, C: float = 0.1, max_spots: int = 10000, seed: int = 0) -> np.ndarray:
    """[C, G] multinomial logistic-regression coefficients (NaN rows for classes absent from y)."""
    from sklearn.linear_model import LogisticRegression
    s = _subsample(len(y), max_spots, seed)
    m = LogisticRegression(max_iter=500, C=C).fit(X[s], y[s])
    out = np.full((n_classes, X.shape[1]), np.nan, dtype=np.float64)
    coef = m.coef_ if len(m.classes_) > 2 else np.vstack([-m.coef_[0], m.coef_[0]])
    out[m.classes_] = coef
    return out


def de_auc_scores(X, y, n_classes: int, max_spots: int = 8000, seed: int = 0) -> np.ndarray:
    """[C, G] one-vs-rest AUC per gene (Mann-Whitney), i.e. a Wilcoxon-style marker score."""
    s = _subsample(len(y), max_spots, seed)
    Xs, ys = X[s], y[s]
    R = rankdata(Xs, axis=0)
    out = np.full((n_classes, X.shape[1]), np.nan)
    for c in range(n_classes):
        m = ys == c
        n1, n0 = int(m.sum()), int((~m).sum())
        if n1 < 3 or n0 < 3:
            continue
        out[c] = (R[m].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out
