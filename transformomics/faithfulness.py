"""Is learned neighbour attention more *informative* than trivial neighbour weightings?

The claim to test (instead of asserting 'improved interpretability'):

    H: the neighbours the model attends to are the ones that share the query's true domain,
       and it does so better than simple non-learned weightings.

Metric (label task, ground-truth domains needed): for each spot, the share of neighbour weight that
lands on neighbours with the SAME true label as the query. Compared against
  * uniform weights                        (= fraction of neighbours that are same-label)
  * inverse-distance weights               (a pure spatial prior)
  * expression-cosine weights              (a pure similarity prior - the honest baseline:
                                            if attention cannot beat this, it adds no insight)

This is only informative where neighbours are a mixture of same/different labels, so the summary
is restricted to spots with 0 < fraction-different < 1 (boundary-like spots).

The complementary causal check (deleting top-attended neighbours) lives in ``interpret.deletion_test``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _norm_rows(w, valid):
    w = np.where(valid, np.maximum(w, 0.0), 0.0)
    s = w.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = w / s
    out[(s <= 0).ravel()] = np.nan
    return out


def baseline_neighbour_weights(kind: str, X, idx, nbr_idx, nbr_dist, eps: float = 1e-6):
    """[n,k] non-learned weights for spots ``idx``."""
    nb = nbr_idx[idx]
    if kind == "uniform":
        return np.ones(nb.shape)
    if kind == "inv_dist":
        return 1.0 / (nbr_dist[idx] + eps)
    if kind == "expr_cosine":
        q = X[idx]
        nq = np.linalg.norm(q, axis=1, keepdims=True) + eps
        n = X[nb]
        nn_ = np.linalg.norm(n, axis=2) + eps
        cos = np.einsum("ng,nkg->nk", q, n) / (nq * nn_)
        return np.maximum(cos, 0.0) + eps
    raise ValueError(kind)


def same_label_mass(weights, labels, idx, nbr_idx) -> np.ndarray:
    """Share of (renormalised) neighbour weight on neighbours with the query's label; NaN if undefined."""
    nb = nbr_idx[idx]
    nl, ql = labels[nb], labels[idx][:, None]
    valid = (nl >= 0) & (ql >= 0)
    w = _norm_rows(np.asarray(weights, dtype=float), valid)
    return np.nansum(np.where(nl == ql, w, 0.0), axis=1) * np.where(np.isnan(w).all(1), np.nan, 1.0)


def attention_vs_baselines(attn_nbr, idx, labels, nbr_idx, nbr_dist, X, group, bscore=None) -> pd.DataFrame:
    """Per-spot same-label mass for attention and baselines.

    ``attn_nbr`` [n,k] attention (or rollout) weights on the k neighbours of spots ``idx``.
    Only spots with 0 < fraction-different < 1 are kept (otherwise the metric cannot discriminate).
    """
    idx = np.asarray(idx)
    nb = nbr_idx[idx]
    nl, ql = labels[nb], labels[idx][:, None]
    valid = (nl >= 0) & (ql >= 0)
    frac_diff = np.where(valid.sum(1) > 0, ((nl != ql) & valid).sum(1) / np.maximum(valid.sum(1), 1), np.nan)
    df = pd.DataFrame({
        "spot": idx, "group": np.asarray(group)[idx], "frac_diff": frac_diff,
        "attention": same_label_mass(attn_nbr, labels, idx, nbr_idx),
        "uniform": same_label_mass(baseline_neighbour_weights("uniform", X, idx, nbr_idx, nbr_dist), labels, idx, nbr_idx),
        "inv_dist": same_label_mass(baseline_neighbour_weights("inv_dist", X, idx, nbr_idx, nbr_dist), labels, idx, nbr_idx),
        "expr_cosine": same_label_mass(baseline_neighbour_weights("expr_cosine", X, idx, nbr_idx, nbr_dist), labels, idx, nbr_idx),
    })
    if bscore is not None:
        df["bscore"] = np.asarray(bscore)[idx]
    return df[(df["frac_diff"] > 0) & (df["frac_diff"] < 1)].reset_index(drop=True)


def summarize_vs_baselines(df: pd.DataFrame, baselines=("uniform", "inv_dist", "expr_cosine")) -> pd.DataFrame:
    """Mean attention-minus-baseline per baseline, overall and per group (donor/section)."""
    rows = []
    for b in baselines:
        d = (df["attention"] - df[b]).dropna()
        per = df.assign(diff=df["attention"] - df[b]).groupby("group")["diff"].mean()
        rows.append({"baseline": b, "n_spots": int(len(d)), "mean_attention": float(df["attention"].mean()),
                     "mean_baseline": float(df[b].mean()), "mean_diff": float(d.mean()) if len(d) else np.nan,
                     "n_groups": int(len(per)), "n_groups_attention_better": int((per > 0).sum()),
                     "group_diff_min": float(per.min()) if len(per) else np.nan,
                     "group_diff_max": float(per.max()) if len(per) else np.nan})
    return pd.DataFrame(rows)
