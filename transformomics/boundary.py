"""Boundary analysis: do neighbours matter more at domain boundaries?

Pre-specified definitions (change them in the config BEFORE looking at results, not after):

* ``boundary_score``  = fraction of a spot's labelled neighbours whose domain label differs.
* ``boundary``        = score >= ``min_frac`` (default 2/6);  ``interior`` = score == 0;
  everything in between (e.g. exactly 1 of 6 neighbours differs) is excluded as ambiguous.

Two outcomes are tested, because attention is only descriptive:
  (a) attention allocated to neighbours (last-layer CLS row and attention rollout);
  (b) **benefit** = per-spot loss without the neighbour information minus loss with it
      (``loss_shuffled - loss_real`` and ``loss_none - loss_real``).

The key control is the same analysis on the shuffled-neighbour model: if attention to neighbours
rises at boundaries there too, the effect is not about spatial structure. ``attn_excess`` =
attention(real) - attention(shuffled) per spot removes that shared component.

Confounds adjusted for (regression covariates): own-expression ambiguity, distance to the k-th
neighbour (tissue-edge proxy), true-label fixed effects and section fixed effects.

CAVEAT on p-values: the permutation test shuffles the boundary indicator within (section x label)
strata and treats spots as exchangeable. Neighbouring spots are not independent, so these p-values
are anti-conservative. Lean on the per-group (donor/section) consistency columns instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def boundary_score(labels, nbr_idx) -> np.ndarray:
    """Fraction of labelled neighbours with a different label (NaN if undefined)."""
    lab = np.asarray(labels)
    nl = lab[nbr_idx]                                   # [N,k]
    valid = nl >= 0
    diff = (nl != lab[:, None]) & valid
    nv = valid.sum(1)
    score = np.where(nv > 0, diff.sum(1) / np.maximum(nv, 1), np.nan)
    score[lab < 0] = np.nan
    return score


def boundary_class(score, min_frac: float = 2 / 6, tol: float = 1e-9) -> np.ndarray:
    """'boundary' | 'interior' | 'excluded' per spot."""
    s = np.asarray(score, dtype=float)
    out = np.full(len(s), "excluded", dtype=object)
    out[s == 0] = "interior"
    out[s >= min_frac - tol] = "boundary"
    return out


# ------------------------------------------------------------------ regression helpers
def build_design(df: pd.DataFrame, cont_cols=(), cat_cols=()) -> np.ndarray:
    """Intercept + standardised continuous covariates + one-hot (first level dropped)."""
    cols = [np.ones(len(df))]
    for c in cont_cols:
        v = df[c].to_numpy(dtype=float)
        if np.all(np.isnan(v)):
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        sd = v.std()
        cols.append((v - v.mean()) / sd if sd > 0 else np.zeros_like(v))
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype(str), drop_first=True, dtype=float)
        cols.extend([d[x].to_numpy() for x in d.columns])
    return np.column_stack(cols)


class _Resid:
    """Frisch-Waugh-Lovell helper: coefficient of b in y ~ b + X, fast for many permutations."""

    def __init__(self, y, X):
        self.X = X
        self.pinv = np.linalg.pinv(X)
        self.y_res = y - X @ (self.pinv @ y)

    def coef(self, b):
        b_res = b - self.X @ (self.pinv @ b)
        den = float(b_res @ b_res)
        return float(b_res @ self.y_res) / den if den > 1e-12 else np.nan


def boundary_effect(y, is_boundary, X, perm_strata=None, n_perm: int = 999, seed: int = 0) -> dict:
    """Adjusted boundary-minus-interior effect on ``y`` with a stratified permutation p (one-sided >)."""
    y = np.asarray(y, dtype=float)
    b = np.asarray(is_boundary, dtype=float)
    out = {"n_boundary": int(b.sum()), "n_interior": int((1 - b).sum()),
           "mean_boundary": float(y[b == 1].mean()) if b.sum() else np.nan,
           "mean_interior": float(y[b == 0].mean()) if (1 - b).sum() else np.nan}
    out["raw_diff"] = out["mean_boundary"] - out["mean_interior"]
    if b.sum() < 5 or (1 - b).sum() < 5:
        return {**out, "adj_effect": np.nan, "p_perm": np.nan}
    r = _Resid(y, X)
    obs = r.coef(b)
    rng = np.random.default_rng(seed)
    strata = np.zeros(len(b), dtype=int) if perm_strata is None else pd.factorize(np.asarray(perm_strata))[0]
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    ge = 0
    for _ in range(n_perm):
        bp = b.copy()
        for g in groups:
            bp[g] = b[g][rng.permutation(len(g))]
        v = r.coef(bp)
        ge += (v >= obs - 1e-12)
    return {**out, "adj_effect": obs, "p_perm": (1 + ge) / (n_perm + 1)}


def effect_by_group(df, outcome, group_col, cont_cols=(), cat_cols=()) -> pd.DataFrame:
    """Adjusted effect within each group (donor/section) separately -> consistency check."""
    rows = []
    for g, d in df.groupby(group_col):
        d = d.dropna(subset=[outcome])
        b = (d["bclass"] == "boundary").to_numpy(dtype=float)
        if b.sum() < 5 or (1 - b).sum() < 5:
            rows.append({"group": g, "n_boundary": int(b.sum()), "n_interior": int((1 - b).sum()),
                         "adj_effect": np.nan})
            continue
        cats = [c for c in cat_cols if d[c].nunique() > 1]
        X = build_design(d, cont_cols, cats)
        rows.append({"group": g, "n_boundary": int(b.sum()), "n_interior": int((1 - b).sum()),
                     "adj_effect": _Resid(d[outcome].to_numpy(float), X).coef(b)})
    return pd.DataFrame(rows)


def analyze_outcomes(df: pd.DataFrame, outcomes, group_col: str = "group",
                     cont_cols=("ambig", "dist_k"), cat_cols=("label", "section"),
                     n_perm: int = 999, seed: int = 0):
    """Run the boundary-vs-interior test for each outcome column.

    ``df`` needs: ``bclass`` ('boundary'/'interior'/'excluded'), the outcome columns, covariate
    columns, ``group``, ``section``, ``label``. Returns (summary_df, per_group_df).
    """
    d0 = df[df["bclass"].isin(["boundary", "interior"])].copy()
    cont = [c for c in cont_cols if c in d0 and d0[c].notna().any()]
    summ, per_group = [], []
    for o in outcomes:
        if o not in d0 or d0[o].notna().sum() == 0:
            continue
        d = d0.dropna(subset=[o])
        cats = [c for c in cat_cols if c in d and d[c].nunique() > 1]
        X = build_design(d, cont, cats)
        strata = d["section"].astype(str) + "|" + d["label"].astype(str)
        res = boundary_effect(d[o].to_numpy(float), (d["bclass"] == "boundary").to_numpy(), X,
                              perm_strata=strata.to_numpy(), n_perm=n_perm, seed=seed)
        pg = effect_by_group(d, o, group_col, cont, cats)
        pg.insert(0, "outcome", o)
        per_group.append(pg)
        e = pg["adj_effect"].dropna()
        res.update({"outcome": o, "n_groups": int(len(e)), "n_groups_positive": int((e > 0).sum()),
                    "group_effect_min": float(e.min()) if len(e) else np.nan,
                    "group_effect_max": float(e.max()) if len(e) else np.nan})
        summ.append(res)
    cols = ["outcome", "n_boundary", "n_interior", "mean_boundary", "mean_interior", "raw_diff",
            "adj_effect", "p_perm", "n_groups", "n_groups_positive", "group_effect_min", "group_effect_max"]
    summary = pd.DataFrame(summ)[cols] if summ else pd.DataFrame(columns=cols)
    return summary, (pd.concat(per_group, ignore_index=True) if per_group else pd.DataFrame())
