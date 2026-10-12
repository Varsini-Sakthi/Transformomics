"""Gene-programme (module) scoring and spatial-structure statistics.

This is the bridge to the microglia work: the programmes (homeostatic, phagocytic,
NRF2/Hmox1, Apoc1, chemokine, ...) are defined as gene lists in a YAML file, scored per
spot/cell, and then (a) tested for spatial autocorrelation and (b) used as prediction
targets for the Transformomics.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import yaml


def load_programs(path_or_dict) -> dict:
    """Load ``{programme_name: [genes]}`` from a YAML file (top-level key ``programs``) or dict."""
    if isinstance(path_or_dict, dict):
        d = path_or_dict
    else:
        with open(path_or_dict) as f:
            d = yaml.safe_load(f)
    d = d.get("programs", d)
    return {k: [str(g) for g in v] for k, v in d.items()}


def _col_mean(X, idx):
    sub = X[:, idx]
    return np.asarray(sub.mean(1)).ravel() if sp.issparse(sub) else sub.mean(1)


def score_programs(logX, genes, programs: dict, n_bins: int = 25, n_ctrl: int = 50,
                   min_genes: int = 3, seed: int = 0):
    """Control-subtracted module score (as in Seurat/Scanpy ``score_genes``).

    score = mean(programme genes) - mean(expression-matched control genes).
    Programmes with fewer than ``min_genes`` detected genes are dropped with a warning.
    Returns (scores [N, P] float32, kept_names, detected_gene_counts dict).
    """
    genes = np.asarray(genes).astype(str)
    gidx = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    mean = np.asarray(logX.mean(0)).ravel() if sp.issparse(logX) else logX.mean(0)
    order = np.argsort(mean, kind="stable")
    bins = np.empty(G, dtype=int)
    bins[order] = np.arange(G) * n_bins // G
    rng = np.random.default_rng(seed)

    cols, names, counts = [], [], {}
    for name, gl in programs.items():
        idx = [gidx[g] for g in dict.fromkeys(gl) if g in gidx]
        counts[name] = (len(idx), len(set(gl)))
        if len(idx) < min_genes:
            warnings.warn(f"programme '{name}': only {len(idx)}/{len(set(gl))} genes detected "
                          f"(< {min_genes}); dropped.")
            continue
        ctrl = []
        for i in idx:
            pool = np.where(bins == bins[i])[0]
            pool = pool[~np.isin(pool, idx)]
            if len(pool):
                ctrl.extend(rng.choice(pool, size=min(n_ctrl, len(pool)), replace=False))
        ctrl = np.unique(ctrl).astype(int)
        cols.append(_col_mean(logX, idx) - _col_mean(logX, ctrl))
        names.append(name)
    if not cols:
        raise ValueError("No programme had enough detected genes. Check gene-name case/species.")
    return np.stack(cols, 1).astype(np.float32), names, counts


def morans_i(values, nbr_idx, n_perm: int = 199, seed: int = 0):
    """Moran's I with row-standardized kNN weights and a permutation p-value.

    Returns (I, p). Positive I = similar values cluster in space.
    """
    v = np.asarray(values, dtype=np.float64)
    z = v - v.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return 0.0, 1.0
    k = nbr_idx.shape[1]

    def stat(zz):
        return (zz[:, None] * zz[nbr_idx]).sum() / k / (zz ** 2).sum()

    obs = stat(z)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (1 + (perms >= obs).sum()) / (n_perm + 1)
    return float(obs), float(p)


def spatial_summary(scores, names, nbr_idx, n_perm: int = 199):
    """Per-programme Moran's I table (list of dicts)."""
    rows = []
    for j, n in enumerate(names):
        i, p = morans_i(scores[:, j], nbr_idx, n_perm=n_perm, seed=j)
        rows.append({"programme": n, "morans_I": i, "perm_p": p})
    return rows
