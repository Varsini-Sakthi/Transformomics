"""Loading and preprocessing of spatial expression data (AnnData in, numpy out)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Prepared:
    """Everything downstream code needs, as plain arrays."""

    logX_all: object          # [N, G_all] log-normalized (dense or sparse), used for programme scoring
    genes_all: np.ndarray     # [G_all]
    X: np.ndarray             # [N, G_in] model input (HVGs, programme genes optionally removed, z-scored)
    genes_in: np.ndarray      # [G_in]
    coords: np.ndarray        # [N, 2]
    obs_names: np.ndarray


def normalize_log1p(X, target_sum: float = 1e4):
    """Library-size normalize then log1p. Accepts dense or sparse; returns same kind."""
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32)
        lib = np.asarray(X.sum(1)).ravel()
        lib[lib == 0] = 1.0
        X = sp.diags(target_sum / lib) @ X
        X = X.tocsr()
        X.data = np.log1p(X.data)
        return X
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X * (target_sum / lib))


def select_hvg(logX, n_top: int = 2000, n_bins: int = 20) -> np.ndarray:
    """Simple Seurat-style HVG selection: dispersion z-scored within mean-expression bins."""
    if sp.issparse(logX):
        mean = np.asarray(logX.mean(0)).ravel()
        sq = np.asarray(logX.multiply(logX).mean(0)).ravel()
    else:
        mean = logX.mean(0)
        sq = (logX ** 2).mean(0)
    var = np.maximum(sq - mean ** 2, 0)
    disp = var / np.maximum(mean, 1e-6)
    disp[mean < 1e-3] = 0
    order = np.argsort(mean)
    bins = np.empty(len(mean), dtype=int)
    bins[order] = np.arange(len(mean)) * n_bins // len(mean)
    z = np.zeros_like(disp)
    for b in range(n_bins):
        m = bins == b
        if m.sum() > 1:
            sd = disp[m].std()
            z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1.0)
    n_top = min(n_top, len(z))
    return np.sort(np.argsort(-z)[:n_top])


def prepare(adata, n_hvg: int = 2000, target_sum: float = 1e4, min_cells: int = 3,
            already_log: bool = False, exclude_genes=(), spatial_key: str = "spatial") -> Prepared:
    """AnnData (raw counts in .X unless ``already_log``) -> :class:`Prepared`.

    ``exclude_genes`` (e.g. every gene in a programme being predicted) are removed from
    the model input so the target cannot be read off its own genes. Programme scoring
    still uses the full gene set (``logX_all``).
    """
    X = adata.X
    genes = np.asarray(adata.var_names).astype(str)
    if sp.issparse(X):
        keep = np.asarray((X > 0).sum(0)).ravel() >= min_cells
    else:
        keep = (np.asarray(X) > 0).sum(0) >= min_cells
    X, genes = X[:, keep], genes[keep]
    logX = X if already_log else normalize_log1p(X, target_sum)

    hv = select_hvg(logX, n_hvg)
    excl = set(exclude_genes)
    hv = np.array([i for i in hv if genes[i] not in excl], dtype=int)
    Xin = logX[:, hv]
    Xin = Xin.toarray() if sp.issparse(Xin) else np.asarray(Xin)
    mu, sd = Xin.mean(0), Xin.std(0)
    sd[sd < 1e-6] = 1.0
    Xin = np.clip((Xin - mu) / sd, -10, 10).astype(np.float32)

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :2]
    return Prepared(logX, genes, Xin, genes[hv], coords, np.asarray(adata.obs_names))
