"""Spatial graph utilities: kNN neighbors, scale-free offsets, spatial-block folds."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def knn_graph(coords: np.ndarray, k: int = 6):
    """Return (neighbor_idx [N,k], dist [N,k], offsets [N,k,2]) excluding self.

    Offsets and distances are divided by the median nearest-neighbor distance so the
    model sees technology-independent units (1.0 ~ one spot/cell spacing).
    """
    coords = np.asarray(coords, dtype=np.float64)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dist, idx = nbrs.kneighbors(coords)
    idx, dist = idx[:, 1:], dist[:, 1:]
    unit = np.median(dist[:, 0])
    if not np.isfinite(unit) or unit <= 0:
        unit = 1.0
    offsets = (coords[idx] - coords[:, None, :]) / unit
    return idx.astype(np.int64), (dist / unit).astype(np.float32), offsets.astype(np.float32)


def spatial_block_folds(coords: np.ndarray, n_blocks: int = 24, n_folds: int = 4, seed: int = 0):
    """Assign each spot to a fold via contiguous spatial blocks (KMeans on coordinates).

    Random spot-level splits leak because neighboring spots are highly correlated;
    holding out whole spatial blocks gives a much more honest estimate.
    Returns an int array [N] of fold ids in [0, n_folds).
    """
    km = KMeans(n_clusters=n_blocks, n_init=3, random_state=seed).fit(coords)
    blocks = km.labels_
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    block_to_fold = np.empty(n_blocks, dtype=int)
    block_to_fold[order] = np.arange(n_blocks) % n_folds
    return block_to_fold[blocks]


def make_split(fold_of_spot, nbr_idx, fold: int, n_folds: int, buffer: bool = True, query_mask=None):
    """Train/val/test index arrays for one outer fold.

    test = ``fold``; val = ``(fold+1) % n_folds``; train = the rest.
    With ``buffer`` any training spot that is a neighbor of (or has a neighbor in) a
    val/test spot is dropped, so no train query ever shares tokens with held-out spots.
    ``query_mask`` (bool [N]) restricts which spots may be queries at all.
    """
    n = len(fold_of_spot)
    te = fold_of_spot == fold
    va = fold_of_spot == (fold + 1) % n_folds
    tr = ~(te | va)
    if buffer:
        held = np.where(te | va)[0]
        in_nbrs_of_held = np.zeros(n, dtype=bool)
        in_nbrs_of_held[nbr_idx[held].ravel()] = True
        has_held_nbr = (te | va)[nbr_idx].any(axis=1)
        tr &= ~(in_nbrs_of_held | has_held_nbr)
    if query_mask is not None:
        te, va, tr = te & query_mask, va & query_mask, tr & query_mask
    return np.where(tr)[0], np.where(va)[0], np.where(te)[0]
