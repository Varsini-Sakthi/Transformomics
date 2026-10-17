"""Multi-section data handling for real Visium experiments.

Why this exists: the single-section pipeline in ``run.py`` can only do spatial-block CV inside
one tissue slice. Real claims need held-out *sections/donors*. Rules enforced here:

* every section gets its **own** kNN graph (neighbours never cross sections), and offsets are
  scaled by that section's own median nearest-neighbour spacing;
* HVG selection and z-scoring use **training spots only** (no peeking at held-out sections);
* splits are leave-one-group-out, where a group is a donor (default) or a section. Adjacent
  sections from one donor are near-duplicates, so they must never straddle train/test.

Input: one ``.h5ad`` per section (raw counts in ``.X``, coords in ``.obsm['spatial']``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import normalize_log1p, select_hvg
from .graph import knn_graph, spatial_block_folds
from .programs import score_programs


@dataclass
class SectionSpec:
    path: str | None = None
    section_id: str = ""
    donor_id: str | None = None
    label_key: str | None = None
    adata: object = None            # optional in-memory AnnData (used by synthetic data / tests)


@dataclass
class Combined:
    logX: object                    # [N, G] log-normalised (sparse CSR or dense), common genes
    genes: np.ndarray               # [G]
    coords: np.ndarray              # [N, 2] raw coordinates (each section in its own frame)
    obs_names: np.ndarray           # [N] "<section>:<barcode>"
    section: np.ndarray             # [N] section id (str)
    group: np.ndarray               # [N] split unit: donor id if available else section id
    nbr_idx: np.ndarray             # [N, k] GLOBAL neighbour indices (always within the same section)
    offsets: np.ndarray             # [N, k, 2] scale-free offsets
    nbr_dist: np.ndarray            # [N, k] scale-free distances
    dist_k: np.ndarray              # [N] distance to the k-th neighbour (large near tissue edges)
    labels: np.ndarray | None       # [N] int codes, -1 = unlabeled
    label_names: list | None
    meta: dict = field(default_factory=dict)


def _read(spec: SectionSpec, spatial_key: str):
    if spec.adata is not None:
        return spec.adata
    import anndata as ad
    return ad.read_h5ad(spec.path)


def load_sections(specs, k: int = 6, target_sum: float = 1e4, min_cells: int = 3,
                  already_log: bool = False, spatial_key: str = "spatial", split_by: str = "donor",
                  permute_coords_seed: int | None = None) -> Combined:
    """Load several sections into one :class:`Combined` object.

    ``permute_coords_seed`` shuffles coordinates among the spots of each section *before* the graph
    is built. This destroys real spatial structure while keeping the graph geometry and the CV
    machinery identical -> an empirical null for 'real vs shuffled neighbours' gaps.
    """
    specs = [s if isinstance(s, SectionSpec) else SectionSpec(**s) for s in specs]
    if not specs:
        raise ValueError("no sections given")
    ids = [s.section_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"section_id values must be unique, got {ids}")

    raw = []
    for s in specs:
        a = _read(s, spatial_key)
        genes = np.asarray(a.var_names).astype(str)
        _, first = np.unique(genes, return_index=True)           # drop duplicate gene symbols
        keep = np.sort(first)
        raw.append((a, genes[keep], keep))
    others = [set(r[1]) for r in raw[1:]]
    common = [g for g in raw[0][1] if all(g in o for o in others)]
    if len(common) < 100:
        warnings.warn(f"only {len(common)} genes shared across sections; check gene naming")
    common = np.array(common)

    blocks, coords, names, sec, grp, labs = [], [], [], [], [], []
    for s, (a, g_u, keep) in zip(specs, raw):
        pos = {g: i for i, g in zip(keep, g_u)}
        cols = np.array([pos[g] for g in common])
        X = a.X[:, cols]
        blocks.append(X if already_log else normalize_log1p(X, target_sum))
        coords.append(np.asarray(a.obsm[spatial_key], dtype=np.float32)[:, :2])
        names.append(np.array([f"{s.section_id}:{o}" for o in a.obs_names]))
        sec.append(np.full(a.n_obs, s.section_id, dtype=object))
        grp.append(np.full(a.n_obs, (s.donor_id or s.section_id) if split_by == "donor" else s.section_id,
                           dtype=object))
        if s.label_key:
            v = a.obs[s.label_key]
            labs.append(np.array([None if pd.isna(x) else str(x) for x in v], dtype=object))
        else:
            labs.append(None)

    sparse = any(sp.issparse(b) for b in blocks)
    logX = sp.vstack([sp.csr_matrix(b) for b in blocks]).tocsr() if sparse else np.vstack(blocks)
    nnz = np.asarray((logX > 0).sum(0)).ravel()
    keepg = nnz >= min_cells
    logX, common = logX[:, keepg], common[keepg]

    coords_all, sec_all = np.vstack(coords), np.concatenate(sec)
    if permute_coords_seed is not None:
        rng = np.random.default_rng(permute_coords_seed)
        for s in ids:
            m = np.where(sec_all == s)[0]
            coords_all[m] = coords_all[m][rng.permutation(len(m))]

    N = len(sec_all)
    nbr = np.zeros((N, k), dtype=np.int64)
    off = np.zeros((N, k, 2), dtype=np.float32)
    dst = np.zeros((N, k), dtype=np.float32)
    for s in ids:
        m = np.where(sec_all == s)[0]
        if len(m) <= k:
            raise ValueError(f"section {s} has only {len(m)} spots (< k+1)")
        i, d, o = knn_graph(coords_all[m], k)
        nbr[m], dst[m], off[m] = m[i], d, o                      # local -> global indices

    labels = names_l = None
    if all(l is not None for l in labs):
        flat = np.concatenate(labs)
        names_l = sorted({x for x in flat if x is not None})
        code = {n: i for i, n in enumerate(names_l)}
        labels = np.array([code[x] if x is not None else -1 for x in flat], dtype=np.int64)
    elif any(l is not None for l in labs):
        warnings.warn("label_key given for only some sections; labels ignored")

    return Combined(logX, common, coords_all, np.concatenate(names), sec_all, np.concatenate(grp),
                    nbr, off, dst, dst[:, -1].copy(), labels, names_l,
                    meta={"k": k, "split_by": split_by, "permuted": permute_coords_seed is not None})


@dataclass
class Inputs:
    X: np.ndarray            # [N, G_in] z-scored (train statistics)
    genes_in: np.ndarray     # [G_in]
    mean_expr: np.ndarray    # [G_in] mean log-expression over TRAIN spots (for expression matching)


def prepare_inputs(comb: Combined, train_mask: np.ndarray, n_hvg: int = 2000, exclude_genes=()) -> Inputs:
    """HVG selection + z-scoring using training spots only; applied to every spot."""
    logX = comb.logX
    hv = select_hvg(logX[np.where(train_mask)[0]], n_hvg)
    excl = set(exclude_genes)
    hv = np.array([i for i in hv if comb.genes[i] not in excl], dtype=int)
    Xin = logX[:, hv]
    Xin = Xin.toarray() if sp.issparse(Xin) else np.asarray(Xin)
    mu, sd = Xin[train_mask].mean(0), Xin[train_mask].std(0)
    sd[sd < 1e-6] = 1.0
    Xz = np.clip((Xin - mu) / sd, -10, 10).astype(np.float32)
    return Inputs(Xz, comb.genes[hv], mu.astype(np.float32))


def score_programs_by_section(comb: Combined, programs: dict, **kw):
    """Module scores computed within each section (avoids cross-section batch offsets)."""
    N = comb.logX.shape[0]
    out, names = None, None
    for s in np.unique(comb.section):
        m = np.where(comb.section == s)[0]
        sc, nm, counts = score_programs(comb.logX[m], comb.genes, programs, **kw)
        if out is None:
            out, names = np.zeros((N, len(nm)), dtype=np.float32), nm
        elif nm != names:
            raise ValueError(f"programmes kept differ between sections ({names} vs {nm})")
        out[m] = sc
    return out, names


def group_splits(comb: Combined, val_mode: str = "auto", n_blocks: int = 24, n_folds: int = 4,
                 seed: int = 0, query_mask: np.ndarray | None = None):
    """Leave-one-group-out splits. Yields dicts with keys test_group, tr, va, te (index arrays).

    val_mode:
      'group' - validation = next group (cyclic); needs >= 3 groups, ideally >= 4.
      'block' - validation = one spatial-block fold inside the training groups, with a neighbour
                buffer. With few groups this keeps more training data, BUT replicate sections of
                one donor are near-duplicates, so validation loss is optimistic (early stopping is
                weaker). The held-out TEST group is never affected.
      'auto'  - 'group' if >= 4 groups else 'block'.
    """
    groups = sorted(set(comb.group))
    if len(groups) < 2:
        raise ValueError("need at least 2 groups (donors or sections) for leave-one-group-out")
    if val_mode == "auto":
        val_mode = "group" if len(groups) >= 4 else "block"
        if val_mode == "block":
            warnings.warn(f"only {len(groups)} groups: using validation='block' (see group_splits docstring)")
    if val_mode == "group" and len(groups) < 3:
        raise ValueError("val_mode='group' needs >= 3 groups")

    fold = None
    if val_mode == "block":
        fold = np.zeros(len(comb.group), dtype=int)
        for s in np.unique(comb.section):
            m = np.where(comb.section == s)[0]
            nb = min(n_blocks, max(2, len(m) // 20))
            fold[m] = spatial_block_folds(comb.coords[m], nb, n_folds, seed)

    for gi, g in enumerate(groups):
        te = comb.group == g
        if val_mode == "group":
            va = comb.group == groups[(gi + 1) % len(groups)]
            tr = ~(te | va)
        else:
            rest = ~te
            va = rest & (fold == 0)
            tr = rest & ~va
            held = np.where(va)[0]
            near = np.zeros(len(tr), dtype=bool)
            near[comb.nbr_idx[held].ravel()] = True
            tr &= ~(near | va[comb.nbr_idx].any(1))
        if query_mask is not None:
            te, va, tr = te & query_mask, va & query_mask, tr & query_mask
        yield {"test_group": g, "tr": np.where(tr)[0], "va": np.where(va)[0], "te": np.where(te)[0]}
