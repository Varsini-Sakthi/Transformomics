"""Synthetic Visium-like data for smoke tests and demos (NOT real biology).

Four latent programme activities vary smoothly in space plus spot-level noise; programme
genes and ~100 'context' genes both load on the smooth fields, with Poisson sampling noise.
So a spot's own expression is a noisy readout of the field and its neighbors carry extra
independent readouts of the same field -- a setting where neighborhood context *should*
help. Use it to check the pipeline, never to claim anything about microglia.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

PROGRAM_NAMES = ["homeostatic", "phagocytic", "nrf2_hmox1", "apoc1_chemokine"]


def make_synthetic(n_rows: int = 50, n_cols: int = 56, n_genes: int = 600, genes_per_program: int = 10,
                   seed: int = 0, smooth_strength: float = 1.0):
    rng = np.random.default_rng(seed)
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    x = (cc + 0.5 * (rr % 2)).ravel().astype(float)
    y = (rr * np.sqrt(3) / 2).ravel().astype(float)
    coords = np.stack([x, y], 1) * 100.0
    N = len(x)

    def smooth_field():
        f = np.zeros(N)
        for _ in range(3):
            kx, ky = rng.normal(0, 0.12, 2)
            f += np.sin(kx * x + ky * y + rng.uniform(0, 2 * np.pi))
        return (f - f.mean()) / f.std()

    P = len(PROGRAM_NAMES)
    fields = np.stack([smooth_field() for _ in range(P)], 1)           # [N,P]
    idio = rng.normal(0, 0.5, (N, P))                                   # spot-level noise
    activity = smooth_strength * fields + idio

    gene_names, logmu_cols = [], []
    base = rng.normal(-0.5, 0.8, n_genes)
    n_ctx = 100
    for g in range(n_genes):
        lm = np.full(N, base[g])
        pg = g // genes_per_program
        if g < P * genes_per_program:
            lm = lm + 0.5 * activity[:, g // genes_per_program]
            gene_names.append(f"{PROGRAM_NAMES[pg][:3].title()}{g % genes_per_program}")
        elif g < P * genes_per_program + n_ctx:
            w = rng.normal(0, 0.35, P)
            lm = lm + smooth_strength * fields @ w
            gene_names.append(f"Ctx{g}")
        else:
            gene_names.append(f"Gene{g}")
        logmu_cols.append(lm)
    logmu = np.stack(logmu_cols, 1)
    lib = np.exp(rng.normal(0, 0.3, (N, 1))) * 3.0
    counts = rng.poisson(lib * np.exp(logmu)).astype(np.float32)

    adata = ad.AnnData(counts, obs=pd.DataFrame(index=[f"spot{i}" for i in range(N)]),
                       var=pd.DataFrame(index=gene_names))
    adata.obsm["spatial"] = coords.astype(np.float32)
    programs = {PROGRAM_NAMES[p]: [gene_names[p * genes_per_program + i] for i in range(genes_per_program)]
                for p in range(P)}
    return adata, programs
