"""Synthetic multi-section Visium-like data with KNOWN domains, boundaries and marker genes.

Purpose: a positive/negative control for the multi-section pipeline (loaders, donor-out CV,
boundary analysis, marker enrichment, faithfulness). It proves the plumbing can detect a planted
effect; it says nothing about real tissue.

Generative model per section (hex lattice, spacing 1):
  * D domains = soft Voronoi cells around random seed points (seeds shared within a donor, jittered
    per section, so same-donor sections are adjacent-replicate-like);
  * each domain has ``markers_per_domain`` marker genes; a spot's expression is a MIXTURE of domain
    profiles with weights softmax(-distance_to_seed / mix_temp);
  * ``mix_temp`` small  -> sharp boundaries, own expression nearly sufficient (neighbours should
    help little); ``mix_temp`` large -> mixed boundary spots that only neighbours can disambiguate;
  * donor-specific gene offsets (batch), lognormal library sizes, Poisson counts.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from .multisection import SectionSpec


def make_synthetic_sections(n_sections: int = 6, n_donors: int = 3, side: int = 26, n_genes: int = 500,
                            n_domains: int = 4, markers_per_domain: int = 8, mix_temp: float = 1.2,
                            marker_strength: float = 1.0, donor_shift: float = 0.25, lib_mean: float = 2500.0,
                            seed: int = 0):
    """Returns (specs [SectionSpec with in-memory adata], markers {domain_name: [genes]})."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:04d}" for i in range(n_genes)], dtype=object)   # object: names must not be truncated
    markers = {}
    prof = rng.normal(0, 0.1, (n_domains, n_genes))
    for d in range(n_domains):
        sl = slice(d * markers_per_domain, (d + 1) * markers_per_domain)
        genes[sl] = [f"DOM{d}_MK{j}" for j in range(markers_per_domain)]
        prof[d, sl] += marker_strength
        markers[f"D{d}"] = list(genes[sl])
    base = rng.normal(0, 1.0, n_genes)
    donor_off = {f"donor{j}": rng.normal(0, donor_shift, n_genes) for j in range(n_donors)}
    donor_seeds = {f"donor{j}": rng.uniform(0.15, 0.85, (n_domains, 2)) * side for j in range(n_donors)}

    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    xy = np.column_stack([(cols + 0.5 * (rows % 2)).ravel(), (rows * np.sqrt(3) / 2).ravel()])
    xy = xy[:, ::1]
    specs = []
    for s in range(n_sections):
        donor = f"donor{s % n_donors}"
        seeds = donor_seeds[donor] + rng.normal(0, 0.6, (n_domains, 2))
        dist = np.linalg.norm(xy[:, None, :] - seeds[None, :, :], axis=2)            # [N, D]
        w = np.exp(-(dist - dist.min(1, keepdims=True)) / max(mix_temp, 1e-3))
        w /= w.sum(1, keepdims=True)
        lab = w.argmax(1)
        logmu = base + donor_off[donor] + w @ prof + rng.normal(0, 0.1, (len(xy), n_genes))
        p = np.exp(logmu - logmu.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        lib = rng.lognormal(np.log(lib_mean), 0.25, len(xy))
        counts = rng.poisson(p * lib[:, None]).astype(np.float32)
        sid = f"S{s}"
        a = ad.AnnData(X=sp.csr_matrix(counts),
                       obs=pd.DataFrame({"domain": [f"D{d}" for d in lab]},
                                        index=[f"{sid}_{i}" for i in range(len(xy))]),
                       var=pd.DataFrame(index=genes))
        a.obsm["spatial"] = (xy * 100.0 + 1000.0 + s).astype(np.float32)
        specs.append(SectionSpec(section_id=sid, donor_id=donor, label_key="domain", adata=a))
    return specs, markers


def write_synthetic_sections(out_dir: str, **kw) -> dict:
    """Write one h5ad per section + manifest.yaml + markers.yaml; returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    specs, markers = make_synthetic_sections(**kw)
    entries = []
    for s in specs:
        path = os.path.join(out_dir, f"{s.section_id}.h5ad")
        s.adata.write_h5ad(path)
        entries.append({"path": path, "section_id": s.section_id, "donor_id": s.donor_id, "label_key": "domain"})
    yaml.safe_dump({"sections": entries}, open(os.path.join(out_dir, "manifest.yaml"), "w"))
    yaml.safe_dump({"markers": markers}, open(os.path.join(out_dir, "markers.yaml"), "w"))
    return {"sections": entries, "markers": markers}
