"""Download 10x Visium example datasets via squidpy and write per-section .h5ad + manifest.

    python scripts/fetch_10x.py --out data/10x V1_Human_Lymph_Node V1_Breast_Cancer_Block_A_Section_1

Verified sample ids: V1_Human_Lymph_Node, V1_Breast_Cancer_Block_A_Section_1, V1_Mouse_Brain_Sagittal_Posterior.
If an id errors, download 'Feature / barcode matrix HDF5 (filtered)' + 'Spatial imaging data' from the 10x
dataset page, unpack, and use scanpy.read_visium on that folder instead.
These datasets have no ground-truth labels, so they support the programme task / ablations, not label metrics.
"""
import argparse
import os

import squidpy as sq
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("sample_ids", nargs="+")
ap.add_argument("--out", default="data/10x")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
entries = []
for sid in a.sample_ids:
    x = sq.datasets.visium(sid)
    x.var_names_make_unique()
    path = os.path.join(a.out, f"{sid}.h5ad")
    x.write_h5ad(path)
    print(sid, x.n_obs, "spots", x.n_vars, "genes ->", path)
    entries.append({"path": path, "section_id": sid, "donor_id": None, "label_key": None})
yaml.safe_dump({"sections": entries}, open(os.path.join(a.out, "manifest.yaml"), "w"), sort_keys=False)
