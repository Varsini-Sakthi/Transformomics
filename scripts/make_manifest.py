"""Build manifest.yaml for run_multi from a folder of per-section .h5ad files.

    python scripts/make_manifest.py data/dlpfc --meta data/dlpfc/sections.csv \
        --label-key layer_guess_reordered --xcol spatial_x --ycol spatial_y --gene-col gene_name

sections.csv columns: section_id,donor_id   (file is <section_id>.h5ad inside the folder)
Optional fixes applied in place: coordinates from obs columns -> obsm['spatial']; var_names from a var column.
Prints a summary so you can eyeball spots/labels per section before running anything.
"""
import argparse
import os

import anndata as ad
import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--meta", required=True)
ap.add_argument("--label-key", default=None)
ap.add_argument("--xcol")
ap.add_argument("--ycol")
ap.add_argument("--gene-col", default=None)
a = ap.parse_args()

meta = pd.read_csv(a.meta, dtype=str)
entries = []
for _, r in meta.iterrows():
    path = os.path.join(a.folder, f"{r['section_id']}.h5ad")
    x = ad.read_h5ad(path)
    changed = False
    if a.xcol and a.ycol and "spatial" not in x.obsm:
        x.obsm["spatial"] = x.obs[[a.xcol, a.ycol]].to_numpy(dtype=float)
        changed = True
    if a.gene_col and a.gene_col in x.var:
        x.var_names = x.var[a.gene_col].astype(str).values
        x.var_names_make_unique()
        changed = True
    if changed:
        x.write_h5ad(path)
    assert "spatial" in x.obsm, f"{path}: no obsm['spatial'] (use --xcol/--ycol)"
    lab = ""
    if a.label_key:
        assert a.label_key in x.obs, f"{path}: label column '{a.label_key}' not in obs: {list(x.obs.columns)[:15]}"
        vc = {k: int(v) for k, v in x.obs[a.label_key].value_counts(dropna=False).items()}
        lab = f" labels={vc}"
    print(f"{r['section_id']} donor={r['donor_id']} spots={x.n_obs} genes={x.n_vars}{lab}")
    entries.append({"path": path, "section_id": r["section_id"], "donor_id": r["donor_id"], "label_key": a.label_key})
out = os.path.join(a.folder, "manifest.yaml")
yaml.safe_dump({"sections": entries}, open(out, "w"), sort_keys=False)
print("wrote", out)
