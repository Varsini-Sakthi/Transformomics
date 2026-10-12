"""Experiment driver.

    python -m transformomics.run --config configs/microglia.yaml --out results/
    python -m transformomics.run --synthetic --out results_synth/      # pipeline smoke test

For every outer spatial-block fold x seed it trains/evaluates:
    transformer_real      true spatial neighbors
    transformer_shuffled  random spots as neighbors (control: same capacity, no spatial info)
    transformer_none      neighbors masked out (single-spot transformer)
    ridge_own / logreg_own            baseline on own expression
    ridge_own_nbr_mean / ...          baseline on own + mean neighbor expression
and writes results.csv, summary.csv, paired_comparisons.csv, programme_spatial_stats.csv, summary.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from . import baselines
from .data import prepare
from .graph import knn_graph, make_split, spatial_block_folds
from .programs import load_programs, score_programs, spatial_summary
from .train import Tensors, classification_metrics, fit_predict, regression_metrics

DEFAULTS = {
    "task": "program",                       # 'program' (predict programme scores) | 'label' (classify obs[label_key])
    "data": {"path": None, "label_key": None, "spatial_key": "spatial", "already_log": False},
    "preprocess": {"n_hvg": 2000, "target_sum": 1e4, "min_cells": 3},
    "programs_file": None,
    "mask_program_genes": True,              # remove programme genes from model input (avoid trivial readout)
    "restrict": None,                        # e.g. {markers: [P2ry12, Tmem119], quantile: 0.5}
    "neighbors": {"k": 6},
    "model": {"embed_dim": 64, "heads": 4, "layers": 2, "ff_dim": 128, "dropout": 0.1, "dist_bias": True},
    "train": {"epochs": 60, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-2, "patience": 8, "rot_aug": True},
    "cv": {"n_blocks": 24, "n_folds": 4, "seeds": [0, 1, 2], "buffer": True},
    "variants": ["real", "shuffled", "none"],
    "baselines": ["own", "own_nbr_mean"],
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def run(cfg: dict, adata, programs: dict | None, out_dir: str, verbose: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    task = cfg["task"]
    k = cfg["neighbors"]["k"]
    cv = cfg["cv"]

    # ---- data + targets
    excl = [g for gl in (programs or {}).values() for g in gl] if cfg["mask_program_genes"] and programs else []
    prep = prepare(adata, n_hvg=cfg["preprocess"]["n_hvg"], target_sum=cfg["preprocess"]["target_sum"],
                   min_cells=cfg["preprocess"]["min_cells"], already_log=cfg["data"]["already_log"],
                   exclude_genes=excl, spatial_key=cfg["data"]["spatial_key"])
    nbr_idx, _, offsets = knn_graph(prep.coords, k)

    names, prog_scores, counts = None, None, {}
    if programs:
        prog_scores, names, counts = score_programs(prep.logX_all, prep.genes_all, programs)
        pd.DataFrame([{"programme": n, "genes_detected": counts[n][0], "genes_listed": counts[n][1]}
                      for n in counts]).to_csv(f"{out_dir}/programme_gene_coverage.csv", index=False)
        pd.DataFrame(spatial_summary(prog_scores, names, nbr_idx)).to_csv(
            f"{out_dir}/programme_spatial_stats.csv", index=False)

    query_mask = None
    if cfg.get("restrict"):
        r = cfg["restrict"]
        ms, _, _ = score_programs(prep.logX_all, prep.genes_all, {"m": r["markers"]}, min_genes=1)
        query_mask = ms[:, 0] >= np.quantile(ms[:, 0], r.get("quantile", 0.5))

    if task == "program":
        if prog_scores is None:
            raise ValueError("task=program needs programmes (programs_file or inline).")
        y = prog_scores
        n_classes = None
        y_t = torch.tensor((y - y.mean(0)) / np.where(y.std(0) == 0, 1, y.std(0)), dtype=torch.float32)
    else:
        lab = adata.obs[cfg["data"]["label_key"]].astype("category")
        y = lab.cat.codes.values.astype(int)
        names, n_classes = list(lab.cat.categories), len(lab.cat.categories)
        y_t = torch.tensor(y, dtype=torch.long)

    T = Tensors(torch.tensor(prep.X), nbr_idx, torch.tensor(offsets), y_t)
    folds = spatial_block_folds(prep.coords, cv["n_blocks"], cv["n_folds"], seed=0)

    rows = []
    for f in range(cv["n_folds"]):
        tr, va, te = make_split(folds, nbr_idx, f, cv["n_folds"], cv["buffer"], query_mask)
        if verbose:
            print(f"[fold {f}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        # per-fold standardisation of regression targets uses training stats only
        if task == "program":
            mu, sd = y[tr].mean(0), np.where(y[tr].std(0) == 0, 1, y[tr].std(0))
            T.y = torch.tensor((y - mu) / sd, dtype=torch.float32)
            y_true = (y[te] - mu) / sd
        else:
            y_true = y[te]

        def score(pred):
            return regression_metrics(y_true, pred, names) if task == "program" else classification_metrics(y_true, pred)

        for kind in cfg["baselines"]:
            pred = baselines.fit_predict_baseline(prep.X, nbr_idx, y if task == "label" else (y - mu) / sd,
                                                  tr, te, task, kind)
            m = score(pred)
            rows.append({"fold": f, "seed": -1, "method": f"{'ridge' if task == 'program' else 'logreg'}_{kind}", **m})
        for seed in cv["seeds"]:
            for mode in cfg["variants"]:
                pred, ex = fit_predict(T, tr, va, te, task, cfg["model"], cfg["train"], seed, mode,
                                       n_classes=n_classes, want_attn=(mode == "real"))
                m = score(pred)
                rows.append({"fold": f, "seed": seed, "method": f"transformer_{mode}", **m, **ex})
                if verbose:
                    print(f"  seed {seed} {mode:9s} score={m['score']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{out_dir}/results.csv", index=False)
    summarize(res, out_dir, task, cfg)
    return res


def summarize(res: pd.DataFrame, out_dir: str, task: str, cfg: dict):
    # one number per (fold, method): average over seeds, so folds are the unit of replication
    fm = res.groupby(["fold", "method"])["score"].mean().unstack("method")
    summ = pd.DataFrame({"mean": fm.mean(), "sd_over_folds": fm.std(), "n_folds": fm.count()}).round(4)
    summ.to_csv(f"{out_dir}/summary.csv")

    pairs = [("transformer_real", "transformer_shuffled"), ("transformer_real", "transformer_none")]
    stem = "ridge" if task == "program" else "logreg"
    pairs += [("transformer_real", f"{stem}_own_nbr_mean"), ("transformer_real", f"{stem}_own"),
              (f"{stem}_own_nbr_mean", f"{stem}_own")]
    prow = []
    for a, b in pairs:
        if a in fm and b in fm:
            d = (fm[a] - fm[b]).dropna()
            prow.append({"A": a, "B": b, "mean_diff_A_minus_B": round(d.mean(), 4),
                         "sd_over_folds": round(d.std(), 4), "folds_A_better": f"{(d > 0).sum()}/{len(d)}"})
    pc = pd.DataFrame(prow)
    pc.to_csv(f"{out_dir}/paired_comparisons.csv", index=False)

    metric = "mean Pearson r across programmes" if task == "program" else "macro-F1"
    lines = [f"# Transformomics v2 results\n", f"Primary metric: {metric}; units of replication = "
             f"{fm.shape[0]} spatial-block folds (seeds averaged within fold).\n",
             "## Mean score per method\n", summ.to_markdown(), "\n## Paired comparisons (A - B)\n",
             pc.to_markdown(index=False) if len(pc) else "(none)", "\n## Reading this honestly\n"]
    d = {(r.A, r.B): r for r in pc.itertuples()} if len(pc) else {}
    r_sh = d.get(("transformer_real", "transformer_shuffled"))
    if r_sh is not None:
        lines.append(f"- Real vs shuffled neighbors: diff = {r_sh.mean_diff_A_minus_B:+.4f} "
                     f"({r_sh.folds_A_better} folds favour real). Only a positive, consistent gap "
                     f"supports 'the model uses spatial context'.")
    r_nm = d.get(("transformer_real", f"{stem}_own_nbr_mean"))
    if r_nm is not None:
        lines.append(f"- Transformer vs linear own+neighbor-mean baseline: diff = {r_nm.mean_diff_A_minus_B:+.4f} "
                     f"({r_nm.folds_A_better} folds favour transformer). If this is <= 0, simple "
                     f"neighborhood smoothing explains the benefit and the attention model is not needed.")
    lines.append("- Attention weights are descriptive, not proof of importance; rely on the ablations above.")
    lines.append(f"- With {fm.shape[0]} folds from one section, differences are indicative only; confirm "
                 f"on held-out sections/donors before drawing biological conclusions.")
    if "nbr_attn_mass" in res:
        a = res.loc[res.method == "transformer_real", "nbr_attn_mass"].mean()
        lines.append(f"- Mean CLS attention mass on neighbor tokens (last layer): {a:.3f} "
                     f"(uniform attention over all {cfg['neighbors']['k'] + 2} tokens would give "
                     f"~{cfg['neighbors']['k'] / (cfg['neighbors']['k'] + 2):.2f}; compare, don't over-read).")
    with open(f"{out_dir}/summary.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(f"{out_dir}/config_used.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data (smoke test)")
    ap.add_argument("--quick", action="store_true", help="fewer seeds/epochs")
    a = ap.parse_args()

    user = yaml.safe_load(open(a.config)) if a.config else {}
    cfg = _merge(DEFAULTS, user)
    if a.quick:
        cfg["cv"]["seeds"], cfg["train"]["epochs"] = [0], 25

    if a.synthetic:
        from .synthetic import make_synthetic
        adata, programs = make_synthetic()
    else:
        import anndata as ad
        adata = ad.read_h5ad(cfg["data"]["path"])
        programs = load_programs(cfg["programs_file"]) if cfg.get("programs_file") else None
    run(cfg, adata, programs, a.out)
    print(open(f"{a.out}/summary.md").read())


if __name__ == "__main__":
    main()
