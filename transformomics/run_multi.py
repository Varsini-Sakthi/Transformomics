"""Multi-section experiment driver (real Visium data; leave-one-donor/section-out).

    python -m transformomics.run_multi --config configs/dlpfc.yaml --out results_dlpfc/
    python -m transformomics.run_multi --synthetic --quick --out results_synth_multi/   # plumbing test
    python -m transformomics.run_multi --config configs/dlpfc.yaml --null-coords 1 --out results_null/

For each held-out group it trains transformer_real / _shuffled / _none (x seeds) and the linear
baselines, stores PER-SPOT losses and attention, then runs the pre-specified analyses:
boundary vs interior, marker enrichment of attribution, attention-vs-baseline faithfulness and the
deletion test. See ``analysis_multi.py`` / ``boundary.py`` for exactly what is computed.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import log_softmax

from . import baselines
from .analysis_multi import (boundary_report, build_spot_table, deletion_report, metrics_tables,
                             write_summary)
from .boundary import boundary_score
from .faithfulness import attention_vs_baselines, summarize_vs_baselines
from .interpret import class_attributions, deletion_test, spot_attention
from .marker_enrichment import de_auc_scores, enrichment_table, logreg_scores
from .multisection import SectionSpec, group_splits, load_sections, prepare_inputs, score_programs_by_section
from .programs import load_programs
from .run import _merge
from .train import Tensors, classification_metrics, fit_predict, regression_metrics

DEFAULTS = {
    "task": "label",                          # 'label' (classify obs[label_key]) | 'program' (regress programme scores)
    "sections": [],                           # list of {path, section_id, donor_id, label_key}
    "manifest": None,                         # or a YAML file containing {sections: [...]}
    "markers_file": None,                     # YAML {markers: {class_name: [genes]}} for enrichment
    "programs_file": None,
    "mask_program_genes": True,
    "restrict": None,                         # e.g. {markers: [P2ry12, Tmem119], quantile: 0.5}; applied PER SECTION
    "data": {"spatial_key": "spatial", "already_log": False},
    "preprocess": {"n_hvg": 2000, "target_sum": 1e4, "min_cells": 3},
    "neighbors": {"k": 6},
    "split": {"by": "donor", "val_mode": "auto", "n_blocks": 24, "n_folds": 4},
    "model": {"embed_dim": 64, "heads": 4, "layers": 2, "ff_dim": 128, "dropout": 0.1, "dist_bias": True},
    "train": {"epochs": 60, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-2, "patience": 8, "rot_aug": True},
    "seeds": [0, 1, 2],
    "variants": ["real", "shuffled", "none"],
    "baselines": ["own", "own_nbr_mean"],
    "analyses": {
        "boundary": {"enabled": True, "min_frac": 2 / 6, "n_perm": 999},
        "saliency": {"enabled": True, "max_spots_per_class": 200, "ig_steps": 16, "n_null": 1000, "min_spots": 30},
        "faithfulness": {"enabled": True, "deletion_m": [1, 2, 3]},
    },
    "null": {"permute_coords_seed": None},
}


# ------------------------------------------------------------------ per-spot helpers
def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def _spot_df(g, idx, seed, method, pred, y_true, task, proba=False, ex=None):
    if task == "label":
        if proba:
            loss = -np.log(np.clip(pred[np.arange(len(y_true)), y_true], 1e-12, 1.0))
            ent = _entropy(pred)
        else:
            loss = -log_softmax(pred, axis=1)[np.arange(len(y_true)), y_true]
            ent = np.full(len(idx), np.nan)
    else:
        loss = ((pred - y_true) ** 2).mean(1)
        ent = np.full(len(idx), np.nan)
    ar = al = np.full(len(idx), np.nan)
    if ex is not None and "attn_rollout" in ex:
        ar, al = ex["attn_rollout"][:, 2:].sum(1), ex["attn_last"][:, 2:].sum(1)
    return pd.DataFrame({"group": g, "spot": idx, "seed": seed, "method": method, "loss": loss,
                         "entropy": ent, "attn_roll": ar, "attn_last": al})


def run_multi(cfg: dict, specs, programs: dict | None, markers: dict | None, out_dir: str, verbose: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    task, k = cfg["task"], cfg["neighbors"]["k"]
    an = cfg["analyses"]
    comb = load_sections(specs, k=k, target_sum=cfg["preprocess"]["target_sum"],
                         min_cells=cfg["preprocess"]["min_cells"], already_log=cfg["data"]["already_log"],
                         spatial_key=cfg["data"]["spatial_key"], split_by=cfg["split"]["by"],
                         permute_coords_seed=cfg["null"]["permute_coords_seed"])
    N = comb.logX.shape[0]
    excl = [g for gl in (programs or {}).values() for g in gl] if cfg["mask_program_genes"] and programs else []

    if task == "program":
        if not programs:
            raise ValueError("task=program needs programs_file")
        y_all, names = score_programs_by_section(comb, programs)
        n_classes, query_mask = None, None
    else:
        if comb.labels is None:
            raise ValueError("task=label needs label_key on every section")
        names, n_classes = comb.label_names, len(comb.label_names)
        y_all, query_mask = comb.labels, comb.labels >= 0

    if cfg.get("restrict"):                   # query spots = top quantile of a marker score within each section
        r = cfg["restrict"]
        ms, _ = score_programs_by_section(comb, {"m": r["markers"]}, min_genes=1)
        qm = np.zeros(N, dtype=bool)
        for sec in np.unique(comb.section):
            mi = np.where(comb.section == sec)[0]
            qm[mi] = ms[mi, 0] >= np.quantile(ms[mi, 0], r.get("quantile", 0.5))
        query_mask = qm if query_mask is None else (query_mask & qm)

    bscore_all = boundary_score(comb.labels, comb.nbr_idx) if comb.labels is not None else None
    splits = list(group_splits(comb, cfg["split"]["val_mode"], cfg["split"]["n_blocks"], cfg["split"]["n_folds"],
                               seed=0, query_mask=query_mask))
    spot_parts, metric_rows, sal_tabs, faith_parts, del_by_group = [], [], [], [], {}

    for sp_ in splits:
        g, tr, va, te = sp_["test_group"], sp_["tr"], sp_["va"], sp_["te"]
        if len(te) == 0 or len(tr) < 20 or len(va) == 0:
            warnings.warn(f"skipping group {g}: train={len(tr)} val={len(va)} test={len(te)}")
            continue
        if verbose:
            print(f"[test {cfg['split']['by']} {g}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        tmask = np.zeros(N, dtype=bool)
        tmask[tr] = True
        inp = prepare_inputs(comb, tmask, cfg["preprocess"]["n_hvg"], excl)

        if task == "program":
            mu, sd = y_all[tr].mean(0), np.where(y_all[tr].std(0) == 0, 1, y_all[tr].std(0))
            y_used = ((y_all - mu) / sd).astype(np.float32)
            y_t, y_true = torch.tensor(y_used), y_used[te]
        else:
            y_used, y_t, y_true = comb.labels, torch.tensor(comb.labels, dtype=torch.long), comb.labels[te]
        T = Tensors(torch.from_numpy(inp.X), comb.nbr_idx, torch.from_numpy(comb.offsets), y_t)

        def score(pred):
            return regression_metrics(y_true, pred, names) if task == "program" else classification_metrics(y_true, pred)

        for kind in cfg["baselines"]:
            pred = baselines.fit_predict_baseline(inp.X, comb.nbr_idx, y_used, tr, te, task, kind)
            name = f"{'ridge' if task == 'program' else 'logreg'}_{kind}"
            metric_rows.append({"group": g, "seed": -1, "method": name, **score(pred)})
            spot_parts.append(_spot_df(g, te, -1, name, pred, y_true, task, proba=(task == "label")))

        kept, att_real = None, []
        for seed in cfg["seeds"]:
            for mode in cfg["variants"]:
                first = (mode == "real" and kept is None)
                pred, ex = fit_predict(T, tr, va, te, task, cfg["model"], cfg["train"], seed, mode,
                                       n_classes=n_classes, want_attn=(mode != "none"),
                                       want_spot_attn=(mode != "none"), return_model=first)
                m = score(pred)
                metric_rows.append({"group": g, "seed": seed, "method": f"transformer_{mode}", **m,
                                    "nbr_attn_mass": ex.get("nbr_attn_mass", np.nan)})
                spot_parts.append(_spot_df(g, te, seed, f"transformer_{mode}", pred, y_true, task, ex=ex))
                if mode == "real":
                    att_real.append(ex["attn_rollout"][:, 2:])
                if first:
                    kept = ex["model"]
                if verbose:
                    print(f"  seed {seed} {mode:9s} score={m['score']:.3f}", flush=True)

        if kept is None:
            continue
        if an["faithfulness"]["enabled"] and comb.labels is not None:
            att_mean = np.mean(att_real, axis=0)
            fdf = attention_vs_baselines(att_mean, te, comb.labels, comb.nbr_idx, comb.nbr_dist, inp.X,
                                         comb.group, bscore_all)
            fdf["test_group"] = g
            faith_parts.append(fdf)
            att_one = spot_attention(kept, T, comb.nbr_idx, te)["rollout"][:, 2:]
            del_by_group[g] = (te, deletion_test(kept, T, comb.nbr_idx, te, att_one, task,
                                                 ms=an["faithfulness"]["deletion_m"]))
        if an["saliency"]["enabled"] and task == "label" and markers:
            s = an["saliency"]
            cl = comb.labels[te]
            idx_by_class = {c: te[cl == c] for c in range(n_classes) if (cl == c).sum() >= s["min_spots"]}
            if idx_by_class:
                Q, Nn, _ = class_attributions(kept, T, comb.nbr_idx, idx_by_class, n_classes,
                                              max_spots=s["max_spots_per_class"], steps=s["ig_steps"])
                ytr = comb.labels[tr]
                mats = {"ig_query": Q, "ig_neighbours": Nn, "ig_total": Q + Nn,
                        "logreg_coef": logreg_scores(inp.X[tr], ytr, n_classes),
                        "de_auc": de_auc_scores(inp.X[tr], ytr, n_classes)}
                tab = enrichment_table(mats, names, inp.genes_in, markers, inp.mean_expr, n_null=s["n_null"])
                ok = set(tab[(tab["method"] == "ig_query") & tab["auc"].notna()]["class"])
                tab = tab[tab["class"].isin(ok)].copy()          # same classes for every method
                tab.insert(0, "test_group", g)
                sal_tabs.append(tab)

    # ---------------------------------------------------------------- outputs
    res = pd.DataFrame(metric_rows)
    res.to_csv(f"{out_dir}/results.csv", index=False)
    spot_rows = pd.concat(spot_parts, ignore_index=True)
    wide = build_spot_table(spot_rows, comb, min_frac=an["boundary"]["min_frac"])
    wide.to_csv(f"{out_dir}/per_spot_table.csv.gz", index=False)
    summ, pairs = metrics_tables(res, task)
    summ.to_csv(f"{out_dir}/summary.csv")
    pairs.to_csv(f"{out_dir}/paired_comparisons.csv", index=False)

    bsum = bpg = pd.DataFrame()
    if an["boundary"]["enabled"]:
        bsum, bpg = boundary_report(wide, an["boundary"]["n_perm"])
        bsum.to_csv(f"{out_dir}/boundary_summary.csv", index=False)
        bpg.to_csv(f"{out_dir}/boundary_by_group.csv", index=False)
    sal = pd.concat(sal_tabs, ignore_index=True) if sal_tabs else pd.DataFrame()
    if len(sal):
        sal.to_csv(f"{out_dir}/marker_enrichment.csv", index=False)
    faith = pd.DataFrame()
    if faith_parts:
        fall = pd.concat(faith_parts, ignore_index=True)
        fall.to_csv(f"{out_dir}/faithfulness_per_spot.csv.gz", index=False)
        faith = summarize_vs_baselines(fall)
        faith.to_csv(f"{out_dir}/faithfulness_summary.csv", index=False)
    dele = deletion_report(del_by_group, wide) if del_by_group else pd.DataFrame()
    if len(dele):
        dele.to_csv(f"{out_dir}/deletion_summary.csv", index=False)
    write_summary(out_dir, cfg, summ, pairs, bsum, bpg, sal, faith, dele, n_groups=res["group"].nunique())
    json.dump(cfg, open(f"{out_dir}/config_used.json", "w"), indent=2, default=str)
    return res, wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default="results_multi")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic multi-section data (plumbing test)")
    ap.add_argument("--quick", action="store_true", help="1 seed, 25 epochs, fewer permutations")
    ap.add_argument("--null-coords", type=int, default=None, metavar="SEED",
                    help="permute coordinates within sections (artefact/noise-floor run)")
    a = ap.parse_args()
    cfg = _merge(DEFAULTS, yaml.safe_load(open(a.config)) if a.config else {})
    if a.quick:
        cfg["seeds"], cfg["train"]["epochs"] = [0], 25
        cfg["analyses"]["boundary"]["n_perm"] = 199
        cfg["analyses"]["saliency"]["n_null"] = 200
    if a.null_coords is not None:
        cfg["null"]["permute_coords_seed"] = a.null_coords

    programs = load_programs(cfg["programs_file"]) if cfg["programs_file"] else None
    markers = yaml.safe_load(open(cfg["markers_file"])).get("markers") if cfg["markers_file"] else None
    if a.synthetic:
        from .synthetic_sections import make_synthetic_sections
        specs, markers = make_synthetic_sections()
    else:
        sections = cfg["sections"] or (yaml.safe_load(open(cfg["manifest"]))["sections"] if cfg["manifest"] else [])
        if not sections:
            raise SystemExit("no sections: set `sections:` or `manifest:` in the config")
        specs = [SectionSpec(**s) for s in sections]
    run_multi(cfg, specs, programs, markers, a.out)
    print(f"done -> {a.out}/summary.md")


if __name__ == "__main__":
    main()
