"""Post-hoc analyses on per-spot results from ``run_multi`` (no torch needed -> unit-testable).

Input is a long table ``spot_rows`` with one row per (spot, seed, method):
    spot (global index), group, seed, method, loss, entropy, attn_roll, attn_last
where attn_* is the total CLS attention / rollout mass on the k neighbour tokens (NaN if n/a).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import analyze_outcomes, boundary_class, boundary_score

BOUNDARY_OUTCOMES = ["attn_roll_real", "attn_last_real", "attn_roll_shuffled", "attn_last_shuffled",
                     "attn_excess_roll", "attn_excess_last", "benefit_vs_shuffled", "benefit_vs_none"]


def build_spot_table(spot_rows: pd.DataFrame, comb, min_frac: float = 2 / 6, ambig_method: str = "logreg_own"):
    """One row per test spot with seed-averaged losses/attention and boundary annotations."""
    g = (spot_rows.groupby(["spot", "method"])
         .agg(loss=("loss", "mean"), attn_roll=("attn_roll", "mean"), attn_last=("attn_last", "mean"),
              entropy=("entropy", "mean")).reset_index())
    wide = pd.DataFrame(index=np.sort(g["spot"].unique()))
    wide.index.name = "spot"
    for m, d in g.groupby("method"):
        d = d.set_index("spot")
        wide[f"loss_{m}"] = d["loss"]
        if d["attn_roll"].notna().any():
            tag = m.replace("transformer_", "")
            wide[f"attn_roll_{tag}"] = d["attn_roll"]
            wide[f"attn_last_{tag}"] = d["attn_last"]
        if m == ambig_method:
            wide["ambig"] = d["entropy"]
    idx = wide.index.to_numpy()
    wide["group"] = comb.group[idx]
    wide["section"] = comb.section[idx]
    wide["dist_k"] = comb.dist_k[idx]
    if comb.labels is not None:
        names = np.array(comb.label_names + ["NA"], dtype=object)
        wide["label"] = names[np.where(comb.labels[idx] < 0, len(comb.label_names), comb.labels[idx])]
        bs = boundary_score(comb.labels, comb.nbr_idx)
        wide["bscore"] = bs[idx]
        wide["bclass"] = boundary_class(bs, min_frac)[idx]
    if {"loss_transformer_real", "loss_transformer_shuffled"} <= set(wide):
        wide["benefit_vs_shuffled"] = wide["loss_transformer_shuffled"] - wide["loss_transformer_real"]
    if {"loss_transformer_real", "loss_transformer_none"} <= set(wide):
        wide["benefit_vs_none"] = wide["loss_transformer_none"] - wide["loss_transformer_real"]
    for t in ("roll", "last"):
        if {f"attn_{t}_real", f"attn_{t}_shuffled"} <= set(wide):
            wide[f"attn_excess_{t}"] = wide[f"attn_{t}_real"] - wide[f"attn_{t}_shuffled"]
    return wide.reset_index()


def boundary_report(wide: pd.DataFrame, n_perm: int = 999, seed: int = 0):
    if "bclass" not in wide:
        return pd.DataFrame(), pd.DataFrame()
    outs = [o for o in BOUNDARY_OUTCOMES if o in wide]
    return analyze_outcomes(wide, outs, group_col="group", n_perm=n_perm, seed=seed)


def deletion_report(del_by_group: dict, wide: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deletion losses. ``del_by_group[g] = (spot_idx, {'full': arr, (m,strategy): arr})``."""
    bmap = wide.set_index("spot")["bclass"] if "bclass" in wide else None
    rows = []
    for g, (idx, res) in del_by_group.items():
        for key, arr in res.items():
            if key == "full":
                continue
            m, strat = key
            for i, (s, loss, full) in enumerate(zip(idx, arr, res["full"])):
                rows.append((g, s, m, strat, loss, full))
    df = pd.DataFrame(rows, columns=["group", "spot", "m", "strategy", "loss", "loss_full"])
    if df.empty:
        return df
    df["delta"] = df["loss"] - df["loss_full"]
    df["bclass"] = df["spot"].map(bmap) if bmap is not None else "all"
    out = []
    for subset, d in (("all", df), ("boundary", df[df["bclass"] == "boundary"]),
                      ("interior", df[df["bclass"] == "interior"])):
        if d.empty:
            continue
        t = d.groupby(["m", "strategy"]).agg(n=("delta", "size"), mean_delta_loss=("delta", "mean")).reset_index()
        t.insert(0, "subset", subset)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def deletion_verdict(rep: pd.DataFrame) -> list[str]:
    """Plain-language reading of the deletion table (top vs random vs bottom removal)."""
    lines = []
    for subset in ("all", "boundary"):
        d = rep[rep["subset"] == subset] if len(rep) else rep
        if d is None or d.empty:
            continue
        for m, dm in d.groupby("m"):
            v = dm.set_index("strategy")["mean_delta_loss"]
            if {"top", "random", "bottom"} <= set(v.index):
                ok = v["top"] > v["random"] > v["bottom"]
                lines.append(f"- {subset}, m={m}: removing top-attended raises loss by {v['top']:+.4f}, random "
                             f"{v['random']:+.4f}, bottom {v['bottom']:+.4f} -> "
                             f"{'ordering consistent with faithful attention' if ok else 'ordering NOT consistent with faithful attention'}.")
    return lines


def metrics_tables(res: pd.DataFrame, task: str):
    """summary + paired comparisons with held-out GROUPS as the unit of replication."""
    gm = res.groupby(["group", "method"])["score"].mean().unstack("method")
    summ = pd.DataFrame({"mean": gm.mean(), "sd_over_groups": gm.std(), "n_groups": gm.count()}).round(4)
    stem = "ridge" if task == "program" else "logreg"
    pairs = [("transformer_real", "transformer_shuffled"), ("transformer_real", "transformer_none"),
             ("transformer_real", f"{stem}_own_nbr_mean"), ("transformer_real", f"{stem}_own"),
             (f"{stem}_own_nbr_mean", f"{stem}_own")]
    rows = []
    for a, b in pairs:
        if a in gm and b in gm:
            d = (gm[a] - gm[b]).dropna()
            rows.append({"A": a, "B": b, "mean_diff_A_minus_B": round(d.mean(), 4),
                         "sd_over_groups": round(d.std(), 4) if len(d) > 1 else np.nan,
                         "groups_A_better": f"{(d > 0).sum()}/{len(d)}"})
    return summ, pd.DataFrame(rows)


def write_summary(out_dir: str, cfg: dict, summ, pairs, boundary, per_group, sal, faith, deletion, n_groups: int):
    L = ["# Transformomics multi-section results\n",
         f"Task: **{cfg['task']}**. Replication unit: **{n_groups} held-out {cfg['split']['by']}s** "
         f"(leave-one-{cfg['split']['by']}-out).",
         "With so few groups, treat every p-value as descriptive; look at the per-group consistency columns.\n"]
    if cfg.get("null", {}).get("permute_coords_seed") is not None:
        L.append("> **NULL RUN**: coordinates were permuted within sections. Any 'real vs shuffled' gap here "
                 "is an artefact/noise floor, not signal.\n")
    L += ["## Mean score per method\n", summ.to_markdown(), "\n## Paired comparisons (A - B)\n",
          pairs.to_markdown(index=False) if len(pairs) else "(none)", ""]
    if len(boundary):
        L += ["## Boundary vs interior (adjusted for ambiguity, tissue-edge distance, label, section)\n",
              boundary.round(4).to_markdown(index=False),
              "\nRead: `attn_excess_*` (real minus shuffled attention) and `benefit_vs_*` are the informative rows; "
              "attention in the shuffled model is the null for 'attends to neighbours at boundaries'. "
              "`n_groups_positive` shows whether the effect holds in each held-out group.\n"]
    if len(sal):
        s = (sal.groupby("method").agg(classes=("class", "nunique"), mean_auc=("auc", "mean"),
                                       median_p=("p", "median"), frac_q_lt_0_1=("q_bh", lambda x: (x < 0.1).mean()))
             .round(4).reset_index())
        L += ["## Marker enrichment of attribution scores (rank-AUC; 0.5 = chance)\n", s.to_markdown(index=False),
              "\nCompare `ig_*` against the `logreg_coef` and `de_auc` baselines; attribution only adds value if it "
              "enriches markers MORE than those simple methods. Few markers per class => low power.\n"]
    if len(faith):
        L += ["## Do attended neighbours share the query's true domain? (spots with mixed neighbourhoods)\n",
              faith.round(4).to_markdown(index=False),
              "\nAttention only provides 'insight' if it beats `expr_cosine` (a free, non-learned baseline).\n"]
    if len(deletion):
        L += ["## Deletion test (loss change when hiding m neighbours)\n"] + deletion_verdict(deletion) + [""]
    L += ["## Caveats\n", "- Attention/saliency are descriptive; the causal evidence is the ablation and deletion results.",
          "- Boundary spots are also expression-ambiguous mixtures; the adjusted effect controls for this only partly.",
          "- Run the same pipeline with `--null-coords` to see the artefact floor before trusting any gap."]
    with open(f"{out_dir}/summary.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
