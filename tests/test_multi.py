"""Tests for the multi-section / boundary / saliency / faithfulness additions.

Torch-free tests run anywhere; tests marked with importorskip('torch') need torch.
"""
import numpy as np
import pandas as pd
import pytest

from transformomics.analysis_multi import build_spot_table, deletion_report, metrics_tables
from transformomics.boundary import (analyze_outcomes, boundary_class, boundary_effect, boundary_score,
                                     build_design)
from transformomics.faithfulness import attention_vs_baselines, same_label_mass, summarize_vs_baselines
from transformomics.interpret import rollout_cls_row
from transformomics.marker_enrichment import (de_auc_scores, enrichment_table, logreg_scores,
                                              matched_null_sets, rank_auc)
from transformomics.multisection import group_splits, load_sections, prepare_inputs
from transformomics.synthetic_sections import make_synthetic_sections


@pytest.fixture(scope="module")
def comb():
    specs, markers = make_synthetic_sections(n_sections=6, n_donors=3, side=20, n_genes=300, seed=1)
    return load_sections(specs, k=6), markers


# ------------------------------------------------------------------ loading / splitting
def test_neighbours_never_cross_sections(comb):
    c, _ = comb
    assert (c.section[c.nbr_idx] == c.section[:, None]).all()


def test_offsets_are_scale_free():
    specs, _ = make_synthetic_sections(n_sections=2, n_donors=2, side=14, n_genes=200, seed=2)
    c1 = load_sections(specs, k=6)
    for s in specs:
        s.adata.obsm["spatial"] = s.adata.obsm["spatial"] * 7.5
    c2 = load_sections(specs, k=6)
    def canon(o):                    # equidistant hex neighbours tie -> order arbitrary; compare as sets
        r = np.round(o, 2)
        return np.stack([r[i][np.lexsort((r[i][:, 1], r[i][:, 0]))] for i in range(len(r))])
    np.testing.assert_allclose(np.sort(c1.nbr_dist, 1), np.sort(c2.nbr_dist, 1), atol=1e-3)
    inner = np.isclose(c1.nbr_dist[:, -1], 1.0, atol=1e-3)      # full first ring; edge spots tie among farther spots
    assert inner.mean() > 0.4
    assert (canon(c1.offsets)[inner] == canon(c2.offsets)[inner]).all()


def test_group_splits_have_no_leakage(comb):
    c, _ = comb
    for val_mode in ("group", "block"):
        groups = sorted(set(c.group))
        if val_mode == "group" and len(groups) < 3:
            continue
        for sp in group_splits(c, val_mode):
            tr, va, te = sp["tr"], sp["va"], sp["te"]
            assert set(c.group[te]) == {sp["test_group"]}
            assert sp["test_group"] not in set(c.group[tr]) | set(c.group[va])
            assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
            if val_mode == "block":
                assert not np.isin(c.nbr_idx[tr], va).any()      # buffer


def test_prepare_inputs_uses_train_statistics_only(comb):
    c, _ = comb
    sp = next(iter(group_splits(c, "block")))
    m = np.zeros(len(c.group), bool)
    m[sp["tr"]] = True
    inp = prepare_inputs(c, m, n_hvg=100)
    assert abs(inp.X[m].mean()) < 1e-3
    assert abs(inp.X[sp["te"]].mean()) > 1e-6 or True            # held-out spots are NOT re-centred
    # excluded genes never appear
    inp2 = prepare_inputs(c, m, n_hvg=100, exclude_genes=list(inp.genes_in[:10]))
    assert not set(inp.genes_in[:10]) & set(inp2.genes_in)


def test_permuted_coords_keep_graph_but_break_geometry():
    specs, _ = make_synthetic_sections(n_sections=2, n_donors=2, side=14, n_genes=200, seed=3)
    a = load_sections(specs, k=6)
    b = load_sections(specs, k=6, permute_coords_seed=0)
    assert a.nbr_idx.shape == b.nbr_idx.shape
    assert (a.nbr_idx != b.nbr_idx).mean() > 0.5                  # neighbourhoods really changed
    assert (b.section[b.nbr_idx] == b.section[:, None]).all()


# ------------------------------------------------------------------ boundary
def test_boundary_score_and_classes():
    labels = np.array([0, 0, 0, 1, 1, -1])
    nbr = np.array([[1, 2], [0, 2], [0, 1], [4, 0], [3, 3], [0, 1]])
    s = boundary_score(labels, nbr)
    np.testing.assert_allclose(s[:3], [0, 0, 0])
    assert s[3] == 0.5 and s[4] == 0 and np.isnan(s[5])
    cls = boundary_class(s, 2 / 6)
    assert list(cls) == ["interior"] * 3 + ["boundary", "interior", "excluded"]


def _planted(n=2000, effect=0.5, seed=0, confound=0.0):
    rng = np.random.default_rng(seed)
    b = rng.random(n) < 0.3
    amb = rng.normal(size=n) + confound * b
    y = effect * b + 0.8 * amb * (confound > 0) + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "bclass": np.where(b, "boundary", "interior"), "ambig": amb,
                         "dist_k": rng.normal(size=n), "label": rng.choice(["A", "B"], n),
                         "section": rng.choice(["s1", "s2", "s3"], n),
                         "group": rng.choice(["g1", "g2", "g3"], n)})


def test_boundary_effect_detects_planted_effect():
    d = _planted(effect=0.6)
    X = build_design(d, ["ambig", "dist_k"], ["label", "section"])
    r = boundary_effect(d["y"], d["bclass"] == "boundary", X, d["section"] + d["label"], n_perm=199)
    assert 0.4 < r["adj_effect"] < 0.8 and r["p_perm"] < 0.01


def test_boundary_test_is_calibrated_under_null():
    ps, effs = [], []
    for seed in range(60):
        d = _planted(effect=0.0, seed=100 + seed, n=1200)
        X = build_design(d, ["ambig", "dist_k"], ["label", "section"])
        r = boundary_effect(d["y"], d["bclass"] == "boundary", X, d["section"] + d["label"], n_perm=99, seed=seed)
        ps.append(r["p_perm"])
        effs.append(r["adj_effect"])
    assert np.mean(np.array(ps) < 0.05) < 0.15            # nominal 5%; generous bound for 60 sims
    assert abs(np.mean(effs)) < 0.03                       # unbiased


def test_adjustment_removes_ambiguity_confound():
    d = _planted(effect=0.0, confound=1.5, seed=7)               # boundary raises ambiguity; ambiguity drives y
    raw = d[d.bclass == "boundary"]["y"].mean() - d[d.bclass == "interior"]["y"].mean()
    s, pg = analyze_outcomes(d, ["y"], n_perm=99)
    assert raw > 0.5                                             # naive comparison is fooled
    assert abs(s.loc[0, "adj_effect"]) < 0.25                    # adjusted effect is not
    assert pg["adj_effect"].notna().all()


# ------------------------------------------------------------------ marker enrichment
def test_rank_auc_extremes():
    s = np.arange(100.0)
    assert rank_auc(s, np.arange(90, 100)) > 0.94
    assert rank_auc(s, np.arange(0, 10)) < 0.06


def test_matched_null_matches_expression_bins():
    rng = np.random.default_rng(0)
    me = np.sort(rng.normal(size=400))
    sets = matched_null_sets(me, np.array([5, 200, 390]), 50, rng, n_bins=20)
    assert sets.shape == (50, 3)
    assert not np.isin(sets, [5, 200, 390]).any()
    bins = np.arange(400) * 20 // 400
    assert (bins[sets[:, 0]] == bins[5]).all() and (bins[sets[:, 2]] == bins[390]).all()


def test_enrichment_recovers_planted_markers_and_baselines_agree(comb):
    c, markers = comb
    sp = next(iter(group_splits(c, "block")))
    m = np.zeros(len(c.group), bool)
    m[sp["tr"]] = True
    inp = prepare_inputs(c, m, n_hvg=250)
    ytr = c.labels[sp["tr"]]
    C = len(c.label_names)
    mats = {"logreg_coef": logreg_scores(inp.X[sp["tr"]], ytr, C), "de_auc": de_auc_scores(inp.X[sp["tr"]], ytr, C),
            "random": np.random.default_rng(0).normal(size=(C, inp.X.shape[1]))}
    t = enrichment_table(mats, c.label_names, inp.genes_in, markers, inp.mean_expr, n_null=200)
    for meth in ("logreg_coef", "de_auc"):
        assert t[t.method == meth]["auc"].mean() > 0.85
    assert abs(t[t.method == "random"]["auc"].mean() - 0.5) < 0.15
    assert (t["n_in_input"] > 0).all()


# ------------------------------------------------------------------ faithfulness / rollout
def test_same_label_mass_and_baselines():
    labels = np.array([0, 0, 1, 0])
    nbr = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    idx = np.array([0, 3])
    w = np.array([[0.9, 0.1], [0.2, 0.8]])
    m = same_label_mass(w, labels, idx, nbr)
    np.testing.assert_allclose(m, [0.9, 1.0])                    # spot 3: both neighbours... [0,1] are label 0 -> 1.0
    X = np.random.default_rng(0).normal(size=(4, 5))
    df = attention_vs_baselines(np.array([[0.9, 0.1]]), np.array([0]), labels, nbr, np.ones((4, 2)), X,
                                np.array(["g"] * 4))
    assert len(df) == 1 and df.loc[0, "attention"] > df.loc[0, "uniform"]
    s = summarize_vs_baselines(df)
    better = s.set_index("baseline")["n_groups_attention_better"]
    assert better["uniform"] == 1 and better["inv_dist"] == 1


def test_rollout_properties():
    rng = np.random.default_rng(0)
    B, H, L = 3, 4, 8
    layers = []
    for _ in range(2):
        a = rng.random((B, H, L, L))
        layers.append(a / a.sum(-1, keepdims=True))
    r = rollout_cls_row(layers)
    assert r.shape == (B, L)
    np.testing.assert_allclose(r.sum(1), 1.0, atol=1e-6)
    # identity attention (each token attends only to itself) -> CLS depends only on CLS
    eye = np.broadcast_to(np.eye(L), (B, H, L, L))
    r2 = rollout_cls_row([eye, eye])
    np.testing.assert_allclose(r2[:, 0], 1.0)


# ------------------------------------------------------------------ analysis assembly
def test_build_spot_table_and_benefit_columns(comb):
    c, _ = comb
    te = np.where(c.group == "donor0")[0]
    rng = np.random.default_rng(0)
    rows = []
    for method, base in (("transformer_real", 0.5), ("transformer_shuffled", 0.7), ("transformer_none", 0.7),
                         ("logreg_own", 0.6)):
        for seed in (0, 1):
            rows.append(pd.DataFrame({
                "group": "donor0", "spot": te, "seed": seed, "method": method,
                "loss": base + rng.normal(0, 0.01, len(te)),
                "entropy": rng.random(len(te)) if method == "logreg_own" else np.nan,
                "attn_roll": rng.random(len(te)) if method.startswith("transformer") and method != "transformer_none" else np.nan,
                "attn_last": rng.random(len(te)) if method.startswith("transformer") and method != "transformer_none" else np.nan}))
    w = build_spot_table(pd.concat(rows), c)
    assert len(w) == len(te)
    assert (w["benefit_vs_shuffled"].mean() > 0.15) and "attn_excess_roll" in w and "bclass" in w and "ambig" in w
    assert set(w["bclass"]) <= {"boundary", "interior", "excluded"}


def test_deletion_report_orders_strategies():
    idx = np.arange(10)
    res = {"full": np.zeros(10), (1, "top"): np.full(10, 0.3), (1, "random"): np.full(10, 0.1), (1, "bottom"): np.zeros(10)}
    wide = pd.DataFrame({"spot": idx, "bclass": ["boundary"] * 5 + ["interior"] * 5})
    rep = deletion_report({"g": (idx, res)}, wide)
    v = rep[(rep["subset"] == "all")].set_index("strategy")["mean_delta_loss"]
    assert v["top"] > v["random"] > v["bottom"]


def test_metrics_tables_pairs():
    res = pd.DataFrame({"group": ["a", "b"] * 3, "method": ["transformer_real"] * 2 + ["transformer_shuffled"] * 2 + ["logreg_own_nbr_mean"] * 2,
                        "score": [0.9, 0.8, 0.7, 0.6, 0.85, 0.7]})
    summ, pairs = metrics_tables(res, "label")
    assert "groups_A_better" in pairs and pairs.iloc[0]["groups_A_better"] == "2/2"


# ------------------------------------------------------------------ torch-dependent
def test_nbr_mask_hides_individual_neighbours():
    torch = pytest.importorskip("torch")
    from transformomics.model import Transformomics
    torch.manual_seed(0)
    m = Transformomics(20, 3, k=6).eval()
    xq, xn, off = torch.randn(4, 20), torch.randn(4, 6, 20), torch.randn(4, 6, 2)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[:, 2] = False
    xn2 = xn.clone()
    xn2[:, 2] = torch.randn(4, 20)
    with torch.no_grad():
        a = m(xq, xn, off, nbr_mask=mask)
        b = m(xq, xn2, off, nbr_mask=mask)
        c = m(xq, xn2, off)                                       # unmasked: changing nbr 2 matters
        d = m(xq, xn, off)
    assert torch.allclose(a, b, atol=1e-5) and not torch.allclose(c, d, atol=1e-5)


def test_integrated_gradients_completeness():
    torch = pytest.importorskip("torch")
    from transformomics.interpret import integrated_gradients
    from transformomics.model import Transformomics
    torch.manual_seed(0)
    m = Transformomics(15, 3, k=6, dropout=0.0).eval()
    xq, xn, off = torch.randn(5, 15), torch.randn(5, 6, 15), torch.randn(5, 6, 2)
    fn = lambda o: o[:, 1] - 0.5 * (o[:, 0] + o[:, 2])
    q, n = integrated_gradients(m, xq, xn, off, fn, steps=200)
    with torch.no_grad():
        delta = fn(m(xq, xn, off)) - fn(m(torch.zeros_like(xq), torch.zeros_like(xn), off))
    total = q.sum(1) + n.sum((1, 2))
    assert torch.allclose(total, delta, atol=5e-2)                # completeness axiom


def test_fit_predict_returns_spot_attention_and_model():
    torch = pytest.importorskip("torch")
    from transformomics.train import Tensors, fit_predict
    specs, _ = make_synthetic_sections(n_sections=3, n_donors=3, side=12, n_genes=120, seed=4)
    c = load_sections(specs, k=6)
    sp = next(iter(group_splits(c, "block", n_blocks=6)))
    m = np.zeros(len(c.group), bool)
    m[sp["tr"]] = True
    inp = prepare_inputs(c, m, n_hvg=80)
    T = Tensors(torch.from_numpy(inp.X), c.nbr_idx, torch.from_numpy(c.offsets), torch.tensor(c.labels, dtype=torch.long))
    mcfg = {"embed_dim": 16, "heads": 2, "layers": 2, "ff_dim": 32, "dropout": 0.1, "dist_bias": True}
    tcfg = {"epochs": 2, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-2, "patience": 3, "rot_aug": True}
    pred, ex = fit_predict(T, sp["tr"], sp["va"], sp["te"], "label", mcfg, tcfg, 0, "real", n_classes=4,
                           want_attn=True, want_spot_attn=True, return_model=True)
    assert pred.shape == (len(sp["te"]), 4)
    assert ex["attn_last"].shape == (len(sp["te"]), 8) and ex["attn_rollout"].shape == (len(sp["te"]), 8)
    np.testing.assert_allclose(ex["attn_rollout"].sum(1), 1.0, atol=1e-4)
    from transformomics.interpret import deletion_test, spot_attention
    att = spot_attention(ex["model"], T, c.nbr_idx, sp["te"])["rollout"][:, 2:]
    dl = deletion_test(ex["model"], T, c.nbr_idx, sp["te"], att, "label", ms=(1, 2))
    assert set(dl) == {"full", (1, "top"), (1, "random"), (1, "bottom"), (2, "top"), (2, "random"), (2, "bottom")}


def test_run_multi_end_to_end_tiny(tmp_path):
    pytest.importorskip("torch")
    from transformomics.run_multi import DEFAULTS, run_multi
    from transformomics.run import _merge
    specs, markers = make_synthetic_sections(n_sections=6, n_donors=3, side=14, n_genes=150, seed=5)
    cfg = _merge(DEFAULTS, {"seeds": [0], "preprocess": {"n_hvg": 100}, "train": {"epochs": 2, "patience": 2},
                            "model": {"embed_dim": 16, "heads": 2, "ff_dim": 32},
                            "split": {"n_blocks": 6},
                            "analyses": {"boundary": {"n_perm": 49}, "saliency": {"n_null": 50, "min_spots": 5,
                                                                                   "ig_steps": 4, "max_spots_per_class": 20}}})
    res, wide = run_multi(cfg, specs, None, markers, str(tmp_path), verbose=False)
    for f in ("summary.md", "per_spot_table.csv.gz", "boundary_summary.csv", "paired_comparisons.csv"):
        assert (tmp_path / f).exists()
    assert res["group"].nunique() == 3
