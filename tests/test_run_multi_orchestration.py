"""Fast orchestration test for run_multi: replaces the model-dependent calls with numpy fakes that PLANT
a boundary effect, then checks the whole pipeline (splits -> per-spot tables -> boundary / saliency /
faithfulness / deletion reports) recovers it and the controls stay flat. No model training happens."""
import numpy as np
import pytest

pytest.importorskip("torch")
import transformomics.run_multi as rm                                   # noqa: E402
from transformomics.boundary import boundary_score                       # noqa: E402
from transformomics.run import _merge                                    # noqa: E402
from transformomics.synthetic_sections import make_synthetic_sections    # noqa: E402

K = 6


def _install_fakes(monkeypatch):
    rng = np.random.default_rng(0)

    def fit(T, tr, va, te, task, mcfg, tcfg, seed, mode, n_classes=None, want_attn=False,
            want_spot_attn=False, return_model=False):
        y = np.asarray(T.y)[te]
        bs = np.nan_to_num(boundary_score(np.asarray(T.y), T.nbr_idx)[te])
        logits = rng.normal(size=(len(te), n_classes))
        boost = 1.0 + (3.0 * bs if mode == "real" else 0.0)              # real helps more at boundaries
        logits[np.arange(len(te)), y] += 2.0 * boost
        ex = {"val_loss": 0.5}
        if want_attn and mode != "none":
            mass = (0.3 + (0.4 * bs if mode == "real" else 0.0)) + rng.normal(0, 0.03, len(te))
            a = np.zeros((len(te), 2 + K))
            a[:, 0], a[:, 1] = 0.05, 1 - 0.05 - mass
            w = rng.uniform(0.5, 1.5, (len(te), K))
            a[:, 2:] = w / w.sum(1, keepdims=True) * mass[:, None]
            ex.update(attn_last=a, attn_rollout=a, nbr_attn_mass=float(mass.mean()))
        if return_model:
            ex["model"], ex["nbr"] = "FAKE", T.nbr_idx
        return logits, ex

    def spot_attention(model, T, nbr_idx, idx, batch=256):
        a = rng.dirichlet(np.ones(2 + K), len(idx))
        return {"last": a, "rollout": a}

    def deletion(model, T, nbr_idx, idx, scores, task, ms=(1,), **kw):
        n = len(idx)
        return {"full": np.zeros(n), **{(m, s): np.full(n, v * m) for m in ms
                                        for s, v in (("top", .3), ("random", .1), ("bottom", 0.))}}

    def class_attr(model, T, nbr_idx, idx_by_class, n_classes, **kw):
        G = T.X.shape[1]
        Q = np.full((n_classes, G), np.nan)
        N = Q.copy()
        for c in idx_by_class:
            Q[c], N[c] = rng.normal(size=G), rng.normal(size=G)
        return Q, N, np.ones(n_classes, int)

    monkeypatch.setattr(rm, "fit_predict", fit)
    monkeypatch.setattr(rm, "spot_attention", spot_attention)
    monkeypatch.setattr(rm, "deletion_test", deletion)
    monkeypatch.setattr(rm, "class_attributions", class_attr)


def test_pipeline_recovers_planted_boundary_effect_and_keeps_controls_flat(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    specs, markers = make_synthetic_sections(n_sections=6, n_donors=3, side=22, n_genes=300, seed=3, mix_temp=1.5)
    cfg = _merge(rm.DEFAULTS, {"seeds": [0, 1], "preprocess": {"n_hvg": 150}, "split": {"n_blocks": 8},
                               "analyses": {"boundary": {"n_perm": 99}, "saliency": {"n_null": 50, "min_spots": 10}}})
    res, wide = rm.run_multi(cfg, specs, None, markers, str(tmp_path), verbose=False)
    for f in ("summary.md", "per_spot_table.csv.gz", "boundary_summary.csv", "boundary_by_group.csv",
              "marker_enrichment.csv", "faithfulness_summary.csv", "deletion_summary.csv", "paired_comparisons.csv"):
        assert (tmp_path / f).exists(), f
    import pandas as pd
    b = pd.read_csv(tmp_path / "boundary_summary.csv").set_index("outcome")
    assert b.loc["attn_excess_roll", "adj_effect"] > 0.1 and b.loc["attn_excess_roll", "n_groups_positive"] == 3
    assert b.loc["benefit_vs_shuffled", "adj_effect"] > 0.3
    assert abs(b.loc["attn_roll_shuffled", "adj_effect"]) < 0.02          # shuffled-model control is flat
    f = pd.read_csv(tmp_path / "faithfulness_summary.csv").set_index("baseline")
    assert abs(f.loc["expr_cosine", "mean_diff"]) < 0.05                   # random attention == no advantage
    assert res["group"].nunique() == 3


def test_restrict_and_group_val_mode_run(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    specs, markers = make_synthetic_sections(n_sections=8, n_donors=4, side=16, n_genes=200, seed=4)
    cfg = _merge(rm.DEFAULTS, {"seeds": [0], "preprocess": {"n_hvg": 100},
                               "restrict": {"markers": ["DOM0_MK0", "DOM0_MK1", "DOM1_MK0"], "quantile": 0.4},
                               "split": {"val_mode": "group"},
                               "analyses": {"boundary": {"n_perm": 29}, "saliency": {"enabled": False}}})
    res, wide = rm.run_multi(cfg, specs, None, markers, str(tmp_path), verbose=False)
    assert res["group"].nunique() == 4 and len(wide) > 0
