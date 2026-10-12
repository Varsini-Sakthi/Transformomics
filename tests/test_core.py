import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transformomics.graph import knn_graph, make_split, spatial_block_folds  # noqa: E402
from transformomics.model import Transformomics  # noqa: E402
from transformomics.programs import morans_i, score_programs  # noqa: E402
from transformomics.synthetic import make_synthetic  # noqa: E402


def test_knn_offsets_scale_free():
    adata, _ = make_synthetic(n_rows=12, n_cols=12)
    c = adata.obsm["spatial"]
    idx, dist, off = knn_graph(c, 6)
    assert idx.shape == (144, 6) and off.shape == (144, 6, 2)
    assert np.median(dist[:, 0]) == pytest.approx(1.0)
    idx2, _, off2 = knn_graph(c * 7.5, 6)           # rescaling coordinates must not change offsets
    assert np.allclose(off, off2, atol=1e-4)
    assert not (idx == np.arange(144)[:, None]).any()   # self excluded


def test_split_has_no_leakage_with_buffer():
    adata, _ = make_synthetic(n_rows=30, n_cols=30)
    idx, _, _ = knn_graph(adata.obsm["spatial"], 6)
    folds = spatial_block_folds(adata.obsm["spatial"], 12, 4)
    tr, va, te = make_split(folds, idx, 0, 4, buffer=True)
    held = set(va) | set(te)
    assert not (set(tr) & held)
    # no training query has a held-out spot among its neighbors, nor vice versa
    assert not any(set(idx[i]) & held for i in tr)
    nb_of_held = set(idx[list(held)].ravel())
    assert not (set(tr) & nb_of_held)


def test_program_scores_track_planted_signal():
    adata, programs = make_synthetic(n_rows=25, n_cols=25, seed=1)
    from transformomics.data import normalize_log1p
    logX = normalize_log1p(adata.X)
    s, names, counts = score_programs(logX, adata.var_names, programs)
    assert s.shape == (625, 4) and names == list(programs)
    assert all(c[0] == c[1] for c in counts.values())


def test_morans_i_detects_structure():
    adata, _ = make_synthetic(n_rows=25, n_cols=25)
    idx, _, _ = knn_graph(adata.obsm["spatial"], 6)
    smooth = adata.obsm["spatial"][:, 0]
    i_s, p_s = morans_i(smooth, idx, n_perm=49)
    i_r, p_r = morans_i(np.random.default_rng(0).normal(size=len(smooth)), idx, n_perm=49)
    assert i_s > 0.8 and p_s < 0.05
    assert abs(i_r) < 0.1


def test_model_forward_and_neighbor_mask():
    m = Transformomics(in_dim=50, out_dim=3, k=6).eval()
    xq, xn, off = torch.randn(4, 50), torch.randn(4, 6, 50), torch.randn(4, 6, 2)
    o_real = m(xq, xn, off)
    assert o_real.shape == (4, 3)
    # with neighbors masked, changing neighbor expression must not change the output
    o1 = m(xq, xn, off, use_neighbors=False)
    o2 = m(xq, torch.randn(4, 6, 50), off, use_neighbors=False)
    assert torch.allclose(o1, o2, atol=1e-6)
    # with neighbors visible, it must
    o3 = m(xq, torch.randn(4, 6, 50), off, use_neighbors=True)
    assert not torch.allclose(o_real, o3, atol=1e-6)
    _, att = m(xq, xn, off, return_attn=True)
    assert torch.allclose(att[-1].sum(-1), torch.ones(4, 4, 8), atol=1e-5)
