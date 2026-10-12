"""Training / prediction for the Transformomics plus metric helpers."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .model import Transformomics


@dataclass
class Tensors:
    X: torch.Tensor        # [N,G]
    nbr_idx: np.ndarray    # [N,k]
    offsets: torch.Tensor  # [N,k,2]
    y: torch.Tensor        # [N,P] float (regression) or [N] long (classification)


def _batch(T: Tensors, idx, nbr_idx):
    ni = torch.as_tensor(nbr_idx[idx])
    return T.X[idx], T.X[ni], T.offsets[idx], T.y[idx]


def _rotate(off):
    """Random rotation of neighbor offsets (tissue orientation should not matter)."""
    th = torch.rand(off.shape[0], 1, device=off.device) * 2 * np.pi
    c, s = torch.cos(th), torch.sin(th)
    x, y = off[..., 0], off[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], -1)


def shuffled_neighbors(n: int, k: int, seed: int) -> np.ndarray:
    """Replace every spot's neighbors with random spots (destroys spatial information)."""
    rng = np.random.default_rng(10_000 + seed)
    return rng.integers(0, n, size=(n, k))


def fit_predict(T: Tensors, tr, va, te, task: str, mcfg: dict, tcfg: dict, seed: int,
                mode: str = "real", n_classes: int | None = None, want_attn: bool = False):
    """Train on ``tr``, early-stop on ``va`` loss, predict ``te``.

    mode: 'real'     - true spatial neighbors
          'shuffled' - random spots as neighbors (same compute, no spatial info)
          'none'     - neighbor tokens masked out of attention (single-spot model)
    Returns (pred [len(te), out], extras dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = T.nbr_idx.shape[1]
    nbr = T.nbr_idx if mode != "shuffled" else shuffled_neighbors(len(T.X), k, seed)
    use_nb = mode != "none"

    out_dim = T.y.shape[1] if task == "program" else int(n_classes)
    model = Transformomics(T.X.shape[1], out_dim, k=k, **mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.MSELoss() if task == "program" else nn.CrossEntropyLoss()
    bs = tcfg["batch_size"]

    def run_eval(idx):
        model.eval()
        tot, outs = 0.0, []
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                b = idx[s:s + 256]
                xq, xn, off, y = _batch(T, b, nbr)
                o = model(xq, xn, off, use_neighbors=use_nb)
                tot += loss_fn(o, y).item() * len(b)
                outs.append(o)
        return tot / max(len(idx), 1), torch.cat(outs).numpy() if outs else np.zeros((0, out_dim))

    best, best_state, bad = float("inf"), None, 0
    for _ in range(tcfg["epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            if len(b) < 2:
                continue
            xq, xn, off, y = _batch(T, b, nbr)
            if tcfg.get("rot_aug", True):
                off = _rotate(off)
            opt.zero_grad()
            loss = loss_fn(model(xq, xn, off, use_neighbors=use_nb), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl, _ = run_eval(va)
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    _, pred = run_eval(te)

    extras = {"val_loss": best}
    if want_attn and use_nb and len(te):
        model.eval()
        masses = []
        with torch.no_grad():
            for s in range(0, len(te), 256):
                b = te[s:s + 256]
                xq, xn, off, _ = _batch(T, b, nbr)
                _, att = model(xq, xn, off, use_neighbors=True, return_attn=True)
                cls_row = att[-1][:, :, 0, :].mean(1)          # last layer, mean over heads
                masses.append(cls_row[:, 2:].sum(1))           # CLS -> all neighbor tokens
        extras["nbr_attn_mass"] = float(torch.cat(masses).mean())
    return pred, extras


# ---------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, names):
    out, rs, r2s = {}, [], []
    for j, n in enumerate(names):
        a, b = y_true[:, j], y_pred[:, j]
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
        ss_res, ss_tot = ((a - b) ** 2).sum(), ((a - a.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[f"r_{n}"] = float(r)
        rs.append(r)
        r2s.append(r2)
    out["score"] = float(np.mean(rs))          # primary metric: mean Pearson r across programmes
    out["mean_r2"] = float(np.mean(r2s))
    return out


def classification_metrics(y_true, logits):
    pred = np.asarray(logits).argmax(1)
    return {"score": float(f1_score(y_true, pred, average="macro")),   # primary: macro-F1
            "acc": float(accuracy_score(y_true, pred))}
