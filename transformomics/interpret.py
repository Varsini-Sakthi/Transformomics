"""Model-side interpretation: attention rollout, integrated gradients, deletion test.

Token order everywhere: [CLS, query, nbr1..nbrk]. ``T`` is any object with ``.X`` (torch [N,G]),
``.offsets`` (torch [N,k,2]) and ``.y`` (torch) - i.e. ``train.Tensors``.

torch is imported lazily so the numpy-only helpers (``rollout_cls_row``) work without it.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ attention rollout (numpy)
def rollout_cls_row(attn_layers, residual: float = 0.5) -> np.ndarray:
    """CLS row of attention rollout (Abnar & Zuidema 2020).

    ``attn_layers``: list over layers of arrays [B, H, L, L] (softmax weights). Heads are averaged,
    the residual connection is modelled as ``residual*I + (1-residual)*A`` and layers are multiplied
    from first to last. Returns [B, L]: how much each *input* token influences the final CLS state.
    Unlike last-layer attention, this accounts for information the query token already absorbed
    from its neighbours in earlier layers.
    """
    R = None
    for a in attn_layers:
        a = np.asarray(a).mean(1)                              # [B, L, L]
        L = a.shape[-1]
        a = (1.0 - residual) * a + residual * np.eye(L)[None]
        a = a / a.sum(-1, keepdims=True)
        R = a if R is None else a @ R
    return R[:, 0, :]


def spot_attention(model, T, nbr_idx, idx, batch: int = 256) -> dict:
    """Per-spot attention for ``idx``: {'last': [n,2+k], 'rollout': [n,2+k]} (numpy)."""
    import torch
    model.eval()
    lasts, rolls = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            xq, xn, off = T.X[b], T.X[torch.as_tensor(nbr_idx[b])], T.offsets[b]
            _, att = model(xq, xn, off, use_neighbors=True, return_attn=True)
            lasts.append(att[-1][:, :, 0, :].mean(1).cpu().numpy())
            rolls.append(rollout_cls_row([a.cpu().numpy() for a in att]))
    return {"last": np.vstack(lasts), "rollout": np.vstack(rolls)}


# ------------------------------------------------------------------ integrated gradients
def integrated_gradients(model, xq, xn, off, out_fn, steps: int = 16):
    """Signed integrated gradients (zero baseline = the average spot, since inputs are z-scored).

    ``out_fn(output [B,out]) -> [B]`` picks the scalar to explain (e.g. a class logit minus the
    mean of the other logits). Returns (attr_query [B,G], attr_nbr [B,k,G]); positive = pushes the
    chosen output up. The path scales query AND neighbour expression together.
    """
    import torch
    model.eval()
    gq, gn = torch.zeros_like(xq), torch.zeros_like(xn)
    for i in range(steps):
        a = (i + 0.5) / steps
        q = (xq * a).detach().requires_grad_(True)
        n = (xn * a).detach().requires_grad_(True)
        score = out_fn(model(q, n, off, use_neighbors=True)).sum()
        g1, g2 = torch.autograd.grad(score, [q, n])
        gq += g1 / steps
        gn += g2 / steps
    return (xq * gq).detach(), (xn * gn).detach()


def mean_attribution(model, T, nbr_idx, idx, out_fn, max_spots: int = 200, steps: int = 16,
                     batch: int = 64, seed: int = 0):
    """Mean signed IG over (a sample of) spots ``idx`` -> (query [G], neighbours-summed [G], n_used)."""
    import torch
    idx = np.asarray(idx)
    if len(idx) == 0:
        return None, None, 0
    if len(idx) > max_spots:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_spots, replace=False))
    aq = torch.zeros(T.X.shape[1])
    an = torch.zeros(T.X.shape[1])
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        xq, xn, off = T.X[b], T.X[torch.as_tensor(nbr_idx[b])], T.offsets[b]
        q, n = integrated_gradients(model, xq, xn, off, out_fn, steps)
        aq += q.sum(0)
        an += n.sum((0, 1))                                    # sum over the k neighbour tokens
    return (aq / len(idx)).numpy(), (an / len(idx)).numpy(), len(idx)


def class_attributions(model, T, nbr_idx, idx_by_class: dict, n_classes: int, **kw):
    """Class-contrastive IG: for class c, explain ``logit_c - mean(other logits)`` on spots of true class c.

    Returns (query [C,G], neighbours [C,G], n_used [C]); rows are NaN for classes without spots.
    """
    G = T.X.shape[1]
    Q = np.full((n_classes, G), np.nan, dtype=np.float32)
    N = np.full((n_classes, G), np.nan, dtype=np.float32)
    used = np.zeros(n_classes, dtype=int)
    for c, idx in idx_by_class.items():
        def out_fn(o, c=c):
            return o[:, c] - (o.sum(1) - o[:, c]) / max(n_classes - 1, 1)
        q, n, u = mean_attribution(model, T, nbr_idx, idx, out_fn, **kw)
        if u:
            Q[c], N[c], used[c] = q, n, u
    return Q, N, used


# ------------------------------------------------------------------ deletion test
def per_spot_loss(out, y, task: str):
    import torch.nn.functional as F
    if task == "label":
        return F.cross_entropy(out, y, reduction="none")
    return ((out - y) ** 2).mean(1)


def deletion_test(model, T, nbr_idx, idx, scores, task: str, ms=(1, 2, 3), seed: int = 0,
                  batch: int = 256) -> dict:
    """Causal check of attention: hide m neighbours chosen by ``scores`` and measure per-spot loss.

    ``scores`` [n,k] = importance per neighbour (e.g. attention rollout). Strategies:
      'top' (most important first), 'random', 'bottom' (least important first).
    If attention is faithful, loss rises most for 'top', least for 'bottom'. Note masked inputs are
    a little out-of-distribution (the model never trained with missing tokens), which is why the
    random-removal arm is the reference rather than the absolute change.
    Returns {'full': [n], (m, strategy): [n]} of per-spot losses (numpy).
    """
    import torch
    idx = np.asarray(idx)
    scores = np.asarray(scores)
    n, k = scores.shape
    rng = np.random.default_rng(seed)
    order_desc = np.argsort(-scores, axis=1)
    order_asc = np.argsort(scores, axis=1)
    rand_order = np.argsort(rng.random((n, k)), axis=1)
    model.eval()

    def run(keep):
        res = []
        with torch.no_grad():
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                xq, xn, off, y = T.X[b], T.X[torch.as_tensor(nbr_idx[b])], T.offsets[b], T.y[b]
                o = model(xq, xn, off, use_neighbors=True,
                          nbr_mask=None if keep is None else torch.as_tensor(keep[s:s + batch]))
                res.append(per_spot_loss(o, y, task).cpu().numpy())
        return np.concatenate(res)

    out = {"full": run(None)}
    for m in ms:
        if m >= k:
            continue
        for name, order in (("top", order_desc), ("random", rand_order), ("bottom", order_asc)):
            keep = np.ones((n, k), dtype=bool)
            np.put_along_axis(keep, order[:, :m], False, axis=1)
            out[(m, name)] = run(keep)
    return out
