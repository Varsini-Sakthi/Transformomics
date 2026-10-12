"""Transformomics v2 model.

Differences from the v1 notebook model (see legacy/):
  * neighbors are described by *relative* offsets (dx, dy, dist) in units of the median
    nearest-neighbor spacing, not absolute normalized coordinates -> transferable across
    sections/technologies;
  * optional learned, per-head distance bias on attention (nearer tokens favoured);
  * the neighbor context can be switched off (``use_neighbors=False``) for ablations;
  * the head can be regression (several programme scores) or classification.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float, dist_bias: bool):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.dist_alpha = nn.Parameter(torch.zeros(heads)) if dist_bias else None

    def forward(self, x, dist, key_mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        if self.dist_alpha is not None:
            att = att - F.softplus(self.dist_alpha).view(1, -1, 1, 1) * dist.unsqueeze(1)
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        w = att.softmax(-1)
        out = (self.drop(w) @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, w


class Transformomics(nn.Module):
    """Tokens: [CLS, query spot, neighbor_1..neighbor_k]."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 6, embed_dim: int = 64, heads: int = 4,
                 layers: int = 2, ff_dim: int = 128, dropout: float = 0.1, dist_bias: bool = True):
        super().__init__()
        self.k = k
        self.expr_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, embed_dim))
        self.type_embed = nn.Embedding(2, embed_dim)            # 0 = query, 1 = neighbor
        self.pos_mlp = nn.Sequential(nn.Linear(3, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, ff_dim, dropout, dist_bias) for _ in range(layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, xq, xn, offsets, use_neighbors: bool = True, return_attn: bool = False):
        """xq [B,G], xn [B,k,G], offsets [B,k,2] -> output [B,out_dim] (and attention list)."""
        B, k, _ = xn.shape
        expr = self.expr_proj(torch.cat([xq.unsqueeze(1), xn], 1))             # [B,1+k,d]
        zero = offsets.new_zeros(B, 1, 2)
        pos = torch.cat([zero, offsets], 1)                                    # [B,1+k,2]
        dnorm = pos.norm(dim=-1, keepdim=True)
        pos_emb = self.pos_mlp(torch.cat([pos, dnorm], -1))
        types = torch.cat([torch.zeros(B, 1), torch.ones(B, k)], 1).long().to(xq.device)
        x = expr + pos_emb + self.type_embed(types)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)                      # [B,2+k,d]

        full_pos = torch.cat([zero, pos], 1)
        dist = torch.cdist(full_pos, full_pos)
        dist[:, 0, :] = 0
        dist[:, :, 0] = 0
        key_mask = torch.ones(B, 2 + k, dtype=torch.bool, device=xq.device)
        if not use_neighbors:
            key_mask[:, 2:] = False

        attns = []
        for blk in self.blocks:
            x, w = blk(x, dist, key_mask)
            attns.append(w)
        out = self.head(self.norm(x[:, 0]))
        return (out, attns) if return_attn else out
