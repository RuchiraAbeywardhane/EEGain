"""
BIH-GCN Stage 1 – Local Region-Level Graph
For each brain region k:
  - Build a fully-connected intra-region adjacency
  - Apply 2-layer GCN
  - Attention-pool all channel nodes → single region embedding z_rk

Input : list of K tensors [B, n_ch_k, proj_dim]
Output: [B, K, gcn1_out]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ── Simple GCN layer ──────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, N, in_dim]
        adj : [N, N]  normalised adjacency (precomputed)
        """
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.gelu(self.norm(self.linear(x)))


# ── Attention Pooling ─────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.q     = nn.Linear(dim, dim)
        self.k     = nn.Linear(dim, dim)
        self.v     = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.scale   = (dim // n_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, N, dim]
        returns: [B, dim]  pooled
        """
        B, N, D = x.shape
        cls = x.mean(dim=1, keepdim=True)              # [B, 1, D]  query = mean
        q = self.q(cls)                                # [B, 1, D]
        k = self.k(x)                                  # [B, N, D]
        v = self.v(x)                                  # [B, N, D]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, 1, N]
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).squeeze(1)                   # [B, D]
        return out


# ── Stage 1 module ────────────────────────────────────────────────────────────

def _build_adj(n: int, device) -> torch.Tensor:
    """Normalised fully-connected adjacency (with self-loops)."""
    if n == 1:
        return torch.ones(1, 1, device=device)
    A = torch.ones(n, n, device=device)
    D = A.sum(dim=1)
    D_inv_sqrt = D.pow(-0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


class LocalRegionGCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.proj_dim

        self.gcn1 = GCNLayer(in_dim,          cfg.gcn1_hidden)
        self.gcn2 = GCNLayer(cfg.gcn1_hidden, cfg.gcn1_out)
        self.pool = AttentionPool(cfg.gcn1_out, cfg.gcn1_heads)

    def _forward_region(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_ch_k, proj_dim] → [B, gcn1_out]"""
        n   = x.size(1)
        adj = _build_adj(n, x.device)
        x   = self.gcn1(x, adj)
        x   = self.gcn2(x, adj)
        return self.pool(x)                             # [B, gcn1_out]

    def forward(self, regions: List[torch.Tensor]) -> torch.Tensor:
        """
        regions : list of K tensors [B, n_ch_k, proj_dim]
        returns : [B, K, gcn1_out]
        """
        embeddings = [self._forward_region(r) for r in regions]
        return torch.stack(embeddings, dim=1)           # [B, K, gcn1_out]
