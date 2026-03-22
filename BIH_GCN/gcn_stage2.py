"""
BIH-GCN Stage 2 – Global Region-Level Graph
Input : z_regions [B, K, gcn1_out]   (K region embeddings from Stage 1)
Output: z_fused   [B, gcn2_out]      (single EEG graph embedding)

A learnable global node is appended → graph has K+1 nodes.
After 2-layer GCN, the global node's output is the final embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Re-use GCNLayer from stage1 ───────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.gelu(self.norm(self.linear(x)))


def _build_adj(n: int, device) -> torch.Tensor:
    A = torch.ones(n, n, device=device)
    D = A.sum(dim=1).pow(-0.5)
    return D.unsqueeze(1) * A * D.unsqueeze(0)


# ── Stage 2 module ────────────────────────────────────────────────────────────

class GlobalRegionGCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.gcn1_out

        # Learnable global node (shared across batch)
        self.global_token = nn.Parameter(torch.randn(1, 1, in_dim))

        self.gcn1 = GCNLayer(in_dim,          cfg.gcn2_hidden)
        self.gcn2 = GCNLayer(cfg.gcn2_hidden, cfg.gcn2_out)

        self.out_dim = cfg.gcn2_out

    def forward(self, z_regions: torch.Tensor) -> torch.Tensor:
        """
        z_regions : [B, K, gcn1_out]
        returns   : [B, gcn2_out]
        """
        B, K, D = z_regions.shape

        # Append global node → [B, K+1, D]
        g = self.global_token.expand(B, 1, D)
        x = torch.cat([z_regions, g], dim=1)    # [B, K+1, D]

        adj = _build_adj(K + 1, x.device)       # [K+1, K+1]

        x = self.gcn1(x, adj)                   # [B, K+1, gcn2_hidden]
        x = self.gcn2(x, adj)                   # [B, K+1, gcn2_out]

        # Return the global node's embedding as the fused EEG representation
        return x[:, -1, :]                       # [B, gcn2_out]
