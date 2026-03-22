"""
Projection Layer + Anatomical Masking
Input : z_e  [B, C, D]
Output: list of K tensors, each [B, n_ch_k, proj_dim]  (one per brain region)
"""

import torch
import torch.nn as nn
from typing import Dict, List


class ProjectionLayer(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D] → [B, C, proj_dim]"""
        return self.proj(x)


class AnatomicalMask(nn.Module):
    """
    Splits projected channel embeddings into K brain regions.
    Returns a list of tensors [B, n_ch_k, proj_dim] ordered by region.
    """

    def __init__(self, brain_regions: Dict[str, List[int]]):
        super().__init__()
        self.region_names   = list(brain_regions.keys())
        self.region_indices = [
            torch.tensor(idx, dtype=torch.long)
            for idx in brain_regions.values()
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x : [B, C, proj_dim]
        returns: list of K tensors, k-th tensor is [B, |R_k|, proj_dim]
        """
        regions = []
        for idx in self.region_indices:
            idx = idx.to(x.device)
            regions.append(x[:, idx, :])     # [B, |R_k|, proj_dim]
        return regions
