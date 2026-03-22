"""
EEG-only BIH_GCN Model
----------------------
Raw EEG [B, C, T]
  → Spectrogram       [B, C, F, T']
  → Mamba Encoder     [B, C, D]          z_e
  → Projection        [B, C, proj_dim]
  → Anatomical Mask   list of K [B, n_k, proj_dim]
  → GCN Stage 1       [B, K, gcn1_out]   z_regions
  → GCN Stage 2       [B, gcn2_out]      z_fused
  → Classifier        [B, n_classes]
"""

import torch
import torch.nn as nn

from BIH_GCN.config       import BIHGCNConfig
from BIH_GCN.spectrogram  import EEGSpectrogram
from BIH_GCN.mamba_encoder import MambaEncoder
from BIH_GCN.brain_regions import ProjectionLayer, AnatomicalMask
from BIH_GCN.gcn_stage1   import LocalRegionGCN
from BIH_GCN.gcn_stage2   import GlobalRegionGCN


class BIHGCN(nn.Module):
    def __init__(self, cfg: BIHGCNConfig):
        super().__init__()
        self.cfg = cfg

        # ── Stage 0: Spectrogram ──────────────────────────────────────────────
        self.spectrogram = EEGSpectrogram(cfg)

        # ── Stage 1: Mamba Encoder ────────────────────────────────────────────
        self.mamba = MambaEncoder(cfg)

        # ── Projection + Anatomical Masking ───────────────────────────────────
        self.projection = ProjectionLayer(cfg.mamba_d_model, cfg.proj_dim)
        self.mask       = AnatomicalMask(cfg.brain_regions)

        # ── BIH-GCN Stage 1: Local Region GCN ────────────────────────────────
        self.gcn1 = LocalRegionGCN(cfg)

        # ── BIH-GCN Stage 2: Global Region GCN ───────────────────────────────
        self.gcn2 = GlobalRegionGCN(cfg)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(cfg.gcn2_out, cfg.gcn2_out // 2),
            nn.GELU(),
            nn.Dropout(cfg.clf_dropout),
            nn.Linear(cfg.gcn2_out // 2, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, T]  raw EEG
        returns: [B, n_classes]  logits
        """
        # Spectrogram
        spec = self.spectrogram(x)          # [B, C, F, T']

        # Mamba encoder → z_e
        z_e  = self.mamba(spec)             # [B, C, D]

        # Project + anatomical split
        z_p  = self.projection(z_e)         # [B, C, proj_dim]
        regs = self.mask(z_p)               # list of K: [B, n_k, proj_dim]

        # GCN Stage 1 → region embeddings
        z_r  = self.gcn1(regs)              # [B, K, gcn1_out]

        # GCN Stage 2 → fused EEG embedding
        z_f  = self.gcn2(z_r)              # [B, gcn2_out]

        # Classify
        return self.classifier(z_f)         # [B, n_classes]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return fused EEG embedding without classification head."""
        spec = self.spectrogram(x)
        z_e  = self.mamba(spec)
        z_p  = self.projection(z_e)
        regs = self.mask(z_p)
        z_r  = self.gcn1(regs)
        return self.gcn2(z_r)               # [B, gcn2_out]
