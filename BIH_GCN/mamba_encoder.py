"""
Stage 1 – Mamba Encoder
Input : [B, C, F, T']      spectrogram per channel
Output: [B, C, D]          one embedding vector per channel  (z_e)

Each channel's spectrogram is flattened along the time axis → sequence of
F-dim tokens → fed through a stack of Mamba (SSM) layers → mean-pooled → z_e.

Falls back to a plain Transformer if mamba-ssm is not installed.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


# ── Mamba block ───────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model = d_model,
                d_state = d_state,
                d_conv  = d_conv,
                expand  = expand,
            )
        else:
            # Fallback: single Transformer encoder layer
            self.ssm = nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = max(1, d_model // 64),
                dim_feedforward = d_model * expand,
                batch_first     = True,
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.ssm(x))


# ── Full encoder ──────────────────────────────────────────────────────────────

class MambaEncoder(nn.Module):
    """
    Encodes a spectrogram sequence [B*C, T', F] → [B*C, D] via Mamba layers.
    """

    def __init__(self, cfg):
        super().__init__()
        self.input_proj = nn.Linear(cfg.stft_n_mels, cfg.mamba_d_model)

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model = cfg.mamba_d_model,
                d_state = cfg.mamba_d_state,
                d_conv  = cfg.mamba_d_conv,
                expand  = cfg.mamba_expand,
            )
            for _ in range(cfg.mamba_n_layers)
        ])

        self.out_norm = nn.LayerNorm(cfg.mamba_d_model)
        self.d_model  = cfg.mamba_d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, F, T']
        returns: [B, C, D]
        """
        B, C, F, Tp = x.shape
        x = x.permute(0, 1, 3, 2)          # [B, C, T', F]
        x = x.reshape(B * C, Tp, F)        # [B*C, T', F]  ← sequence of freq tokens

        x = self.input_proj(x)              # [B*C, T', D]
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        x = x.mean(dim=1)                   # mean-pool over time → [B*C, D]

        return x.reshape(B, C, self.d_model)  # [B, C, D]
