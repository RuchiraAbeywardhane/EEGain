"""
Parallel Convolutional Recurrent Neural Network (PCRNN).

Two modes, auto-selected by PCRConfig:
    dataset="deap"       → CNNSpatialBranch  (2D, 32-ch, 9×9 grid) + LSTMTemporalBranch
    dataset="emognition" → CNN1DTemporalBranch (1D, 4-ch)          + LSTMTemporalBranch
"""

import torch
import torch.nn as nn
from .config import PCRConfig


# ════════════════════════════════════════════════════════════════════════════
#  DEAP BRANCH — 2D Spatial CNN
# ════════════════════════════════════════════════════════════════════════════

class CNNSpatialBranch(nn.Module):
    """
    2-D spatial feature extractor for DEAP (32 channels, 9×9 grid).
    Processes every frame independently with shared conv weights,
    concatenates all frames along depth, fuses with 1×1 conv → SFV.
    """

    def __init__(
        self,
        window_size: int = 128,
        grid_size: int = 9,
        conv_filters: list = None,
        kernel_size: int = 4,
        reduce_filters: int = 13,
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128]

        pad = kernel_size // 2

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,               conv_filters[0], kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(conv_filters[0]),
            nn.ELU(),
            nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(conv_filters[1]),
            nn.ELU(),
            nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(conv_filters[2]),
            nn.ELU(),
        )

        # Measure real spatial output via dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size)
            dummy_out = self.conv_layers(dummy)
            _, C_out, H_out, W_out = dummy_out.shape

        self._H_out  = H_out
        self._W_out  = W_out
        self._C_feat = C_out

        fused_channels = window_size * C_out
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, reduce_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_filters),
            nn.ELU(),
        )
        self.sfv_size = reduce_filters * H_out * W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, W, H, W_grid]  →  sfv : [B, sfv_size]"""
        B, W, H, W_grid = x.shape
        frames = x.view(B * W, 1, H, W_grid)
        feat   = self.conv_layers(frames)                      # [B*W, C, H, W]
        _, C_feat, H_out, W_out = feat.shape
        feat   = feat.view(B, W * C_feat, H_out, W_out)       # [B, W*C, H, W]
        fused  = self.fuse_conv(feat)                          # [B, 13, H, W]
        return fused.view(B, -1)                               # [B, sfv_size]


# ════════════════════════════════════════════════════════════════════════════
#  EMOGNITION BRANCH — 1D Temporal CNN
# ════════════════════════════════════════════════════════════════════════════

class CNN1DTemporalBranch(nn.Module):
    """
    1-D temporal feature extractor for Emognition (4 channels, 256 Hz).

    Treats the window as a 1-D sequence of shape [B, C, W] and applies
    three Conv1d layers to extract temporal patterns across all 4 channels.

    Output: Spatial-equivalent Feature Vector (SFV) of shape [B, sfv_size]
    """

    def __init__(
        self,
        n_channels: int = 4,
        window_size: int = 256,
        conv_filters: list = None,
        kernel_size: int = 4,
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128]

        pad = kernel_size // 2

        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_channels,      conv_filters[0], kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ELU(),
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ELU(),
            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ELU(),
        )

        # Measure real output length via dummy forward pass
        with torch.no_grad():
            dummy    = torch.zeros(1, n_channels, window_size)
            dummy_out = self.conv_layers(dummy)         # [1, 128, L_out]
            _, C_out, L_out = dummy_out.shape

        self.pool     = nn.AdaptiveAvgPool1d(1)         # collapse time → [B, 128, 1]
        self.sfv_size = C_out                           # = conv_filters[-1] = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, W, C]  (LSTM-format input, W=window_size, C=n_channels)
        →  sfv : [B, sfv_size]
        """
        # Conv1d expects [B, C, W]
        x   = x.permute(0, 2, 1)          # [B, C, W]
        out = self.conv_layers(x)          # [B, 128, L_out]
        out = self.pool(out)               # [B, 128, 1]
        return out.squeeze(-1)             # [B, 128]


# ════════════════════════════════════════════════════════════════════════════
#  SHARED LSTM BRANCH
# ════════════════════════════════════════════════════════════════════════════

class LSTMTemporalBranch(nn.Module):
    """
    Temporal feature extractor (shared by both DEAP and Emognition paths).
    2-layer LSTM → last hidden state → FC → TFV.
    """

    def __init__(
        self,
        n_channels: int = 32,
        hidden_size: int = 32,
        num_layers: int = 2,
        tfv_size: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_channels,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, tfv_size),
            nn.ELU(),
        )
        self.tfv_size = tfv_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, W, C]  →  tfv : [B, tfv_size]"""
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# ════════════════════════════════════════════════════════════════════════════
#  PCRNN — unified model, auto-selects CNN branch from config
# ════════════════════════════════════════════════════════════════════════════

class PCRNN(nn.Module):
    """
    Parallel CNN-Recurrent Neural Network.

    DEAP mode       (cfg.dataset="deap"):
        CNN branch  = CNNSpatialBranch     input x_2d [B, W, 9, 9]
        LSTM branch = LSTMTemporalBranch   input x_1d [B, W, 32]

    Emognition mode (cfg.dataset="emognition"):
        CNN branch  = CNN1DTemporalBranch  input x_1d [B, W, 4]  (x_2d ignored)
        LSTM branch = LSTMTemporalBranch   input x_1d [B, W, 4]

    In both modes forward() accepts (x_2d, x_1d) for API consistency.
    """

    def __init__(self, cfg: PCRConfig):
        super().__init__()
        self.is_emognition = cfg.dataset.lower() == "emognition"

        tfv_size = cfg.lstm_hidden

        if self.is_emognition:
            # window_size scaled to 256 Hz
            ws = cfg.window_size * (cfg.sampling_rate // 128)
            self.cnn_branch = CNN1DTemporalBranch(
                n_channels   = cfg.n_eeg_channels,
                window_size  = ws,
                conv_filters = cfg.cnn_filters,
                kernel_size  = cfg.cnn_kernel,
            )
        else:
            self.cnn_branch = CNNSpatialBranch(
                window_size    = cfg.window_size,
                grid_size      = cfg.grid_size,
                conv_filters   = cfg.cnn_filters,
                kernel_size    = cfg.cnn_kernel,
                reduce_filters = cfg.cnn_reduce_filters,
            )

        self.lstm_branch = LSTMTemporalBranch(
            n_channels  = cfg.n_eeg_channels,
            hidden_size = cfg.lstm_hidden,
            num_layers  = cfg.lstm_layers,
            tfv_size    = tfv_size,
            dropout     = cfg.dropout,
        )

        joint_size = self.cnn_branch.sfv_size + self.lstm_branch.tfv_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(joint_size, cfg.n_classes),
        )

    def forward(self, x_2d: torch.Tensor, x_1d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_2d : [B, W, H, W_grid]  – 2D frames  (DEAP) or dummy zeros (Emognition)
            x_1d : [B, W, C]          – 1D EEG sequence  (both datasets)
        Returns:
            logits : [B, n_classes]
        """
        if self.is_emognition:
            # x_2d is dummy zeros for Emognition — CNN branch uses x_1d directly
            sfv = self.cnn_branch(x_1d)
        else:
            sfv = self.cnn_branch(x_2d)

        tfv    = self.lstm_branch(x_1d)
        joint  = torch.cat([sfv, tfv], dim=1)
        return self.classifier(joint)
