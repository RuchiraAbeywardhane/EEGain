"""
Parallel Convolutional Recurrent Neural Network (PCRNN) for EEG emotion recognition.

Architecture (from the paper):
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: one 1-second window                                 │
    │                                                             │
    │  CNN Branch (spatial)      LSTM Branch (temporal)           │
    │  ─────────────────         ──────────────────────           │
    │  [W, 9, 9] per frame       [W, 32]  (raw 1-D EEG)          │
    │  Conv2d 32 @4×4 + BN+ELU  LSTM(32) × 2 layers             │
    │  Conv2d 64 @4×4 + BN+ELU  → last hidden state              │
    │  Conv2d 128@4×4 + BN+ELU  → FC → TFV                       │
    │  Concat depth dim                                           │
    │  Conv2d 13 @1×1           ───────────────────              │
    │  Flatten → SFV             TFV                              │
    │                                                             │
    │  [SFV ‖ TFV]  → Dropout → FC → Softmax                    │
    └─────────────────────────────────────────────────────────────┘

Shapes used in this implementation
    W   = window_size = 128  (time steps)
    H,W_grid = 9             (spatial grid)
    C   = 32                 (EEG channels for LSTM)
    n_classes = 2

CNN branch processes every frame independently (shared weights across time),
then concatenates the resulting feature maps along the depth dimension before
applying the 1×1 fusion convolution.
"""

import torch
import torch.nn as nn
from .config import PCRConfig


class CNNSpatialBranch(nn.Module):
    """
    Spatial feature extractor.

    Takes a sequence of 2-D EEG frames  [B, W, H, W_grid]  and processes
    each frame through three conv layers (shared weights).  The resulting
    feature maps from all frames are then concatenated along the channel
    dimension and fused with a 1×1 convolution.

    Output: Spatial Feature Vector (SFV) of shape [B, sfv_size]
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

        pad = kernel_size // 2  # pad=2 for 4×4 kernel

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

        # ── Measure real spatial output size with a dummy pass ──────────
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size)
            dummy_out = self.conv_layers(dummy)          # [1, 128, H_out, W_out]
            _, C_out, H_out, W_out = dummy_out.shape

        self._H_out = H_out
        self._W_out = W_out
        self._C_feat = C_out

        fused_channels = window_size * C_out            # W × 128
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, reduce_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_filters),
            nn.ELU(),
        )

        # Real SFV size derived from actual conv output dimensions
        self.sfv_size = reduce_filters * H_out * W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, W, H, W_grid]  – sequence of 2-D EEG frames

        Returns:
            sfv : [B, sfv_size]
        """
        B, W, H, W_grid = x.shape

        # Process every frame with shared conv layers
        # Reshape to [B*W, 1, H, W_grid]
        frames = x.view(B * W, 1, H, W_grid)
        feat = self.conv_layers(frames)          # [B*W, 128, H, W_grid]

        # Reshape back: [B, W*128, H, W_grid]
        _, C_feat, H_out, W_out = feat.shape
        feat = feat.view(B, W * C_feat, H_out, W_out)

        # 1×1 fusion → [B, reduce_filters, H, W_grid]
        fused = self.fuse_conv(feat)

        # Flatten → SFV [B, sfv_size]
        sfv = fused.view(B, -1)
        return sfv


class LSTMTemporalBranch(nn.Module):
    """
    Temporal feature extractor.

    Takes the raw 1-D EEG sequence  [B, W, C]  and passes it through
    two stacked LSTM layers.  Only the last hidden state is used.
    A fully connected layer maps it to the Temporal Feature Vector (TFV).

    Output: TFV  [B, tfv_size]
    """

    def __init__(
        self,
        n_channels: int = 32,
        hidden_size: int = 32,
        num_layers: int = 2,
        tfv_size: int = 64,
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
        """
        Args:
            x : [B, W, C]  – raw 1-D EEG sequence

        Returns:
            tfv : [B, tfv_size]
        """
        # lstm_out: [B, W, hidden],  (h_n, c_n): [num_layers, B, hidden]
        _, (h_n, _) = self.lstm(x)

        # Take the last layer's hidden state: [B, hidden]
        last_hidden = h_n[-1]

        # Project to TFV
        tfv = self.fc(last_hidden)
        return tfv


class PCRNN(nn.Module):
    """
    Parallel CNN-Recurrent Neural Network.

    Combines spatial features from the CNN branch and temporal features
    from the LSTM branch, then classifies with a shared fully connected layer.

    Input per forward():
        x_2d : [B, W, H, W_grid]  – preprocessed 2-D EEG frame sequence
        x_1d : [B, W, C]          – raw 1-D EEG for LSTM

    Output:
        logits : [B, n_classes]
    """

    def __init__(self, cfg: PCRConfig):
        super().__init__()

        # TFV size = hidden_size (paper uses hidden=32 and keeps TFV at 32)
        tfv_size = cfg.lstm_hidden

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

    def forward(
        self,
        x_2d: torch.Tensor,
        x_1d: torch.Tensor,
    ) -> torch.Tensor:
        sfv = self.cnn_branch(x_2d)     # [B, sfv_size]
        tfv = self.lstm_branch(x_1d)    # [B, tfv_size]

        joint = torch.cat([sfv, tfv], dim=1)   # [B, sfv+tfv]
        logits = self.classifier(joint)         # [B, n_classes]
        return logits
