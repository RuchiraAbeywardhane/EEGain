"""
CNN-LSTM model for EEG-based emotion recognition.

Architecture
------------
Input: (B, 1, C, T)  — standard EEGain 4-D format
       C = 4 EEG channels (TP9, AF7, AF8, TP10)
       T = time samples

Stage 1 – Per-channel 1D CNN branches
    Four independent CNN branches, one per channel.
    Each branch processes its channel as a 1-D temporal signal and
    produces a compressed feature sequence of shape (B, cnn_out, T').
    Using separate branches forces the model to learn
    channel-specific spectral / temporal patterns before mixing them.

Stage 2 – Feature sequence concatenation
    The four branch outputs are concatenated along the channel dimension:
    (B, 4 * cnn_out, T')

Stage 3 – Bidirectional LSTM
    The concatenated sequence is treated as a time series of
    (4 * cnn_out)-dimensional feature vectors.
    A 2-layer BiLSTM captures long-range temporal dependencies.
    Only the final hidden state is used for classification.

Stage 4 – Classifier head
    Linear → GELU → Dropout → Linear → num_classes logits
    No Softmax: nn.CrossEntropyLoss applies log-softmax internally.
"""

import logging
import torch
import torch.nn as nn

from ._registry import register_model

logger = logging.getLogger("Model")


class _ChannelCNNBranch(nn.Module):
    """
    Single 1-D CNN branch that processes one EEG channel.

    Layers
    ------
    Conv1 (k=7, stride=1) → BN → GELU → MaxPool(2)   # coarse features
    Conv2 (k=5, stride=1) → BN → GELU → MaxPool(2)   # mid-level features
    Conv3 (k=3, stride=1) → BN → GELU                 # fine features
    Dropout

    Padding keeps the temporal length predictable:
        L_out = L_in // 4   (two MaxPool2 layers)

    Args:
        in_channels  : always 1 (one EEG channel)
        out_channels : number of feature maps per conv layer (= cnn_out)
        dropout_rate : dropout applied after the final conv
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # --- block 1: coarse temporal patterns (theta / alpha) ---
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # --- block 2: mid-level patterns (beta / low-gamma) ---
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # --- block 3: fine-grained high-frequency patterns ---
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),

            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)  →  (B, out_channels, T//4)
        return self.net(x)


@register_model
class CNNLSTMEmognition(nn.Module):
    """
    4-branch 1D CNN + bidirectional LSTM for 4-class emotion recognition
    on the Emognition dataset (MUSE headband: TP9, AF7, AF8, TP10).

    Args:
        num_classes  : number of emotion classes (default 4)
        num_channels : number of EEG channels (default 4 for Emognition)
        cnn_out      : feature maps per CNN branch (default 32)
        lstm_hidden  : hidden units per LSTM direction (default 64)
        lstm_layers  : number of stacked LSTM layers (default 2)
        dropout_rate : dropout rate used throughout (default 0.5)
        **kwargs     : absorbs EEGain pipeline kwargs (input_size, sampling_r, …)
    """

    def __init__(
        self,
        num_classes:  int   = 4,
        num_channels: int   = 4,
        cnn_out:      int   = 32,
        lstm_hidden:  int   = 64,
        lstm_layers:  int   = 2,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        log_info = "\n--".join(
            [f"{n}={v}" for n, v in locals().items()
             if n not in ("self", "__class__", "kwargs")]
        )
        logger.info(f"Using model: \n{self.__class__.__name__}(\n--{log_info})")

        self.num_channels = num_channels
        self.cnn_out      = cnn_out
        self.lstm_hidden  = lstm_hidden

        # --- 4 independent CNN branches (one per EEG channel) ---
        self.branches = nn.ModuleList([
            _ChannelCNNBranch(
                in_channels=1,
                out_channels=cnn_out,
                dropout_rate=dropout_rate * 0.6,   # lighter dropout inside CNN
            )
            for _ in range(num_channels)
        ])

        # --- BiLSTM: input = concatenated branch features per time-step ---
        # After CNN branches: (B, num_channels * cnn_out, T')
        # Reshape to (B, T', num_channels * cnn_out) for LSTM
        lstm_input_size = num_channels * cnn_out
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        # --- Classifier head ---
        # BiLSTM final state: 2 * lstm_hidden (forward + backward)
        lstm_out_size = lstm_hidden * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, lstm_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_hidden, num_classes),
            # NO Softmax — nn.CrossEntropyLoss applies log-softmax internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, C, T)  — standard EEGain 4-D input

        Returns:
            logits: (B, num_classes)
        """
        B, _, C, T = x.shape

        # --- per-channel CNN ---
        # Each branch takes (B, 1, T) and returns (B, cnn_out, T')
        branch_outs = []
        for ch_idx, branch in enumerate(self.branches):
            # Extract channel ch_idx: (B, 1, T)
            x_ch = x[:, 0, ch_idx, :].unsqueeze(1)   # (B, 1, T)
            branch_outs.append(branch(x_ch))           # (B, cnn_out, T')

        # Concatenate along feature dim: (B, C * cnn_out, T')
        features = torch.cat(branch_outs, dim=1)

        # Reshape to (B, T', C * cnn_out) for LSTM
        features = features.permute(0, 2, 1)   # (B, T', C * cnn_out)

        # --- BiLSTM ---
        lstm_out, _ = self.lstm(features)      # (B, T', 2 * lstm_hidden)

        # Use the last time-step's output as the sequence representation
        final = lstm_out[:, -1, :]             # (B, 2 * lstm_hidden)

        # --- classify ---
        return self.classifier(final)          # (B, num_classes)
