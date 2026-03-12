"""
Configuration for the Parallel CNN-LSTM (PCRNN) pipeline on DEAP.
All hyper-parameters from the paper are defined here as a single dataclass.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PCRConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path: str = ""                  # path to DEAP .dat files
    label_type: str = "V"               # "V" = Valence, "A" = Arousal
    n_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5  # inclusive: <= low, > high

    # ── Signal properties ─────────────────────────────────────────────────────
    sampling_rate: int = 128             # Hz
    n_eeg_channels: int = 32            # EEG channels only (drop peripherals)
    trial_duration: int = 60            # seconds of EEG per trial
    baseline_duration: int = 3          # seconds of pre-stimulus baseline
    n_trials: int = 40                  # trials per subject

    # ── Baseline removal ──────────────────────────────────────────────────────
    baseline_segment_len: int = 128     # samples per baseline segment (L=128 → 1 s)

    # ── 2-D spatial frame ─────────────────────────────────────────────────────
    grid_size: int = 9                  # 9×9 scalp topography grid

    # ── Windowing ─────────────────────────────────────────────────────────────
    window_size: int = 128              # samples per window  (1 s @ 128 Hz)
    window_step: int = 128             # non-overlapping windows

    # ── CNN branch ────────────────────────────────────────────────────────────
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel: int = 4                 # 4×4 convolution
    cnn_reduce_filters: int = 13        # 1×1 conv to fuse spatial features

    # ── LSTM branch ───────────────────────────────────────────────────────────
    lstm_hidden: int = 32
    lstm_layers: int = 2

    # ── Joint classifier ──────────────────────────────────────────────────────
    dropout: float = 0.5

    # ── Training ─────────────────────────────────────────────────────────────
    n_folds: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    early_stopping_patience: int = 7
    log_dir: str = "pcr_logs/"
    seed: int = 42
