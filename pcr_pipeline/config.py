"""
Configuration for the Parallel CNN-LSTM (PCRNN) pipeline.
Supports both DEAP and Emognition datasets.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PCRConfig:
    # ── Dataset selection ─────────────────────────────────────────────────────
    dataset: str = "deap"                # "deap" or "emognition"
    data_path: str = ""

    # ── DEAP-specific labels ──────────────────────────────────────────────────
    label_type: str = "V"                # "V" = Valence, "A" = Arousal (DEAP only)
    ground_truth_threshold: float = 4.5  # binary split threshold (DEAP only)

    # ── Emognition-specific labels ────────────────────────────────────────────
    # Only these 4 emotions are used. Files for other emotions are ignored.
    #   0 = ENTHUSIASM
    #   1 = NEUTRAL
    #   2 = FEAR
    #   3 = SADNESS
    emognition_class_map: dict = field(default_factory=lambda: {
        "ENTHUSIASM": 0,
        "NEUTRAL":    1,
        "FEAR":       2,
        "SADNESS":    3,
    })

    # ── Shared label config (auto-set by dataset loader) ─────────────────────
    n_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["enthusiasm", "neutral", "fear", "sadness"])

    # ── Signal properties ─────────────────────────────────────────────────────
    sampling_rate: int = 128             # Hz  — 128 for DEAP, 256 for Emognition
    n_eeg_channels: int = 32            # 32 for DEAP, 4 for Emognition
    n_trials: int = 40                  # trials per subject (DEAP=40, Emognition varies)

    # ── Emognition trial trimming ─────────────────────────────────────────────
    # The first `lead_in_duration` seconds of every Emognition clip are
    # emotionally neutral (video not yet started / fade-in).  They are
    # discarded BEFORE baseline extraction.
    # Change this value if you want a shorter/longer lead-in trim.
    lead_in_duration: float = 5.0       # seconds to discard from clip start

    # ── Baseline removal ──────────────────────────────────────────────────────
    # For DEAP  : baseline is the 3 s pre-stimulus segment already in the file.
    # For Emognition: baseline = the `baseline_duration` seconds that immediately
    #                 follow the lead-in trim (i.e. seconds 5–8 of the raw clip).
    baseline_duration: float = 3.0      # seconds used as baseline
    baseline_segment_len: int = 128     # samples per averaging segment (1 s @ 128 Hz)
                                        # auto-scaled to sampling_rate in the loader

    # ── 2-D spatial frame (DEAP / 32-ch only) ────────────────────────────────
    grid_size: int = 9                  # 9×9 scalp topography grid

    # ── Windowing ─────────────────────────────────────────────────────────────
    window_size: int = 128              # samples per window  (1 s @ 128 Hz)
    window_step: int = 128             # stride between windows (128 = no overlap)

    # ── CNN branch ────────────────────────────────────────────────────────────
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel: int = 4                 # 4×4 for 2D (DEAP); kernel length for 1D (Emognition)
    cnn_reduce_filters: int = 13        # 1×1 fusion filters (2D path only)

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

    # ── Emognition quality filter ─────────────────────────────────────────────
    emognition_use_baseline_reduction: bool = False  # spectral InvBase reduction
    emognition_min_trial_seconds: float = 10.0       # skip trials shorter than this
