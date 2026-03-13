"""
Configuration for the WEAVE + SVM pipeline.

WEAVE = Wavelet Entropy and AVErage wavelet coefficient.

Supports:
    dataset="deap"       – 32 channels, 128 Hz, binary valence / arousal
    dataset="emognition" – 4 channels,  256 Hz, 4-class emotion
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class WEAVEConfig:
    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset: str = "deap"               # "deap" | "emognition"
    data_path: str = ""

    # ── DEAP-specific ──────────────────────────────────────────────────────────
    label_type: str = "V"               # "V"=Valence  "A"=Arousal
    ground_truth_threshold: float = 5.0 # paper uses >5 → High, ≤5 → Low
    n_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ["Low", "High"])

    # ── Emognition-specific ────────────────────────────────────────────────────
    emognition_class_map: dict = field(default_factory=lambda: {
        "ENTHUSIASM": 0,
        "NEUTRAL":    1,
        "FEAR":       2,
        "SADNESS":    3,
    })
    # lead-in seconds to discard (emotionally-neutral fade-in)
    lead_in_duration: float  = 5.0
    baseline_duration: float = 3.0

    # ── Signal properties (auto-set by dataset) ────────────────────────────────
    sampling_rate: int   = 128          # Hz  – 128 DEAP, 256 Emognition
    n_eeg_channels: int  = 32           # 32 DEAP, 4 Emognition

    # ── Segmentation ──────────────────────────────────────────────────────────
    # Paper: 6-second non-overlapping windows
    segment_duration: float = 6.0       # seconds per segment
    # segment_samples is derived: int(segment_duration * sampling_rate)

    # ── DWT feature extraction ────────────────────────────────────────────────
    wavelet: str = "db5"                # Daubechies-5

    # Frequency bands to RETAIN (Hz).
    # The correct DWT level for each band is auto-computed from sampling_rate.
    #   DEAP  (128 Hz): alpha=9-16, beta=17-32, gamma=33-64  → levels 3,2,1
    #   Emog  (256 Hz): same Hz ranges                       → levels 4,3,2
    retained_bands: List[str] = field(default_factory=lambda: ["alpha", "beta", "gamma"])

    # Band definitions in Hz (used to select DWT levels automatically)
    band_hz: dict = field(default_factory=lambda: {
        "delta": (0.5,  4.0),
        "theta": (4.0,  8.0),
        "alpha": (8.0, 16.0),
        "beta":  (16.0, 32.0),
        "gamma": (32.0, 64.0),
    })

    # ── Channel selection ──────────────────────────────────────────────────────
    # NMI-based iterative reduction: 32 → 16 (DEAP) / 4 → 2 (Emognition)
    min_channels: int = 16             # stop reduction here
    # For Emognition (4 ch) we override this to 2 in run.py

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm_kernel: str   = "rbf"
    svm_C: float      = 1.0
    svm_gamma: str    = "scale"        # scikit-learn default

    # ── Evaluation ────────────────────────────────────────────────────────────
    n_repetitions: int  = 30           # paper: 30 random train/test splits
    test_size: float    = 0.2          # 80/20 split per repetition
    seed: int           = 42

    # ── Output ────────────────────────────────────────────────────────────────
    log_dir: str = "weave_logs/"
