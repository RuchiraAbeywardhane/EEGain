"""
Configuration for the EEG-only BIH_GCN pipeline.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BIHGCNConfig:

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset: str         = "deap"          # "deap" | "emognition"
    data_path: str       = ""
    label_type: str      = "V"             # "V"=Valence | "A"=Arousal (DEAP)
    ground_truth_threshold: float = 5.0
    lead_in_duration: float  = 5.0
    baseline_duration: float = 3.0

    # ── Signal ────────────────────────────────────────────────────────────────
    sampling_rate: int   = 128             # Hz  (128 DEAP, 256 Emognition)
    n_eeg_channels: int  = 32
    segment_duration: float = 6.0         # seconds
    n_classes: int       = 2
    class_names: List[str] = field(default_factory=lambda: ["Low", "High"])

    # ── Spectrogram ───────────────────────────────────────────────────────────
    stft_n_fft: int      = 64             # smaller FFT → fewer freq bins → faster
    stft_hop: int        = 16
    stft_n_mels: int     = 32             # must be ≤ n_fft//2 + 1 = 33

    # ── Mamba Encoder ─────────────────────────────────────────────────────────
    mamba_d_model: int   = 64             # reduced from 128 – less overfit on small data
    mamba_n_layers: int  = 2              # reduced from 4
    mamba_d_state: int   = 16
    mamba_d_conv: int    = 4
    mamba_expand: int    = 2

    # ── Projection ────────────────────────────────────────────────────────────
    proj_dim: int        = 64             # reduced from 128

    # ── Anatomical Brain Regions ──────────────────────────────────────────────
    # Keys = region names, values = 0-based channel indices (DEAP 32-ch layout)
    brain_regions: dict = field(default_factory=lambda: {
        "frontal"   : [0, 1, 2, 3, 16, 17, 18, 19],
        "temporal"  : [4, 5, 10, 11, 20, 21, 26, 27],
        "parietal"  : [6, 7, 8, 9, 22, 23, 24, 25],
        "occipital" : [12, 13, 14, 15, 28, 29, 30, 31],
    })

    # ── BIH-GCN Stage 1 (local, per-region) ──────────────────────────────────
    gcn1_hidden: int     = 64             # reduced from 128
    gcn1_out: int        = 64
    gcn1_heads: int      = 2              # reduced from 4

    # ── BIH-GCN Stage 2 (global, cross-region) ────────────────────────────────
    gcn2_hidden: int     = 64             # reduced from 128
    gcn2_out: int        = 64

    # ── Classifier ────────────────────────────────────────────────────────────
    clf_dropout: float   = 0.4

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int      = 16             # smaller batch → more gradient updates
    lr: float            = 3e-4           # reduced from 1e-3
    weight_decay: float  = 1e-3           # stronger regularisation
    epochs: int          = 150            # more epochs with early stopping
    patience: int        = 20             # early stopping patience
    seed: int            = 42
    test_size: float     = 0.2
    n_repetitions: int   = 10

    # ── Output ────────────────────────────────────────────────────────────────
    log_dir: str         = "BIH_GCN_logs/"

    # ── Derived (filled at runtime) ───────────────────────────────────────────
    @property
    def segment_samples(self) -> int:
        return int(self.segment_duration * self.sampling_rate)

    @property
    def n_regions(self) -> int:
        return len(self.brain_regions)
