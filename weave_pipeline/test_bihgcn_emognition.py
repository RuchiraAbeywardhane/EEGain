"""
test_bihgcn_emognition.py
─────────────────────────
Standalone test script for the BIH-GCN EEG pipeline on the Emognition dataset.

4-class emotion recognition:
    0 = enthusiasm
    1 = neutral
    2 = fear
    3 = sadness

Pipeline:
  1. Load raw EEG  → shape (N, 4, samples)
  2. Segments are 1536 samples (6 s @ 256 Hz) — native Emognition rate
  3. Bandpass 1–50 Hz + 50 Hz notch + per-channel z-score normalisation
  4. STFT spectrogram → (batch, 4, freq_bins, time_frames)
       n_fft=128, hop=32  →  (65 freq × 45 time frames)
  5. CNN encoder     → (batch, 128) per channel
  6. BIH-GCN
       • Local graph  : intra-region GCN + attention pooling
       • Global graph : [global_emb | frontal_emb | temporal_emb] GCN
  7. FC head         → 4-class softmax
  8. Loss            : 0.9 × weighted-CE  +  0.1 × focal loss
  9. Evaluate with accuracy + macro-F1 over 30 random 80/20 splits
     Checkpoint selection uses a held-out VAL set (10% of train trials),
     NOT the test set.

Usage
-----
    python -m weave_pipeline.test_bihgcn_emognition --data_path /path/to/emog
    python -m weave_pipeline.test_bihgcn_emognition --data_path /path/to/emog --subject 22
    python -m weave_pipeline.test_bihgcn_emognition --data_path /path/to/emog --all_subjects --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np

# ── Optional heavy imports (fail fast with a clear message) ──────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    sys.exit(f"[ERROR] PyTorch is required: {e}")

try:
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, butter as _b
except ImportError as e:
    sys.exit(f"[ERROR] SciPy is required: {e}")

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    sys.exit(f"[ERROR] scikit-learn is required: {e}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Attempt to reuse the existing Emognition loader ──────────────────────────
try:
    from weave_pipeline.config  import WEAVEConfig
    from weave_pipeline.dataset import load_data as _weave_load
    _HAS_WEAVE = True
except ImportError:
    _HAS_WEAVE = False

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BIH-GCN.Test")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FS             = 256          # Emognition Muse headset sampling rate (correct)
SEG_SAMPLES    = 1536         # 6 s × 256 Hz  (matches loader's segment_duration=6s)
N_CHANNELS     = 4
N_CLASSES      = 4
EMOTION_NAMES  = ["enthusiasm", "neutral", "fear", "sadness"]

# Electrode → region mapping for Emognition 4-channel cap
# Channels 0,1 → Frontal  |  Channels 2,3 → Temporal
FRONTAL_CH  = [0, 1]
TEMPORAL_CH = [2, 3]

# STFT parameters — chosen to give ~45 time frames from 1536-sample segments:
#   time_frames = (1536 - 128) // 32 + 1 = 45
#   freq_bins   = 128 // 2 + 1          = 65
STFT_N_FFT      = 128
STFT_HOP        = 32
FREQ_BINS       = STFT_N_FFT // 2 + 1   # 65
CNN_EMBED_DIM   = 128
GCN_HIDDEN      = 64


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lo: float = 1.0, hi: float = 50.0,
                    fs: int = FS, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass.  data: (C, T)"""
    nyq  = fs / 2.0
    sos  = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(data: np.ndarray, f0: float = 50.0,
                 fs: int = FS, Q: float = 30.0) -> np.ndarray:
    """Zero-phase notch at f0 Hz.  data: (C, T)"""
    b, a = iirnotch(f0 / (fs / 2.0), Q)
    return filtfilt(b, a, data, axis=-1)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score.  data: (C, T)"""
    mu  = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    return (data - mu) / std


def preprocess(data: np.ndarray) -> np.ndarray:
    """Full preprocessing chain.  data: (C, T) → (C, T)"""
    data = bandpass_filter(data)
    data = notch_filter(data)
    data = zscore_normalize(data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def segment_trial(eeg: np.ndarray, seg_len: int = SEG_SAMPLES,
                  stride: int = SEG_SAMPLES) -> np.ndarray:
    """
    Split (C, T) into non-overlapping windows.
    Returns (n_seg, C, seg_len).
    """
    C, T    = eeg.shape
    n_segs  = (T - seg_len) // stride + 1
    segs    = np.stack(
        [eeg[:, i * stride: i * stride + seg_len] for i in range(n_segs)], axis=0
    )
    return segs  # (n_seg, C, seg_len)


# ─────────────────────────────────────────────────────────────────────────────
#  SPECTROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectrograms(segments: np.ndarray,
                         n_fft: int   = STFT_N_FFT,
                         hop:   int   = STFT_HOP) -> np.ndarray:
    """
    segments : (N, C, T)
    returns  : (N, C, freq_bins, time_frames)   – log-power spectrogram
    """
    N, C, T     = segments.shape
    window      = np.hanning(n_fft)
    freq_bins   = n_fft // 2 + 1
    time_frames = (T - n_fft) // hop + 1

    specs = np.zeros((N, C, freq_bins, time_frames), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            sig = segments[n, c]
            # manual STFT via stride tricks
            frames = np.array([
                sig[t: t + n_fft] * window
                for t in range(0, T - n_fft + 1, hop)
            ])                                          # (time_frames, n_fft)
            fft_out = np.fft.rfft(frames, n=n_fft)     # (time_frames, freq_bins)
            power   = np.abs(fft_out) ** 2             # power spectrum
            log_pow = np.log1p(power).T                # (freq_bins, time_frames)
            specs[n, c] = log_pow[:, :time_frames]

    return specs


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class ChannelCNNEncoder(nn.Module):
    """
    Shared CNN applied independently to each channel's spectrogram.
    Input  : (batch × C, 1, freq_bins, time_frames)
    Output : (batch × C, embed_dim)

    MaxPool is applied along the frequency axis only (kernel (2,1)) so that
    the time dimension – which can be as small as 3 frames with 512-sample
    segments – is never down-sampled and never collapses to 0.
    """
    def __init__(self, freq_bins: int, time_frames: int,
                 embed_dim: int = CNN_EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 1)),   # freq: /2,  time: unchanged
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 1)),   # freq: /4 total, time: unchanged
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),        # global average → (128, 1, 1)
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*C, 1, freq_bins, time_frames)
        h = self.cnn(x).flatten(1)   # (B*C, 128)
        return self.fc(h)             # (B*C, embed_dim)


class GraphConvLayer(nn.Module):
    """Simple spectral GCN layer:  H' = σ( D^{-1} A H W )"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N_nodes, in_dim)
        adj : (N_nodes, N_nodes)  – row-normalised adjacency
        """
        return F.relu(torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1),
                                self.W(x)))


def _row_norm_adj(n: int, device) -> torch.Tensor:
    """Fully-connected self-looped row-normalised adjacency matrix."""
    A   = torch.ones(n, n, device=device)
    deg = A.sum(dim=1, keepdim=True)
    return A / deg


class LocalRegionGCN(nn.Module):
    """
    Intra-region graph: nodes = EEG channel embeddings in the region.
    Outputs a single region embedding via learned attention pooling.
    """
    def __init__(self, in_dim: int, hidden: int = GCN_HIDDEN):
        super().__init__()
        self.gcn     = GraphConvLayer(in_dim, hidden)
        self.attn_w  = nn.Linear(hidden, 1)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        node_feats : (B, n_nodes, in_dim)
        returns    : (B, hidden)  region embedding
        """
        adj  = _row_norm_adj(node_feats.size(1), node_feats.device)
        h    = self.gcn(node_feats, adj)            # (B, n_nodes, hidden)
        attn = F.softmax(self.attn_w(h), dim=1)    # (B, n_nodes, 1)
        return (attn * h).sum(dim=1)                # (B, hidden)


class GlobalBIHGCN(nn.Module):
    """
    Global graph: 3 nodes → [global_emb, frontal_emb, temporal_emb]
    All-connected + self-loops, single GCN pass.
    Output: global node representation (B, hidden).
    """
    def __init__(self, in_dim: int, hidden: int = GCN_HIDDEN):
        super().__init__()
        self.gcn = GraphConvLayer(in_dim, hidden)

    def forward(self, global_emb: torch.Tensor,
                frontal_emb: torch.Tensor,
                temporal_emb: torch.Tensor) -> torch.Tensor:
        """All tensors: (B, in_dim)  →  returns (B, hidden)"""
        nodes = torch.stack([global_emb, frontal_emb, temporal_emb], dim=1)  # (B,3,in_dim)
        adj   = _row_norm_adj(3, nodes.device)
        h     = self.gcn(nodes, adj)    # (B, 3, hidden)
        return h[:, 0, :]               # global node


class BIHGCN(nn.Module):
    """
    Full BIH-GCN model.

    Input  : spectrograms  (B, C, freq_bins, time_frames)
    Output : logits        (B, n_classes)
    """
    def __init__(self, freq_bins: int, time_frames: int,
                 n_classes: int        = N_CLASSES,
                 embed_dim: int        = CNN_EMBED_DIM,
                 gcn_hidden: int       = GCN_HIDDEN,
                 frontal_ch: List[int] = FRONTAL_CH,
                 temporal_ch: List[int]= TEMPORAL_CH,
                 dropout: float        = 0.4):
        super().__init__()
        self.n_channels  = len(frontal_ch) + len(temporal_ch)
        self.frontal_ch  = frontal_ch
        self.temporal_ch = temporal_ch

        # ── Shared CNN encoder ────────────────────────────────────────────────
        self.cnn_enc = ChannelCNNEncoder(freq_bins, time_frames, embed_dim)

        # ── Global average over all channels ─────────────────────────────────
        self.global_proj = nn.Linear(embed_dim, gcn_hidden)

        # ── Local region GCNs ─────────────────────────────────────────────────
        self.frontal_gcn  = LocalRegionGCN(embed_dim, gcn_hidden)
        self.temporal_gcn = LocalRegionGCN(embed_dim, gcn_hidden)

        # ── Global BIH-GCN ────────────────────────────────────────────────────
        self.global_gcn = GlobalBIHGCN(gcn_hidden, gcn_hidden)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, n_classes),
        )

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        """specs: (B, C, F, T)"""
        B, C, Freq, Time = specs.shape

        # ── CNN per channel ───────────────────────────────────────────────────
        x_flat   = specs.view(B * C, 1, Freq, Time)
        ch_embs  = self.cnn_enc(x_flat).view(B, C, -1)   # (B, C, embed_dim)

        # ── Global embedding (mean across channels) ───────────────────────────
        global_emb = F.relu(self.global_proj(ch_embs.mean(dim=1)))  # (B, gcn_hidden)

        # ── Local region embeddings ───────────────────────────────────────────
        frontal_nodes  = ch_embs[:, self.frontal_ch,  :]  # (B, 2, embed_dim)
        temporal_nodes = ch_embs[:, self.temporal_ch, :]  # (B, 2, embed_dim)

        frontal_emb  = self.frontal_gcn(frontal_nodes)    # (B, gcn_hidden)
        temporal_emb = self.temporal_gcn(temporal_nodes)  # (B, gcn_hidden)

        # ── Global BIH-GCN ────────────────────────────────────────────────────
        final_emb = self.global_gcn(global_emb, frontal_emb, temporal_emb)  # (B, gcn_hidden)

        return self.classifier(final_emb)   # (B, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p  = F.log_softmax(logits, dim=-1)
        p      = log_p.exp()
        target_log_p = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
        target_p     = p.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_w = (1 - target_p) ** self.gamma
        loss    = -(focal_w * target_log_p)
        if self.weight is not None:
            cls_w = self.weight[targets]
            loss  = loss * cls_w
        return loss.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  class_weights: torch.Tensor,
                  ce_w: float = 0.9, focal_w: float = 0.1) -> torch.Tensor:
    ce    = F.cross_entropy(logits, targets, weight=class_weights)
    focal = FocalLoss(gamma=2.0, weight=class_weights)(logits, targets)
    return ce_w * ce + focal_w * focal


# ─────────────────────────────────────────────────────────────────────────────
#  DUMMY DATA GENERATOR  (used when real data cannot be loaded)
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_data(n_segments: int = 200,
                    n_channels: int = N_CHANNELS,
                    seg_samples: int = SEG_SAMPLES,
                    n_classes:  int = N_CLASSES,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic EEG for smoke-testing when no real data is available."""
    rng = np.random.default_rng(seed)
    segs  = rng.standard_normal((n_segments, n_channels, seg_samples)).astype(np.float32)
    # roughly balanced labels
    per_cls = n_segments // n_classes
    labels  = np.array(
        [c for c in range(n_classes) for _ in range(per_cls)]
        + list(range(n_segments - per_cls * n_classes))
    )
    rng.shuffle(labels)
    trial_ids = np.repeat(np.arange(n_segments // 4), 4)[:n_segments]
    return segs, labels.astype(np.int64), trial_ids


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING  (reuse existing loader or stub)
# ─────────────────────────────────────────────────────────────────────────────

def _rewindow(segments: np.ndarray, labels: np.ndarray,
              trial_ids: np.ndarray,
              target: int = SEG_SAMPLES) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If the loader returned segments longer than `target` samples, split each
    segment into non-overlapping sub-windows of exactly `target` samples.
    Labels and trial_ids are broadcast to match.
    If segments are already `target` samples wide, returns inputs unchanged.
    """
    N, C, T = segments.shape
    if T == target:
        return segments, labels, trial_ids

    n_sub = T // target          # number of sub-windows per segment
    if n_sub < 1:
        raise ValueError(
            f"Segment length {T} is shorter than target {target}. "
            "Reduce SEG_SAMPLES or check the loader's segment_duration."
        )

    new_segs  = np.zeros((N * n_sub, C, target), dtype=segments.dtype)
    new_labs  = np.zeros(N * n_sub, dtype=labels.dtype)
    new_tids  = np.zeros(N * n_sub, dtype=trial_ids.dtype)

    for i in range(N):
        for j in range(n_sub):
            idx = i * n_sub + j
            new_segs[idx] = segments[i, :, j * target: (j + 1) * target]
            new_labs[idx] = labels[i]
            new_tids[idx] = trial_ids[i]

    logger.info(
        f"Re-windowed {N} × {T}-sample segments → "
        f"{N * n_sub} × {target}-sample segments  (×{n_sub} per original)"
    )
    return new_segs, new_labs, new_tids


def load_emognition_data(data_path: str,
                         subject_id: Optional[str] = None,
                         lead_in: float = 5.0,
                         baseline_dur: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw Emognition EEG.
    Falls back to dummy data if the weave_pipeline loader is unavailable
    or the data path does not exist.
    Returns (segments [N,4,SEG_SAMPLES], labels [N], trial_ids [N]).
    """
    if not _HAS_WEAVE or not os.path.isdir(data_path):
        logger.warning("Real data unavailable – using DUMMY data for smoke-test.")
        return make_dummy_data()

    cfg = WEAVEConfig(
        dataset            = "emognition",
        data_path          = data_path,
        sampling_rate      = FS,                  # ← 256 Hz (correct)
        n_eeg_channels     = N_CHANNELS,
        n_classes          = N_CLASSES,
        class_names        = EMOTION_NAMES,
        segment_duration   = SEG_SAMPLES / FS,    # ← 6.0 s
        lead_in_duration   = lead_in,
        baseline_duration  = baseline_dur,
    )

    try:
        raw_segs, raw_labels, trial_ids = _weave_load(cfg, subject_id=subject_id)
    except Exception as exc:
        logger.warning(f"Loader raised {exc}  →  falling back to DUMMY data.")
        return make_dummy_data()

    # ── Drop any segments whose label index is outside [0, N_CLASSES-1] ──────
    valid_mask = raw_labels < N_CLASSES
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        logger.warning(f"Dropping {n_dropped} segments with out-of-range labels.")
        raw_segs, raw_labels, trial_ids = (
            raw_segs[valid_mask], raw_labels[valid_mask], trial_ids[valid_mask]
        )

    # ── Re-window to SEG_SAMPLES if the loader returned different-length segments ──
    raw_segs, raw_labels, trial_ids = _rewindow(raw_segs, raw_labels, trial_ids)

    # ── Preprocess each segment ───────────────────────────────────────────────
    logger.info("Preprocessing segments …")
    proc = np.zeros_like(raw_segs, dtype=np.float32)
    for i in range(raw_segs.shape[0]):
        proc[i] = preprocess(raw_segs[i].astype(np.float64))

    return proc, raw_labels.astype(np.int64), trial_ids


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, n_classes: int,
                          device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    class_weights: torch.Tensor,
                    device) -> float:
    model.train()
    total_loss = 0.0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss   = combined_loss(logits, labels, class_weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * specs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_true = [], []
    for specs, labels in loader:
        specs = specs.to(device)
        preds = model(specs).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    return acc, f1


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(spectrograms: np.ndarray,
                   labels: np.ndarray,
                   trial_ids: np.ndarray,
                   args: argparse.Namespace,
                   device) -> dict:
    """
    30 × stratified 80/20 train/test splits (trial-level).
    Within each split, 10% of train trials are held out as a validation set
    for checkpoint selection — the test set is NEVER seen during training.
    Returns summary dict with mean/std of acc and F1.
    """
    N, C, Freq, Time = spectrograms.shape

    # ── Trial-level split indices ─────────────────────────────────────────────
    unique_trials = np.unique(trial_ids)
    trial_labels  = np.array([labels[trial_ids == tid][0] for tid in unique_trials])

    # Outer split: 80% train+val trials / 20% test trials
    sss = StratifiedShuffleSplit(
        n_splits     = args.n_reps,
        test_size    = args.test_size,
        random_state = args.seed,
    )
    # Inner split: carve 10% of train trials as val (for checkpoint selection)
    inner_sss = StratifiedShuffleSplit(
        n_splits     = 1,
        test_size    = 0.125,   # 10% of total ≈ 10/80 of train portion
        random_state = args.seed,
    )

    full_accs = []
    full_f1s  = []

    for rep, (tr_tidx, te_tidx) in enumerate(sss.split(unique_trials, trial_labels)):
        tr_tids_all = unique_trials[tr_tidx]
        te_tids     = unique_trials[te_tidx]
        tr_labels_all = trial_labels[tr_tidx]

        # ── Inner val split (from train trials only) ──────────────────────────
        try:
            tr_inner_idx, val_inner_idx = next(
                inner_sss.split(tr_tids_all, tr_labels_all)
            )
            tr_tids  = tr_tids_all[tr_inner_idx]
            val_tids = tr_tids_all[val_inner_idx]
        except ValueError:
            # Fallback: not enough trials per class for inner split — use all train
            tr_tids  = tr_tids_all
            val_tids = tr_tids_all

        tr_mask  = np.isin(trial_ids, tr_tids)
        val_mask = np.isin(trial_ids, val_tids)
        te_mask  = np.isin(trial_ids, te_tids)

        X_tr,  y_tr  = spectrograms[tr_mask],  labels[tr_mask]
        X_val, y_val = spectrograms[val_mask],  labels[val_mask]
        X_te,  y_te  = spectrograms[te_mask],   labels[te_mask]

        cw = compute_class_weights(y_tr, N_CLASSES, device)

        # ── Tensors ───────────────────────────────────────────────────────────
        tr_loader  = DataLoader(
            TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                          torch.tensor(y_tr,  dtype=torch.long)),
            batch_size=args.batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=args.batch_size, shuffle=False,
        )
        te_loader  = DataLoader(
            TensorDataset(torch.tensor(X_te,  dtype=torch.float32),
                          torch.tensor(y_te,  dtype=torch.long)),
            batch_size=args.batch_size, shuffle=False,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = BIHGCN(
            freq_bins   = Freq,
            time_frames = Time,
            n_classes   = N_CLASSES,
            embed_dim   = CNN_EMBED_DIM,
            gcn_hidden  = GCN_HIDDEN,
            dropout     = args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, tr_loader, optimizer, cw, device)
            # ── Checkpoint selection on VAL set (not test set) ────────────────
            val_acc, _ = evaluate(model, val_loader, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 20 == 0 or epoch == args.epochs:
                logger.info(
                    f"  Rep {rep+1:02d}/{args.n_reps}  "
                    f"Epoch {epoch:03d}/{args.epochs}  "
                    f"loss={tr_loss:.4f}  val_acc={val_acc:.4f}"
                )

        # ── Evaluate best checkpoint on the held-out TEST set ─────────────────
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        acc, f1 = evaluate(model, te_loader, device)
        full_accs.append(acc)
        full_f1s.append(f1)

        logger.info(
            f"  ── Rep {rep+1:02d} FINAL  "
            f"test_acc={acc:.4f}  macro-F1={f1:.4f}  "
            f"(best_val_acc={best_val_acc:.4f})"
        )

    summary = {
        "acc_mean"  : float(np.mean(full_accs)),
        "acc_std"   : float(np.std(full_accs)),
        "f1_mean"   : float(np.mean(full_f1s)),
        "f1_std"    : float(np.std(full_f1s)),
        "all_accs"  : full_accs,
        "all_f1s"   : full_f1s,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_results(summary: dict, path: str, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "=" * 60,
        "  BIH-GCN  –  Emognition  –  5-class Emotion",
        "=" * 60,
        f"  Dataset    : Emognition",
        f"  Subject    : {args.subject or 'ALL'}",
        f"  Emotions   : {EMOTION_NAMES}",
        f"  Epochs     : {args.epochs}",
        f"  Batch size : {args.batch_size}",
        f"  LR         : {args.lr}",
        f"  Dropout    : {args.dropout}",
        f"  Reps       : {args.n_reps}",
        f"  Test size  : {args.test_size}",
        "-" * 60,
        f"  Accuracy   : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}",
        f"  Macro-F1   : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
        "-" * 60,
        "  Per-rep accuracies:",
    ]
    for i, (a, f) in enumerate(zip(summary["all_accs"], summary["all_f1s"]), 1):
        lines.append(f"    Rep {i:02d}:  acc={a:.4f}  f1={f:.4f}")
    lines.append("=" * 60)

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info(f"Results saved → {path}")

    # Also print to console
    for ln in lines:
        logger.info(ln)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIH-GCN – Emognition 5-class emotion recognition",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--data_path",   default="DUMMY",
                   help="Path to Emognition dataset root. "
                        "Use 'DUMMY' to run on synthetic data.")
    p.add_argument("--subject",     default=None,
                   help="Single subject ID (string).  Omit = all subjects.")
    p.add_argument("--all_subjects", action="store_true",
                   help="Pool all subjects (default when --subject omitted).")
    p.add_argument("--lead_in",     default=5.0,  type=float)
    p.add_argument("--baseline_dur",default=3.0,  type=float)
    # Training
    p.add_argument("--epochs",      default=50,   type=int)
    p.add_argument("--batch_size",  default=16,   type=int)
    p.add_argument("--lr",          default=1e-4, type=float)
    p.add_argument("--dropout",     default=0.4,  type=float)
    # Evaluation
    p.add_argument("--n_reps",      default=30,   type=int,
                   help="Random train/test repetitions (paper: 30)")
    p.add_argument("--test_size",   default=0.2,  type=float)
    p.add_argument("--seed",        default=42,   type=int)
    # Output
    p.add_argument("--log_dir",     default="weave_logs/",
                   help="Directory to write result files")
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  BIH-GCN  –  Emognition  –  5-class Emotion")
    logger.info("=" * 60)
    logger.info(f"  Device       : {device}")
    logger.info(f"  Classes      : {EMOTION_NAMES}")
    logger.info(f"  Segment      : {SEG_SAMPLES} samples ({SEG_SAMPLES/FS:.1f}s @ {FS}Hz)")
    logger.info(f"  Spectrogram  : n_fft={STFT_N_FFT}  hop={STFT_HOP}  "
                f"→ ({FREQ_BINS} freq × time_frames)")
    logger.info(f"  CNN embed    : {CNN_EMBED_DIM}-dim   GCN hidden: {GCN_HIDDEN}")
    logger.info(f"  Regions      : Frontal={FRONTAL_CH}  Temporal={TEMPORAL_CH}")
    logger.info(f"  Epochs       : {args.epochs}  |  batch={args.batch_size}  "
                f"|  lr={args.lr}")
    logger.info(f"  Reps         : {args.n_reps}  |  test_size={args.test_size}")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    subject_id = args.subject  # None → all subjects
    segments, labels, trial_ids = load_emognition_data(
        args.data_path, subject_id,
        lead_in      = args.lead_in,
        baseline_dur = args.baseline_dur,
    )

    logger.info(
        f"Loaded  {segments.shape[0]} segments  |  shape {segments.shape}  |  "
        f"class dist {np.bincount(labels, minlength=N_CLASSES).tolist()}"
    )

    # Guard: need enough trial-level data
    unique_tids  = np.unique(trial_ids)
    tid_labels   = np.array([labels[trial_ids == t][0] for t in unique_tids])
    min_per_cls  = min(
        int((tid_labels == c).sum()) for c in range(N_CLASSES)
        if (tid_labels == c).sum() > 0
    )
    if min_per_cls < 2:
        logger.error(
            f"Only {min_per_cls} trial(s) in at least one class — "
            "not enough for stratified splitting. "
            "Use --all_subjects or check your data."
        )
        return

    # ── Compute spectrograms ──────────────────────────────────────────────────
    logger.info("Computing STFT spectrograms …")
    specs = compute_spectrograms(segments)
    logger.info(f"Spectrogram tensor shape: {specs.shape}")   # (N, 4, 129, time_frames)

    # ── Run evaluation ────────────────────────────────────────────────────────
    summary = run_evaluation(specs, labels, trial_ids, args, device)

    # ── Save results ──────────────────────────────────────────────────────────
    subj_tag = f"subject_{args.subject}" if args.subject else "all_subjects"
    log_path = os.path.join(args.log_dir, f"bihgcn_emognition_{subj_tag}.txt")
    save_results(summary, log_path, args)

    logger.info("=" * 60)
    logger.info(f"  FINAL  Accuracy : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    logger.info(f"  FINAL  Macro-F1 : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    logger.info(f"  (Chance level for 4 classes = 0.25)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
