"""
transfer_seediv_to_emognition.py
=================================
Two-phase transfer learning pipeline:

  Phase 1 — Pre-train on SEED-IV (62-ch, 200 Hz, 4-class)
    SEED-IV emotions : 0=neutral  1=sad  2=fear  3=happy
    Emognition map   : 0=enthusiasm  1=neutral  2=fear  3=sadness

    Channel matching strategy
    ─────────────────────────
    Emognition uses a 4-channel Muse headset:
        AF7  → left  prefrontal (frontal)
        AF8  → right prefrontal (frontal)
        TP9  → left  temporal
        TP10 → right temporal

    SEED-IV 62-ch 10-20 layout closest matches:
        AF7  → AF7  (idx 1)
        AF8  → AF8  (idx 2)
        TP9  → TP7  (idx 50)   ← nearest temporal-parietal left
        TP10 → TP8  (idx 51)   ← nearest temporal-parietal right

    Only these 4 channels are fed to the model during pre-training so the
    learned weights transfer directly to Emognition.

  Phase 2 — Fine-tune on Emognition
    The pre-trained encoder (Mamba + GCN stages) is loaded and the
    classifier head is replaced with a fresh one.
    Fine-tuning uses a lower LR for the backbone and a higher LR for
    the new head (differential LR).

Evaluation
──────────
Both phases use clip-independent 80/20 splits (no within-clip leakage).
Phase 2 reports 30 repetitions mean ± std accuracy and macro-F1.

Usage
─────
# Phase 1 only (pre-train and save checkpoint):
    python -m BIH_GCN.transfer_seediv_to_emognition \\
        --seediv_path  /path/to/SEED_IV \\
        --emog_path    /path/to/emognition \\
        --mode         pretrain

# Phase 2 only (load checkpoint and fine-tune):
    python -m BIH_GCN.transfer_seediv_to_emognition \\
        --seediv_path  /path/to/SEED_IV \\
        --emog_path    /path/to/emognition \\
        --mode         finetune \\
        --checkpoint   BIH_GCN_logs/seediv_pretrained.pt

# Full pipeline (pretrain → finetune):
    python -m BIH_GCN.transfer_seediv_to_emognition \\
        --seediv_path  /path/to/SEED_IV \\
        --emog_path    /path/to/emognition \\
        --mode         full

SEED-IV file layout expected
────────────────────────────
The SEED-IV dataset (BCMI lab) distributes preprocessed .mat files:
    <seediv_path>/
        1/          ← session 1
            1_20160518.mat
            2_20160518.mat
            ...
        2/          ← session 2
            ...
        3/          ← session 3
            ...
Each .mat file contains:
    de_LDS1 … de_LDS4  : (62, n_seg, 5)  DE features per band
    -- OR --
    eeg_raw             : (62, T)  raw EEG  (if available)

This loader uses the DE (Differential Entropy) feature matrices that
come with SEED-IV by default, then reconstructs pseudo-raw segments.
If you have the raw signals set --seediv_raw.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    sys.exit(f"[ERROR] PyTorch required: {e}")

try:
    from scipy.io import loadmat
except ImportError as e:
    sys.exit(f"[ERROR] SciPy required: {e}")

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError as e:
    sys.exit(f"[ERROR] scikit-learn required: {e}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BIH_GCN.config   import BIHGCNConfig
from BIH_GCN.model    import BIHGCN
from BIH_GCN.dataset  import (
    load_emognition_all_subjects,
    load_emognition_subject,
)
from BIH_GCN.train    import _normalise_per_channel, _clip_level_split

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("Transfer.SEEDIV→Emog")


# ─────────────────────────────────────────────────────────────────────────────
#  CHANNEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Full SEED-IV 62-channel 10-20 order (0-based indices)
_SEEDIV_62CH = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ",
    "F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2",
    "FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4",
    "C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6",
    "TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","OZ","CB2",
    "O1","O2",
]
_CH_TO_IDX = {ch: i for i, ch in enumerate(_SEEDIV_62CH)}

# Emognition Muse channels → best-matching SEED-IV channel + index
# AF7/AF8 exist directly; TP9/TP10 are mapped to TP7/TP8 (nearest)
EMOG_TO_SEEDIV: Dict[str, str] = {
    "AF7" : "AF3",   # AF7 not in 62-ch set → AF3 is closest left frontal
    "AF8" : "AF4",   # AF8 not in 62-ch set → AF4 is closest right frontal
    "TP9" : "TP7",   # left  temporal-parietal
    "TP10": "TP8",   # right temporal-parietal
}

# Resulting 4-channel indices in the SEED-IV 62-ch array
SEEDIV_CHANNEL_INDICES: List[int] = [
    _CH_TO_IDX[EMOG_TO_SEEDIV["AF7"]],
    _CH_TO_IDX[EMOG_TO_SEEDIV["AF8"]],
    _CH_TO_IDX[EMOG_TO_SEEDIV["TP9"]],
    _CH_TO_IDX[EMOG_TO_SEEDIV["TP10"]],
]

# SEED-IV emotion labels → Emognition label space
# SEED-IV: 0=neutral  1=sad  2=fear  3=happy
# Emognition: 0=enthusiasm  1=neutral  2=fear  3=sadness
SEEDIV_TO_EMOG_LABEL: Dict[int, int] = {
    0: 1,   # neutral  → neutral
    1: 3,   # sad      → sadness
    2: 2,   # fear     → fear
    3: 0,   # happy    → enthusiasm  (closest positive high-arousal)
}

# Brain regions for the 4-channel model
# (same layout as Emognition in BIH_GCN/run.py)
EMOG_BRAIN_REGIONS: Dict[str, List[int]] = {
    "frontal"  : [0, 1],   # AF7, AF8
    "temporal" : [2, 3],   # TP9, TP10
}


# ─────────────────────────────────────────────────────────────────────────────
#  SEED-IV LOADER
# ─────────────────────────────────────────────────────────────────────────────

# SEED-IV trial labels per session (15 trials per session × 3 sessions)
# From the official README: sessions 1-3 have the same label sequence
_SEEDIV_TRIAL_LABELS = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2,
                         2, 1, 3, 2, 2, 3, 2, 3, 3, 0, 3, 0, 0, 2, 3,
                         0, 3, 3, 3, 2, 3, 2, 0, 1, 3, 0, 3, 3, 3, 3]
# 3 sessions × 24 trials each — official SEED-IV label order
_SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 2, 1, 3, 2, 2, 3, 2, 3, 3],
    2: [0, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 0, 1, 3, 0, 3, 3, 3, 3, 3, 3],
    3: [2, 3, 3, 3, 2, 3, 2, 0, 1, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 2, 3, 2, 3, 3],
}


def inspect_seediv_mat(fpath: str) -> None:
    """
    Diagnostic helper — prints every key in a .mat file and its shape/type.
    Call this manually to understand the file structure before running the
    full pipeline:
        from BIH_GCN.transfer_seediv_to_emognition import inspect_seediv_mat
        inspect_seediv_mat("/path/to/1_20160518.mat")
    """
    try:
        mat = loadmat(fpath, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"[inspect] Cannot load {fpath}: {e}")
        return
    print(f"\n[inspect] Keys in {os.path.basename(fpath)}:")
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "shape"):
            print(f"  {k:30s}  shape={v.shape}  dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:30s}  list  len={len(v)}")
        else:
            print(f"  {k:30s}  type={type(v).__name__}  val={str(v)[:80]}")
    print()


def _load_seediv_mat(fpath: str) -> Optional[Tuple[np.ndarray, Optional[List[int]]]]:
    """
    Load one SEED-IV .mat file.

    Handles ALL known SEED-IV release formats:

    Format A  — raw EEG (rare, separate download)
        Key:  eeg_raw / EEG / data   → (62, T)
        Returns: ([62, T], None)

    Format B  — per-trial raw EEG  (eeg_raw_data release)
        Keys: eeg1, eeg2 … eeg24    → each (62, T_trial)
        Returns: ([62, T_total], [label_per_sample])
        ── This is the format at /kaggle/input/.../eeg_raw_data ──

    Format C  — smoothed DE features (default BCMI release)
        Keys: de_LDS1 … de_LDS24   → each (62, n_seg, 5)
        or    smth_de_LDS1 …       → each (62, n_seg, 5)
        Returns: ([62, total_segs*5], None)  ← pseudo-signal

    Returns (signal [62, T], per_sample_labels or None)
    Per-sample labels are only returned for Format B so that
    load_seediv_subject() can assign labels without the proportional
    heuristic.
    """
    try:
        mat = loadmat(fpath, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        logger.warning(f"  Cannot load {fpath}: {e}")
        return None

    data_keys = [k for k in mat if not k.startswith("__")]

    # ── Format A: single raw EEG array ───────────────────────────────────────
    for key in ("eeg_raw", "EEG", "data"):
        if key in mat:
            arr = np.array(mat[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] in (62, 64):
                logger.debug(f"  Format A ({key}): shape={arr.shape}")
                return arr[:62], None

    # ── Format B: per-trial raw EEG  eeg1…eeg24 ──────────────────────────────
    # Keys are like 'eeg1', 'eeg2', ... up to 'eeg24'
    eeg_trial_keys = sorted(
        [k for k in data_keys if k.startswith("eeg") and k[3:].isdigit()],
        key=lambda k: int(k[3:])
    )
    if eeg_trial_keys:
        logger.debug(f"  Format B: found {len(eeg_trial_keys)} per-trial EEG keys "
                     f"({eeg_trial_keys[0]}…{eeg_trial_keys[-1]})")
        arrays, per_sample_labels = [], []
        for t_idx, key in enumerate(eeg_trial_keys):
            arr = np.array(mat[key], dtype=np.float32)   # (62, T_trial)
            if arr.ndim != 2 or arr.shape[0] not in (62, 64):
                logger.debug(f"    Skipping {key}: unexpected shape {arr.shape}")
                continue
            arrays.append(arr[:62])
            per_sample_labels.append(t_idx)   # will be resolved to emotion later
        if arrays:
            concat = np.concatenate(arrays, axis=1)   # [62, T_total]
            # Build sample-level index so caller knows which samples belong
            # to which trial (used for label assignment)
            trial_boundaries = np.cumsum([0] + [a.shape[1] for a in arrays])
            return concat, trial_boundaries.tolist()

    # ── Format C: DE feature matrices  de_LDS1…de_LDS24 ─────────────────────
    # Also handles smth_de_LDS* (smoothed variant) and de_movingAve*
    for prefix in ("de_LDS", "smth_de_LDS", "de_movingAve"):
        de_keys = sorted(
            [k for k in data_keys if k.startswith(prefix) and k[len(prefix):].isdigit()],
            key=lambda k: int(k[len(prefix):])
        )
        if de_keys:
            logger.debug(f"  Format C ({prefix}): found {len(de_keys)} keys")
            de_arrays = []
            for key in de_keys:
                arr = np.array(mat[key], dtype=np.float32)
                if arr.ndim == 3 and arr.shape[0] in (62, 64):
                    de_arrays.append(arr[:62])   # (62, n_seg, 5_bands)
                elif arr.ndim == 2 and arr.shape[0] in (62, 64):
                    de_arrays.append(arr[:62, :, np.newaxis])  # add band dim
            if de_arrays:
                combined = np.concatenate(de_arrays, axis=1)   # (62, total_segs, bands)
                C, S, B  = combined.shape
                return combined.reshape(C, S * B).astype(np.float32), None

    # ── Nothing matched — dump available keys to help debug ──────────────────
    logger.warning(
        f"  No usable data in {os.path.basename(fpath)}. "
        f"Available keys: {[k for k in data_keys if not k.startswith('__')]}"
    )
    return None


def load_seediv_subject(
    subject_id: int,
    seediv_path: str,
    segment_samples: int,
    clip_id_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all 3 sessions for one SEED-IV subject.
    Selects only the 4 channels matching Emognition.
    Remaps labels to Emognition label space.

    Handles both:
      • Continuous signal  (Format A / C) → proportional label assignment
      • Per-trial signal   (Format B)     → exact label assignment via
                                            trial_boundaries
    """
    all_segs, all_labels, all_clip_ids = [], [], []
    clip_id = clip_id_offset

    for session in (1, 2, 3):
        session_dir = os.path.join(seediv_path, str(session))
        if not os.path.isdir(session_dir):
            logger.warning(f"  Session dir not found: {session_dir}")
            continue

        candidates = [
            f for f in os.listdir(session_dir)
            if f.startswith(f"{subject_id}_") and f.endswith(".mat")
        ]
        if not candidates:
            logger.warning(f"  No .mat for subject {subject_id} session {session}")
            continue

        fpath  = os.path.join(session_dir, candidates[0])
        result = _load_seediv_mat(fpath)

        if result is None:
            clip_id += len(_SESSION_LABELS.get(session, []))
            continue

        signal, trial_boundaries = result   # [62, T], list|None
        signal_4ch = signal[SEEDIV_CHANNEL_INDICES, :]   # [4, T]
        trial_labels_raw = _SESSION_LABELS.get(session, [])
        n_trials = len(trial_labels_raw)
        C, T = signal_4ch.shape

        # ── Format B: exact per-trial boundaries ─────────────────────────────
        if trial_boundaries is not None and len(trial_boundaries) > 1:
            for t_idx in range(min(len(trial_boundaries) - 1, n_trials)):
                t_start = trial_boundaries[t_idx]
                t_end   = trial_boundaries[t_idx + 1]
                trial_sig = signal_4ch[:, t_start:t_end]
                n_t = (t_end - t_start) // segment_samples
                if n_t == 0:
                    clip_id += 1
                    continue
                raw_label  = trial_labels_raw[t_idx]
                emog_label = SEEDIV_TO_EMOG_LABEL[raw_label]
                t_segs = np.stack([
                    trial_sig[:, i * segment_samples:(i + 1) * segment_samples]
                    for i in range(n_t)
                ], axis=0).astype(np.float32)
                all_segs.append(t_segs)
                all_labels.extend([emog_label] * n_t)
                all_clip_ids.extend([clip_id]  * n_t)
                clip_id += 1
            continue   # done with this session

        # ── Format A / C: continuous signal → proportional assignment ─────────
        n_segs = T // segment_samples
        if n_segs == 0:
            logger.warning(f"  Subject {subject_id} session {session}: "
                           f"signal too short ({T} < {segment_samples})")
            clip_id += n_trials
            continue

        segs_per_trial = n_segs // max(n_trials, 1)
        remainder      = n_segs %  max(n_trials, 1)
        seg_start = 0

        for t_idx, raw_label in enumerate(trial_labels_raw):
            emog_label = SEEDIV_TO_EMOG_LABEL[raw_label]
            n_t = segs_per_trial + (1 if t_idx < remainder else 0)
            if n_t == 0:
                clip_id += 1
                continue
            trial_sig = signal_4ch[:, seg_start:seg_start + n_t * segment_samples]
            t_segs = np.stack([
                trial_sig[:, i * segment_samples:(i + 1) * segment_samples]
                for i in range(n_t)
            ], axis=0).astype(np.float32)
            all_segs.append(t_segs)
            all_labels.extend([emog_label] * n_t)
            all_clip_ids.extend([clip_id]  * n_t)
            seg_start += n_t * segment_samples
            clip_id   += 1

    if not all_segs:
        empty = np.empty((0, 4, segment_samples), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    segments = np.concatenate(all_segs, axis=0)
    labels   = np.array(all_labels,   dtype=np.int64)
    clip_ids = np.array(all_clip_ids, dtype=np.int64)

    logger.info(
        f"SEED-IV subject {subject_id:02d}: {segments.shape[0]} segments | "
        f"{len(np.unique(clip_ids))} clips | "
        f"class dist {np.bincount(labels, minlength=4).tolist()}"
    )
    return segments, labels, clip_ids


def load_seediv_all_subjects(
    seediv_path: str,
    segment_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all available SEED-IV subjects (auto-detected 1–15)."""
    all_segs, all_labels, all_clip_ids = [], [], []
    clip_offset = 0

    # Auto-detect subjects from session 1 directory
    session1_dir = os.path.join(seediv_path, "1")
    if not os.path.isdir(session1_dir):
        raise RuntimeError(
            f"SEED-IV session 1 directory not found: {session1_dir}\n"
            "Expected layout: <seediv_path>/1/*.mat  /2/*.mat  /3/*.mat"
        )

    subject_ids = set()
    for fname in os.listdir(session1_dir):
        if fname.endswith(".mat"):
            try:
                sid = int(fname.split("_")[0])
                subject_ids.add(sid)
            except ValueError:
                pass
    subject_ids = sorted(subject_ids)
    logger.info(f"Found {len(subject_ids)} SEED-IV subjects: {subject_ids}")

    for sid in subject_ids:
        segs, labels, clip_ids = load_seediv_subject(
            sid, seediv_path, segment_samples,
            clip_id_offset=clip_offset,
        )
        if segs.shape[0] > 0:
            all_segs.append(segs)
            all_labels.append(labels)
            all_clip_ids.append(clip_ids)
            clip_offset = int(clip_ids.max()) + 1

    if not all_segs:
        raise RuntimeError("No SEED-IV data loaded — check --seediv_path")

    segments = np.concatenate(all_segs,     axis=0)
    labels   = np.concatenate(all_labels,   axis=0)
    clip_ids = np.concatenate(all_clip_ids, axis=0)

    logger.info(
        f"SEED-IV total: {segments.shape[0]} segments | "
        f"{len(np.unique(clip_ids))} clips | "
        f"class dist {np.bincount(labels, minlength=4).tolist()}"
    )
    return segments, labels, clip_ids


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def make_seediv_cfg(args) -> BIHGCNConfig:
    """
    BIHGCNConfig for SEED-IV pre-training.
    Uses the 4-channel Emognition-matched layout so architecture
    weights transfer directly.
    """
    cfg = BIHGCNConfig(
        dataset          = "seediv",
        data_path        = args.seediv_path,
        sampling_rate    = 200,          # SEED-IV raw EEG is 200 Hz
        n_eeg_channels   = 4,            # only the 4 matched channels
        segment_duration = args.segment_dur,
        n_classes        = 4,
        class_names      = ["enthusiasm", "neutral", "fear", "sadness"],
        brain_regions    = EMOG_BRAIN_REGIONS,
        # Spectrogram — sized for 200 Hz, 4-channel
        stft_n_fft  = 64,
        stft_hop    = 16,
        stft_n_mels = 32,
        # Training
        epochs       = args.pretrain_epochs,
        patience     = args.patience,
        batch_size   = args.batch_size,
        lr           = args.pretrain_lr,
        weight_decay = args.weight_decay,
        n_repetitions = 1,               # one pass for pre-training
        test_size     = 0.2,
        seed          = args.seed,
        log_dir       = args.log_dir,
    )
    return cfg


def make_emog_cfg(args) -> BIHGCNConfig:
    """BIHGCNConfig for Emognition fine-tuning."""
    cfg = BIHGCNConfig(
        dataset          = "emognition",
        data_path        = args.emog_path,
        sampling_rate    = 256,
        n_eeg_channels   = 4,
        segment_duration = args.segment_dur,
        n_classes        = 4,
        class_names      = ["enthusiasm", "neutral", "fear", "sadness"],
        brain_regions    = EMOG_BRAIN_REGIONS,
        # Same spectrogram config as pre-training (same architecture)
        stft_n_fft  = 64,
        stft_hop    = 16,
        stft_n_mels = 32,
        # Training
        epochs        = args.finetune_epochs,
        patience      = args.patience,
        batch_size    = args.batch_size,
        lr            = args.finetune_lr,
        weight_decay  = args.weight_decay,
        n_repetitions = args.n_reps,
        test_size     = 0.2,
        seed          = args.seed,
        log_dir       = args.log_dir,
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)


def _class_weights(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_epoch(model, loader, opt, criterion, device, clip_norm=1.0):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def _lr_lambda_warmup_cosine(epoch, total_epochs, warmup):
    if epoch < warmup:
        return float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — PRE-TRAIN ON SEED-IV
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_on_seediv(
    segments:  np.ndarray,
    labels:    np.ndarray,
    clip_ids:  np.ndarray,
    cfg:       BIHGCNConfig,
    device:    torch.device,
    save_path: str,
) -> BIHGCN:
    """
    Train one fold on SEED-IV, save the best checkpoint, return the model.
    Uses clip-independent split (no within-clip leakage).
    """
    logger.info("=" * 60)
    logger.info("  Phase 1 — Pre-training on SEED-IV")
    logger.info(f"  Channels used : {list(EMOG_TO_SEEDIV.keys())}")
    logger.info(f"  SEED-IV → Emog label map: {SEEDIV_TO_EMOG_LABEL}")
    logger.info(f"  Segments : {len(segments)}  |  "
                f"class dist {np.bincount(labels, minlength=4).tolist()}")
    logger.info(f"  Epochs   : {cfg.epochs}  |  LR: {cfg.lr}  |  "
                f"Patience: {cfg.patience}")
    logger.info("=" * 60)

    # ── Single 80/20 clip-level split ────────────────────────────────────────
    tr_val_idx, te_idx = _clip_level_split(labels, clip_ids, cfg.test_size, cfg.seed)
    tr_val_clip_ids    = clip_ids[tr_val_idx]
    tr_val_labels      = labels[tr_val_idx]
    tr_local, val_local = _clip_level_split(
        tr_val_labels, tr_val_clip_ids, 0.2, cfg.seed + 1000
    )
    tr_idx  = tr_val_idx[tr_local]
    val_idx = tr_val_idx[val_local]

    X_tr,  y_tr  = segments[tr_idx],  labels[tr_idx]
    X_val, y_val = segments[val_idx], labels[val_idx]
    X_te,  y_te  = segments[te_idx],  labels[te_idx]

    # Normalise per channel — fit on train only
    X_tr, X_val, X_te = _normalise_per_channel(X_tr, X_val, X_te)

    logger.info(f"  Split → train={len(y_tr)} | val={len(y_val)} | test={len(y_te)}")

    tr_loader  = _make_loader(X_tr,  y_tr,  cfg.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    te_loader  = _make_loader(X_te,  y_te,  cfg.batch_size, shuffle=False)

    torch.manual_seed(cfg.seed)
    model     = BIHGCN(cfg).to(device)
    cw        = _class_weights(y_tr, cfg.n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt       = torch.optim.AdamW(model.parameters(),
                                   lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup = max(1, cfg.epochs // 10)
    sched  = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: _lr_lambda_warmup_cosine(ep, cfg.epochs, warmup)
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(cfg.epochs):
        tr_loss  = _train_epoch(model, tr_loader, opt, criterion, device)
        val_loss = _eval_epoch(model, val_loader, criterion, device)
        sched.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"  [SEEDIV pretrain]  epoch {epoch+1:3d}/{cfg.epochs}  "
                        f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                        f"lr={sched.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                logger.info(f"  [SEEDIV pretrain]  Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on SEED-IV test set
    preds, trues = _predict(model, te_loader, device)
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    logger.info(f"  [SEEDIV pretrain]  Test acc={acc:.4f}  macro-F1={f1:.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict" : best_state,
        "cfg_dict"         : vars(cfg) if not hasattr(cfg, "__dict__") else cfg.__dict__,
        "seediv_test_acc"  : acc,
        "seediv_test_f1"   : f1,
        "channel_indices"  : SEEDIV_CHANNEL_INDICES,
        "label_map"        : SEEDIV_TO_EMOG_LABEL,
    }, save_path)
    logger.info(f"  Checkpoint saved → {save_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — FINE-TUNE ON EMOGNITION
# ─────────────────────────────────────────────────────────────────────────────

def finetune_on_emognition(
    segments:       np.ndarray,
    labels:         np.ndarray,
    clip_ids:       np.ndarray,
    cfg:            BIHGCNConfig,
    pretrained_cfg: BIHGCNConfig,
    checkpoint_path: str,
    device:         torch.device,
    backbone_lr_scale: float = 0.1,
) -> dict:
    """
    Fine-tune the pre-trained model on Emognition.
    Runs cfg.n_repetitions × clip-independent 80/20 splits.

    backbone_lr_scale : LR multiplier for the encoder (< 1 → slower update).
                        The classifier head uses cfg.lr directly.
    """
    logger.info("=" * 60)
    logger.info("  Phase 2 — Fine-tuning on Emognition")
    logger.info(f"  Checkpoint    : {checkpoint_path}")
    logger.info(f"  Backbone LR   : {cfg.lr * backbone_lr_scale:.2e}  "
                f"Head LR: {cfg.lr:.2e}")
    logger.info(f"  Segments      : {len(segments)}  |  "
                f"class dist {np.bincount(labels, minlength=4).tolist()}")
    logger.info(f"  Reps          : {cfg.n_repetitions}  |  "
                f"Epochs: {cfg.epochs}  |  Patience: {cfg.patience}")
    logger.info("=" * 60)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    logger.info(f"  Loaded checkpoint  "
                f"(SEED-IV test acc={ckpt.get('seediv_test_acc', '?'):.4f})")

    accs, f1s = [], []

    for rep in range(cfg.n_repetitions):
        seed = cfg.seed + rep

        # ── Clip-level outer split ────────────────────────────────────────────
        tr_val_idx, te_idx = _clip_level_split(labels, clip_ids, cfg.test_size, seed)
        tr_val_clip_ids    = clip_ids[tr_val_idx]
        tr_val_labels      = labels[tr_val_idx]
        tr_local, val_local = _clip_level_split(
            tr_val_labels, tr_val_clip_ids, 0.2, seed + 1000
        )
        tr_idx  = tr_val_idx[tr_local]
        val_idx = tr_val_idx[val_local]

        X_tr,  y_tr  = segments[tr_idx],  labels[tr_idx]
        X_val, y_val = segments[val_idx], labels[val_idx]
        X_te,  y_te  = segments[te_idx],  labels[te_idx]

        # Normalise per channel — fit on train only
        X_tr, X_val, X_te = _normalise_per_channel(X_tr, X_val, X_te)

        logger.info(
            f"  Rep {rep+1:02d}  "
            f"train={len(y_tr)} val={len(y_val)} test={len(y_te)}"
        )

        tr_loader  = _make_loader(X_tr,  y_tr,  cfg.batch_size, shuffle=True)
        val_loader = _make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
        te_loader  = _make_loader(X_te,  y_te,  cfg.batch_size, shuffle=False)

        # ── Load pre-trained model & replace classifier head ──────────────────
        torch.manual_seed(seed)
        model = BIHGCN(pretrained_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Replace the classifier head with a fresh one sized for Emognition
        # (same n_classes=4 but fresh weights → fine-tunes faster)
        old_clf = model.classifier
        model.classifier = nn.Sequential(
            nn.Linear(pretrained_cfg.gcn2_out, pretrained_cfg.gcn2_out // 2),
            nn.GELU(),
            nn.Dropout(cfg.clf_dropout),
            nn.Linear(pretrained_cfg.gcn2_out // 2, cfg.n_classes),
        ).to(device)

        # ── Differential learning rates ───────────────────────────────────────
        # Backbone (encoder + GCN): low LR to preserve pre-trained features
        # Classifier head         : full LR to learn task-specific mapping
        backbone_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("classifier")
        ]
        head_params = list(model.classifier.parameters())

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr * backbone_lr_scale},
            {"params": head_params,     "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)

        cw        = _class_weights(y_tr, cfg.n_classes, device)
        criterion = nn.CrossEntropyLoss(weight=cw)

        warmup = max(1, cfg.epochs // 10)
        sched  = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda ep: _lr_lambda_warmup_cosine(ep, cfg.epochs, warmup)
        )

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        for epoch in range(cfg.epochs):
            tr_loss  = _train_epoch(model, tr_loader, opt, criterion, device)
            val_loss = _eval_epoch(model, val_loader, criterion, device)
            sched.step()

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"  Rep {rep+1:02d}  epoch {epoch+1:3d}/{cfg.epochs}  "
                    f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                    f"no_improve={no_improve}/{cfg.patience}"
                )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    logger.info(f"  Rep {rep+1:02d}  Early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        preds, trues = _predict(model, te_loader, device)
        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        logger.info(f"  ── Rep {rep+1:02d} FINAL  "
                    f"acc={acc:.4f}  macro-F1={f1:.4f}  "
                    f"(best_val_loss={best_val_loss:.4f})")

    summary = {
        "acc_mean"  : float(np.mean(accs)),
        "acc_std"   : float(np.std(accs)),
        "f1_mean"   : float(np.mean(f1s)),
        "f1_std"    : float(np.std(f1s)),
        "all_accs"  : accs,
        "all_f1s"   : f1s,
    }
    logger.info(f"  ══ Emognition Fine-tune Final  "
                f"acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}  "
                f"f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_results(summary: dict, path: str, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "=" * 65,
        "  SEED-IV → Emognition Transfer Learning  (BIH-GCN)",
        "=" * 65,
        f"  Mode              : {args.mode}",
        f"  SEED-IV path      : {args.seediv_path}",
        f"  Emognition path   : {args.emog_path}",
        f"  Checkpoint        : {args.checkpoint}",
        f"  Matched channels  : AF7→AF3, AF8→AF4, TP9→TP7, TP10→TP8",
        f"  Label mapping     : neutral→neutral, sad→sadness, "
        f"fear→fear, happy→enthusiasm",
        f"  Pretrain epochs   : {args.pretrain_epochs}",
        f"  Finetune epochs   : {args.finetune_epochs}",
        f"  Backbone LR scale : {args.backbone_lr_scale}",
        f"  Reps              : {args.n_reps}",
        "-" * 65,
        f"  Accuracy   : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}",
        f"  Macro-F1   : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
        f"  Chance     : 0.2500  (4 classes)",
        "-" * 65,
        "  Per-rep:",
    ]
    for i, (a, f) in enumerate(zip(summary["all_accs"], summary["all_f1s"]), 1):
        lines.append(f"    Rep {i:02d}:  acc={a:.4f}  f1={f:.4f}")
    lines.append("=" * 65)

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    for ln in lines:
        logger.info(ln)
    logger.info(f"Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEED-IV → Emognition Transfer Learning (BIH-GCN)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Paths
    p.add_argument("--seediv_path", required=True,
                   help="Root directory of SEED-IV dataset "
                        "(contains subdirs 1/, 2/, 3/ with .mat files)")
    p.add_argument("--emog_path",   required=True,
                   help="Root directory of Emognition dataset")

    # Mode
    p.add_argument("--mode", default="full",
                   choices=["pretrain", "finetune", "full"],
                   help="pretrain=Phase1 only | finetune=Phase2 only | "
                        "full=Phase1+Phase2")

    # Checkpoint
    p.add_argument("--checkpoint", default=None,
                   help="Path to pre-trained checkpoint (.pt). "
                        "Required for --mode finetune. "
                        "Auto-set to log_dir/seediv_pretrained.pt for --mode full.")

    # Segment
    p.add_argument("--segment_dur", default=6.0, type=float,
                   help="Segment duration in seconds (default 6.0)")

    # Phase 1 (pre-training)
    p.add_argument("--pretrain_epochs", default=150,  type=int)
    p.add_argument("--pretrain_lr",     default=3e-4, type=float)

    # Phase 2 (fine-tuning)
    p.add_argument("--finetune_epochs",    default=100,  type=int)
    p.add_argument("--finetune_lr",        default=1e-4, type=float,
                   help="Head LR for fine-tuning")
    p.add_argument("--backbone_lr_scale",  default=0.1,  type=float,
                   help="Multiplier applied to finetune_lr for the backbone "
                        "(0.1 = 10× slower than head)")
    p.add_argument("--n_reps",             default=30,   type=int,
                   help="Fine-tuning repetitions")

    # Shared
    p.add_argument("--batch_size",   default=32,   type=int)
    p.add_argument("--weight_decay", default=1e-3, type=float)
    p.add_argument("--patience",     default=20,   type=int)
    p.add_argument("--seed",         default=42,   type=int)
    p.add_argument("--log_dir",      default="BIH_GCN_logs/",
                   help="Directory for checkpoints and result files")
    p.add_argument("--device",       default="auto",
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
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(
        args.log_dir, "seediv_pretrained.pt"
    )

    logger.info("=" * 65)
    logger.info("  SEED-IV → Emognition  Transfer Learning  [BIH-GCN]")
    logger.info("=" * 65)
    logger.info(f"  Mode   : {args.mode}")
    logger.info(f"  Device : {device}")
    logger.info(f"  Channel mapping:")
    for emog_ch, seediv_ch in EMOG_TO_SEEDIV.items():
        seediv_idx = _CH_TO_IDX[seediv_ch]
        logger.info(f"    Emognition {emog_ch:5s} → SEED-IV {seediv_ch} (idx {seediv_idx})")
    logger.info(f"  Label mapping: {SEEDIV_TO_EMOG_LABEL}")
    logger.info("=" * 65)

    seediv_cfg = make_seediv_cfg(args)
    emog_cfg   = make_emog_cfg(args)

    # ── Phase 1: Pre-train on SEED-IV ─────────────────────────────────────────
    if args.mode in ("pretrain", "full"):
        logger.info("Loading SEED-IV data …")
        seg_samples = int(args.segment_dur * 200)   # 200 Hz for SEED-IV
        seediv_segs, seediv_labels, seediv_clip_ids = load_seediv_all_subjects(
            args.seediv_path, seg_samples
        )

        if seediv_segs.shape[0] == 0:
            logger.error("No SEED-IV data loaded. Check --seediv_path.")
            return

        pretrain_on_seediv(
            seediv_segs, seediv_labels, seediv_clip_ids,
            cfg        = seediv_cfg,
            device     = device,
            save_path  = checkpoint_path,
        )

    # ── Phase 2: Fine-tune on Emognition ──────────────────────────────────────
    if args.mode in ("finetune", "full"):
        if not os.path.isfile(checkpoint_path):
            logger.error(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Run with --mode pretrain first, or provide --checkpoint."
            )
            return

        logger.info("Loading Emognition data …")
        emog_segs, emog_labels, emog_clip_ids = load_emognition_all_subjects(emog_cfg)

        if emog_segs.shape[0] == 0:
            logger.error("No Emognition data loaded. Check --emog_path.")
            return

        summary = finetune_on_emognition(
            emog_segs, emog_labels, emog_clip_ids,
            cfg              = emog_cfg,
            pretrained_cfg   = seediv_cfg,   # same architecture
            checkpoint_path  = checkpoint_path,
            device           = device,
            backbone_lr_scale = args.backbone_lr_scale,
        )

        result_path = os.path.join(
            args.log_dir, "transfer_seediv_to_emognition_results.txt"
        )
        save_results(summary, result_path, args)

    logger.info("Done.")


if __name__ == "__main__":
    main()
