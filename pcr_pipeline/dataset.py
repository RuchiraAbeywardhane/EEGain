"""
DEAP and Emognition dataset loaders for the PCR pipeline.

Reads the raw DEAP .dat pickle files and applies the full preprocessing
pipeline (baseline removal → 2-D spatial frames → Z-score → windowing).

Each subject file contains:
    data   : [40 trials, 40 channels, 8064 samples]
               – first 32 channels are EEG
               – channels 32-39 are peripheral signals (ignored)
               – first 384 samples (3 s) of each trial are the baseline
               – remaining 7680 samples (60 s) are the stimulus period
    labels : [40 trials, 4]  – valence, arousal, dominance, liking

The loader returns two parallel arrays per subject:
    X_2d : [N_windows, window_size, grid_size, grid_size]  – CNN branch input
    X_1d : [N_windows, window_size, n_channels]            – LSTM branch input
    y    : [N_windows]                                     – class labels
"""

import os
import glob
import json
import pickle
import logging
import numpy as np
from typing import Tuple, List

from .config import PCRConfig
from .preprocessing import (
    preprocess_trial,
    preprocess_trial_emognition,
)

logger = logging.getLogger("PCR.Dataset")

# ── DEAP constants ────────────────────────────────────────────────────────────
_LABEL_IDX        = {"V": 0, "A": 1, "D": 2, "L": 3}
_BASELINE_SAMPLES = 384    # 3 s × 128 Hz
_N_EEG_CHANNELS   = 32

# ── Emognition constants ──────────────────────────────────────────────────────
_EMOG_CHANNELS    = ["TP9", "AF7", "AF8", "TP10"]
_EMOG_SRATE       = 256    # Hz


# ════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset


class DEAPWindowDataset(Dataset):
    """PyTorch Dataset for pre-processed windows (DEAP or Emognition)."""

    def __init__(self, X_2d: np.ndarray, X_1d: np.ndarray, y: np.ndarray):
        self.X_2d = torch.from_numpy(X_2d).float()
        self.X_1d = torch.from_numpy(X_1d).float()
        self.y    = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_2d[idx], self.X_1d[idx], self.y[idx]


def get_subject_ids(cfg: PCRConfig) -> List:
    """Return sorted subject IDs for whichever dataset is configured."""
    if cfg.dataset.lower() == "emognition":
        return get_emognition_subject_ids(cfg)
    return get_deap_subject_ids(cfg)


def load_subject_trials(
    subject_id,
    cfg: PCRConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Dispatch to the correct loader based on cfg.dataset."""
    if cfg.dataset.lower() == "emognition":
        return load_emognition_subject_trials(subject_id, cfg)
    return load_deap_subject_trials(subject_id, cfg)


# ════════════════════════════════════════════════════════════════════════════
#  DEAP LOADER
# ════════════════════════════════════════════════════════════════════════════

def _deap_subject_filename(subject_id: int) -> str:
    return f"s{subject_id:02d}.dat"


def get_deap_subject_ids(cfg: PCRConfig) -> List[int]:
    ids = []
    for fname in os.listdir(cfg.data_path):
        if fname.startswith("s") and fname.endswith(".dat"):
            try:
                ids.append(int(fname[1:-4]))
            except ValueError:
                pass
    return sorted(ids)


def load_deap_subject_trials(
    subject_id: int,
    cfg: PCRConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Load all 40 DEAP trials for one subject, one array per trial.

    Returns:
        trials_2d    : list of [n_windows, W, 9, 9]
        trials_1d    : list of [n_windows, W, 32]
        trial_labels : list of int  (0=low, 1=high)
    """
    fpath = os.path.join(cfg.data_path, _deap_subject_filename(subject_id))
    with open(fpath, "rb") as f:
        subject_data = pickle.load(f, encoding="latin-1")

    raw_data   = subject_data["data"]    # [40, 40, 8064]
    raw_labels = subject_data["labels"]  # [40, 4]

    label_idx  = _LABEL_IDX.get(cfg.label_type.upper(), 0)
    threshold  = cfg.ground_truth_threshold
    seg_len    = cfg.baseline_segment_len

    trials_2d, trials_1d, trial_labels = [], [], []

    for trial_idx in range(cfg.n_trials):
        full_signal  = raw_data[trial_idx, :_N_EEG_CHANNELS, :]  # [32, 8064]
        baseline_eeg = full_signal[:, :_BASELINE_SAMPLES]
        trial_eeg    = full_signal[:, _BASELINE_SAMPLES:]

        windows_2d, windows_1d = preprocess_trial(
            trial_eeg    = trial_eeg,
            baseline_eeg = baseline_eeg,
            segment_len  = seg_len,
            grid_size    = cfg.grid_size,
            window_size  = cfg.window_size,
            step         = cfg.window_step,
        )

        raw_score = float(raw_labels[trial_idx, label_idx])
        label = 0 if raw_score <= threshold else 1

        trials_2d.append(windows_2d)
        trials_1d.append(windows_1d)
        trial_labels.append(label)

        logger.debug(
            f"  DEAP subj {subject_id:02d} | trial {trial_idx:02d} | "
            f"label={label} (score={raw_score:.1f}) | windows={windows_2d.shape[0]}"
        )

    logger.info(
        f"DEAP subject {subject_id:02d}: {cfg.n_trials} trials | "
        f"class dist: {[trial_labels.count(c) for c in range(cfg.n_classes)]}"
    )
    return trials_2d, trials_1d, trial_labels


def load_subject(
    subject_id: int,
    cfg: PCRConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat version of DEAP loader (kept for compatibility)."""
    trials_2d, trials_1d, trial_labels = load_deap_subject_trials(subject_id, cfg)
    X_2d = np.concatenate(trials_2d, axis=0)
    X_1d = np.concatenate(trials_1d, axis=0)
    y    = np.array(
        [lbl for lbl, t in zip(trial_labels, trials_2d)
         for _ in range(t.shape[0])],
        dtype=np.int64,
    )
    return X_2d, X_1d, y


# ════════════════════════════════════════════════════════════════════════════
#  EMOGNITION LOADER
# ════════════════════════════════════════════════════════════════════════════

def _emog_to_num(x):
    """Convert a JSON value (list or scalar) to a float64 numpy array."""
    import pandas as pd
    if isinstance(x, list):
        if not x:
            return np.array([], dtype=np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, dtype=np.float64)
    return np.asarray([x], dtype=np.float64)


def _emog_interp_nan(a: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN/inf values in a 1-D array."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _emog_read_json(fpath: str) -> np.ndarray:
    """
    Load one Emognition JSON file and return a quality-filtered
    raw EEG array of shape [4, T] (channels: TP9, AF7, AF8, TP10).
    Returns None if the signal is empty or all samples are bad quality.

    Quality filter: keeps only samples where HeadBandOn==1 AND all
    four HSI scores are <= 2  (1=good contact, 2=ok, 3/4=poor).
    """
    with open(fpath, "r") as f:
        data = json.load(f)

    tp9  = _emog_interp_nan(_emog_to_num(data.get("RAW_TP9",  [])))
    af7  = _emog_interp_nan(_emog_to_num(data.get("RAW_AF7",  [])))
    af8  = _emog_interp_nan(_emog_to_num(data.get("RAW_AF8",  [])))
    tp10 = _emog_interp_nan(_emog_to_num(data.get("RAW_TP10", [])))

    L = min(len(tp9), len(af7), len(af8), len(tp10))
    if L == 0:
        return None

    # ── Quality mask ──────────────────────────────────────────────────────
    hsi_tp9  = _emog_to_num(data.get("HSI_TP9",  []))[:L]
    hsi_af7  = _emog_to_num(data.get("HSI_AF7",  []))[:L]
    hsi_af8  = _emog_to_num(data.get("HSI_AF8",  []))[:L]
    hsi_tp10 = _emog_to_num(data.get("HSI_TP10", []))[:L]
    head_on  = _emog_to_num(data.get("HeadBandOn", []))[:L]

    mask = (
        np.isfinite(tp9[:L]) & np.isfinite(af7[:L]) &
        np.isfinite(af8[:L]) & np.isfinite(tp10[:L])
    )
    if len(head_on) == L and len(hsi_tp9) == L:
        mask &= (
            (head_on == 1) &
            np.isfinite(hsi_tp9)  & (hsi_tp9  <= 2) &
            np.isfinite(hsi_af7)  & (hsi_af7  <= 2) &
            np.isfinite(hsi_af8)  & (hsi_af8  <= 2) &
            np.isfinite(hsi_tp10) & (hsi_tp10 <= 2)
        )

    tp9, af7, af8, tp10 = (
        tp9[:L][mask], af7[:L][mask],
        af8[:L][mask], tp10[:L][mask],
    )
    if len(tp9) == 0:
        return None

    signal = np.stack([tp9, af7, af8, tp10], axis=0).astype(np.float32)  # [4, T]
    # mean-centre each channel
    signal -= signal.mean(axis=1, keepdims=True)
    return signal


def get_emognition_subject_ids(cfg: PCRConfig) -> List[str]:
    """
    Scan the data directory and return a sorted list of unique subject IDs.
    Subject ID is the first part of the filename before the first underscore,
    e.g. "P01" from "P01_ENTHUSIASM_STIMULUS_MUSE.json".
    """
    patterns = [
        os.path.join(cfg.data_path, "*_STIMULUS_MUSE.json"),
        os.path.join(cfg.data_path, "*", "*_STIMULUS_MUSE.json"),
        os.path.join(cfg.data_path, "*", "*", "*_STIMULUS_MUSE.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    subjects = set()
    for fpath in files:
        fname = os.path.basename(fpath)
        if "BASELINE" in fname:
            continue
        parts = fname.split("_")
        if len(parts) >= 2:
            emotion = parts[1].upper()
            if emotion in cfg.emognition_class_map:
                subjects.add(parts[0])
    return sorted(subjects)


def _emog_find_trials(subject_id: str, cfg: PCRConfig) -> List[Tuple[str, int]]:
    """
    Return list of (file_path, label_int) for all valid trials of one subject.
    Baseline files are excluded.
    """
    patterns = [
        os.path.join(cfg.data_path, f"{subject_id}_*_STIMULUS_MUSE.json"),
        os.path.join(cfg.data_path, "*",    f"{subject_id}_*_STIMULUS_MUSE.json"),
        os.path.join(cfg.data_path, "*", "*", f"{subject_id}_*_STIMULUS_MUSE.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    result = []
    for fpath in files:
        fname = os.path.basename(fpath)
        if "BASELINE" in fname:
            continue
        parts = fname.split("_")
        if len(parts) < 2:
            continue
        emotion = parts[1].upper()
        label = cfg.emognition_class_map.get(emotion)
        if label is None:
            continue
        result.append((fpath, label))
    return result


def load_emognition_subject_trials(
    subject_id: str,
    cfg: PCRConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Load all trials for one Emognition subject, keeping each trial separate
    so the CV can split at the trial level (no leakage).

    Preprocessing per trial:
        1. Discard first `lead_in_duration` seconds  (emotionally neutral lead-in)
        2. Next `baseline_duration` seconds          → baseline
        3. Remainder                                 → stimulus EEG
        4. Baseline removal → Z-score → sliding window

    Returns:
        trials_2d    : list of [n_windows, W, 1, 1]  – dummy zeros (1D model)
        trials_1d    : list of [n_windows, W, 4]
        trial_labels : list of int
    """
    trial_files = _emog_find_trials(subject_id, cfg)
    if not trial_files:
        logger.warning(f"Emognition: no trials found for subject {subject_id}")
        return [], [], []

    # Window size and step in samples (scale to 256 Hz)
    window_size = cfg.window_size  * (cfg.sampling_rate // 128)
    window_step = cfg.window_step  * (cfg.sampling_rate // 128)

    # Minimum trial length needed: lead_in + baseline + at least 1 window
    min_samples = int(
        (cfg.lead_in_duration + cfg.baseline_duration) * cfg.sampling_rate
    ) + window_size

    trials_2d, trials_1d, trial_labels = [], [], []
    skipped = 0

    for fpath, label in trial_files:
        fname = os.path.basename(fpath)

        # ── Load & quality-filter ─────────────────────────────────────────
        signal = _emog_read_json(fpath)   # [4, T] or None
        if signal is None:
            logger.warning(f"  Skipping {fname}: empty after quality filter")
            skipped += 1
            continue

        # ── Minimum length check ──────────────────────────────────────────
        if signal.shape[1] < min_samples:
            logger.warning(
                f"  Skipping {fname}: {signal.shape[1]} samples < "
                f"minimum {min_samples} "
                f"(lead_in={cfg.lead_in_duration}s + "
                f"baseline={cfg.baseline_duration}s + 1 window)"
            )
            skipped += 1
            continue

        # ── Preprocessing ─────────────────────────────────────────────────
        try:
            windows_2d, windows_1d = preprocess_trial_emognition(
                trial_eeg        = signal,
                sampling_rate    = cfg.sampling_rate,
                lead_in_duration = cfg.lead_in_duration,
                baseline_duration= cfg.baseline_duration,
                window_size      = window_size,
                step             = window_step,
            )
        except ValueError as e:
            logger.warning(f"  Skipping {fname}: {e}")
            skipped += 1
            continue

        if windows_1d.shape[0] == 0:
            logger.warning(f"  Skipping {fname}: 0 windows produced")
            skipped += 1
            continue

        trials_2d.append(windows_2d)
        trials_1d.append(windows_1d)
        trial_labels.append(label)

        logger.debug(
            f"  Emognition {subject_id} | {fname} | "
            f"label={label} | windows={windows_1d.shape[0]}"
        )

    logger.info(
        f"Emognition subject {subject_id}: "
        f"{len(trial_labels)} trials loaded, {skipped} skipped | "
        f"class dist: {[trial_labels.count(c) for c in range(cfg.n_classes)]}"
    )
    return trials_2d, trials_1d, trial_labels
