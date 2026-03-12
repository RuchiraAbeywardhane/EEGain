"""
DEAP dataset loader for the PCR pipeline.

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
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List

from .config import PCRConfig
from .preprocessing import preprocess_trial

logger = logging.getLogger("PCR.Dataset")


# Indices into DEAP labels array
_LABEL_IDX = {"V": 0, "A": 1, "D": 2, "L": 3}

# DEAP raw layout
_BASELINE_SAMPLES = 384   # 3 s × 128 Hz
_TRIAL_SAMPLES    = 7680  # 60 s × 128 Hz
_N_EEG_CHANNELS   = 32


def _load_subject_file(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin-1")


def _subject_filename(subject_id: int) -> str:
    """Return DEAP filename for a 1-indexed subject id, e.g. s01.dat"""
    return f"s{subject_id:02d}.dat"


def load_subject(
    subject_id: int,
    cfg: PCRConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, preprocess and window all 40 trials for one DEAP subject.

    Args:
        subject_id : 1-indexed subject number (1..32)
        cfg        : PCRConfig

    Returns:
        X_2d : float32  [N, window_size, grid_size, grid_size]
        X_1d : float32  [N, window_size, n_eeg_channels]
        y    : int64    [N]
    """
    fpath = os.path.join(cfg.data_path, _subject_filename(subject_id))
    subject_data = _load_subject_file(fpath)

    raw_data   = subject_data["data"]    # [40, 40, 8064]
    raw_labels = subject_data["labels"]  # [40, 4]

    label_idx = _LABEL_IDX.get(cfg.label_type.upper(), 0)
    threshold  = cfg.ground_truth_threshold

    all_2d, all_1d, all_y = [], [], []

    for trial_idx in range(cfg.n_trials):
        full_signal = raw_data[trial_idx, :_N_EEG_CHANNELS, :]  # [32, 8064]

        # Split baseline and stimulus
        baseline_eeg = full_signal[:, :_BASELINE_SAMPLES]        # [32, 384]
        trial_eeg    = full_signal[:, _BASELINE_SAMPLES:]        # [32, 7680]

        # Full preprocessing pipeline
        windows_2d, windows_1d = preprocess_trial(
            trial_eeg    = trial_eeg,
            baseline_eeg = baseline_eeg,
            segment_len  = cfg.baseline_segment_len,
            grid_size    = cfg.grid_size,
            window_size  = cfg.window_size,
            step         = cfg.window_step,
        )  # [N, W, 9, 9],  [N, W, 32]

        # Label: 0 = low, 1 = high
        raw_score = float(raw_labels[trial_idx, label_idx])
        label = 0 if raw_score <= threshold else 1

        n_windows = windows_2d.shape[0]
        all_2d.append(windows_2d)
        all_1d.append(windows_1d)
        all_y.append(np.full(n_windows, label, dtype=np.int64))

        logger.debug(
            f"  Subject {subject_id:02d} | trial {trial_idx:02d} | "
            f"label={label} (score={raw_score:.1f}) | windows={n_windows}"
        )

    X_2d = np.concatenate(all_2d, axis=0)  # [N_total, W, 9, 9]
    X_1d = np.concatenate(all_1d, axis=0)  # [N_total, W, 32]
    y    = np.concatenate(all_y,  axis=0)  # [N_total]

    logger.info(
        f"Subject {subject_id:02d}: {X_2d.shape[0]} windows | "
        f"class distribution: {np.bincount(y).tolist()}"
    )
    return X_2d, X_1d, y


def get_subject_ids(cfg: PCRConfig) -> List[int]:
    """Return sorted list of subject IDs found in the data directory."""
    ids = []
    for fname in os.listdir(cfg.data_path):
        if fname.startswith("s") and fname.endswith(".dat"):
            try:
                ids.append(int(fname[1:-4]))
            except ValueError:
                pass
    return sorted(ids)


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset


class DEAPWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-processed DEAP windows.

    Each sample is a tuple:
        (x_2d, x_1d, label)
        x_2d : FloatTensor  [window_size, grid_size, grid_size]
        x_1d : FloatTensor  [window_size, n_channels]
        label: LongTensor   scalar
    """

    def __init__(self, X_2d: np.ndarray, X_1d: np.ndarray, y: np.ndarray):
        self.X_2d = torch.from_numpy(X_2d).float()
        self.X_1d = torch.from_numpy(X_1d).float()
        self.y    = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_2d[idx], self.X_1d[idx], self.y[idx]
