"""
Preprocessing pipeline for DEAP EEG data.

Steps (in order):
    1. baseline_removal      – subtract averaged baseline segments from each trial
    2. to_2d_spatial_frames  – map 32-ch vector → 9×9 topographic grid (per time-step)
    3. zscore_normalize      – per-frame Z-score using only electrode (non-zero) positions
    4. sliding_window        – segment the 2-D frame sequence into fixed-length windows
                               and extract matching 1-D windows for the LSTM branch

DEAP electrode layout (32 channels → 9×9 grid, 10-20 system):
    Row/col indices are 0-based.
    Positions without electrodes remain 0.
"""

import numpy as np
from typing import Tuple, List


# ── 10-20 electrode → (row, col) in a 9×9 grid ──────────────────────────────
# Based on the standard 10-20 layout projected onto a 9×9 matrix.
# DEAP channel order:
#   Fp1 AF3 F3 F7 FC5 FC1 C3 T7 CP5 CP1 P3 P7 PO3 O1 Oz Pz
#   Fp2 AF4 Fz F4 F8  FC6 FC2 Cz C4  T8  CP6 CP2 P4 P8 PO4 O2
ELECTRODE_POSITIONS = {
    # channel_name : (row, col)   — 0-indexed in a 9×9 grid
    "Fp1": (0, 3), "Fp2": (0, 5),
    "AF3": (1, 3), "AF4": (1, 5),
    "F7":  (2, 0), "F3":  (2, 2), "Fz":  (2, 4), "F4":  (2, 6), "F8":  (2, 8),
    "FC5": (3, 1), "FC1": (3, 3), "FC2": (3, 5), "FC6": (3, 7),
    "T7":  (4, 0), "C3":  (4, 2), "Cz":  (4, 4), "C4":  (4, 6), "T8":  (4, 8),
    "CP5": (5, 1), "CP1": (5, 3), "CP2": (5, 5), "CP6": (5, 7),
    "P7":  (6, 0), "P3":  (6, 2), "Pz":  (6, 4), "P4":  (6, 6), "P8":  (6, 8),
    "PO3": (7, 3), "PO4": (7, 5),
    "O1":  (8, 3), "Oz":  (8, 4), "O2":  (8, 5),
}

# DEAP channel order (first 32 channels, 0-indexed)
DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8",  "FC6", "FC2", "Cz",
    "C4",  "T8",  "CP6", "CP2", "P4", "P8",  "PO4", "O2",
]


# ── 1. Baseline Removal ──────────────────────────────────────────────────────

def baseline_removal(
    trial: np.ndarray,
    baseline: np.ndarray,
    segment_len: int = 128,
) -> np.ndarray:
    """
    Subtract a baseline mean from every sample of a trial.

    The baseline (3 s × 128 Hz = 384 samples) is split into non-overlapping
    segments of length `segment_len` (default 128 = 1 s).  Those segments are
    averaged element-wise to form BaseMean [channels × segment_len], which is
    then tiled and subtracted from the full trial.

    Args:
        trial      : np.ndarray  shape [C, T_trial]   – raw EEG trial
        baseline   : np.ndarray  shape [C, T_baseline] – pre-stimulus baseline
        segment_len: int – length of each baseline segment in samples

    Returns:
        np.ndarray  shape [C, T_trial] – baseline-removed trial
    """
    C, T_base = baseline.shape
    n_segments = T_base // segment_len

    # Stack baseline segments: [n_segments, C, segment_len]
    segments = np.stack(
        [baseline[:, i * segment_len:(i + 1) * segment_len] for i in range(n_segments)],
        axis=0,
    )
    base_mean = segments.mean(axis=0)  # [C, segment_len]

    # Tile BaseMean to cover the full trial length
    C_trial, T_trial = trial.shape
    n_tiles = T_trial // segment_len
    remainder = T_trial % segment_len

    tiled = np.tile(base_mean, (1, n_tiles))        # [C, n_tiles * segment_len]
    if remainder > 0:
        tiled = np.concatenate([tiled, base_mean[:, :remainder]], axis=1)

    base_removed = trial - tiled[:, :T_trial]
    return base_removed


# ── 2. 1-D → 2-D Spatial Mapping ─────────────────────────────────────────────

def build_position_index(
    channel_names: List[str] = DEAP_CHANNELS,
    grid_size: int = 9,
) -> np.ndarray:
    """
    Build a [C, 2] integer array mapping each channel index → (row, col).
    Channels not found in ELECTRODE_POSITIONS are assigned to (0, 0) as a fallback.

    Returns:
        pos_index : np.ndarray  shape [C, 2]
    """
    pos_index = np.zeros((len(channel_names), 2), dtype=np.int32)
    for ch_idx, name in enumerate(channel_names):
        if name in ELECTRODE_POSITIONS:
            r, c = ELECTRODE_POSITIONS[name]
            pos_index[ch_idx] = [r, c]
    return pos_index


# Pre-compute position index once at import time
_POS_INDEX = build_position_index()


def to_2d_spatial_frames(
    eeg: np.ndarray,
    pos_index: np.ndarray = None,
    grid_size: int = 9,
) -> np.ndarray:
    """
    Convert a [C, T] EEG signal into a [T, grid_size, grid_size] sequence of
    2-D spatial frames by placing each channel value at its scalp position.

    Args:
        eeg       : np.ndarray  shape [C, T]
        pos_index : np.ndarray  shape [C, 2]  – (row, col) per channel
        grid_size : int – spatial grid side length

    Returns:
        frames : np.ndarray  shape [T, grid_size, grid_size]  (float32)
    """
    if pos_index is None:
        pos_index = _POS_INDEX

    C, T = eeg.shape
    frames = np.zeros((T, grid_size, grid_size), dtype=np.float32)

    rows = pos_index[:, 0]  # [C]
    cols = pos_index[:, 1]  # [C]

    # Vectorised assignment: for every time step, fill grid positions
    # frames[t, row[c], col[c]] = eeg[c, t]  for all c
    frames[:, rows, cols] = eeg.T  # eeg.T is [T, C]; broadcast over channels

    return frames


# ── 3. Z-score Normalisation ──────────────────────────────────────────────────

def zscore_normalize_frames(frames: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalisation to each 2-D frame independently.
    Only non-zero elements (actual electrode positions) are used to compute
    the mean and standard deviation.

    Args:
        frames : np.ndarray  shape [T, H, W]  – spatial EEG frames

    Returns:
        normalized : np.ndarray  shape [T, H, W]  (float32)
    """
    normalized = np.empty_like(frames, dtype=np.float32)
    for t in range(frames.shape[0]):
        frame = frames[t]
        mask = frame != 0.0
        if mask.any():
            mu  = frame[mask].mean()
            sig = frame[mask].std()
            sig = sig if sig > 1e-8 else 1e-8
            normalized[t] = np.where(mask, (frame - mu) / sig, 0.0)
        else:
            normalized[t] = frame
    return normalized


# ── 4. Sliding Window Segmentation ────────────────────────────────────────────

def sliding_window(
    frames_2d: np.ndarray,
    eeg_1d: np.ndarray,
    window_size: int = 128,
    step: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment both the 2-D frame sequence and the raw 1-D EEG into fixed-length
    windows.

    Args:
        frames_2d   : np.ndarray  shape [T, H, W]   – spatial frames (CNN input)
        eeg_1d      : np.ndarray  shape [C, T]       – raw EEG (LSTM input)
        window_size : int – number of time steps per window
        step        : int – stride between windows

    Returns:
        windows_2d : np.ndarray  shape [N, window_size, H, W]
        windows_1d : np.ndarray  shape [N, window_size, C]
    """
    T = frames_2d.shape[0]
    starts = range(0, T - window_size + 1, step)

    windows_2d = np.stack([frames_2d[s:s + window_size] for s in starts])   # [N, W, H, W_grid]
    windows_1d = np.stack([eeg_1d[:, s:s + window_size].T for s in starts]) # [N, W, C]

    return windows_2d.astype(np.float32), windows_1d.astype(np.float32)


# ── Full preprocessing pipeline for a single trial ────────────────────────────

def preprocess_trial(
    trial_eeg: np.ndarray,
    baseline_eeg: np.ndarray,
    segment_len: int = 128,
    grid_size: int = 9,
    window_size: int = 128,
    step: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing for one DEAP trial.

    Args:
        trial_eeg    : np.ndarray  shape [C, T_trial]    – raw EEG (first 32 ch)
        baseline_eeg : np.ndarray  shape [C, T_baseline] – 3-s pre-stimulus signal
        segment_len  : int – baseline segment length in samples
        grid_size    : int – spatial grid side length
        window_size  : int – samples per window
        step         : int – stride between windows

    Returns:
        windows_2d : np.ndarray  shape [N, window_size, grid_size, grid_size]
        windows_1d : np.ndarray  shape [N, window_size, C]
    """
    # Step 1 – baseline removal
    eeg_clean = baseline_removal(trial_eeg, baseline_eeg, segment_len)

    # Step 2 – convert to 2-D spatial frames
    frames = to_2d_spatial_frames(eeg_clean, grid_size=grid_size)

    # Step 3 – Z-score normalisation per frame
    frames = zscore_normalize_frames(frames)

    # Step 4 – sliding window segmentation
    windows_2d, windows_1d = sliding_window(frames, eeg_clean, window_size, step)

    return windows_2d, windows_1d
