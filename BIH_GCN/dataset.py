"""
Dataset loader for the EEG-only BIH_GCN pipeline.

Supports:
  - DEAP       : 32 subjects, 32 EEG channels, 128 Hz, 40 trials × 60s
  - Emognition : multiple subjects, 4 EEG channels, 256 Hz

Returns:
  segments  : np.ndarray  [N, C, T]   float32
  labels    : np.ndarray  [N]         int
"""

import os
import glob
import logging
import numpy as np
import pickle
from typing import Optional, Tuple

from BIH_GCN.config import BIHGCNConfig

logger = logging.getLogger("BIH_GCN.Dataset")


# ── Segmentation helper ───────────────────────────────────────────────────────

def _segment(data: np.ndarray, seg_samples: int,
             step: Optional[int] = None) -> np.ndarray:
    """
    data : [C, T]
    returns: [N_seg, C, seg_samples]  non-overlapping windows
    """
    if step is None:
        step = seg_samples
    C, T  = data.shape
    starts = range(0, T - seg_samples + 1, step)
    return np.stack([data[:, s:s + seg_samples] for s in starts], axis=0)


# ── DEAP ──────────────────────────────────────────────────────────────────────

def _load_deap(cfg: BIHGCNConfig,
               subject_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads DEAP .dat files.
    Returns (segments [N,C,T], labels [N]).
    """
    seg_samples = cfg.segment_samples
    files = sorted(glob.glob(os.path.join(cfg.data_path, "s*.dat")))

    if not files:
        raise FileNotFoundError(f"No DEAP .dat files found in {cfg.data_path}")

    if subject_id is not None:
        tag   = f"s{subject_id:02d}.dat"
        files = [f for f in files if os.path.basename(f) == tag]
        if not files:
            raise FileNotFoundError(f"Subject file {tag} not found.")

    all_segs, all_labels = [], []

    for fpath in files:
        logger.info(f"  Loading {os.path.basename(fpath)} …")
        with open(fpath, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")

        eeg    = data["data"][:, :32, 128 * 3:]   # [40, 32, T]  skip baseline
        rating = data["labels"]                    # [40, 4]

        dim = 0 if cfg.label_type == "V" else 1
        for trial_idx in range(eeg.shape[0]):
            score = rating[trial_idx, dim]
            label = int(score > cfg.ground_truth_threshold)
            segs  = _segment(eeg[trial_idx], seg_samples)  # [N_seg, 32, T]
            all_segs.append(segs)
            all_labels.extend([label] * len(segs))

    segments = np.concatenate(all_segs, axis=0).astype(np.float32)
    labels   = np.array(all_labels, dtype=np.int64)
    return segments, labels


# ── Emognition ────────────────────────────────────────────────────────────────

# Mapping from Emognition emotion string → integer class
_EMOG_LABEL_MAP = {
    "enthusiasm": 0,
    "neutral"   : 1,
    "fear"      : 2,
    "sadness"   : 3,
}


def _load_emognition(cfg: BIHGCNConfig,
                     subject_id: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads Emognition .npz files (expected format):
      each file has keys: 'eeg' [C, T], 'label' str, 'srate' int
    Returns (segments [N,C,T], labels [N]).
    """
    seg_samples  = cfg.segment_samples
    lead_samples = int(cfg.lead_in_duration  * cfg.sampling_rate)

    pattern = os.path.join(cfg.data_path, "**", "*.npz")
    files   = sorted(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError(f"No .npz files found under {cfg.data_path}")

    if subject_id is not None:
        files = [f for f in files if subject_id in os.path.basename(f)]
        if not files:
            raise FileNotFoundError(f"No files for subject '{subject_id}'.")

    all_segs, all_labels = [], []

    for fpath in files:
        logger.info(f"  Loading {os.path.basename(fpath)} …")
        d     = np.load(fpath, allow_pickle=True)
        eeg   = d["eeg"]                        # [C, T]
        label_str = str(d["label"])
        label = _EMOG_LABEL_MAP.get(label_str.lower(), -1)
        if label == -1:
            logger.warning(f"  Unknown label '{label_str}' — skipping.")
            continue

        eeg  = eeg[:, lead_samples:]            # discard lead-in
        segs = _segment(eeg, seg_samples)       # [N_seg, C, T]
        all_segs.append(segs)
        all_labels.extend([label] * len(segs))

    if not all_segs:
        return np.empty((0, cfg.n_eeg_channels, cfg.segment_samples),
                        dtype=np.float32), np.empty(0, dtype=np.int64)

    segments = np.concatenate(all_segs, axis=0).astype(np.float32)
    labels   = np.array(all_labels, dtype=np.int64)
    return segments, labels


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(cfg: BIHGCNConfig,
              subject_id=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG segments and labels for the configured dataset.

    Parameters
    ----------
    cfg        : BIHGCNConfig
    subject_id : int (DEAP) | str (Emognition) | None = all subjects

    Returns
    -------
    segments : [N, C, T]  float32
    labels   : [N]        int64
    """
    if cfg.dataset == "deap":
        segments, labels = _load_deap(cfg, subject_id)
    elif cfg.dataset == "emognition":
        segments, labels = _load_emognition(cfg, subject_id)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    logger.info(f"Loaded {len(segments)} segments  shape={segments.shape}  "
                f"class dist={np.bincount(labels, minlength=cfg.n_classes).tolist()}")
    return segments, labels
