"""
Dataset loader for the EEG-only BIH_GCN pipeline.

Supports:
  - DEAP       : 32 subjects, 32 EEG channels, 128 Hz, 40 trials × 60s
  - Emognition : 43 subjects, 4 EEG channels (TP9,AF7,AF8,TP10), 256 Hz
                 Folder structure:
                   <data_path>/
                     Participant_<id>/
                       EEG_<id>_<emotion>.csv   (columns: timestamps + 4 EEG ch)
                       OR a single EEG CSV with a label column

Returns:
  segments  : np.ndarray  [N, C, T]   float32
  labels    : np.ndarray  [N]         int
"""

import os
import glob
import logging
import numpy as np
import pickle
import pandas as pd
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
    C, T = data.shape
    starts = range(0, T - seg_samples + 1, step)
    segs = [data[:, s:s + seg_samples] for s in starts]
    if not segs:
        return np.empty((0, C, seg_samples), dtype=np.float32)
    return np.stack(segs, axis=0)


# ── DEAP ──────────────────────────────────────────────────────────────────────

def _load_deap(cfg: BIHGCNConfig,
               subject_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
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

        eeg    = data["data"][:, :32, 128 * 3:]
        rating = data["labels"]
        dim    = 0 if cfg.label_type == "V" else 1

        for trial_idx in range(eeg.shape[0]):
            score = rating[trial_idx, dim]
            label = int(score > cfg.ground_truth_threshold)
            segs  = _segment(eeg[trial_idx], seg_samples)
            if segs.shape[0] == 0:
                continue
            all_segs.append(segs)
            all_labels.extend([label] * len(segs))

    segments = np.concatenate(all_segs, axis=0).astype(np.float32)
    labels   = np.array(all_labels, dtype=np.int64)
    return segments, labels


# ── Emognition ────────────────────────────────────────────────────────────────

_EMOG_LABEL_MAP = {
    "enthusiasm": 0,
    "neutral"   : 1,
    "fear"      : 2,
    "sadness"   : 3,
}

# EEG column names used in Emognition CSVs (Muse headset)
_EEG_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]


def _find_emog_files(data_path: str, subject_id: Optional[str]):
    """
    Recursively search for all JSON files under data_path.
    Emognition layout:
      <data_path>/
        Participant_<id>/
          <emotion>_<id>.json   OR   <id>_<emotion>.json
    """
    all_json = sorted(glob.glob(
        os.path.join(data_path, "**", "*.json"), recursive=True
    ))

    if not all_json:
        raise FileNotFoundError(f"No JSON files found under {data_path}")

    if subject_id is not None:
        all_json = [f for f in all_json if subject_id in os.path.basename(f)
                    or subject_id in os.path.basename(os.path.dirname(f))]
        if not all_json:
            raise FileNotFoundError(
                f"No JSON files found for subject '{subject_id}' under {data_path}"
            )

    return all_json


def _infer_label_from_path(fpath: str) -> int:
    """Try to infer emotion label from the file/folder name."""
    name = (os.path.basename(fpath) + " " +
            os.path.basename(os.path.dirname(fpath))).lower()
    for emotion, idx in _EMOG_LABEL_MAP.items():
        if emotion in name:
            return idx
    return -1


def _safe_stack(arrays: list) -> np.ndarray:
    """
    Stack arrays along the first axis, truncating to the minimum length.
    """
    min_len = min(len(arr) for arr in arrays)
    return np.stack([arr[:min_len] for arr in arrays], axis=0)


def _load_eeg_from_json(fpath: str) -> Tuple[Optional[np.ndarray], int]:
    """
    Load EEG data from an Emognition JSON file.

    Expected JSON structures (tries all):
      A) { "eeg": [[ch0_t0, ch1_t0, ...], ...],  "label": "fear" }
      B) { "RAW_TP9": [...], "RAW_AF7": [...], ... }
      C) { "data": { "RAW_TP9": [...], ... }, "label": "fear" }
      D) list of records: [{"RAW_TP9": v, "RAW_AF7": v, ...}, ...]

    Returns (eeg [C, T] float32  or  None,  label int  or  -1)
    """
    import json
    try:
        with open(fpath, "r") as fh:
            raw = json.load(fh)
    except Exception as e:
        logger.warning(f"  Could not read {fpath}: {e}")
        return None, -1

    label = -1
    eeg   = None

    # ── Extract label from JSON if present ────────────────────────────────────
    if isinstance(raw, dict):
        for key in ("label", "emotion", "class", "stimulus"):
            if key in raw:
                label = _EMOG_LABEL_MAP.get(str(raw[key]).lower(), -1)
                break

    # ── Structure A: {"eeg": [[sample0], [sample1], ...]} ────────────────────
    if isinstance(raw, dict) and "eeg" in raw:
        arr = np.array(raw["eeg"], dtype=np.float32)
        if arr.ndim == 2:
            eeg = arr.T if arr.shape[1] <= 8 else arr

    # ── Structure B/C: flat or nested dict of channel arrays ─────────────────
    elif isinstance(raw, dict):
        data_dict = raw.get("data", raw)
        present   = [c for c in _EEG_COLS if c in data_dict]
        if len(present) == 4:
            eeg = _safe_stack([data_dict[c] for c in present])
        else:
            num_keys = [k for k, v in data_dict.items()
                        if isinstance(v, (list, np.ndarray))
                        and not any(x in k.lower()
                                    for x in ["time", "unix", "marker",
                                              "label", "trigger", "index"])]
            if len(num_keys) >= 4:
                eeg = _safe_stack([data_dict[k] for k in num_keys[:4]])

    # ── Structure D: list of per-sample dicts ─────────────────────────────────
    elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        df      = pd.DataFrame(raw)
        present = [c for c in _EEG_COLS if c in df.columns]
        if len(present) == 4:
            eeg = df[present].values.T.astype(np.float32)
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols
                        if not any(k in c.lower()
                                   for k in ["time", "unix", "marker", "label",
                                             "trigger", "index", "id"])]
            if len(num_cols) >= 4:
                eeg = df[num_cols[:4]].values.T.astype(np.float32)

    if eeg is not None:
        # Ensure shape is exactly [4, T] — drop extra channels
        if eeg.shape[0] > 4:
            eeg = eeg[:4, :]

    return eeg, label


def _load_emognition(cfg: BIHGCNConfig,
                     subject_id: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    seg_samples  = cfg.segment_samples
    lead_samples = int(cfg.lead_in_duration * cfg.sampling_rate)

    files = _find_emog_files(cfg.data_path, subject_id)
    logger.info(f"  Found {len(files)} JSON file(s)")

    all_segs, all_labels = [], []

    for fpath in files:
        # ── Load EEG + label from JSON ─────────────────────────────────────────
        eeg, label_from_json = _load_eeg_from_json(fpath)

        # Prefer label embedded in JSON; fall back to path-inferred label
        label = label_from_json if label_from_json != -1 \
                else _infer_label_from_path(fpath)

        if label == -1:
            logger.warning(f"  Skipping (no emotion label): "
                           f"{os.path.basename(fpath)}")
            continue

        if eeg is None:
            logger.warning(f"  Skipping (could not parse EEG): "
                           f"{os.path.basename(fpath)}")
            continue

        if eeg.shape[0] != cfg.n_eeg_channels:
            logger.warning(f"  Expected {cfg.n_eeg_channels} ch, "
                           f"got {eeg.shape[0]} — skipping {os.path.basename(fpath)}")
            continue

        # ── Drop lead-in, segment ─────────────────────────────────────────────
        eeg  = eeg[:, lead_samples:]
        segs = _segment(eeg, seg_samples)
        if segs.shape[0] == 0:
            logger.warning(f"  Too short for one segment: {os.path.basename(fpath)}")
            continue

        logger.info(f"  {os.path.basename(fpath):50s}  "
                    f"label={label}  segs={segs.shape[0]}")
        all_segs.append(segs)
        all_labels.extend([label] * segs.shape[0])

    if not all_segs:
        logger.error("No usable Emognition data found. "
                     "Run a quick diagnostic:\n"
                     "  import json; d=json.load(open('<any>.json')); print(type(d), list(d.keys()) if isinstance(d,dict) else d[0])")
        return (np.empty((0, cfg.n_eeg_channels, cfg.segment_samples),
                         dtype=np.float32),
                np.empty(0, dtype=np.int64))

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
