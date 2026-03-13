"""
Dataset loaders for the WEAVE pipeline.

Returns raw EEG segments of shape [N, C, T] and labels [N] — no neural-net
preprocessing.  Feature extraction is done separately in features.py.

DEAP
----
File : s{id:02d}.dat  (pickle, latin-1)
  data   [40, 40, 8064] – first 32 ch EEG; first 384 samples = 3 s baseline
  labels [40,  4]       – valence, arousal, dominance, liking  (1–9 scale)

  Steps:
    1. Separate baseline (first 384 samples) from stimulus (remaining 7680 s)
    2. Baseline-removal  (same averaging method as pcr_pipeline)
    3. Z-score per channel
    4. Segment into non-overlapping windows of `segment_duration` seconds
    5. Binary label: score > threshold → 1 (High), else → 0 (Low)

Emognition
----------
JSON files (STIMULUS_MUSE only, 4 EEG channels at 256 Hz):
  Steps:
    1. Quality filter  (HSI ≤ 2, HeadBandOn = 1)
    2. Discard lead-in (first `lead_in_duration` s)
    3. Extract baseline (next `baseline_duration` s)
    4. Baseline-removal
    5. Z-score per channel
    6. Segment into non-overlapping windows of `segment_duration` seconds
    7. Label from class_map (ENTHUSIASM=0, NEUTRAL=1, FEAR=2, SADNESS=3)
"""

import os
import glob
import json
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional

from .config import WEAVEConfig

logger = logging.getLogger("WEAVE.Dataset")

# ── DEAP constants ────────────────────────────────────────────────────────────
_LABEL_IDX        = {"V": 0, "A": 1, "D": 2, "L": 3}
_BASELINE_SAMPLES = 384     # 3 s × 128 Hz
_N_EEG_CHANNELS   = 32
_DEAP_SRATE       = 128


# ════════════════════════════════════════════════════════════════════════════
#  SHARED SIGNAL UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _baseline_removal(trial: np.ndarray, baseline: np.ndarray,
                      segment_len: int) -> np.ndarray:
    """
    Subtract averaged baseline from trial.
    baseline shape : [C, T_base]
    trial    shape : [C, T_trial]
    Returns        : [C, T_trial]
    """
    C, T_base = baseline.shape
    n_seg     = T_base // segment_len
    if n_seg == 0:
        # baseline shorter than one segment — just subtract global mean
        base_mean = baseline.mean(axis=1, keepdims=True)  # [C, 1]
        return trial - base_mean

    segs     = np.stack(
        [baseline[:, i * segment_len:(i + 1) * segment_len]
         for i in range(n_seg)], axis=0)
    base_mean = segs.mean(axis=0)                     # [C, segment_len]

    C_t, T_trial = trial.shape
    n_tiles   = T_trial // segment_len
    remainder = T_trial %  segment_len
    tiled     = np.tile(base_mean, (1, n_tiles))
    if remainder > 0:
        tiled = np.concatenate([tiled, base_mean[:, :remainder]], axis=1)
    return trial - tiled[:, :T_trial]


def _zscore_1d(eeg: np.ndarray) -> np.ndarray:
    """Per-channel Z-score normalisation.  eeg : [C, T]"""
    eeg = eeg.astype(np.float32, copy=True)
    for c in range(eeg.shape[0]):
        mu  = eeg[c].mean()
        sig = eeg[c].std()
        eeg[c] = (eeg[c] - mu) / max(sig, 1e-8)
    return eeg


def _segment_signal(eeg: np.ndarray, segment_samples: int) -> np.ndarray:
    """
    Cut [C, T] into non-overlapping windows of `segment_samples` samples.
    Trailing samples that don't fill a complete window are discarded.

    Returns : [N, C, segment_samples]
    """
    C, T    = eeg.shape
    n_segs  = T // segment_samples
    if n_segs == 0:
        return np.empty((0, C, segment_samples), dtype=np.float32)
    segs = np.stack(
        [eeg[:, i * segment_samples:(i + 1) * segment_samples]
         for i in range(n_segs)], axis=0)
    return segs.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  DEAP LOADER
# ════════════════════════════════════════════════════════════════════════════

def get_deap_subject_ids(cfg: WEAVEConfig) -> List[int]:
    ids = []
    for fname in os.listdir(cfg.data_path):
        if fname.startswith("s") and fname.endswith(".dat"):
            try:
                ids.append(int(fname[1:-4]))
            except ValueError:
                pass
    return sorted(ids)


def load_deap_subject(
    subject_id: int,
    cfg: WEAVEConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all 40 DEAP trials for one subject, extract segments.

    Returns:
        segments : np.ndarray  shape [N, 32, segment_samples]
        labels   : np.ndarray  shape [N]  int  (0=Low, 1=High)
    """
    fpath = os.path.join(cfg.data_path, f"s{subject_id:02d}.dat")
    with open(fpath, "rb") as f:
        subject_data = pickle.load(f, encoding="latin-1")

    raw_data   = subject_data["data"]    # [40, 40, 8064]
    raw_labels = subject_data["labels"]  # [40, 4]

    label_col      = _LABEL_IDX.get(cfg.label_type.upper(), 0)
    threshold      = cfg.ground_truth_threshold
    segment_samples = int(cfg.segment_duration * _DEAP_SRATE)
    segment_len    = _DEAP_SRATE                               # 1-s baseline avg

    all_segs, all_labels = [], []

    for trial_idx in range(raw_data.shape[0]):
        full  = raw_data[trial_idx, :_N_EEG_CHANNELS, :]      # [32, 8064]
        base  = full[:, :_BASELINE_SAMPLES]                    # [32, 384]
        trial = full[:, _BASELINE_SAMPLES:]                    # [32, 7680]

        trial_clean = _baseline_removal(trial, base, segment_len)
        trial_clean = _zscore_1d(trial_clean)

        segs = _segment_signal(trial_clean, segment_samples)   # [N_s, 32, T_seg]
        if segs.shape[0] == 0:
            continue

        raw_score = float(raw_labels[trial_idx, label_col])
        label     = 1 if raw_score > threshold else 0

        all_segs.append(segs)
        all_labels.extend([label] * segs.shape[0])

    if not all_segs:
        logger.warning(f"DEAP subject {subject_id:02d}: no segments produced")
        return np.empty((0, _N_EEG_CHANNELS, segment_samples), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    segments = np.concatenate(all_segs, axis=0)
    labels   = np.array(all_labels, dtype=np.int64)

    logger.info(
        f"DEAP subject {subject_id:02d}: {segments.shape[0]} segments | "
        f"class dist {np.bincount(labels).tolist()}"
    )
    return segments, labels


def load_deap_all_subjects(
    cfg: WEAVEConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool all DEAP subjects into one segment matrix."""
    ids = get_deap_subject_ids(cfg)
    all_segs, all_labels = [], []
    for sid in ids:
        segs, labels = load_deap_subject(sid, cfg)
        if segs.shape[0]:
            all_segs.append(segs)
            all_labels.append(labels)
    if not all_segs:
        raise RuntimeError("No DEAP segments loaded — check data_path.")
    return np.concatenate(all_segs, axis=0), np.concatenate(all_labels, axis=0)


# ════════════════════════════════════════════════════════════════════════════
#  EMOGNITION LOADER  (reuses helpers from pcr_pipeline.dataset)
# ════════════════════════════════════════════════════════════════════════════

def _emog_to_num(x) -> np.ndarray:
    import pandas as pd
    if isinstance(x, list):
        if not x:
            return np.array([], dtype=np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, dtype=np.float64)
    return np.asarray([x], dtype=np.float64)


def _emog_interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():   return a
    if not m.any(): return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _emog_is_stimulus_muse(fname: str) -> bool:
    stem  = fname.replace(".json", "")
    parts = stem.split("_")
    if len(parts) < 4:
        return False
    return parts[2].upper() == "STIMULUS" and parts[3].upper() == "MUSE"


def _emog_parse_emotion(fname: str) -> str:
    return fname.split("_")[1].upper()


def _emog_parse_subject(fname: str) -> str:
    return fname.split("_")[0]


def _emog_read_json(fpath: str) -> Optional[np.ndarray]:
    """
    Load one STIMULUS_MUSE JSON, quality-filter, return [4, T] float32 or None.
    Channels: TP9, AF7, AF8, TP10.
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

    tp9  = tp9[:L][mask];  af7  = af7[:L][mask]
    af8  = af8[:L][mask];  tp10 = tp10[:L][mask]
    if len(tp9) == 0:
        return None

    signal = np.stack([tp9, af7, af8, tp10], axis=1).T.astype(np.float32)
    signal -= signal.mean(axis=1, keepdims=True)
    return signal


def get_emognition_subject_ids(cfg: WEAVEConfig) -> List[str]:
    patterns = [
        os.path.join(cfg.data_path, "*",  "*.json"),
        os.path.join(cfg.data_path,       "*.json"),
        os.path.join(cfg.data_path, "*", "*", "*.json"),
    ]
    files    = sorted({p for pat in patterns for p in glob.glob(pat)})
    subjects = set()
    for fpath in files:
        fname = os.path.basename(fpath)
        if not _emog_is_stimulus_muse(fname):
            continue
        emotion = _emog_parse_emotion(fname)
        if emotion in cfg.emognition_class_map:
            subjects.add(_emog_parse_subject(fname))
    return sorted(subjects)


def _emog_find_trials(subject_id: str, cfg: WEAVEConfig) -> List[Tuple[str, int]]:
    patterns = [
        os.path.join(cfg.data_path, subject_id, "*.json"),
        os.path.join(cfg.data_path,              "*.json"),
        os.path.join(cfg.data_path, "*", subject_id, "*.json"),
    ]
    files  = sorted({p for pat in patterns for p in glob.glob(pat)})
    result = []
    for fpath in files:
        fname = os.path.basename(fpath)
        if _emog_parse_subject(fname) != subject_id:
            continue
        if not _emog_is_stimulus_muse(fname):
            continue
        emotion = _emog_parse_emotion(fname)
        label   = cfg.emognition_class_map.get(emotion)
        if label is None:
            continue
        result.append((fpath, label))
    return sorted(result)


def load_emognition_subject(
    subject_id: str,
    cfg: WEAVEConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all STIMULUS_MUSE trials for one Emognition subject.

    Returns:
        segments : [N, 4, segment_samples]
        labels   : [N]  int
    """
    srate          = 256   # Emognition MUSE sampling rate
    segment_samples = int(cfg.segment_duration * srate)
    lead_in_samples = int(cfg.lead_in_duration  * srate)
    baseline_samples = int(cfg.baseline_duration * srate)
    segment_len     = srate    # 1-s baseline averaging segments

    min_needed = lead_in_samples + baseline_samples + segment_samples

    trial_files = _emog_find_trials(subject_id, cfg)
    if not trial_files:
        logger.warning(f"Emognition: no trials found for subject '{subject_id}'")
        return np.empty((0, 4, segment_samples), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    all_segs, all_labels = [], []
    skipped = 0

    for fpath, label in trial_files:
        fname  = os.path.basename(fpath)
        signal = _emog_read_json(fpath)
        if signal is None:
            logger.warning(f"  Skipping {fname}: empty after quality filter")
            skipped += 1
            continue

        if signal.shape[1] < min_needed:
            logger.warning(
                f"  Skipping {fname}: only {signal.shape[1]} samples "
                f"(need ≥ {min_needed})"
            )
            skipped += 1
            continue

        # discard lead-in
        trimmed  = signal[:, lead_in_samples:]
        baseline = trimmed[:, :baseline_samples]
        stimulus = trimmed[:, baseline_samples:]

        try:
            clean = _baseline_removal(stimulus, baseline, segment_len)
        except Exception as e:
            logger.warning(f"  Skipping {fname}: baseline removal error: {e}")
            skipped += 1
            continue

        clean = _zscore_1d(clean)
        segs  = _segment_signal(clean, segment_samples)  # [N_s, 4, T_seg]

        if segs.shape[0] == 0:
            logger.warning(f"  Skipping {fname}: 0 segments produced")
            skipped += 1
            continue

        all_segs.append(segs)
        all_labels.extend([label] * segs.shape[0])

        logger.debug(
            f"  Emognition {subject_id} | {fname} | "
            f"label={label} | segments={segs.shape[0]}"
        )

    if not all_segs:
        return np.empty((0, 4, segment_samples), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    segments = np.concatenate(all_segs, axis=0)
    labels   = np.array(all_labels, dtype=np.int64)

    logger.info(
        f"Emognition subject {subject_id}: {segments.shape[0]} segments "
        f"loaded, {skipped} skipped | "
        f"class dist {np.bincount(labels, minlength=cfg.n_classes).tolist()}"
    )
    return segments, labels


def load_emognition_all_subjects(
    cfg: WEAVEConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool all Emognition subjects into one segment matrix."""
    ids = get_emognition_subject_ids(cfg)
    all_segs, all_labels = [], []
    for sid in ids:
        segs, labels = load_emognition_subject(sid, cfg)
        if segs.shape[0]:
            all_segs.append(segs)
            all_labels.append(labels)
    if not all_segs:
        raise RuntimeError("No Emognition segments loaded — check data_path.")
    return np.concatenate(all_segs, axis=0), np.concatenate(all_labels, axis=0)


# ════════════════════════════════════════════════════════════════════════════
#  UNIFIED API
# ════════════════════════════════════════════════════════════════════════════

def load_data(
    cfg: WEAVEConfig,
    subject_id=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load segments and labels for the configured dataset.

    Args:
        cfg        : WEAVEConfig
        subject_id : int (DEAP) | str (Emognition) | None → all subjects

    Returns:
        segments : [N, C, T]
        labels   : [N]
    """
    is_emog = cfg.dataset.lower() == "emognition"

    if subject_id is not None:
        if is_emog:
            return load_emognition_subject(str(subject_id), cfg)
        else:
            return load_deap_subject(int(subject_id), cfg)
    else:
        if is_emog:
            return load_emognition_all_subjects(cfg)
        else:
            return load_deap_all_subjects(cfg)
