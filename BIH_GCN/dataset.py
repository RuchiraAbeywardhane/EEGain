"""
Dataset loaders for the BIH_GCN pipeline.

Returns raw EEG segments [N, C, T], labels [N], clip_ids [N].
clip_ids is an integer array where every window from the same
original JSON file / DEAP trial shares the same ID.
train.py uses clip_ids to split at the CLIP level so that NO
window from a test clip ever appears in the train set.
"""

import os
import glob
import json
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional

from BIH_GCN.config import BIHGCNConfig

logger = logging.getLogger("BIH_GCN.Dataset")

# ── DEAP constants ─────────────────────────────────────────────────────────────
_LABEL_IDX        = {"V": 0, "A": 1, "D": 2, "L": 3}
_BASELINE_SAMPLES = 384      # 3 s × 128 Hz
_N_EEG_CHANNELS   = 32
_DEAP_SRATE       = 128

# ── Emognition class map ───────────────────────────────────────────────────────
_EMOG_CLASS_MAP = {
    "ENTHUSIASM": 0,
    "NEUTRAL":    1,
    "FEAR":       2,
    "SADNESS":    3,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _baseline_removal(trial: np.ndarray, baseline: np.ndarray,
                      segment_len: int) -> np.ndarray:
    C, T_base = baseline.shape
    n_seg     = T_base // segment_len
    if n_seg == 0:
        return trial - baseline.mean(axis=1, keepdims=True)
    segs      = np.stack(
        [baseline[:, i * segment_len:(i + 1) * segment_len] for i in range(n_seg)],
        axis=0)
    base_mean = segs.mean(axis=0)
    C_t, T_trial = trial.shape
    n_tiles   = T_trial // segment_len
    remainder = T_trial %  segment_len
    tiled     = np.tile(base_mean, (1, n_tiles))
    if remainder > 0:
        tiled = np.concatenate([tiled, base_mean[:, :remainder]], axis=1)
    return trial - tiled[:, :T_trial]


# ⚠️  _zscore_1d is intentionally NOT called in the dataset loader.
# Normalisation must happen AFTER the train/test split in train.py
# to avoid data leakage.
def _zscore_1d(eeg: np.ndarray) -> np.ndarray:
    eeg = eeg.astype(np.float32, copy=True)
    for c in range(eeg.shape[0]):
        mu  = eeg[c].mean()
        sig = eeg[c].std()
        eeg[c] = (eeg[c] - mu) / max(sig, 1e-8)
    return eeg


def _segment_signal(eeg: np.ndarray, segment_samples: int) -> np.ndarray:
    C, T   = eeg.shape
    n_segs = T // segment_samples
    if n_segs == 0:
        return np.empty((0, C, segment_samples), dtype=np.float32)
    return np.stack(
        [eeg[:, i * segment_samples:(i + 1) * segment_samples]
         for i in range(n_segs)], axis=0
    ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  DEAP LOADER
# ══════════════════════════════════════════════════════════════════════════════

def get_deap_subject_ids(cfg: BIHGCNConfig) -> List[int]:
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
    cfg: BIHGCNConfig,
    clip_id_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpath = os.path.join(cfg.data_path, f"s{subject_id:02d}.dat")
    with open(fpath, "rb") as f:
        subject_data = pickle.load(f, encoding="latin-1")

    raw_data   = subject_data["data"]    # [40, 40, 8064]
    raw_labels = subject_data["labels"]  # [40, 4]

    label_col       = _LABEL_IDX.get(cfg.label_type.upper(), 0)
    segment_samples = cfg.segment_samples
    segment_len     = _DEAP_SRATE

    all_segs, all_labels, all_clip_ids = [], [], []

    for trial_idx in range(raw_data.shape[0]):
        full  = raw_data[trial_idx, :_N_EEG_CHANNELS, :]
        base  = full[:, :_BASELINE_SAMPLES]
        trial = full[:, _BASELINE_SAMPLES:]

        trial_clean = _baseline_removal(trial, base, segment_len)
        # ⚠️  NO _zscore_1d here — normalisation happens after split in train.py
        segs        = _segment_signal(trial_clean, segment_samples)

        if segs.shape[0] == 0:
            continue

        raw_score = float(raw_labels[trial_idx, label_col])
        label     = 1 if raw_score > cfg.ground_truth_threshold else 0
        clip_id   = clip_id_offset + trial_idx   # unique per DEAP trial

        all_segs.append(segs)
        all_labels.extend([label]   * segs.shape[0])
        all_clip_ids.extend([clip_id] * segs.shape[0])

    if not all_segs:
        logger.warning(f"DEAP subject {subject_id:02d}: no segments produced")
        empty = np.empty((0, _N_EEG_CHANNELS, segment_samples), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    segments  = np.concatenate(all_segs, axis=0)
    labels    = np.array(all_labels,   dtype=np.int64)
    clip_ids  = np.array(all_clip_ids, dtype=np.int64)
    logger.info(f"DEAP subject {subject_id:02d}: {segments.shape[0]} segments | "
                f"{len(np.unique(clip_ids))} clips | "
                f"class dist {np.bincount(labels).tolist()}")
    return segments, labels, clip_ids


def load_deap_all_subjects(cfg: BIHGCNConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = get_deap_subject_ids(cfg)
    all_segs, all_labels, all_clip_ids = [], [], []
    clip_offset = 0
    for sid in ids:
        segs, labels, clip_ids = load_deap_subject(sid, cfg, clip_id_offset=clip_offset)
        if segs.shape[0]:
            all_segs.append(segs)
            all_labels.append(labels)
            all_clip_ids.append(clip_ids)
            clip_offset = int(clip_ids.max()) + 1   # next subject's clips start after this
    if not all_segs:
        raise RuntimeError("No DEAP segments loaded — check data_path.")
    return (np.concatenate(all_segs,     axis=0),
            np.concatenate(all_labels,   axis=0),
            np.concatenate(all_clip_ids, axis=0))


# ══════════════════════════════════════════════════════════════════════════════
#  EMOGNITION LOADER
# ══════════════════════════════════════════════════════════════════════════════

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
    if m.all():     return a
    if not m.any(): return np.zeros_like(a)
    idx   = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _emog_is_stimulus_muse(fname: str) -> bool:
    parts = fname.replace(".json", "").split("_")
    return (len(parts) >= 4 and
            parts[2].upper() == "STIMULUS" and
            parts[3].upper() == "MUSE")


def _emog_parse_emotion(fname: str) -> str:
    return fname.split("_")[1].upper()


def _emog_parse_subject(fname: str) -> str:
    return fname.split("_")[0]


def _emog_read_json(fpath: str) -> Optional[np.ndarray]:
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

    mask = (np.isfinite(tp9[:L]) & np.isfinite(af7[:L]) &
            np.isfinite(af8[:L]) & np.isfinite(tp10[:L]))
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
    # ⚠️  NO DC removal (signal -= mean) here — normalisation happens after split in train.py
    return signal   # [4, T]


def get_emognition_subject_ids(cfg: BIHGCNConfig) -> List[str]:
    patterns = [
        os.path.join(cfg.data_path, "*",      "*.json"),
        os.path.join(cfg.data_path,            "*.json"),
        os.path.join(cfg.data_path, "*", "*", "*.json"),
    ]
    files    = sorted({p for pat in patterns for p in glob.glob(pat)})
    subjects = set()
    for fpath in files:
        fname = os.path.basename(fpath)
        if not _emog_is_stimulus_muse(fname):
            continue
        if _emog_parse_emotion(fname) in _EMOG_CLASS_MAP:
            subjects.add(_emog_parse_subject(fname))
    return sorted(subjects)


def _emog_find_trials(subject_id: str, cfg: BIHGCNConfig) -> List[Tuple[str, int]]:
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
        label = _EMOG_CLASS_MAP.get(_emog_parse_emotion(fname))
        if label is None:
            continue
        result.append((fpath, label))
    return sorted(result)


def load_emognition_subject(
    subject_id: str,
    cfg: BIHGCNConfig,
    clip_id_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    srate            = 256
    segment_samples  = cfg.segment_samples
    lead_in_samples  = int(cfg.lead_in_duration  * srate)
    baseline_samples = int(cfg.baseline_duration * srate)
    segment_len      = srate
    min_needed       = lead_in_samples + baseline_samples + segment_samples

    trial_files = _emog_find_trials(subject_id, cfg)
    if not trial_files:
        logger.warning(f"Emognition: no trials found for subject '{subject_id}'")
        empty = np.empty((0, 4, segment_samples), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    all_segs, all_labels, all_clip_ids = [], [], []
    skipped  = 0
    local_clip_id = clip_id_offset   # each JSON file = one clip

    for fpath, label in trial_files:
        fname  = os.path.basename(fpath)
        signal = _emog_read_json(fpath)

        if signal is None:
            logger.warning(f"  Skipping {fname}: empty after quality filter")
            skipped += 1
            continue
        if signal.shape[1] < min_needed:
            logger.warning(f"  Skipping {fname}: only {signal.shape[1]} samples "
                           f"(need ≥ {min_needed})")
            skipped += 1
            continue

        trimmed  = signal[:, lead_in_samples:]
        baseline = trimmed[:, :baseline_samples]
        stimulus = trimmed[:, baseline_samples:]

        try:
            clean = _baseline_removal(stimulus, baseline, segment_len)
        except Exception as e:
            logger.warning(f"  Skipping {fname}: baseline removal error: {e}")
            skipped += 1
            continue

        # ⚠️  NO _zscore_1d here — normalisation happens after split in train.py
        segs = _segment_signal(clean, segment_samples)

        if segs.shape[0] == 0:
            logger.warning(f"  Skipping {fname}: 0 segments produced")
            skipped += 1
            continue

        all_segs.append(segs)
        all_labels.extend([label]          * segs.shape[0])
        all_clip_ids.extend([local_clip_id] * segs.shape[0])  # all segs of this file share one clip id
        local_clip_id += 1                                      # next file = new clip id

        logger.debug(f"  {subject_id} | {fname} | clip_id={local_clip_id-1} | "
                     f"label={label} | segs={segs.shape[0]}")

    if not all_segs:
        empty = np.empty((0, 4, segment_samples), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    segments  = np.concatenate(all_segs, axis=0)
    labels    = np.array(all_labels,   dtype=np.int64)
    clip_ids  = np.array(all_clip_ids, dtype=np.int64)
    logger.info(
        f"Emognition subject {subject_id}: {segments.shape[0]} segments | "
        f"{len(np.unique(clip_ids))} clips | {skipped} skipped | "
        f"class dist {np.bincount(labels, minlength=cfg.n_classes).tolist()}"
    )
    return segments, labels, clip_ids


def load_emognition_all_subjects(cfg: BIHGCNConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = get_emognition_subject_ids(cfg)
    logger.info(f"Found {len(ids)} Emognition subjects: {ids}")
    all_segs, all_labels, all_clip_ids = [], [], []
    clip_offset = 0
    for sid in ids:
        segs, labels, clip_ids = load_emognition_subject(sid, cfg, clip_id_offset=clip_offset)
        if segs.shape[0]:
            all_segs.append(segs)
            all_labels.append(labels)
            all_clip_ids.append(clip_ids)
            clip_offset = int(clip_ids.max()) + 1   # next subject's clips start after this
    if not all_segs:
        raise RuntimeError("No Emognition segments loaded — check data_path.")
    return (np.concatenate(all_segs,     axis=0),
            np.concatenate(all_labels,   axis=0),
            np.concatenate(all_clip_ids, axis=0))


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED API
# ══════════════════════════════════════════════════════════════════════════════

def load_data(
    cfg: BIHGCNConfig,
    subject_id=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EEG segments, labels, and clip_ids for the configured dataset.

    Args:
        cfg        : BIHGCNConfig
        subject_id : int (DEAP) | str (Emognition) | None → all subjects

    Returns:
        segments : [N, C, T]  float32
        labels   : [N]        int64
        clip_ids : [N]        int64  — all segments from the same recording
                                       share the same clip_id. Used to split
                                       at the CLIP level in train.py.
    """
    is_emog = cfg.dataset.lower() == "emognition"

    if subject_id is not None:
        if is_emog:
            segments, labels, clip_ids = load_emognition_subject(str(subject_id), cfg)
        else:
            segments, labels, clip_ids = load_deap_subject(int(subject_id), cfg)
    else:
        if is_emog:
            segments, labels, clip_ids = load_emognition_all_subjects(cfg)
        else:
            segments, labels, clip_ids = load_deap_all_subjects(cfg)

    n_clips = len(np.unique(clip_ids))
    logger.info(f"Total: {len(segments)} segments | {n_clips} clips | "
                f"shape={segments.shape} | "
                f"class dist={np.bincount(labels, minlength=cfg.n_classes).tolist()}")
    return segments, labels, clip_ids
