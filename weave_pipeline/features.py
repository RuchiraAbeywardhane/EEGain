"""
WEAVE Feature Extraction
========================
WEAVE = Wavelet Entropy and AVErage wavelet coefficient.

For each EEG segment and each channel:
    1.  Decompose signal with DWT (Daubechies-5, db5)
    2.  Auto-map frequency bands (alpha / beta / gamma) to DWT detail levels
        based on the actual sampling rate — works for both 128 Hz (DEAP) and
        256 Hz (Emognition).
    3.  For each retained band extract:
          a. Wavelet Entropy  – energy-normalised Shannon entropy of coefficients
          b. Mean Coefficient – arithmetic mean of absolute coefficient values
    4.  Concatenate → 6 features per channel (2 features × 3 bands)
       → 192 features for 32-ch DEAP,  24 features for 4-ch Emognition

DWT level ↔ frequency band mapping
------------------------------------
At level L of a DWT with sampling rate Fs, the detail subband covers:
    [ Fs / 2^(L+1),  Fs / 2^L ]  Hz

We assign each detail level to whichever named band has the most overlap with
that frequency interval, then keep only the levels that correspond to the
retained bands (alpha, beta, gamma by default).

Example – DEAP (128 Hz, 5-level decomposition):
    Level 1 detail : 32–64 Hz  → gamma
    Level 2 detail : 16–32 Hz  → beta
    Level 3 detail : 8–16  Hz  → alpha
    Level 4 detail : 4–8   Hz  → theta
    Level 5 detail : 2–4   Hz  → delta
    Approximation  : 0–2   Hz  → (sub-delta, discarded)

Example – Emognition (256 Hz, 6-level decomposition):
    Level 1 detail : 64–128 Hz → above gamma (discarded)
    Level 2 detail : 32–64  Hz → gamma
    Level 3 detail : 16–32  Hz → beta
    Level 4 detail : 8–16   Hz → alpha
    Level 5 detail : 4–8    Hz → theta
    Level 6 detail : 2–4    Hz → delta
    Approximation  : 0–2    Hz → (sub-delta, discarded)
"""

import numpy as np
import pywt
from typing import Dict, List, Tuple

from .config import WEAVEConfig


# ── DWT level → frequency band mapping ──────────────────────────────────────

def build_level_band_map(
    sampling_rate: int,
    n_levels: int,
    band_hz: Dict[str, Tuple[float, float]],
) -> Dict[int, str]:
    """
    Return {detail_level: band_name} for every DWT detail level that
    overlaps with a named frequency band.

    Args:
        sampling_rate : signal sampling rate in Hz
        n_levels      : number of DWT decomposition levels
        band_hz       : dict mapping band name → (low_hz, high_hz)

    Returns:
        mapping : dict  {level (int, 1-based) : band_name (str)}
                  Only levels that fall inside a named band are included.
    """
    mapping: Dict[int, str] = {}
    for level in range(1, n_levels + 1):
        # Detail subband frequency range at this level
        f_high = sampling_rate / (2 ** level)
        f_low  = sampling_rate / (2 ** (level + 1))

        best_band   = None
        best_overlap = 0.0
        for band_name, (b_low, b_high) in band_hz.items():
            overlap = max(0.0, min(f_high, b_high) - max(f_low, b_low))
            if overlap > best_overlap:
                best_overlap = overlap
                best_band    = band_name

        if best_band is not None and best_overlap > 0.0:
            mapping[level] = best_band

    return mapping


def get_retained_levels(cfg: WEAVEConfig) -> Dict[str, int]:
    """
    Return {band_name: detail_level} for the retained bands only.

    Uses the auto-scaled mapping so the correct DWT levels are chosen
    regardless of whether sampling_rate is 128 Hz or 256 Hz.

    Returns:
        dict mapping each retained band name → DWT detail level (1-based)
    """
    # Use enough levels to cover down to ~1 Hz
    n_levels = int(np.floor(np.log2(cfg.sampling_rate))) - 1
    full_map = build_level_band_map(cfg.sampling_rate, n_levels, cfg.band_hz)

    # Invert: band_name → level  (keep only retained bands)
    band_to_level: Dict[str, int] = {}
    for level, band in full_map.items():
        if band in cfg.retained_bands and band not in band_to_level:
            band_to_level[band] = level

    # Verify all requested bands are covered
    missing = [b for b in cfg.retained_bands if b not in band_to_level]
    if missing:
        raise ValueError(
            f"Could not map bands {missing} to any DWT level at "
            f"{cfg.sampling_rate} Hz.  Check band_hz ranges in WEAVEConfig."
        )
    return band_to_level


# ── Single-channel WEAVE ─────────────────────────────────────────────────────

def _wavelet_entropy(coeffs: np.ndarray) -> float:
    """
    Shannon entropy of the normalised energy distribution of wavelet coefficients.

    E_j = c_j^2
    p_j = E_j / sum(E)          # probability distribution
    H   = -sum(p_j * log2(p_j)) # entropy (bits)

    Returns 0.0 for an all-zero subband.
    """
    energy = coeffs ** 2
    total  = energy.sum()
    if total < 1e-12:
        return 0.0
    p   = energy / total
    # Avoid log(0): mask out zero probabilities
    p   = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _mean_coefficient(coeffs: np.ndarray) -> float:
    """Mean of absolute wavelet coefficients — signal magnitude proxy."""
    return float(np.mean(np.abs(coeffs)))


def weave_channel(
    signal_1d: np.ndarray,
    band_to_level: Dict[str, int],
    wavelet: str = "db5",
    retained_bands: List[str] = None,
) -> np.ndarray:
    """
    Extract WEAVE features from a single 1-D EEG channel segment.

    Args:
        signal_1d      : np.ndarray  shape [T]  – single-channel EEG segment
        band_to_level  : dict  {band_name → DWT detail level}
        wavelet        : PyWavelets wavelet string (default "db5")
        retained_bands : ordered list of band names to use
                         (determines feature vector order)

    Returns:
        features : np.ndarray  shape [2 * n_bands]
                   [entropy_band0, mean_band0, entropy_band1, mean_band1, …]
    """
    if retained_bands is None:
        retained_bands = list(band_to_level.keys())

    max_level = max(band_to_level.values())

    # Full DWT decomposition up to max_level
    # pywt.wavedec returns [cA_n, cD_n, cD_(n-1), …, cD_1]
    coeffs_list = pywt.wavedec(signal_1d, wavelet, level=max_level)
    # Index mapping: detail level L → coeffs_list[max_level - L + 1]
    #   level 1 → coeffs_list[-1]  (highest-frequency details)
    #   level N → coeffs_list[1]

    features = []
    for band in retained_bands:
        level  = band_to_level[band]
        idx    = max_level - level + 1          # position in wavedec output
        coeffs = coeffs_list[idx]
        features.append(_wavelet_entropy(coeffs))
        features.append(_mean_coefficient(coeffs))

    return np.array(features, dtype=np.float32)


# ── Full-segment WEAVE (all channels) ────────────────────────────────────────

def weave_segment(
    segment: np.ndarray,
    cfg: WEAVEConfig,
    band_to_level: Dict[str, int] = None,
) -> np.ndarray:
    """
    Extract WEAVE features for one EEG segment (all channels).

    Args:
        segment       : np.ndarray  shape [C, T]
        cfg           : WEAVEConfig
        band_to_level : pre-computed {band → level} (pass to avoid recomputing)

    Returns:
        feature_vector : np.ndarray  shape [C * 2 * n_bands]
                         Channels are concatenated in order:
                         [ch0_alpha_ent, ch0_alpha_mean, ch0_beta_ent, … chN_gamma_mean]
    """
    if band_to_level is None:
        band_to_level = get_retained_levels(cfg)

    C = segment.shape[0]
    channel_feats = []
    for c in range(C):
        ch_feat = weave_channel(
            segment[c],
            band_to_level  = band_to_level,
            wavelet        = cfg.wavelet,
            retained_bands = cfg.retained_bands,
        )
        channel_feats.append(ch_feat)

    return np.concatenate(channel_feats, axis=0)   # [C * 2 * n_bands]


# ── Batch extraction (list of segments) ──────────────────────────────────────

def extract_weave_features(
    segments: np.ndarray,
    cfg: WEAVEConfig,
) -> np.ndarray:
    """
    Extract WEAVE features for a batch of EEG segments.

    Args:
        segments : np.ndarray  shape [N, C, T]  – N segments, C channels, T samples
        cfg      : WEAVEConfig

    Returns:
        X : np.ndarray  shape [N, C * 2 * n_bands]
    """
    band_to_level = get_retained_levels(cfg)

    X = np.stack(
        [weave_segment(segments[i], cfg, band_to_level) for i in range(len(segments))],
        axis=0,
    )
    return X.astype(np.float32)
