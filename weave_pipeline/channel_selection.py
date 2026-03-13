"""
NMI-Based Channel Selection
============================
Ranks EEG channels by Normalised Mutual Information (NMI) between their
WEAVE feature block and the emotion labels, then iteratively removes the
lowest-ranked channel until `min_channels` is reached.

NMI definition used here (symmetric, in [0, 1]):
    NMI(X; Y) = 2 * I(X; Y) / (H(X) + H(Y))

Because the WEAVE feature block for each channel is a multi-dimensional
vector, we compute MI as the average MI across the individual features in
that channel's block, then normalise by the average entropy values.

This is computed per-channel:
    score_c = mean over features f in block_c of:
                  MI(f, y) / (0.5 * H(f) + 0.5 * H(y))

sklearn's mutual_info_classif is used — it estimates MI between each
continuous feature and the discrete class label via k-NN density estimation.
"""

import numpy as np
from typing import List, Tuple

from sklearn.feature_selection import mutual_info_classif


def _channel_feature_indices(
    channel_idx: int,
    n_features_per_channel: int,
) -> np.ndarray:
    """Return the column indices in the full feature matrix for one channel."""
    start = channel_idx * n_features_per_channel
    return np.arange(start, start + n_features_per_channel)


def _entropy_discrete_approx(values: np.ndarray, n_bins: int = 20) -> float:
    """
    Approximate Shannon entropy of a continuous variable via histogram binning.
    Used only to normalise MI into [0, 1].
    Returns entropy in nats.
    """
    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _label_entropy(y: np.ndarray) -> float:
    """Shannon entropy of discrete labels (nats)."""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_channel_nmi_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int,
    n_features_per_channel: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute one NMI score per EEG channel.

    Args:
        X                     : np.ndarray  shape [N, n_channels * n_features_per_channel]
        y                     : np.ndarray  shape [N]  – integer class labels
        n_channels            : int
        n_features_per_channel: int  (= 2 * n_retained_bands, e.g. 6 for 3 bands)
        random_state          : int

    Returns:
        scores : np.ndarray  shape [n_channels]
                 NMI score for each channel (higher → more informative)
    """
    # MI of every individual feature with y  (shape [n_total_features])
    mi_per_feature = mutual_info_classif(
        X, y,
        discrete_features=False,
        random_state=random_state,
    )

    H_y = _label_entropy(y)

    scores = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        idx         = _channel_feature_indices(ch, n_features_per_channel)
        mi_values   = mi_per_feature[idx]           # [n_features_per_channel]

        # Average entropy of the channel's features (approximated)
        h_feats = np.array([
            _entropy_discrete_approx(X[:, i]) for i in idx
        ])
        H_X_avg = float(h_feats.mean()) if len(h_feats) > 0 else 1e-8

        denom = H_X_avg + H_y
        if denom < 1e-12:
            scores[ch] = 0.0
        else:
            # NMI = 2*I(X;Y) / (H(X)+H(Y))  averaged over features in block
            scores[ch] = float(2.0 * mi_values.mean() / denom)

    return scores


def rank_channels_by_nmi(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int,
    n_features_per_channel: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank channels from most to least informative using NMI.

    Returns:
        ranked_indices : np.ndarray  shape [n_channels]  – channel indices, best first
        nmi_scores     : np.ndarray  shape [n_channels]  – NMI score per channel
    """
    scores         = compute_channel_nmi_scores(
        X, y, n_channels, n_features_per_channel, random_state
    )
    ranked_indices = np.argsort(scores)[::-1]   # descending
    return ranked_indices, scores


def select_top_channels(
    X: np.ndarray,
    ranked_indices: np.ndarray,
    n_channels: int,
    n_features_per_channel: int,
    keep_k: int,
) -> np.ndarray:
    """
    Return feature matrix keeping only the top-k channels (by NMI rank).

    Args:
        X                     : np.ndarray  shape [N, n_channels * n_features_per_channel]
        ranked_indices        : np.ndarray  shape [n_channels]  – best channel first
        n_channels            : int
        n_features_per_channel: int
        keep_k                : int – number of channels to retain

    Returns:
        X_reduced : np.ndarray  shape [N, keep_k * n_features_per_channel]
    """
    top_channels = ranked_indices[:keep_k]
    cols = np.concatenate([
        _channel_feature_indices(ch, n_features_per_channel)
        for ch in sorted(top_channels)    # keep original channel order
    ])
    return X[:, cols]


def iterative_channel_reduction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_channels: int,
    n_features_per_channel: int,
    min_channels: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute NMI ranking on the training set, then reduce both train and test
    down to `min_channels` channels by dropping the lowest-ranked ones.

    The ranking is computed ONLY on the training set to avoid test-set leakage.

    Args:
        X_train               : [N_train, n_ch * n_feat]
        y_train               : [N_train]
        X_test                : [N_test,  n_ch * n_feat]
        n_channels            : total channels before reduction
        n_features_per_channel: features per channel
        min_channels          : target number of channels after reduction
        random_state          : int

    Returns:
        X_train_reduced : [N_train, min_channels * n_features_per_channel]
        X_test_reduced  : [N_test,  min_channels * n_features_per_channel]
        ranked_indices  : [n_channels]  – full ranking for logging
        keep_k          : int           – min_channels (or n_channels if no reduction)
    """
    keep_k = max(min_channels, 1)
    keep_k = min(keep_k, n_channels)        # can't keep more than we have

    ranked_indices, nmi_scores = rank_channels_by_nmi(
        X_train, y_train,
        n_channels, n_features_per_channel,
        random_state=random_state,
    )

    X_train_reduced = select_top_channels(
        X_train, ranked_indices, n_channels, n_features_per_channel, keep_k
    )
    X_test_reduced = select_top_channels(
        X_test, ranked_indices, n_channels, n_features_per_channel, keep_k
    )

    return X_train_reduced, X_test_reduced, ranked_indices, keep_k
