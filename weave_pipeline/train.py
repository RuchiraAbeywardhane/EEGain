"""
WEAVE + SVM Training and Evaluation
=====================================
Evaluation strategy (as per paper):
    - 30 random train/test splits  (80% train / 20% test)
    - Each repetition uses a DIFFERENT random seed derived from cfg.seed + rep
    - NMI channel ranking is recomputed on the training fold of each repetition
      (no leakage into the test set)

Two classifiers are always run per repetition:
    1. Full channels  – all C channels, C*6 WEAVE features
    2. Reduced        – top `min_channels` channels selected by NMI from train set

Final results report mean ± std accuracy and F1 over the 30 repetitions.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from .config import WEAVEConfig
from .features import extract_weave_features
from .channel_selection import iterative_channel_reduction

logger = logging.getLogger("WEAVE.Train")


# ── SVM factory ──────────────────────────────────────────────────────────────

def _make_svm(cfg: WEAVEConfig) -> SVC:
    """
    Build an SVM with RBF kernel.
    The paper uses SMO; scikit-learn's SVC uses LibSVM which implements SMO
    internally, so this is equivalent.
    """
    return SVC(
        kernel      = cfg.svm_kernel,
        C           = cfg.svm_C,
        gamma       = cfg.svm_gamma,
        decision_function_shape = "ovr",   # one-vs-rest for multi-class
        random_state = None,               # SVC doesn't use random_state for RBF
        max_iter     = -1,                 # no iteration cap
    )


# ── Single repetition ────────────────────────────────────────────────────────

def _run_one_rep(
    X: np.ndarray,
    y: np.ndarray,
    cfg: WEAVEConfig,
    rep_seed: int,
    n_features_per_channel: int,
) -> Dict:
    """
    One random 80/20 split → extract WEAVE → train SVM → evaluate.

    Returns a dict with keys:
        full_acc, full_f1,
        reduced_acc, reduced_f1,
        ranked_channels  (list of ints, best-first)
    """
    # ── Stratified 80/20 split ────────────────────────────────────────────
    sss = StratifiedShuffleSplit(
        n_splits    = 1,
        test_size   = cfg.test_size,
        random_state = rep_seed,
    )
    train_idx, test_idx = next(sss.split(X, y))

    segs_train, y_train = X[train_idx], y[train_idx]
    segs_test,  y_test  = X[test_idx],  y[test_idx]

    # ── WEAVE feature extraction ──────────────────────────────────────────
    X_train = extract_weave_features(segs_train, cfg)   # [N_train, C*n_feat]
    X_test  = extract_weave_features(segs_test,  cfg)   # [N_test,  C*n_feat]

    # ── StandardScaler (fit on train only) ───────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_channels = cfg.n_eeg_channels

    # ── Run 1: Full channels ──────────────────────────────────────────────
    svm_full = _make_svm(cfg)
    svm_full.fit(X_train, y_train)
    pred_full     = svm_full.predict(X_test)
    full_acc      = accuracy_score(y_test, pred_full)
    full_f1       = f1_score(y_test, pred_full,
                             average="weighted", zero_division=0)

    # ── Run 2: NMI channel reduction (train set only) ─────────────────────
    (X_tr_red, X_te_red,
     ranked_indices, keep_k) = iterative_channel_reduction(
        X_train               = X_train,
        y_train               = y_train,
        X_test                = X_test,
        n_channels            = n_channels,
        n_features_per_channel = n_features_per_channel,
        min_channels          = cfg.min_channels,
        random_state          = rep_seed,
    )

    svm_red = _make_svm(cfg)
    svm_red.fit(X_tr_red, y_train)
    pred_red      = svm_red.predict(X_te_red)
    reduced_acc   = accuracy_score(y_test, pred_red)
    reduced_f1    = f1_score(y_test, pred_red,
                             average="weighted", zero_division=0)

    return {
        "full_acc":        full_acc,
        "full_f1":         full_f1,
        "reduced_acc":     reduced_acc,
        "reduced_f1":      reduced_f1,
        "ranked_channels": ranked_indices.tolist(),
        "kept_channels":   keep_k,
    }


# ── 30-repetition evaluation ─────────────────────────────────────────────────

def run_weave_evaluation(
    segments: np.ndarray,
    labels:   np.ndarray,
    cfg: WEAVEConfig,
) -> Dict:
    """
    Run the full WEAVE + SVM evaluation over cfg.n_repetitions random splits.

    Args:
        segments : np.ndarray  shape [N, C, T]
        labels   : np.ndarray  shape [N]
        cfg      : WEAVEConfig

    Returns:
        summary dict with per-repetition and aggregate results
    """
    n_feat_per_ch = 2 * len(cfg.retained_bands)   # entropy + mean per band

    logger.info(
        f"WEAVE evaluation | {cfg.dataset.upper()} | "
        f"N={len(labels)} segments | {cfg.n_eeg_channels} channels | "
        f"{n_feat_per_ch * cfg.n_eeg_channels} total features | "
        f"{cfg.n_repetitions} repetitions"
    )

    rep_results: List[Dict] = []

    for rep in range(cfg.n_repetitions):
        rep_seed = cfg.seed + rep
        result   = _run_one_rep(
            X                      = segments,
            y                      = labels,
            cfg                    = cfg,
            rep_seed               = rep_seed,
            n_features_per_channel = n_feat_per_ch,
        )
        rep_results.append(result)

        print(
            f"  Rep {rep + 1:2d}/{cfg.n_repetitions} | "
            f"Full  acc={result['full_acc']:.4f}  f1={result['full_f1']:.4f} | "
            f"Reduced({result['kept_channels']}ch) "
            f"acc={result['reduced_acc']:.4f}  f1={result['reduced_f1']:.4f}"
        )

    return _summarise(rep_results, cfg)


# ── Summary ───────────────────────────────────────────────────────────────────

def _summarise(rep_results: List[Dict], cfg: WEAVEConfig) -> Dict:
    full_accs    = [r["full_acc"]    for r in rep_results]
    full_f1s     = [r["full_f1"]     for r in rep_results]
    red_accs     = [r["reduced_acc"] for r in rep_results]
    red_f1s      = [r["reduced_f1"]  for r in rep_results]

    # Aggregate channel ranking: average position across repetitions
    n_ch = cfg.n_eeg_channels
    rank_matrix = np.zeros((len(rep_results), n_ch), dtype=np.int32)
    for i, r in enumerate(rep_results):
        rank_matrix[i] = np.array(r["ranked_channels"])
    # Most consistently top channel = smallest average rank position
    avg_rank_pos = np.zeros(n_ch)
    for rep_ranking in rank_matrix:
        for pos, ch in enumerate(rep_ranking):
            avg_rank_pos[ch] += pos
    avg_rank_pos /= len(rep_results)
    consensus_ranking = np.argsort(avg_rank_pos).tolist()

    summary = {
        "rep_results":        rep_results,
        # Full-channel results
        "full_mean_acc":      float(np.mean(full_accs)),
        "full_std_acc":       float(np.std(full_accs)),
        "full_mean_f1":       float(np.mean(full_f1s)),
        "full_std_f1":        float(np.std(full_f1s)),
        # Reduced-channel results
        "reduced_mean_acc":   float(np.mean(red_accs)),
        "reduced_std_acc":    float(np.std(red_accs)),
        "reduced_mean_f1":    float(np.mean(red_f1s)),
        "reduced_std_f1":     float(np.std(red_f1s)),
        "kept_channels":      rep_results[0]["kept_channels"],
        "total_channels":     n_ch,
        "consensus_ranking":  consensus_ranking,
    }

    print(f"\n{'='*65}")
    print(f"  WEAVE + SVM  –  {cfg.n_repetitions}-Repetition Summary")
    print(f"  Dataset  : {cfg.dataset.upper()}"
          + (f"  |  Label: {cfg.label_type}" if cfg.dataset == "deap" else ""))
    print(f"  Segments : {cfg.n_eeg_channels} ch  |  "
          f"{cfg.segment_duration}s windows  |  "
          f"{2 * len(cfg.retained_bands)} feat/ch  |  "
          f"{2 * len(cfg.retained_bands) * cfg.n_eeg_channels} total feat")
    print(f"{'─'*65}")
    print(f"  Full channels ({cfg.n_eeg_channels} ch)")
    print(f"    Accuracy : {summary['full_mean_acc']:.4f} ± {summary['full_std_acc']:.4f}")
    print(f"    F1-score : {summary['full_mean_f1']:.4f} ± {summary['full_std_f1']:.4f}")
    print(f"  Reduced channels ({summary['kept_channels']} ch)")
    print(f"    Accuracy : {summary['reduced_mean_acc']:.4f} ± {summary['reduced_std_acc']:.4f}")
    print(f"    F1-score : {summary['reduced_mean_f1']:.4f} ± {summary['reduced_std_f1']:.4f}")
    print(f"{'='*65}\n")

    return summary


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(summary: Dict, log_path: str, cfg: WEAVEConfig):
    """Write a human-readable results file."""
    import os
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    n_feat_per_ch   = 2 * len(cfg.retained_bands)
    total_feat_full = cfg.n_eeg_channels * n_feat_per_ch
    total_feat_red  = summary["kept_channels"] * n_feat_per_ch

    with open(log_path, "w") as f:
        f.write("WEAVE + SVM Pipeline Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset         : {cfg.dataset.upper()}\n")
        if cfg.dataset.lower() == "deap":
            f.write(f"Label type      : {cfg.label_type} "
                    f"(threshold > {cfg.ground_truth_threshold})\n")
        else:
            f.write(f"Classes         : {cfg.class_names}\n")
        f.write(f"Wavelet         : {cfg.wavelet}\n")
        f.write(f"Retained bands  : {cfg.retained_bands}\n")
        f.write(f"Segment duration: {cfg.segment_duration} s\n")
        f.write(f"Channels        : {cfg.n_eeg_channels}  →  {summary['kept_channels']} "
                f"(after NMI reduction)\n")
        f.write(f"Features        : {total_feat_full} full  →  {total_feat_red} reduced\n")
        f.write(f"Repetitions     : {cfg.n_repetitions}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Repetition Results\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Rep':>4}  {'Full Acc':>9}  {'Full F1':>8}  "
                f"{'Red Acc':>8}  {'Red F1':>7}\n")
        f.write("-" * 50 + "\n")
        for r in summary["rep_results"]:
            idx = summary["rep_results"].index(r) + 1
            f.write(
                f"{idx:4d}  {r['full_acc']:9.4f}  {r['full_f1']:8.4f}  "
                f"{r['reduced_acc']:8.4f}  {r['reduced_f1']:7.4f}\n"
            )
        f.write("-" * 50 + "\n\n")

        f.write("Summary\n")
        f.write("-" * 50 + "\n")
        f.write(f"Full channels ({cfg.n_eeg_channels} ch, {total_feat_full} feat)\n")
        f.write(f"  Accuracy : {summary['full_mean_acc']:.4f} ± {summary['full_std_acc']:.4f}\n")
        f.write(f"  F1-score : {summary['full_mean_f1']:.4f} ± {summary['full_std_f1']:.4f}\n\n")
        f.write(f"Reduced channels ({summary['kept_channels']} ch, {total_feat_red} feat)\n")
        f.write(f"  Accuracy : {summary['reduced_mean_acc']:.4f} ± {summary['reduced_std_acc']:.4f}\n")
        f.write(f"  F1-score : {summary['reduced_mean_f1']:.4f} ± {summary['reduced_std_f1']:.4f}\n\n")

        f.write(f"Consensus NMI channel ranking (best → worst):\n")
        f.write(f"  {summary['consensus_ranking']}\n")

    logger.info(f"Results saved → {log_path}")
