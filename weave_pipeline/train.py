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
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)

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


# ── Per-class metrics helper ──────────────────────────────────────────────────

def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> Dict:
    """
    Return per-class accuracy, precision, recall, and F1.

    Per-class accuracy = (TP + TN) / N  for each class treated as binary
                       = diagonal of the normalised confusion matrix when
                         expressed as the fraction of true-class samples
                         correctly predicted (recall per class).

    We use sklearn's 'macro' with labels= to get one value per class.

    Returns a dict:
        {
          "class_acc"      : [n_classes]  – recall per class (= class-wise accuracy)
          "class_precision": [n_classes]
          "class_recall"   : [n_classes]  – same as class_acc
          "class_f1"       : [n_classes]
          "confusion"      : [n_classes, n_classes]  – raw counts
        }
    """
    labels = list(range(n_classes))

    precision = precision_score(y_true, y_pred, labels=labels,
                                average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, labels=labels,
                                average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, labels=labels,
                                average=None, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class accuracy = recall (proportion of class-i samples predicted correctly)
    return {
        "class_acc"      : recall.tolist(),      # recall == class-wise accuracy
        "class_precision": precision.tolist(),
        "class_recall"   : recall.tolist(),
        "class_f1"       : f1.tolist(),
        "confusion"      : cm.tolist(),
    }


# ── Trial-level split ─────────────────────────────────────────────────────────

def _trial_level_split(
    trial_ids: np.ndarray,
    labels:    np.ndarray,
    test_size: float,
    rng:       np.random.Generator,
) -> tuple:
    """
    Split at the TRIAL level so that every window of a given trial lands
    entirely in train OR entirely in test — never both.

    Strategy
    --------
    1. Get the unique trial IDs and the majority class-label for each trial
       (a trial's label is constant — all windows share the same emotion).
    2. Stratify: for each class, shuffle its trial IDs and take the last
       ceil(n_class_trials * test_size) as test trials.
    3. Expand trial IDs back to window indices.

    Returns
    -------
    train_idx, test_idx : np.ndarray of window-level integer indices
    """
    unique_tids = np.unique(trial_ids)

    # Representative label for each trial (all windows are the same class)
    tid_labels = np.array([
        labels[trial_ids == tid][0] for tid in unique_tids
    ])

    train_tids, test_tids = [], []
    for cls in np.unique(tid_labels):
        cls_tids = unique_tids[tid_labels == cls]
        rng.shuffle(cls_tids)
        n_test   = max(1, int(np.ceil(len(cls_tids) * test_size)))
        test_tids.extend(cls_tids[-n_test:].tolist())
        train_tids.extend(cls_tids[:-n_test].tolist())

    train_set = set(train_tids)
    test_set  = set(test_tids)

    train_idx = np.where(np.isin(trial_ids, list(train_set)))[0]
    test_idx  = np.where(np.isin(trial_ids, list(test_set)))[0]

    return train_idx, test_idx


# ── Single repetition ────────────────────────────────────────────────────────

def _run_one_rep(
    X:         np.ndarray,
    y:         np.ndarray,
    trial_ids: np.ndarray,
    cfg:       WEAVEConfig,
    rep_seed:  int,
    n_features_per_channel: int,
) -> Dict:
    """
    One trial-level 80/20 split → WEAVE extraction → train SVM → evaluate.
    No window from a test trial is ever present in the training set.
    """
    rng = np.random.default_rng(rep_seed)

    train_idx, test_idx = _trial_level_split(
        trial_ids = trial_ids,
        labels    = y,
        test_size = cfg.test_size,
        rng       = rng,
    )

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
    full_pc       = _per_class_metrics(y_test, pred_full, cfg.n_classes)

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
    red_pc        = _per_class_metrics(y_test, pred_red, cfg.n_classes)

    return {
        "full_acc":        full_acc,
        "full_f1":         full_f1,
        "full_per_class":  full_pc,
        "reduced_acc":     reduced_acc,
        "reduced_f1":      reduced_f1,
        "red_per_class":   red_pc,
        "ranked_channels": ranked_indices.tolist(),
        "kept_channels":   keep_k,
        "n_train_windows": len(train_idx),
        "n_test_windows":  len(test_idx),
        "n_train_trials":  len(np.unique(trial_ids[train_idx])),
        "n_test_trials":   len(np.unique(trial_ids[test_idx])),
    }


# ── 30-repetition evaluation ─────────────────────────────────────────────────

def run_weave_evaluation(
    segments:  np.ndarray,
    labels:    np.ndarray,
    trial_ids: np.ndarray,
    cfg:       WEAVEConfig,
) -> Dict:
    """
    Run the full WEAVE + SVM evaluation over cfg.n_repetitions random splits.

    Args:
        segments  : [N, C, T]
        labels    : [N]
        trial_ids : [N]  – integer trial ID per window (from dataset.load_data)
        cfg       : WEAVEConfig

    Returns:
        summary dict
    """
    n_feat_per_ch  = 2 * len(cfg.retained_bands)
    n_unique_trials = len(np.unique(trial_ids))

    logger.info(
        f"WEAVE evaluation | {cfg.dataset.upper()} | "
        f"N={len(labels)} windows | {n_unique_trials} trials | "
        f"{cfg.n_eeg_channels} channels | "
        f"{n_feat_per_ch * cfg.n_eeg_channels} total features | "
        f"{cfg.n_repetitions} repetitions | split=TRIAL-LEVEL"
    )

    rep_results: List[Dict] = []

    for rep in range(cfg.n_repetitions):
        rep_seed = cfg.seed + rep
        result   = _run_one_rep(
            X                      = segments,
            y                      = labels,
            trial_ids              = trial_ids,
            cfg                    = cfg,
            rep_seed               = rep_seed,
            n_features_per_channel = n_feat_per_ch,
        )
        rep_results.append(result)

        print(
            f"  Rep {rep + 1:2d}/{cfg.n_repetitions} | "
            f"train {result['n_train_trials']} trials/{result['n_train_windows']} win  "
            f"test {result['n_test_trials']} trials/{result['n_test_windows']} win | "
            f"Full acc={result['full_acc']:.4f} f1={result['full_f1']:.4f} | "
            f"Reduced({result['kept_channels']}ch) "
            f"acc={result['reduced_acc']:.4f} f1={result['reduced_f1']:.4f}"
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

    # Aggregate per-class metrics across repetitions
    def _agg_pc(key_outer, key_inner):
        mat = np.array([r[key_outer][key_inner] for r in rep_results])  # [R, C]
        return {
            "mean": mat.mean(axis=0).tolist(),   # [C]
            "std":  mat.std(axis=0).tolist(),    # [C]
        }

    full_pc = {
        "acc"      : _agg_pc("full_per_class", "class_acc"),
        "precision": _agg_pc("full_per_class", "class_precision"),
        "recall"   : _agg_pc("full_per_class", "class_recall"),
        "f1"       : _agg_pc("full_per_class", "class_f1"),
    }
    red_pc = {
        "acc"      : _agg_pc("red_per_class", "class_acc"),
        "precision": _agg_pc("red_per_class", "class_precision"),
        "recall"   : _agg_pc("red_per_class", "class_recall"),
        "f1"       : _agg_pc("red_per_class", "class_f1"),
    }

    # Aggregate confusion matrices
    full_cm_mean = np.mean(
        [r["full_per_class"]["confusion"] for r in rep_results], axis=0
    )
    red_cm_mean  = np.mean(
        [r["red_per_class"]["confusion"]  for r in rep_results], axis=0
    )

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
        # Per-class metrics
        "full_per_class":     full_pc,
        "red_per_class":      red_pc,
        "full_cm_mean":       full_cm_mean.tolist(),
        "red_cm_mean":        red_cm_mean.tolist(),
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

    # ── Overall ───────────────────────────────────────────────────────────
    print(f"\n  Overall — Full channels ({cfg.n_eeg_channels} ch)")
    print(f"    Accuracy : {summary['full_mean_acc']:.4f} ± {summary['full_std_acc']:.4f}")
    print(f"    F1-score : {summary['full_mean_f1']:.4f} ± {summary['full_std_f1']:.4f}")
    print(f"\n  Overall — Reduced channels ({summary['kept_channels']} ch)")
    print(f"    Accuracy : {summary['reduced_mean_acc']:.4f} ± {summary['reduced_std_acc']:.4f}")
    print(f"    F1-score : {summary['reduced_mean_f1']:.4f} ± {summary['reduced_std_f1']:.4f}")

    # ── Per-class table printer ───────────────────────────────────────────
    cnames = cfg.class_names

    def _print_pc_table(pc_dict, header):
        print(f"\n  {header}")
        print(f"  {'Class':<14}  {'Accuracy':>12}  {'Precision':>10}  "
              f"{'Recall':>8}  {'F1':>8}")
        print(f"  {'─'*60}")
        for i, name in enumerate(cnames):
            print(
                f"  {name:<14}  "
                f"{pc_dict['acc']['mean'][i]:.4f}±{pc_dict['acc']['std'][i]:.3f}  "
                f"{pc_dict['precision']['mean'][i]:.4f}±{pc_dict['precision']['std'][i]:.3f}  "
                f"{pc_dict['recall']['mean'][i]:.4f}±{pc_dict['recall']['std'][i]:.3f}  "
                f"{pc_dict['f1']['mean'][i]:.4f}±{pc_dict['f1']['std'][i]:.3f}"
            )

    _print_pc_table(full_pc, f"Per-Class — Full channels ({cfg.n_eeg_channels} ch)")
    _print_pc_table(red_pc,  f"Per-Class — Reduced channels ({summary['kept_channels']} ch)")

    # ── Confusion matrices ────────────────────────────────────────────────
    def _print_cm(cm, title):
        print(f"\n  {title}  (mean over {cfg.n_repetitions} reps)")
        print("  " + " " * 14 + "".join(f"  {n[:8]:>8}" for n in cnames))
        for i, name in enumerate(cnames):
            row = "  " + f"{name:<14}" + "".join(
                f"  {cm[i][j]:8.1f}" for j in range(cfg.n_classes)
            )
            print(row)

    _print_cm(full_cm_mean, f"Confusion Matrix — Full ({cfg.n_eeg_channels} ch)")
    _print_cm(red_cm_mean,  f"Confusion Matrix — Reduced ({summary['kept_channels']} ch)")

    print(f"\n{'='*65}\n")

    return summary


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(summary: Dict, log_path: str, cfg: WEAVEConfig):
    """Write a human-readable results file."""
    import os
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    n_feat_per_ch   = 2 * len(cfg.retained_bands)
    total_feat_full = cfg.n_eeg_channels * n_feat_per_ch
    total_feat_red  = summary["kept_channels"] * n_feat_per_ch
    cnames          = cfg.class_names
    n_classes       = cfg.n_classes

    with open(log_path, "w") as f:
        # ── Header ───────────────────────────────────────────────────────
        f.write("WEAVE + SVM Pipeline Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset         : {cfg.dataset.upper()}\n")
        if cfg.dataset.lower() == "deap":
            f.write(f"Label type      : {cfg.label_type} "
                    f"(threshold > {cfg.ground_truth_threshold})\n")
        else:
            f.write(f"Classes         : {cfg.class_names}\n")
        f.write(f"Wavelet         : {cfg.wavelet}\n")
        f.write(f"Retained bands  : {cfg.retained_bands}\n")
        f.write(f"Segment duration: {cfg.segment_duration} s\n")
        f.write(f"Channels        : {cfg.n_eeg_channels}  →  "
                f"{summary['kept_channels']} (after NMI reduction)\n")
        f.write(f"Features        : {total_feat_full} full  →  {total_feat_red} reduced\n")
        f.write(f"Repetitions     : {cfg.n_repetitions}\n")
        f.write("=" * 60 + "\n\n")

        # ── Per-repetition table ──────────────────────────────────────────
        f.write("Per-Repetition Results\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rep':>4}  {'Full Acc':>9}  {'Full F1':>8}  "
                f"{'Red Acc':>8}  {'Red F1':>7}\n")
        f.write("-" * 60 + "\n")
        for idx, r in enumerate(summary["rep_results"], 1):
            f.write(
                f"{idx:4d}  {r['full_acc']:9.4f}  {r['full_f1']:8.4f}  "
                f"{r['reduced_acc']:8.4f}  {r['reduced_f1']:7.4f}\n"
            )
        f.write("-" * 60 + "\n\n")

        # ── Overall summary ───────────────────────────────────────────────
        f.write("Overall Summary\n")
        f.write("-" * 60 + "\n")
        f.write(f"Full channels ({cfg.n_eeg_channels} ch, {total_feat_full} feat)\n")
        f.write(f"  Accuracy : {summary['full_mean_acc']:.4f} ± {summary['full_std_acc']:.4f}\n")
        f.write(f"  F1-score : {summary['full_mean_f1']:.4f} ± {summary['full_std_f1']:.4f}\n\n")
        f.write(f"Reduced channels ({summary['kept_channels']} ch, {total_feat_red} feat)\n")
        f.write(f"  Accuracy : {summary['reduced_mean_acc']:.4f} ± {summary['reduced_std_acc']:.4f}\n")
        f.write(f"  F1-score : {summary['reduced_mean_f1']:.4f} ± {summary['reduced_std_f1']:.4f}\n\n")

        # ── Per-class tables ──────────────────────────────────────────────
        def _write_pc_table(pc_dict, title):
            f.write(f"{title}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<14}  {'Accuracy':>12}  {'Precision':>12}  "
                    f"{'Recall':>12}  {'F1':>12}\n")
            f.write("-" * 60 + "\n")
            for i, name in enumerate(cnames):
                f.write(
                    f"{name:<14}  "
                    f"{pc_dict['acc']['mean'][i]:.4f}±{pc_dict['acc']['std'][i]:.3f}  "
                    f"{pc_dict['precision']['mean'][i]:.4f}±{pc_dict['precision']['std'][i]:.3f}  "
                    f"{pc_dict['recall']['mean'][i]:.4f}±{pc_dict['recall']['std'][i]:.3f}  "
                    f"{pc_dict['f1']['mean'][i]:.4f}±{pc_dict['f1']['std'][i]:.3f}\n"
                )
            f.write("\n")

        _write_pc_table(
            summary["full_per_class"],
            f"Per-Class Metrics — Full channels ({cfg.n_eeg_channels} ch)"
        )
        _write_pc_table(
            summary["red_per_class"],
            f"Per-Class Metrics — Reduced channels ({summary['kept_channels']} ch)"
        )

        # ── Confusion matrices ────────────────────────────────────────────
        def _write_cm(cm, title):
            f.write(f"{title}  (mean over {cfg.n_repetitions} reps)\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'':14}" + "".join(f"  {n[:10]:>10}" for n in cnames) + "\n")
            for i, name in enumerate(cnames):
                f.write(
                    f"{name:<14}"
                    + "".join(f"  {cm[i][j]:10.1f}" for j in range(n_classes))
                    + "\n"
                )
            f.write("\n")

        _write_cm(
            summary["full_cm_mean"],
            f"Confusion Matrix — Full channels ({cfg.n_eeg_channels} ch)"
        )
        _write_cm(
            summary["red_cm_mean"],
            f"Confusion Matrix — Reduced channels ({summary['kept_channels']} ch)"
        )

        # ── Consensus channel ranking ─────────────────────────────────────
        f.write("Consensus NMI channel ranking (best → worst):\n")
        f.write(f"  {summary['consensus_ranking']}\n")

    logger.info(f"Results saved → {log_path}")
