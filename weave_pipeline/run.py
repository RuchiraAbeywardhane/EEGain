"""
Entry point for the WEAVE + SVM pipeline.

Usage examples
--------------
# DEAP – Valence, single subject:
    python -m weave_pipeline.run --dataset deap --data_path /path/to/deap --subject 1

# DEAP – Arousal, all subjects:
    python -m weave_pipeline.run --dataset deap --data_path /path/to/deap --all_subjects --label_type A

# DEAP – Valence, all subjects, 32→16 channel reduction:
    python -m weave_pipeline.run --dataset deap --data_path /path/to/deap --all_subjects --min_channels 16

# Emognition – single subject:
    python -m weave_pipeline.run --dataset emognition --data_path /path/to/emog --subject 22

# Emognition – all subjects (recommended — more segments for stable estimates):
    python -m weave_pipeline.run --dataset emognition --data_path /path/to/emog --all_subjects --min_channels 2

Pipeline summary
----------------
  1. Load raw EEG segments  [N, C, segment_samples]
  2. Extract WEAVE features [N, C × 6]
       – db5 DWT, alpha/beta/gamma bands (auto-scaled to sampling rate)
       – per band: wavelet entropy + mean absolute coefficient
  3. 30 random 80/20 train/test splits (stratified)
     Per split:
       a. Fit StandardScaler on train
       b. Train RBF-SVM on full C×6 features
       c. NMI channel ranking (train set only, no leakage)
       d. Reduce to top `min_channels` channels
       e. Train second RBF-SVM on reduced features
       f. Evaluate both on test set
  4. Report mean ± std accuracy and F1 over 30 repetitions
  5. Save detailed results to weave_logs/
"""

import argparse
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weave_pipeline.config  import WEAVEConfig
from weave_pipeline.dataset import load_data
from weave_pipeline.train   import run_weave_evaluation, save_results

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("WEAVE.Run")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WEAVE + SVM – EEG Emotion Recognition",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset",      default="deap",
                   choices=["deap", "emognition"],
                   help="Dataset to use")
    p.add_argument("--data_path",    required=True,
                   help="Path to dataset root folder")

    # ── DEAP-specific ─────────────────────────────────────────────────────────
    p.add_argument("--label_type",   default="V", choices=["V", "A"],
                   help="(DEAP) Emotion dimension: V=Valence  A=Arousal")
    p.add_argument("--threshold",    default=5.0, type=float,
                   help="(DEAP) Binary label threshold: score > threshold → High\n"
                        "Default 5.0  (paper: >5 = High, ≤5 = Low)")

    # ── Emognition-specific ───────────────────────────────────────────────────
    p.add_argument("--lead_in",      default=5.0, type=float,
                   help="(Emognition) Seconds to discard from clip start. Default=5.0")
    p.add_argument("--baseline_dur", default=3.0, type=float,
                   help="(Emognition) Seconds after lead-in used as baseline. Default=3.0")

    # ── Subject selection ─────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--subject",
                     help="Single subject  (int for DEAP, string for Emognition)")
    grp.add_argument("--all_subjects", action="store_true",
                     help="Pool all subjects found in data_path")

    # ── Feature / channel config ──────────────────────────────────────────────
    p.add_argument("--segment_dur",  default=6.0, type=float,
                   help="EEG window duration in seconds. Default=6.0 (paper)")
    p.add_argument("--wavelet",      default="db5",
                   help="PyWavelets wavelet name. Default=db5 (paper)")
    p.add_argument("--bands",        nargs="+",
                   default=["alpha", "beta", "gamma"],
                   help="Frequency bands to retain. Default: alpha beta gamma")
    p.add_argument("--min_channels", default=None, type=int,
                   help="Target channel count after NMI reduction.\n"
                        "Default: 16 for DEAP (32→16), 2 for Emognition (4→2)")

    # ── SVM ───────────────────────────────────────────────────────────────────
    p.add_argument("--svm_C",        default=1.0,     type=float,
                   help="SVM regularisation parameter C. Default=1.0")
    p.add_argument("--svm_gamma",    default="scale",
                   help="SVM RBF kernel gamma. Default='scale'")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--n_reps",       default=30,   type=int,
                   help="Number of random train/test repetitions. Default=30 (paper)")
    p.add_argument("--test_size",    default=0.2,  type=float,
                   help="Fraction held out as test set per repetition. Default=0.2")
    p.add_argument("--seed",         default=42,   type=int,
                   help="Base random seed (seed+rep used per repetition)")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--log_dir",      default="weave_logs/",
                   help="Directory for result files. Default=weave_logs/")

    return p.parse_args()


# ── Config builder ────────────────────────────────────────────────────────────

def _make_config(args) -> WEAVEConfig:
    is_emog = args.dataset.lower() == "emognition"

    # Determine default min_channels based on dataset if not specified
    if args.min_channels is not None:
        min_ch = args.min_channels
    else:
        min_ch = 2 if is_emog else 16   # 4→2 Emognition, 32→16 DEAP (paper)

    cfg = WEAVEConfig(
        dataset                 = args.dataset.lower(),
        data_path               = args.data_path,
        # DEAP
        label_type              = args.label_type,
        ground_truth_threshold  = args.threshold,
        # Emognition
        lead_in_duration        = args.lead_in,
        baseline_duration       = args.baseline_dur,
        # Segmentation
        segment_duration        = args.segment_dur,
        # Features
        wavelet                 = args.wavelet,
        retained_bands          = args.bands,
        # Channel selection
        min_channels            = min_ch,
        # SVM
        svm_C                   = args.svm_C,
        svm_gamma               = args.svm_gamma,
        # Evaluation
        n_repetitions           = args.n_reps,
        test_size               = args.test_size,
        seed                    = args.seed,
        # Output
        log_dir                 = args.log_dir,
    )

    # ── Dataset-specific auto-config ──────────────────────────────────────────
    if is_emog:
        cfg.sampling_rate  = 256
        cfg.n_eeg_channels = 4
        cfg.n_classes      = 4
        cfg.class_names    = ["enthusiasm", "neutral", "fear", "sadness"]
    else:
        cfg.sampling_rate  = 128
        cfg.n_eeg_channels = 32
        cfg.n_classes      = 2
        cfg.class_names    = ["Low", "High"]

    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = _make_config(args)

    np.random.seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("  WEAVE + SVM Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Dataset      : {cfg.dataset.upper()}")
    logger.info(f"  Sampling rate: {cfg.sampling_rate} Hz")
    logger.info(f"  Channels     : {cfg.n_eeg_channels}")
    logger.info(f"  Segment      : {cfg.segment_duration}s  "
                f"({int(cfg.segment_duration * cfg.sampling_rate)} samples)")
    logger.info(f"  Wavelet      : {cfg.wavelet}  |  Bands: {cfg.retained_bands}")
    logger.info(f"  Features     : {cfg.n_eeg_channels} ch × "
                f"{2 * len(cfg.retained_bands)} feat = "
                f"{cfg.n_eeg_channels * 2 * len(cfg.retained_bands)} total")
    logger.info(f"  NMI reduce   : {cfg.n_eeg_channels} → {cfg.min_channels} channels")
    logger.info(f"  SVM          : kernel={cfg.svm_kernel}  C={cfg.svm_C}  gamma={cfg.svm_gamma}")
    logger.info(f"  Repetitions  : {cfg.n_repetitions}  |  test_size={cfg.test_size}")
    if cfg.dataset == "deap":
        logger.info(f"  Label        : {cfg.label_type}  "
                    f"(>  {cfg.ground_truth_threshold} → High)")
    else:
        logger.info(f"  Classes      : {cfg.class_names}")
        logger.info(f"  Lead-in      : {cfg.lead_in_duration}s  |  "
                    f"Baseline: {cfg.baseline_duration}s")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    is_emog    = cfg.dataset == "emognition"
    subject_id = None

    if args.subject:
        subject_id = args.subject if is_emog else int(args.subject)
        logger.info(f"Loading subject {subject_id} …")
    else:
        logger.info("Loading ALL subjects …")

    segments, labels = load_data(cfg, subject_id=subject_id)

    if segments.shape[0] == 0:
        logger.error("No data loaded — check data_path and subject ID.")
        return

    logger.info(
        f"Loaded {segments.shape[0]} segments  |  shape {segments.shape}  |  "
        f"class dist {np.bincount(labels, minlength=cfg.n_classes).tolist()}"
    )

    # Guard: need at least n_classes samples per class for stratified splitting
    counts = np.bincount(labels, minlength=cfg.n_classes)
    min_count = counts[counts > 0].min()
    if min_count < 2:
        logger.error(
            f"Not enough samples per class for stratified splitting "
            f"(min class count = {min_count}).  "
            f"Use --all_subjects or check your data."
        )
        return

    # ── Run evaluation ────────────────────────────────────────────────────────
    summary = run_weave_evaluation(segments, labels, cfg)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)

    if args.subject:
        subj_tag = (f"subject_{args.subject}"
                    if is_emog else f"subject_{int(args.subject):02d}")
    else:
        subj_tag = "all_subjects"

    if is_emog:
        run_tag = f"emognition_{subj_tag}"
    else:
        run_tag = f"deap_{subj_tag}_{cfg.label_type}"

    log_path = os.path.join(cfg.log_dir, f"weave_results_{run_tag}.txt")
    save_results(summary, log_path, cfg)
    logger.info(f"Done. Results written to {log_path}")


if __name__ == "__main__":
    main()
