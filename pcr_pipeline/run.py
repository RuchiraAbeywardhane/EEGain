"""
Entry point for the PCRNN pipeline.

Usage examples
--------------
# Run 10-fold CV on subject 1 (Valence):
    python -m pcr_pipeline.run --data_path /path/to/deap --subject 1

# Run on all subjects and save a log:
    python -m pcr_pipeline.run --data_path /path/to/deap --all_subjects --label_type V

# Run Arousal on subject 5, custom epochs:
    python -m pcr_pipeline.run --data_path /path/to/deap --subject 5 --label_type A --num_epochs 30

Modes
-----
--subject N      : 10-fold CV on a single subject's 2400 windows
--all_subjects   : concatenate all subjects then run 10-fold CV
                   (cross-subject experiment)
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch

# Allow running as  python -m pcr_pipeline.run  from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcr_pipeline.config  import PCRConfig
from pcr_pipeline.dataset import load_subject_trials, get_subject_ids
from pcr_pipeline.train   import run_10fold_cv, save_results


# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PCR.Run")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCRNN – Parallel CNN-LSTM for DEAP EEG emotion recognition"
    )

    # Dataset
    p.add_argument("--data_path",   required=True,  help="Path to folder containing DEAP .dat files")
    p.add_argument("--label_type",  default="V",    choices=["V", "A"], help="V=Valence, A=Arousal")
    p.add_argument("--threshold",   default=4.5,    type=float, help="Binary label threshold (inclusive low)")

    # Subject selection
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--subject",      type=int,  help="Run 10-fold CV on a single subject (1-indexed)")
    grp.add_argument("--all_subjects", action="store_true", help="Concatenate all subjects then run CV")

    # Training hyper-parameters
    p.add_argument("--n_folds",     default=10,   type=int)
    p.add_argument("--batch_size",  default=64,   type=int)
    p.add_argument("--lr",          default=1e-4, type=float)
    p.add_argument("--num_epochs",  default=50,   type=int)
    p.add_argument("--patience",    default=7,    type=int,  help="Early-stopping patience (epochs)")
    p.add_argument("--seed",        default=42,   type=int)

    # Output
    p.add_argument("--log_dir",     default="pcr_logs/", help="Directory to save result logs")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Build config from CLI args
    cfg = PCRConfig(
        data_path               = args.data_path,
        label_type              = args.label_type,
        ground_truth_threshold  = args.threshold,
        n_folds                 = args.n_folds,
        batch_size              = args.batch_size,
        lr                      = args.lr,
        num_epochs              = args.num_epochs,
        early_stopping_patience = args.patience,
        seed                    = args.seed,
        log_dir                 = args.log_dir,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info(f"Device : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"Label  : {cfg.label_type}  |  Threshold : {cfg.ground_truth_threshold}")
    logger.info(f"Config : {cfg}")

    # ── Load data  (trial-level, no leakage) ─────────────────────────────────
    if args.subject:
        logger.info(f"Loading subject {args.subject:02d} …")
        trials_2d, trials_1d, trial_labels = load_subject_trials(args.subject, cfg)
        log_name = f"subject_{args.subject:02d}_{cfg.label_type}"
    else:
        subject_ids = get_subject_ids(cfg)
        logger.info(f"Found {len(subject_ids)} subjects: {subject_ids}")
        trials_2d, trials_1d, trial_labels = [], [], []
        for sid in subject_ids:
            logger.info(f"  Loading subject {sid:02d} …")
            t2, t1, tl = load_subject_trials(sid, cfg)
            trials_2d.extend(t2)
            trials_1d.extend(t1)
            trial_labels.extend(tl)
        log_name = f"all_subjects_{cfg.label_type}"

    total_windows = sum(t.shape[0] for t in trials_2d)
    logger.info(
        f"Data ready — {len(trials_2d)} trials | {total_windows} total windows | "
        f"class dist (trials): {[trial_labels.count(c) for c in range(cfg.n_classes)]}"
    )

    # ── Run 10-fold CV  (split at trial level) ────────────────────────────────
    summary = run_10fold_cv(trials_2d, trials_1d, trial_labels, cfg)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"results_{log_name}.txt")
    save_results(summary, log_path)
    logger.info(f"Done. Results written to {log_path}")


if __name__ == "__main__":
    main()
