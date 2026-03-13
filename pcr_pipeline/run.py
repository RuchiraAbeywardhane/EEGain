"""
Entry point for the PCRNN pipeline.

Usage examples
--------------
# DEAP – Valence, subject 1:
    python -m pcr_pipeline.run --dataset deap --data_path /path/to/deap --subject 1

# DEAP – Arousal, all subjects:
    python -m pcr_pipeline.run --dataset deap --data_path /path/to/deap --all_subjects --label_type A

# Emognition – subject P01:
    python -m pcr_pipeline.run --dataset emognition --data_path /path/to/emognition --subject P01

# Emognition – all subjects, custom lead-in:
    python -m pcr_pipeline.run --dataset emognition --data_path /path/to/emognition --all_subjects --lead_in 3.0
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcr_pipeline.config  import PCRConfig
from pcr_pipeline.dataset import load_subject_trials, get_subject_ids
from pcr_pipeline.train   import run_10fold_cv, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PCR.Run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCRNN – Parallel CNN-LSTM for EEG emotion recognition"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset",     default="deap",
                   choices=["deap", "emognition"],
                   help="Which dataset to use")
    p.add_argument("--data_path",   required=True,
                   help="Path to dataset folder")

    # ── DEAP-specific ─────────────────────────────────────────────────────────
    p.add_argument("--label_type",  default="V", choices=["V", "A"],
                   help="(DEAP) V=Valence  A=Arousal")
    p.add_argument("--threshold",   default=4.5, type=float,
                   help="(DEAP) Binary label threshold")

    # ── Emognition-specific ───────────────────────────────────────────────────
    p.add_argument("--lead_in",     default=5.0, type=float,
                   help="(Emognition) Seconds to discard from clip start "
                        "(emotionally neutral lead-in).  Default=5.0")
    p.add_argument("--baseline_dur",default=3.0, type=float,
                   help="(Emognition) Seconds after lead-in used as baseline. "
                        "Default=3.0")

    # ── Subject selection ─────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--subject",
                     help="Single subject  (int for DEAP, string for Emognition)")
    grp.add_argument("--all_subjects", action="store_true",
                     help="Run CV over all subjects found in data_path")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--n_folds",     default=10,   type=int)
    p.add_argument("--batch_size",  default=64,   type=int)
    p.add_argument("--lr",          default=1e-4, type=float)
    p.add_argument("--num_epochs",  default=50,   type=int)
    p.add_argument("--patience",    default=7,    type=int)
    p.add_argument("--seed",        default=42,   type=int)
    p.add_argument("--log_dir",     default="pcr_logs/")

    return p.parse_args()


def _make_config(args) -> PCRConfig:
    """Build a PCRConfig from parsed CLI args, setting dataset-specific fields."""
    is_emog = args.dataset.lower() == "emognition"

    cfg = PCRConfig(
        dataset                 = args.dataset.lower(),
        data_path               = args.data_path,
        # DEAP fields
        label_type              = args.label_type,
        ground_truth_threshold  = args.threshold,
        # Emognition fields
        lead_in_duration        = args.lead_in,
        baseline_duration       = args.baseline_dur,
        # Training
        n_folds                 = args.n_folds,
        batch_size              = args.batch_size,
        lr                      = args.lr,
        num_epochs              = args.num_epochs,
        early_stopping_patience = args.patience,
        seed                    = args.seed,
        log_dir                 = args.log_dir,
    )

    # ── Auto-configure dataset-specific parameters ────────────────────────────
    if is_emog:
        cfg.sampling_rate  = 256
        cfg.n_eeg_channels = 4
        cfg.n_classes      = 4
        cfg.class_names    = ["enthusiasm", "neutral", "fear", "sadness"]
        cfg.window_size    = 128          # scaled ×2 inside model → 256 samples = 1s @ 256Hz
        cfg.window_step    = 64           # 50% overlap → ~4× more windows per trial
        cfg.lstm_hidden    = 64           # more capacity for 4-class problem
        cfg.dropout        = 0.3          # less aggressive — dataset is small
    else:
        cfg.sampling_rate  = 128
        cfg.n_eeg_channels = 32
        cfg.n_classes      = 2
        cfg.class_names    = ["low", "high"]
        cfg.window_size    = 128
        cfg.window_step    = 128          # DEAP: no overlap (original behaviour)
        cfg.lstm_hidden    = 32
        cfg.dropout        = 0.5

    return cfg


def main():
    args = parse_args()
    cfg  = _make_config(args)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info(f"Dataset : {cfg.dataset.upper()}")
    logger.info(f"Device  : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    if cfg.dataset == "deap":
        logger.info(f"Label   : {cfg.label_type}  |  Threshold : {cfg.ground_truth_threshold}")
    else:
        logger.info(
            f"Classes : {cfg.class_names}  |  "
            f"Lead-in : {cfg.lead_in_duration}s  |  "
            f"Baseline: {cfg.baseline_duration}s"
        )

    # ── Load data (trial-level, no leakage) ───────────────────────────────────
    is_emog = cfg.dataset == "emognition"

    if args.subject:
        # Convert to int for DEAP, keep as string for Emognition
        subject_id = args.subject if is_emog else int(args.subject)
        logger.info(f"Loading subject {subject_id} …")
        trials_2d, trials_1d, trial_labels = load_subject_trials(subject_id, cfg)
        log_name = (
            f"emognition_subject_{subject_id}"
            if is_emog
            else f"deap_subject_{int(subject_id):02d}_{cfg.label_type}"
        )
    else:
        subject_ids = get_subject_ids(cfg)
        logger.info(f"Found {len(subject_ids)} subjects: {subject_ids}")
        trials_2d, trials_1d, trial_labels = [], [], []
        for sid in subject_ids:
            logger.info(f"  Loading subject {sid} …")
            t2, t1, tl = load_subject_trials(sid, cfg)
            trials_2d.extend(t2)
            trials_1d.extend(t1)
            trial_labels.extend(tl)
        log_name = (
            "emognition_all_subjects"
            if is_emog
            else f"deap_all_subjects_{cfg.label_type}"
        )

    if not trials_2d:
        logger.error("No trials loaded — check your data path and subject ID.")
        return

    total_windows = sum(t.shape[0] for t in trials_2d)
    logger.info(
        f"Data ready — {len(trials_2d)} trials | {total_windows} total windows | "
        f"class dist (trials): {[trial_labels.count(c) for c in range(cfg.n_classes)]}"
    )

    # ── Run 10-fold CV (split at trial level) ─────────────────────────────────
    summary = run_10fold_cv(trials_2d, trials_1d, trial_labels, cfg)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"results_{log_name}.txt")
    save_results(summary, log_path)
    logger.info(f"Done. Results written to {log_path}")


if __name__ == "__main__":
    main()
