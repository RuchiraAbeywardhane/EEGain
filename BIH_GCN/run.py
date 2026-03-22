"""
Entry point for the EEG-only BIH_GCN pipeline.

Usage examples
--------------
# DEAP – Valence, single subject:
    python -m BIH_GCN.run --dataset deap --data_path /path/to/deap --subject 1

# DEAP – Arousal, all subjects:
    python -m BIH_GCN.run --dataset deap --data_path /path/to/deap --all_subjects --label_type A

# Emognition – single subject:
    python -m BIH_GCN.run --dataset emognition --data_path /path/to/emog --subject 22

# Emognition – all subjects:
    python -m BIH_GCN.run --dataset emognition --data_path /path/to/emog --all_subjects
"""

import argparse
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BIH_GCN.config  import BIHGCNConfig
from BIH_GCN.dataset import load_data
from BIH_GCN.train   import run_evaluation, save_results

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("BIH_GCN.Run")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIH_GCN – EEG Emotion Recognition",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--dataset",      default="deap", choices=["deap", "emognition"])
    p.add_argument("--data_path",    required=True)
    p.add_argument("--label_type",   default="V", choices=["V", "A"],
                   help="(DEAP only) V=Valence  A=Arousal")
    p.add_argument("--threshold",    default=5.0, type=float)
    p.add_argument("--lead_in",      default=5.0, type=float)
    p.add_argument("--baseline_dur", default=3.0, type=float)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--subject",      help="Single subject id")
    grp.add_argument("--all_subjects", action="store_true")

    p.add_argument("--segment_dur",  default=6.0,   type=float)
    p.add_argument("--epochs",       default=50,    type=int)
    p.add_argument("--batch_size",   default=32,    type=int)
    p.add_argument("--lr",           default=1e-3,  type=float)
    p.add_argument("--n_reps",       default=10,    type=int)
    p.add_argument("--test_size",    default=0.2,   type=float)
    p.add_argument("--seed",         default=42,    type=int)
    p.add_argument("--log_dir",      default="BIH_GCN_logs/")

    return p.parse_args()


# ── Config builder ────────────────────────────────────────────────────────────

def _make_config(args) -> BIHGCNConfig:
    is_emog = args.dataset.lower() == "emognition"

    cfg = BIHGCNConfig(
        dataset                = args.dataset.lower(),
        data_path              = args.data_path,
        label_type             = args.label_type,
        ground_truth_threshold = args.threshold,
        lead_in_duration       = args.lead_in,
        baseline_duration      = args.baseline_dur,
        segment_duration       = args.segment_dur,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        n_repetitions          = args.n_reps,
        test_size              = args.test_size,
        seed                   = args.seed,
        log_dir                = args.log_dir,
    )

    if is_emog:
        cfg.sampling_rate  = 256
        cfg.n_eeg_channels = 4
        cfg.n_classes      = 4
        cfg.class_names    = ["enthusiasm", "neutral", "fear", "sadness"]
        cfg.brain_regions  = {
            "frontal"  : [0],
            "temporal" : [1],
            "parietal" : [2],
            "occipital": [3],
        }
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
    logger.info("  BIH_GCN Pipeline  [EEG-only]")
    logger.info("=" * 60)
    logger.info(f"  Dataset      : {cfg.dataset.upper()}")
    logger.info(f"  Sampling rate: {cfg.sampling_rate} Hz")
    logger.info(f"  Channels     : {cfg.n_eeg_channels}")
    logger.info(f"  Segment      : {cfg.segment_duration}s "
                f"({cfg.segment_samples} samples)")
    logger.info(f"  Brain regions: {list(cfg.brain_regions.keys())}")
    logger.info(f"  Epochs       : {cfg.epochs}  |  LR: {cfg.lr}  |  "
                f"Batch: {cfg.batch_size}")
    logger.info(f"  Repetitions  : {cfg.n_repetitions}  |  "
                f"test_size={cfg.test_size}")
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
        logger.error("No data loaded — check --data_path and subject id.")
        return

    # ── Run evaluation ────────────────────────────────────────────────────────
    summary = run_evaluation(segments, labels, cfg)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)
    subj_tag = f"subject_{args.subject}" if args.subject else "all_subjects"
    run_tag  = (f"emognition_{subj_tag}" if is_emog
                else f"deap_{subj_tag}_{cfg.label_type}")
    log_path = os.path.join(cfg.log_dir, f"BIH_GCN_results_{run_tag}.txt")
    save_results(summary, log_path, cfg)
    logger.info(f"Done. Results written to {log_path}")


if __name__ == "__main__":
    main()
