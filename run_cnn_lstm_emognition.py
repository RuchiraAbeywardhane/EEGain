"""
run_cnn_lstm_emognition.py
==========================
Standalone training script for the 4-branch 1D CNN + BiLSTM model
on the Emognition dataset (4 emotions: Enthusiasm, Neutral, Sadness, Fear).

Architecture summary
--------------------
  Input  : (B, 1, 4, T)   — 4 MUSE EEG channels, T time-samples per window
  Stage 1: 4 × independent 1-D CNN branches  (one per channel)
             each: Conv1d(k=7)→BN→GELU→Pool → Conv1d(k=5)→BN→GELU→Pool
                    → Conv1d(k=3)→BN→GELU → Dropout
             output: (B, cnn_out, T//4)
  Stage 2: Concatenate branches → (B, 4·cnn_out, T//4)
  Stage 3: BiLSTM (2 layers, hidden=64)  →  take last time-step
  Stage 4: LayerNorm → Linear → GELU → Dropout → Linear(num_classes)

Split strategy: LOSO_Fixed
  Train / val subjects: read from  test_subjects.json  ["Emognition"]["train"]
  Test  subjects      : read from  test_subjects.json  ["Emognition"]["test"]
  Val subjects are carved out of the training pool (subject-level split, no leakage).

Usage
-----
  python run_cnn_lstm_emognition.py --data_path "C:/path/to/emognition"

Optional flags (all have sensible defaults):
  --data_path            Path to Emognition JSON files       (required)
  --window               Segment duration in seconds         (default 4)
  --overlap              Segment overlap in seconds          (default 0)
  --sampling_r           Target sampling rate Hz             (default 256)
  --batch_size           Mini-batch size                     (default 32)
  --num_epochs           Max training epochs                 (default 100)
  --lr                   Learning rate                       (default 3e-4)
  --weight_decay         AdamW weight decay                  (default 0.01)
  --label_smoothing      Cross-entropy label smoothing       (default 0.05)
  --early_stopping_patience  Val-loss patience              (default 15)
  --cnn_out              CNN branch feature maps             (default 32)
  --lstm_hidden          LSTM hidden size per direction      (default 64)
  --lstm_layers          Number of LSTM layers               (default 2)
  --dropout_rate         Dropout rate                        (default 0.5)
  --use_baseline_reduction   InvBase baseline reduction      (default True)
  --train_val_split      Fraction of train subjects for train(default 0.8)
  --random_seed          Random seed                         (default 2025)
  --log_dir              Logging directory                   (default logs/)
  --no_class_weights     Disable weighted cross-entropy      (flag)
"""

import os, sys, copy, json, argparse, logging

# ── make sure the EEGain package on the workspace root is importable ──────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_cnn_lstm")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, classification_report
)

import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import Emognition
from eegain.logger import EmotionLogger
from eegain.models import CNNLSTMEmognition

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_NAMES = ["Enthusiasm", "Neutral", "Sadness", "Fear"]

# Subjects from test_subjects.json  ─ used as fallback if the file is absent
_DEFAULT_TRAIN = [
    "23","24","25","26","27","28","30","31","33","34","35","36","37",
    "38","40","41","42","43","44","45","46","47","48","50","51","52","53","55","57",
]
_DEFAULT_TEST  = ["49","54","56","58","59","60","61","62","63","64"]


def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_subject_split(json_path: str = "test_subjects.json"):
    """Return (train_subjects, test_subjects) lists."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        train = data["Emognition"]["train"]
        test  = data["Emognition"]["test"]
        logger.info(f"Loaded subject split from {json_path} "
                    f"— train: {len(train)}, test: {len(test)}")
        return train, test
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load {json_path} ({e}). Using built-in defaults.")
        return _DEFAULT_TRAIN, _DEFAULT_TEST


def compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    """sklearn-style: w_c = n_samples / (n_classes * count_c)."""
    labels = train_loader.dataset.y
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    labels = labels.long()
    n     = len(labels)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    return (n / (num_classes * counts)).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    bar = tqdm(loader, desc="  Train", leave=False)
    for xb, yb in bar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping — prevents LSTM exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        preds = logits.argmax(1)
        running_loss += loss.item() * xb.size(0)
        n            += xb.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())
        bar.set_postfix(loss=f"{running_loss/n:.4f}")
    return all_preds, all_labels, running_loss / n


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, desc="  Eval"):
    model.eval()
    running_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    bar = tqdm(loader, desc=desc, leave=False)
    for xb, yb in bar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        preds  = logits.argmax(1)
        running_loss += loss.item() * xb.size(0)
        n            += xb.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())
        bar.set_postfix(loss=f"{running_loss/n:.4f}")
    return all_preds, all_labels, running_loss / n


def print_metrics(preds, labels, split: str = "Test"):
    acc   = accuracy_score(labels, preds)
    f1_w  = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m  = f1_score(labels, preds, average="macro",    zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    print(f"\n{'─'*55}")
    print(f"  {split} results")
    print(f"{'─'*55}")
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  F1 (weighted)    : {f1_w*100:.2f}%")
    print(f"  F1 (macro)       : {f1_m*100:.2f}%")
    print(f"  Cohen κ          : {kappa:.4f}")
    print(f"{'─'*55}")
    print(classification_report(
        labels, preds,
        target_names=EMOTION_NAMES,
        zero_division=0,
    ))
    return {"acc": acc, "f1_weighted": f1_w, "f1_macro": f1_m, "kappa": kappa}


# ─────────────────────────────────────────────────────────────────────────────
# Main training procedure
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    setup_seed(args.random_seed)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Args  : {vars(args)}")

    # ── 1. Preprocessing transform ────────────────────────────────────────
    transform = eegain.transforms.Construct([
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(sampling_r=args.sampling_r),
        eegain.transforms.Segment(duration=args.window, overlap=args.overlap),
    ])

    # ── 2. Dataset ────────────────────────────────────────────────────────
    logger.info(f"Loading Emognition dataset from: {args.data_path}")
    dataset = Emognition(
        root=args.data_path,
        transform=transform,
        use_baseline_reduction=args.use_baseline_reduction,
        sampling_r=args.sampling_r,
    )
    logger.info(f"Found {len(dataset.__get_subject_ids__())} subjects")

    # ── 3. Subject split ──────────────────────────────────────────────────
    train_subjects, test_subjects = load_subject_split("test_subjects.json")

    # ── 4. Data loaders (LOSO_Fixed) ─────────────────────────────────────
    loaders = EEGDataloader(dataset, batch_size=args.batch_size).loso_fixed(
        train_subjects,
        test_subjects,
        train_val_split=args.train_val_split,
        random_seed=args.random_seed,
    )

    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    logger.info(f"Windows — train: {n_train}, val: {n_val}, test: {n_test}")

    # ── 5. Model ──────────────────────────────────────────────────────────
    # T = window_seconds × sampling_rate
    T = args.window * args.sampling_r
    model = CNNLSTMEmognition(
        num_classes  = 4,
        num_channels = 4,              # TP9, AF7, AF8, TP10
        cnn_out      = args.cnn_out,
        lstm_hidden  = args.lstm_hidden,
        lstm_layers  = args.lstm_layers,
        dropout_rate = args.dropout_rate,
        # pass through so EEGain registry kwargs are absorbed cleanly
        input_size   = [1, 4, T],
        sampling_r   = args.sampling_r,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: CNNLSTMEmognition  —  {n_params:,} trainable parameters")
    logger.info(f"\n{model}")

    # ── 6. Loss, optimiser, scheduler ────────────────────────────────────
    if args.no_class_weights:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        logger.info("Using uniform class weights")
    else:
        weights = compute_class_weights(train_loader, num_classes=4)
        loss_fn = nn.CrossEntropyLoss(weight=weights,
                                       label_smoothing=args.label_smoothing)
        logger.info(f"Class weights: {weights.cpu().tolist()}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-5,
    )

    # ── 7. Training loop with early stopping ─────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_path = os.path.join(args.log_dir, "best_cnn_lstm_emognition.pt")

    best_val_loss     = float("inf")
    best_state        = None
    patience_counter  = 0

    print(f"\n{'═'*55}")
    print(f"  Training CNNLSTMEmognition  ·  {args.num_epochs} epochs max")
    print(f"  Window: {args.window}s  ·  SR: {args.sampling_r} Hz  ·  T={T}")
    print(f"  Branches: 4 × 1D-CNN (out={args.cnn_out})")
    print(f"  BiLSTM : layers={args.lstm_layers}, hidden={args.lstm_hidden}")
    print(f"{'═'*55}\n")

    for epoch in range(1, args.num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        # ── train ──
        tr_preds, tr_labels, tr_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn)
        tr_acc = accuracy_score(tr_labels, tr_preds)

        # ── validate ──
        va_preds, va_labels, va_loss = eval_one_epoch(
            model, val_loader, loss_fn, desc="   Val ")
        va_acc = accuracy_score(va_labels, va_preds)

        scheduler.step()

        # ── early stopping on val_loss ──
        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, ckpt_path)
            tag = "✓ new best"
        else:
            patience_counter += 1
            tag = f"patience {patience_counter}/{args.early_stopping_patience}"

        print(
            f"Epoch {epoch:>3}/{args.num_epochs}  "
            f"lr={lr_now:.2e}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc*100:.1f}%  "
            f"val_loss={va_loss:.4f} acc={va_acc*100:.1f}%  "
            f"[{tag}]"
        )

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}  "
                  f"(best val_loss={best_val_loss:.4f})")
            break

    # ── 8. Test with best checkpoint ─────────────────────────────────────
    logger.info(f"Loading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    te_preds, te_labels, te_loss = eval_one_epoch(
        model, test_loader, loss_fn, desc="  Test ")

    metrics = print_metrics(te_preds, te_labels, split="Test (best model)")

    # ── 9. Save results ───────────────────────────────────────────────────
    results_path = os.path.join(args.log_dir, "cnn_lstm_emognition_results.json")
    results = {
        "model": "CNNLSTMEmognition",
        "dataset": "Emognition",
        "split": "LOSO_Fixed",
        "emotions": EMOTION_NAMES,
        "hyperparams": vars(args),
        "test_loss": te_loss,
        **metrics,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train 4-branch 1D CNN + BiLSTM on Emognition (4 emotions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_path", type=str, required=True,
                   help="Root folder containing Emognition JSON trial files")
    p.add_argument("--window",    type=int,   default=4,
                   help="Segment window size (seconds)")
    p.add_argument("--overlap",   type=int,   default=0,
                   help="Segment overlap (seconds)")
    p.add_argument("--sampling_r",type=int,   default=256,
                   help="Target sampling rate (Hz)")
    p.add_argument("--use_baseline_reduction", action="store_true", default=True,
                   help="Apply InvBase spectral baseline reduction")
    p.add_argument("--no_baseline_reduction",  dest="use_baseline_reduction",
                   action="store_false",
                   help="Disable InvBase baseline reduction")
    p.add_argument("--train_val_split", type=float, default=0.8,
                   help="Fraction of train subjects kept for training "
                        "(remainder used for validation)")

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--cnn_out",      type=int,   default=32,
                   help="Feature maps per CNN branch")
    p.add_argument("--lstm_hidden",  type=int,   default=64,
                   help="LSTM hidden units per direction")
    p.add_argument("--lstm_layers",  type=int,   default=2,
                   help="Number of stacked BiLSTM layers")
    p.add_argument("--dropout_rate", type=float, default=0.5,
                   help="Dropout rate")

    # ── training ──────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--num_epochs",   type=int,   default=100)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--early_stopping_patience", type=int, default=15)
    p.add_argument("--no_class_weights", action="store_true", default=False,
                   help="Use uniform class weights instead of frequency-based")

    # ── misc ──────────────────────────────────────────────────────────────
    p.add_argument("--random_seed", type=int, default=2025)
    p.add_argument("--log_dir",     type=str, default="logs/")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
