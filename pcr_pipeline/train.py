"""
Training and evaluation loop for the PCRNN pipeline.

Supports:
    - 10-fold cross-validation (within a single subject or across all subjects)
    - Early stopping on validation loss
    - Class-weighted cross-entropy loss
    - ReduceLROnPlateau scheduler
    - Per-fold and summary metrics (accuracy, F1)
"""

import os
import copy
import logging
import numpy as np
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .config import PCRConfig
from .model import PCRNN
from .dataset import DEAPWindowDataset

logger = logging.getLogger("PCR.Train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=n_classes).float().clamp(min=1)
    weights = 1.0 / counts
    return (weights / weights.sum()).to(device)


def _get_loss_fn(train_dataset: DEAPWindowDataset, n_classes: int) -> nn.Module:
    weights = _compute_class_weights(train_dataset.y, n_classes)
    return nn.CrossEntropyLoss(weight=weights)


def _make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


# ── Single epoch passes ───────────────────────────────────────────────────────

def _train_epoch(
    model: PCRNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> Tuple[List, List, float]:
    model.train()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for x_2d, x_1d, y in pbar:
        x_2d, x_1d, y = x_2d.to(device), x_1d.to(device), y.to(device)

        logits = model(x_2d, x_1d)
        loss   = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping helps stabilise LSTM training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        total_loss    += loss.item() * x_2d.size(0)
        total_samples += x_2d.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return all_preds, all_labels, total_loss / total_samples


@torch.no_grad()
def _eval_epoch(
    model: PCRNN,
    loader: DataLoader,
    loss_fn: nn.Module,
    desc: str = "  Val ",
) -> Tuple[List, List, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x_2d, x_1d, y in pbar:
        x_2d, x_1d, y = x_2d.to(device), x_1d.to(device), y.to(device)
        logits = model(x_2d, x_1d)
        loss   = loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        total_loss    += loss.item() * x_2d.size(0)
        total_samples += x_2d.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return all_preds, all_labels, total_loss / total_samples


# ── One complete fold ─────────────────────────────────────────────────────────

def train_fold(
    train_ds: DEAPWindowDataset,
    val_ds:   DEAPWindowDataset,
    test_ds:  DEAPWindowDataset,
    cfg: PCRConfig,
    fold_idx: int,
) -> Dict:
    """
    Train one fold from scratch and return test metrics.

    Returns:
        dict with keys: fold, test_acc, test_f1, test_preds, test_labels
    """
    logger.info(
        f"Fold {fold_idx + 1}/{cfg.n_folds}  |  "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )

    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, shuffle=False)
    test_loader  = _make_loader(test_ds,  cfg.batch_size, shuffle=False)

    model    = PCRNN(cfg).to(device)
    loss_fn  = _get_loss_fn(train_ds, cfg.n_classes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    best_val_loss   = float("inf")
    best_state      = None
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        _, _, train_loss = _train_epoch(model, train_loader, optimizer, loss_fn)
        val_preds, val_labels, val_loss = _eval_epoch(model, val_loader, loss_fn)

        val_acc = accuracy_score(val_labels, val_preds)
        print(
            f"  Epoch {epoch + 1:3d}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"patience={patience_counter}/{cfg.early_stopping_patience}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"  ⛔ Early stopping at epoch {epoch + 1}")
                break

    # Restore best checkpoint and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)

    test_preds, test_labels, test_loss = _eval_epoch(
        model, test_loader, loss_fn, desc="  Test"
    )
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds, average="weighted", zero_division=0)

    print(
        f"\n  ✅ Fold {fold_idx + 1} Test | "
        f"acc={test_acc:.4f}  f1={test_f1:.4f}  loss={test_loss:.4f}\n"
    )
    return {
        "fold":        fold_idx + 1,
        "test_acc":    test_acc,
        "test_f1":     test_f1,
        "test_preds":  test_preds,
        "test_labels": test_labels,
    }


# ── 10-fold cross-validation ──────────────────────────────────────────────────

def run_10fold_cv(
    X_2d: np.ndarray,
    X_1d: np.ndarray,
    y: np.ndarray,
    cfg: PCRConfig,
) -> Dict:
    """
    Perform stratified 10-fold cross-validation on the given data.

    For each fold:
        - 80 % of train indices → training set
        - 20 % of train indices → validation set  (class-balanced)
        - held-out fold         → test set

    Args:
        X_2d : [N, W, H, W_grid]
        X_1d : [N, W, C]
        y    : [N]
        cfg  : PCRConfig

    Returns:
        summary dict with per-fold results and mean ± std metrics
    """
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_2d, y)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1} / {cfg.n_folds}")
        print(f"{'='*60}")

        # Split train_val into train (80%) + val (20%), class-balanced
        train_val_labels = y[train_val_idx]
        unique_classes   = np.unique(train_val_labels)
        train_rel, val_rel = [], []

        rng = np.random.default_rng(cfg.seed + fold_idx)
        for cls in unique_classes:
            cls_mask = np.where(train_val_labels == cls)[0]
            rng.shuffle(cls_mask)
            split = int(len(cls_mask) * 0.8)
            train_rel.extend(cls_mask[:split].tolist())
            val_rel.extend(cls_mask[split:].tolist())

        train_idx = train_val_idx[train_rel]
        val_idx   = train_val_idx[val_rel]

        # Build DEAPWindowDataset subsets directly from numpy slices
        train_ds = DEAPWindowDataset(X_2d[train_idx], X_1d[train_idx], y[train_idx])
        val_ds   = DEAPWindowDataset(X_2d[val_idx],   X_1d[val_idx],   y[val_idx])
        test_ds  = DEAPWindowDataset(X_2d[test_idx],  X_1d[test_idx],  y[test_idx])

        result = train_fold(train_ds, val_ds, test_ds, cfg, fold_idx)
        fold_results.append(result)

    # Summary
    accs = [r["test_acc"] for r in fold_results]
    f1s  = [r["test_f1"]  for r in fold_results]

    summary = {
        "fold_results":  fold_results,
        "mean_acc":      float(np.mean(accs)),
        "std_acc":       float(np.std(accs)),
        "mean_f1":       float(np.mean(f1s)),
        "std_f1":        float(np.std(f1s)),
    }

    print(f"\n{'='*60}")
    print(f"  10-FOLD CV SUMMARY")
    print(f"  Accuracy : {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"  F1-score : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"{'='*60}\n")

    return summary


# ── Save results to text log ──────────────────────────────────────────────────

def save_results(summary: Dict, log_path: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("PCR Pipeline – 10-Fold CV Results\n")
        f.write("=" * 40 + "\n")
        for r in summary["fold_results"]:
            f.write(
                f"Fold {r['fold']:2d} | "
                f"acc={r['test_acc']:.4f} | f1={r['test_f1']:.4f}\n"
            )
        f.write("-" * 40 + "\n")
        f.write(
            f"Mean acc : {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}\n"
        )
        f.write(
            f"Mean F1  : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}\n"
        )
    logger.info(f"Results saved to {log_path}")
