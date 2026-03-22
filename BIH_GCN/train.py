"""
Training & evaluation for the EEG-only BIH_GCN pipeline.
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from BIH_GCN.config import BIHGCNConfig
from BIH_GCN.model  import BIHGCN

logger = logging.getLogger("BIH_GCN.Train")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype)


def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(_to_tensor(X), _to_tensor(y, torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle)   # drop last incomplete batch only during train


def _normalise_per_channel(X_tr: np.ndarray,
                            X_te: np.ndarray):
    """
    Z-score each channel independently.
    Fit statistics on X_tr only → apply to both.
    X_tr / X_te : [N, C, T]
    """
    X_tr = X_tr.copy().astype(np.float32)
    X_te = X_te.copy().astype(np.float32)
    C = X_tr.shape[1]
    for c in range(C):
        mu  = X_tr[:, c, :].mean()
        sig = X_tr[:, c, :].std()
        sig = max(sig, 1e-8)
        X_tr[:, c, :] = (X_tr[:, c, :] - mu) / sig
        X_te[:, c, :] = (X_te[:, c, :] - mu) / sig
    return X_tr, X_te


# ── One repetition ────────────────────────────────────────────────────────────

def _run_one_rep(segments, labels, cfg: BIHGCNConfig, seed: int, device):
    """
    One train/test split → train model → return (acc, f1).
    segments : [N, C, T]  numpy
    labels   : [N]        numpy int
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size,
                                  random_state=seed)
    tr_idx, te_idx = next(sss.split(segments, labels))

    X_tr, y_tr = segments[tr_idx], labels[tr_idx]
    X_te, y_te = segments[te_idx], labels[te_idx]

    # ── Per-channel normalisation (fit on train only — no leakage) ────────────
    X_tr, X_te = _normalise_per_channel(X_tr, X_te)

    tr_loader = _make_loader(X_tr, y_tr, cfg.batch_size, shuffle=True)
    te_loader = _make_loader(X_te, y_te, cfg.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    model = BIHGCN(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg.lr,
                             weight_decay=cfg.weight_decay)

    # Class-weighted CE to handle imbalance
    class_counts = np.bincount(y_tr, minlength=cfg.n_classes).astype(float)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights   = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Warmup for first 10% of epochs, then cosine anneal
    warmup_epochs  = max(1, cfg.epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, cfg.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss   = float("inf")
    best_state  = None
    patience    = cfg.patience
    no_improve  = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.debug(f"    epoch {epoch+1:3d}/{cfg.epochs}  "
                         f"loss={avg_loss:.4f}  "
                         f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping on train loss
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug(f"    Early stop at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    avg = "binary" if cfg.n_classes == 2 else "macro"
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return acc, f1


# ── Full evaluation ───────────────────────────────────────────────────────────

def run_evaluation(segments: np.ndarray, labels: np.ndarray,
                   cfg: BIHGCNConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Repetitions: {cfg.n_repetitions}  |  "
                f"Epochs: {cfg.epochs}  |  LR: {cfg.lr}  |  "
                f"Patience: {cfg.patience}")

    accs, f1s = [], []
    for rep in range(cfg.n_repetitions):
        seed = cfg.seed + rep
        acc, f1 = _run_one_rep(segments, labels, cfg, seed, device)
        accs.append(acc)
        f1s.append(f1)
        logger.info(f"  Rep {rep+1:02d}/{cfg.n_repetitions}  "
                    f"acc={acc:.4f}  f1={f1:.4f}")

    summary = {
        "acc_mean" : float(np.mean(accs)),
        "acc_std"  : float(np.std(accs)),
        "f1_mean"  : float(np.mean(f1s)),
        "f1_std"   : float(np.std(f1s)),
        "acc_all"  : accs,
        "f1_all"   : f1s,
    }
    logger.info(f"  ── Final  acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}"
                f"  f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    return summary


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(summary: dict, path: str, cfg: BIHGCNConfig):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("BIH_GCN EEG-only Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset    : {cfg.dataset.upper()}\n")
        f.write(f"Epochs     : {cfg.epochs}\n")
        f.write(f"LR         : {cfg.lr}\n")
        f.write(f"Batch size : {cfg.batch_size}\n")
        f.write(f"Reps       : {cfg.n_repetitions}\n")
        f.write(f"Accuracy   : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}\n")
        f.write(f"F1         : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}\n")
        f.write("\nPer-rep:\n")
        for i, (a, f1) in enumerate(zip(summary["acc_all"], summary["f1_all"])):
            f.write(f"  rep {i+1:02d}  acc={a:.4f}  f1={f1:.4f}\n")
    logger.info(f"Results saved → {path}")
