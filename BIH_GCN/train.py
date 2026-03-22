"""
Training & evaluation for the EEG-only BIH_GCN pipeline.
Clip-independent evaluation: the train/test split is done at the
CLIP level so that ALL segments from the same recording stay on
the same side of the split.
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
                      drop_last=shuffle)


def _normalise_per_channel(X_tr: np.ndarray, *others: np.ndarray):
    """
    Z-score each channel independently.
    Fit statistics on X_tr only → apply to X_tr and all arrays in *others.
    All arrays : [N, C, T]
    Returns (X_tr_normed, *others_normed)
    """
    X_tr = X_tr.copy().astype(np.float32)
    C    = X_tr.shape[1]
    mus, sigs = [], []
    for c in range(C):
        mu  = X_tr[:, c, :].mean()
        sig = max(X_tr[:, c, :].std(), 1e-8)
        X_tr[:, c, :] = (X_tr[:, c, :] - mu) / sig
        mus.append(mu); sigs.append(sig)

    normed = [X_tr]
    for X in others:
        X = X.copy().astype(np.float32)
        for c in range(C):
            X[:, c, :] = (X[:, c, :] - mus[c]) / sigs[c]
        normed.append(X)
    return tuple(normed)


def _clip_level_split(labels: np.ndarray, clip_ids: np.ndarray,
                      test_size: float, seed: int):
    """
    Split at the clip level.
    Returns (tr_seg_idx, te_seg_idx) — indices into the segment arrays.
    Each unique clip_id is assigned entirely to train or test.
    Stratification is on the per-clip label.
    """
    unique_clips = np.unique(clip_ids)
    clip_labels  = np.array([labels[clip_ids == cid][0] for cid in unique_clips])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=seed)
    tr_clip_local, te_clip_local = next(sss.split(unique_clips, clip_labels))

    tr_clips = set(unique_clips[tr_clip_local])
    te_clips = set(unique_clips[te_clip_local])

    tr_seg_idx = np.where(np.isin(clip_ids, list(tr_clips)))[0]
    te_seg_idx = np.where(np.isin(clip_ids, list(te_clips)))[0]
    return tr_seg_idx, te_seg_idx


# ── One repetition ────────────────────────────────────────────────────────────

def _run_one_rep(segments, labels, clip_ids, cfg: BIHGCNConfig,
                 seed: int, device):
    """
    One clip-level train/val/test split → train → return (acc, f1).
    segments : [N, C, T]  numpy
    labels   : [N]        numpy int
    clip_ids : [N]        numpy int — clip-level grouping
    """
    # ── Outer split: 80% train+val clips / 20% test clips ────────────────────
    tr_val_seg_idx, te_seg_idx = _clip_level_split(
        labels, clip_ids, cfg.test_size, seed
    )

    # ── Inner split: carve val clips from train+val (no test leakage) ─────────
    tr_val_clip_ids = clip_ids[tr_val_seg_idx]
    tr_val_labels   = labels[tr_val_seg_idx]
    tr_seg_local, val_seg_local = _clip_level_split(
        tr_val_labels, tr_val_clip_ids, 0.2, seed + 1000
    )
    tr_seg_idx  = tr_val_seg_idx[tr_seg_local]
    val_seg_idx = tr_val_seg_idx[val_seg_local]

    X_tr,  y_tr  = segments[tr_seg_idx],  labels[tr_seg_idx]
    X_val, y_val = segments[val_seg_idx], labels[val_seg_idx]
    X_te,  y_te  = segments[te_seg_idx],  labels[te_seg_idx]

    logger.debug(
        f"  Clips → train={len(np.unique(clip_ids[tr_seg_idx]))} "
        f"({len(tr_seg_idx)} segs) | "
        f"val={len(np.unique(clip_ids[val_seg_idx]))} "
        f"({len(val_seg_idx)} segs) | "
        f"test={len(np.unique(clip_ids[te_seg_idx]))} "
        f"({len(te_seg_idx)} segs)"
    )

    # ── Per-channel normalisation — fit on train only, apply to val & test ────
    X_tr, X_val, X_te = _normalise_per_channel(X_tr, X_val, X_te)

    tr_loader  = _make_loader(X_tr,  y_tr,  cfg.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    te_loader  = _make_loader(X_te,  y_te,  cfg.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    model = BIHGCN(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg.lr, weight_decay=cfg.weight_decay)

    class_counts = np.bincount(y_tr, minlength=cfg.n_classes).astype(float)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights      = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion    = nn.CrossEntropyLoss(weight=weights)

    warmup_epochs = max(1, cfg.epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, cfg.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(cfg.epochs):
        # train
        model.train()
        tr_loss, n_tr = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item(); n_tr += 1
        scheduler.step()

        # validation
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
                n_val    += 1
        avg_tr  = tr_loss  / max(n_tr,  1)
        avg_val = val_loss / max(n_val, 1)

        if (epoch + 1) % 10 == 0:
            logger.debug(f"    epoch {epoch+1:3d}/{cfg.epochs}  "
                         f"tr={avg_tr:.4f}  val={avg_val:.4f}  "
                         f"lr={scheduler.get_last_lr()[0]:.2e}")

        # early stopping on VALIDATION loss
        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                logger.debug(f"    Early stop at epoch {epoch+1}  "
                             f"best_val={best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Evaluation on held-out test clips ─────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
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
                   clip_ids: np.ndarray,
                   cfg: BIHGCNConfig) -> dict:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clips = len(np.unique(clip_ids))
    logger.info(f"Device: {device}  |  Protocol: clip-independent  |  "
                f"Clips: {n_clips}  |  Segments: {len(segments)}  |  "
                f"Reps: {cfg.n_repetitions}  |  Epochs: {cfg.epochs}  |  "
                f"LR: {cfg.lr}  |  Patience: {cfg.patience}")

    accs, f1s = [], []
    for rep in range(cfg.n_repetitions):
        seed = cfg.seed + rep
        acc, f1 = _run_one_rep(segments, labels, clip_ids, cfg, seed, device)
        accs.append(acc)
        f1s.append(f1)
        logger.info(f"  Rep {rep+1:02d}/{cfg.n_repetitions}  "
                    f"acc={acc:.4f}  f1={f1:.4f}")

    summary = {
        "acc_mean": float(np.mean(accs)),
        "acc_std" : float(np.std(accs)),
        "f1_mean" : float(np.mean(f1s)),
        "f1_std"  : float(np.std(f1s)),
        "acc_all" : accs,
        "f1_all"  : f1s,
    }
    logger.info(f"  ── Final  acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}"
                f"  f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    return summary


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(summary: dict, path: str, cfg: BIHGCNConfig):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("BIH_GCN EEG-only Results  [clip-independent]\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset    : {cfg.dataset.upper()}\n")
        f.write(f"Protocol   : clip-independent\n")
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
