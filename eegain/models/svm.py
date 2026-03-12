import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ._registry import register_model

logger = logging.getLogger("Model")


@register_model
class SVMClassifier(nn.Module):
    """
    SVM classifier wrapped as a nn.Module so it plugs into the existing
    training pipeline without any changes to helpers.py.

    Feature extraction strategy (controlled by --svm_features):
        'flatten'  : raw flatten of the input window  (B, 1, C, T) -> (B, C*T)
        'bandpower': per-channel mean absolute value in 4 bands
                     delta/theta/alpha/beta -> (B, C*4)
        'eegnet'   : use a small frozen EEGNet trunk to get a compact embedding

    Because SVM has no gradient, forward() during "training" epochs just
    accumulates batches.  On the first call to eval() after train() the SVM
    is fitted on the accumulated data.  Subsequent eval() forward() calls
    return one-hot probability tensors so CrossEntropyLoss still works for
    logging purposes.
    """

    def __init__(
        self,
        num_classes,
        channels,
        dropout_rate=0.5,
        sampling_r=256,
        svm_kernel="rbf",
        svm_c=1.0,
        svm_features="bandpower",
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels    = channels
        self.sampling_r  = sampling_r
        self.svm_features = svm_features

        # SVM pipeline: StandardScaler + SVC
        self.svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(kernel=svm_kernel, C=svm_c,
                           probability=True, random_state=42)),
        ])
        self._fitted   = False

        # Buffers to accumulate training data across batches
        self._train_X: list = []
        self._train_y: list = []

        # Small EEGNet trunk for 'eegnet' feature mode
        if svm_features == "eegnet":
            from .eegnet import EEGNet
            self._trunk = EEGNet(
                num_classes=num_classes,
                channels=channels,
                dropout_rate=dropout_rate,
                sampling_r=sampling_r,
                **kwargs,
            )
            # We only use self._trunk.blocks (feature extractor), not classifier
        else:
            self._trunk = None

        # Dummy parameter so PyTorch optimiser doesn't complain about
        # an empty parameter list
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        logger.info(
            f"SVMClassifier: kernel={svm_kernel}, C={svm_c}, "
            f"features={svm_features}, classes={num_classes}"
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract(self, x: torch.Tensor) -> np.ndarray:
        """Convert (B, 1, C, T) tensor to (B, feat_dim) numpy array."""
        with torch.no_grad():
            if self.svm_features == "flatten":
                return x.cpu().numpy().reshape(x.size(0), -1)

            elif self.svm_features == "bandpower":
                return self._bandpower_features(x)

            elif self.svm_features == "eegnet":
                feat = self._trunk.blocks(x)          # (B, F2, 1, T')
                return feat.cpu().numpy().reshape(x.size(0), -1)

            else:
                raise ValueError(f"Unknown svm_features: {self.svm_features}")

    def _bandpower_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute mean absolute amplitude in 4 frequency bands per channel.
        Bands (Hz): delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30
        Returns (B, C*4)
        """
        sr   = self.sampling_r
        data = x.squeeze(1).cpu().numpy()          # (B, C, T)
        B, C, T = data.shape
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
        feats = np.zeros((B, C * len(bands)), dtype=np.float32)

        for b_idx, (lo, hi) in enumerate(bands):
            lo_bin = max(1, int(lo * T / sr))
            hi_bin = min(T // 2, int(hi * T / sr))
            fft_mag = np.abs(np.fft.rfft(data, axis=-1))  # (B, C, T//2+1)
            band_power = fft_mag[:, :, lo_bin:hi_bin].mean(axis=-1)  # (B, C)
            feats[:, b_idx * C : (b_idx + 1) * C] = band_power

        return feats

    # ------------------------------------------------------------------
    # nn.Module forward — dual behaviour: accumulate vs predict
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract(x)

        if self.training:
            # Accumulate — actual fit happens in fit_svm()
            self._train_X.append(feats)
            # Return dummy logits so the loss call in train_one_epoch doesn't crash
            return torch.zeros(x.size(0), self.num_classes,
                               device=x.device, requires_grad=True)

        else:
            # Predict
            if not self._fitted:
                raise RuntimeError(
                    "SVMClassifier: call fit_svm() before eval/test."
                )
            probs = self.svm.predict_proba(feats)           # (B, num_classes)
            return torch.tensor(probs, dtype=torch.float32, device=x.device)

    # ------------------------------------------------------------------
    # Public API — called from helpers.py after training epochs
    # ------------------------------------------------------------------
    def fit_svm(self, labels: torch.Tensor):
        """
        Fit the SVM on all accumulated training batches.
        Call this once after all train_one_epoch() calls complete.
        """
        X = np.concatenate(self._train_X, axis=0)
        y = labels.cpu().numpy() if isinstance(labels, torch.Tensor) \
            else np.array(labels)
        self.svm.fit(X, y)
        self._fitted = True
        self._train_X.clear()
        logger.info(f"SVMClassifier: fitted on {X.shape[0]} samples, "
                    f"{X.shape[1]} features.")

    def reset(self):
        """Clear accumulated data for a new fold."""
        self._train_X.clear()
        self._train_y.clear()
        self._fitted = False
