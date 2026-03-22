"""
Stage 0 – Raw EEG → Spectrogram
Input : [B, C, T]          (batch, channels, time samples)
Output: [B, C, F, T']      (batch, channels, freq bins, time frames)
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class EEGSpectrogram(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stft = T.Spectrogram(
            n_fft       = cfg.stft_n_fft,
            hop_length  = cfg.stft_hop,
            power       = 2.0,
        )
        self.mel = T.MelScale(
            n_mels      = cfg.stft_n_mels,
            sample_rate = cfg.sampling_rate,
            n_stft      = cfg.stft_n_fft // 2 + 1,
        )
        self.log = T.AmplitudeToDB()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, T]
        returns: [B, C, n_mels, T']
        """
        B, C, T = x.shape
        x = x.reshape(B * C, T)          # treat each channel independently
        x = self.stft(x)                  # [B*C, F, T']
        x = self.mel(x)                   # [B*C, n_mels, T']
        x = self.log(x)                   # log scale
        _, F, Tp = x.shape
        return x.reshape(B, C, F, Tp)     # [B, C, n_mels, T']
