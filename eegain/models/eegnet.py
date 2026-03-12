import torch
import torch.nn as nn

from ._registry import register_model


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


@register_model
class EEGNet(nn.Module):
    def initial_block(self, dropout_rate):
        block1 = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernelLength1),
                stride=1,
                padding=(0, self.kernelLength1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # Depth-wiseConv2D =======================
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.channels, 1),
                max_norm=1,
                stride=1,
                padding=(0, 0),
                groups=self.F1,
                bias=False,
            ),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size1), stride=self.pool_size1), # [NEW] Using scaled pool size
            nn.Dropout(p=dropout_rate),
        )

        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.kernelLength2),
                stride=1,
                padding=(0, self.kernelLength2 // 2),
                bias=False,
                groups=self.F1 * self.D,
            ),
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                1,
                padding=(0, 0),
                groups=1,
                bias=False,
                stride=1,
            ),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size2), stride=self.pool_size2), # [NEW] Using scaled pool size
            nn.Dropout(p=dropout_rate),
        )
        return nn.Sequential(block1, block2)

    @staticmethod
    def classifier_block(input_size, n_classes):
        return nn.Sequential(
            nn.Linear(input_size, n_classes, bias=False), nn.Softmax(dim=1)
        )

    @staticmethod
    def calculate_out_size(model, channels, samples):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """

        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(
        self,
        num_classes,
        channels,
        dropout_rate,
        kernel_length1=64,
        kernel_length2=16,
        f1=8,
        d=2,
        f2=16,
        **kwargs
    ):
        super(EEGNet, self).__init__()
        samples = kwargs["sampling_r"]*kwargs["window"]
        self.F1 = f1
        self.F2 = f2
        self.D = d
        self.samples = samples
        self.n_classes = num_classes
        self.channels = channels
        
        # [NEW] Scale kernel length and pooling sizes based on sampling rate
        sampling_rate = kwargs.get("sampling_r", 128)  # Default to 128Hz
        scaling_factor = sampling_rate / 128  # Calculate scaling factor
        
        self.kernelLength1 = int(kernel_length1 * scaling_factor)
        self.kernelLength2 = int(kernel_length2 * scaling_factor)
        self.pool_size1 = int(4 * scaling_factor)  # [NEW] Scale pool size 1
        self.pool_size2 = int(8 * scaling_factor)  # [NEW] Scale pool size 2
        self.dropoutRate = dropout_rate

        self.blocks = self.initial_block(dropout_rate)
        
        # [NEW] Set classifier to None and create it dynamically in the first forward pass
        self.classifierBlock = None
    
    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        
        # [NEW] Create the classifier on the first forward pass with correct size
        # and move to the same device as the input tensor
        if self.classifierBlock is None:
            input_size = x.size(1)
            device = x.device
            print(f"[NEW] Creating classifier with input_size={input_size} on device {device}")
            self.classifierBlock = EEGNet.classifier_block(input_size, self.n_classes).to(device)
            
        x = self.classifierBlock(x)
        return x


@register_model
class MultiScaleEEGNet(nn.Module):
    """
    Multi-scale EEGNet: runs N parallel EEGNet branches, each processing
    a different temporal window length cropped from the same input.
    Branch outputs are concatenated and fed to a shared classifier.

    The input to forward() is always the LONGEST window (max of scales).
    Shorter windows are obtained by centre-cropping along the time axis.

    Args:
        num_classes  : number of output classes
        channels     : number of EEG channels
        dropout_rate : dropout probability
        sampling_r   : sampling rate in Hz
        window_scales: list of window durations in seconds, e.g. [2, 4, 8]
                       The largest value must equal the --window CLI arg.
        f1, d, f2    : EEGNet filter counts (same for all branches)
    """

    def __init__(
        self,
        num_classes,
        channels,
        dropout_rate,
        sampling_r=256,
        window_scales=None,
        f1=8,
        d=2,
        f2=16,
        **kwargs,
    ):
        super().__init__()

        if window_scales is None:
            # default: 2s / 4s / 8s  (largest = --window CLI value)
            window_scales = [2, 4, 8]

        self.window_scales = sorted(window_scales)   # ascending
        self.max_samples   = int(max(window_scales) * sampling_r)
        self.sampling_r    = sampling_r
        self.n_classes     = num_classes

        scaling_factor = sampling_r / 128

        # ---- build one EEGNet feature-extractor per scale ---------------
        self.branches = nn.ModuleList()
        for w in self.window_scales:
            samples = int(w * sampling_r)
            klen1   = int(64 * scaling_factor)
            klen2   = int(16 * scaling_factor)
            ps1     = int(4  * scaling_factor)
            ps2     = int(8  * scaling_factor)

            block1 = nn.Sequential(
                nn.Conv2d(1, f1, (1, klen1),
                          stride=1, padding=(0, klen1 // 2), bias=False),
                nn.BatchNorm2d(f1, momentum=0.01, affine=True, eps=1e-3),
                Conv2dWithConstraint(f1, f1 * d, (channels, 1),
                                     max_norm=1, stride=1, padding=(0, 0),
                                     groups=f1, bias=False),
                nn.BatchNorm2d(f1 * d, momentum=0.01, affine=True, eps=1e-3),
                nn.ELU(),
                nn.AvgPool2d((1, ps1), stride=ps1),
                nn.Dropout(p=dropout_rate),
            )
            block2 = nn.Sequential(
                nn.Conv2d(f1 * d, f1 * d, (1, klen2),
                          stride=1, padding=(0, klen2 // 2),
                          bias=False, groups=f1 * d),
                nn.Conv2d(f1 * d, f2, 1,
                          padding=(0, 0), groups=1, bias=False, stride=1),
                nn.BatchNorm2d(f2, momentum=0.01, affine=True, eps=1e-3),
                nn.ELU(),
                nn.AvgPool2d((1, ps2), stride=ps2),
                nn.Dropout(p=dropout_rate),
            )
            self.branches.append(nn.Sequential(block1, block2))

        # classifier built lazily on first forward pass
        self.classifier = None
        self._f2          = f2

    # ------------------------------------------------------------------
    @staticmethod
    def _centre_crop(x: torch.Tensor, target_samples: int) -> torch.Tensor:
        """Centre-crop x (batch, 1, C, T) to (batch, 1, C, target_samples)."""
        T = x.shape[-1]
        if T == target_samples:
            return x
        if T < target_samples:
            # zero-pad symmetrically if input is shorter (edge case)
            pad = target_samples - T
            return torch.nn.functional.pad(x, (pad // 2, pad - pad // 2))
        start = (T - target_samples) // 2
        return x[..., start : start + target_samples]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, channels, T_max)
        branch_features = []
        for branch, w in zip(self.branches, self.window_scales):
            target = int(w * self.sampling_r)
            x_crop = self._centre_crop(x, target)   # (B, 1, C, target)
            feat   = branch(x_crop)                  # (B, F2, 1, T')
            feat   = feat.flatten(1)                 # (B, feat_dim)
            branch_features.append(feat)

        combined = torch.cat(branch_features, dim=1)  # (B, sum of feat_dims)

        # build classifier once we know the combined feature size
        if self.classifier is None:
            in_size = combined.shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(in_size, self.n_classes, bias=False),
                nn.Softmax(dim=1),
            ).to(combined.device)
            print(f"[MultiScaleEEGNet] classifier input_size={in_size} "
                  f"scales={self.window_scales}s")

        return self.classifier(combined)