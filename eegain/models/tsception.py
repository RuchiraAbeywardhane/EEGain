import logging

import torch
import torch.nn as nn

from ._registry import register_model

logger = logging.getLogger("Model")


@register_model
class TSception(nn.Module):
    @staticmethod
    def conv_block(in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel,
                stride=step,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(
        self, num_classes, input_size, sampling_r, num_t, num_s, hidden, dropout_rate, **kwargs
    ):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()

        log_info = "\n--".join(
            [f"{n}={v}" for n, v in locals().items() if n not in ["self", "__class__"]]
        )
        logger.info(f"Using model: \n{self.__class__.__name__}(\n--{log_info})")

        n_channels = int(input_size[1])

        if kwargs["data_name"] == "MAHNOB":
            self.inception_window = [0.25, 0.125, 0.0625]  # for MAHNOB dataset
            print("[INFO] Using MAHNOB dataset, setting inception_window to [0.25, 0.125, 0.0625]")
        else:
            self.inception_window = [0.5, 0.25, 0.125]  # for DEAP and other datasets

        # [FIX] For low-channel headsets (≤8 ch), add longer temporal windows
        # so theta/alpha (slow emotion signatures) are also captured
        if n_channels <= 8:
            self.inception_window = [1.0, 0.5, 0.25]
            print(f"[INFO] Low-channel device ({n_channels} ch): "
                  f"inception_window set to {self.inception_window}")

        self.pool = 8
        # by setting the convolutional kernel being (1, length) and the strides being 1 we can use conv 2d to
        # achieve the 1d convolution operation
        self.tsception1 = TSception.conv_block(
            1, num_t, (1, int(self.inception_window[0] * sampling_r)), 1, self.pool
        )
        self.tsception2 = TSception.conv_block(
            1, num_t, (1, int(self.inception_window[1] * sampling_r)), 1, self.pool
        )
        self.tsception3 = TSception.conv_block(
            1, num_t, (1, int(self.inception_window[2] * sampling_r)), 1, self.pool
        )

        self.sception1 = TSception.conv_block(
            num_t, num_s, (n_channels, 1), 1, int(self.pool * 0.25)
        )

        # [FIX] sception2: half-spatial split only makes sense when n_channels >= 4
        # and the resulting kernel must be >= 1.  Guard both conditions.
        half_ch = max(1, int(n_channels * 0.5))
        if half_ch < n_channels:
            self.sception2 = TSception.conv_block(
                num_t, num_s, (half_ch, 1), (half_ch, 1), int(self.pool * 0.25)
            )
            self._use_sception2 = True
        else:
            # only 1 channel — spatial split is meaningless, skip it
            self.sception2 = None
            self._use_sception2 = False

        # [FIX] fusion_layer kernel height must not exceed the actual spatial
        # height produced by sception concat.  Clamp to what we will have.
        # sception1 output height = 1 (full collapse)
        # sception2 output height = n_channels // half_ch  (number of groups)
        if self._use_sception2:
            n_spatial_rows = 1 + (n_channels // half_ch)
        else:
            n_spatial_rows = 1
        fusion_kernel_h = min(3, n_spatial_rows)
        self.fusion_layer = TSception.conv_block(num_s, num_s, (fusion_kernel_h, 1), 1, 4)

        # InstanceNorm2d instead of BatchNorm2d:
        # - BatchNorm accumulates running stats from train subjects only.
        #   At val/test time it applies those stats to different subjects → domain shift → flat val loss.
        # - InstanceNorm normalises per sample, per channel — no running stats, immune to subject shift.
        self.BN_t      = nn.InstanceNorm2d(num_t,  affine=True)
        self.BN_s      = nn.InstanceNorm2d(num_s,  affine=True)
        self.BN_fusion = nn.InstanceNorm2d(num_s,  affine=True)

        # Proper 2-layer classifier: num_s -> hidden -> num_classes
        # NO Softmax: nn.CrossEntropyLoss applies log-softmax internally.
        # Having Softmax here feeds log(softmax(x)) into the loss, corrupting gradients.
        self.fc = nn.Sequential(
            nn.Linear(num_s, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        y = self.tsception1(x)
        out = y
        y = self.tsception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.tsception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)

        z = self.sception1(out)
        if self._use_sception2:
            z2 = self.sception2(out)
            out_ = torch.cat((z, z2), dim=2)
        else:
            out_ = z

        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)

        return out
