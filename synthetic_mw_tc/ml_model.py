"""
Compact U-Net for learning a CORRECTION on top of the existing parametric
synthetic algorithm's output -- not a from-scratch MW predictor. Given
the chosen direction (residual/correction learning, see project notes),
this model's target is (real_MW - parametric_backbone), not real_MW
directly.

HONEST, IMPORTANT CAVEAT, different in kind from everything else in this
project: PyTorch isn't installable in this sandbox (no network access),
so unlike almost every other module here, this code has NEVER been
executed -- no forward-pass shape check, no gradient step, nothing. It's
written using standard, well-established U-Net patterns (encoder/decoder
with skip connections, GroupNorm, transposed-conv upsampling) that are
about as close to "boilerplate" as this kind of architecture gets, but
"I'm confident this is correct based on the pattern" is a meaningfully
weaker claim than "I ran this and confirmed it," which is the bar
everything else in this project has been held to. Run a quick shape
sanity check (a few lines, see the __main__ block below) before trusting
this for real training.

SIZING, for a 6GB-VRAM laptop GPU (RTX 3050): base_channels=32 with
4 downsampling levels (32->64->128->256->512 at the bottleneck) should
comfortably fit 128-192px patches at a batch size of 8-16 with mixed
precision (torch.cuda.amp) -- but "should" is an estimate from the
architecture's parameter/activation memory footprint, not a measurement
on your actual hardware. Start smaller (e.g. 96px patches, batch_size=4)
and scale up while watching nvidia-smi, rather than assuming the
estimate is exactly right.
"""
from __future__ import annotations

from ml_constants import MODEL_IN_CHANNELS
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convs, each followed by GroupNorm + ReLU. GroupNorm (not
    BatchNorm) deliberately -- this project's likely batch sizes (4-16,
    per the VRAM sizing note above) are small enough that BatchNorm's
    running statistics can get noisy; GroupNorm doesn't depend on batch
    size at all."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)  # GroupNorm requires groups to divide out_ch evenly
        while out_ch % groups != 0:
            groups -= 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        return x


class MWCorrectionUNet(nn.Module):
    """Predicts a per-pixel correction to add to the parametric
    algorithm's V37/H37/V89/H89 backbone output.

    Default input channels (9): IR (band13), WV (band9), SWIR (band7),
    the backbone's own V37/H37/V89/H89, plus intensity and RMW --
    broadcast to constant-value 2D layers across the whole patch (a
    standard way to inject scalar/global context into a convolutional
    architecture). Added directly because the existing parametric
    algorithm this model is meant to CORRECT already leans heavily on
    both (calibration curves, radial weight profiles keyed on RMW) --
    without seeing the same context, the correction model has no way to
    learn something like "the backbone tends to underestimate 89GHz
    depression specifically for strong, compact systems," since it
    can't tell a 65kt system from a 115kt one from imagery alone with
    any reliability. See ml_train.py's MWCorrectionDataset for how these
    get loaded, normalized, and broadcast from the saved training data
    (both already present in every .npz file training_data_export.py
    produces -- this was a real gap where the data existed and was
    already correctly computed but the model never actually saw it).

    Add VIS (band2) as an additional channel if training with
    daytime-only examples (this project already knows band2 is only
    available in daylight, so the training pipeline needs a consistent
    decision about whether to include it at all, zero-fill it at night,
    or train separate day/night variants -- deliberately left as a
    training-script decision, not baked into the model, since it's a
    data question more than an architecture one).

    Output channels (4): correction_v37, correction_h37, correction_v89,
    correction_h89 -- add these to the backbone to get the corrected
    prediction, everywhere in the image, not just where training
    supervision existed (masking only matters for the LOSS during
    training, not for what the model is capable of predicting).
    """

    def __init__(self, in_channels: int = MODEL_IN_CHANNELS, out_channels: int = 4, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

        # Zero-init the output layer's weights (not bias) so the model
        # starts out predicting a near-zero correction everywhere --
        # meaning at the start of training, output ~= backbone (the
        # existing parametric algorithm), and training only pulls it
        # away from that where the data actually supports a correction.
        # This is a deliberate choice for a RESIDUAL-learning setup:
        # without it, an untrained/undertrained model could inject
        # large, wrong "corrections" that make things worse than the
        # existing algorithm alone, especially early in training or in
        # regions unlike anything in the training set.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self._pad_and_cat(self.up4(b), e4)
        d4 = self.dec4(d4)
        d3 = self._pad_and_cat(self.up3(d4), e3)
        d3 = self.dec3(d3)
        d2 = self._pad_and_cat(self.up2(d3), e2)
        d2 = self.dec2(d2)
        d1 = self._pad_and_cat(self.up1(d2), e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    @staticmethod
    def _pad_and_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle any off-by-one spatial mismatch between the upsampled
        path and the skip connection -- can happen for odd input sizes
        propagating through repeated stride-2 pooling. Pads `up` to
        match `skip`'s spatial size before concatenating along channels."""
        diff_h = skip.size(2) - up.size(2)
        diff_w = skip.size(3) - up.size(3)
        if diff_h != 0 or diff_w != 0:
            up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([skip, up], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Minimal shape sanity check -- run this directly (`python ml_model.py`)
    # after installing torch, before trusting the architecture for real
    # training. This is NOT a substitute for actually training and
    # validating the model, just confirms the tensor shapes flow through
    # correctly end to end.
    model = MWCorrectionUNet(in_channels=MODEL_IN_CHANNELS, out_channels=4, base_channels=32)
    print(f"Parameters: {count_parameters(model):,}")

    for size in (96, 128, 160):
        x = torch.randn(2, 9, size, size)
        y = model(x)
        expected = (2, 4, size, size)
        status = "OK" if tuple(y.shape) == expected else "MISMATCH"
        print(f"input {tuple(x.shape)} -> output {tuple(y.shape)} (expected {expected}) [{status}]")
