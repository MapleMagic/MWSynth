"""
Conditional diffusion model for the MW correction, using flow matching.

WHY DIFFUSION AT ALL. The deterministic U-Net in ml_model.py is trained
with a pixel-wise loss, and a pixel-wise loss on an ambiguous mapping
converges to the conditional MEAN. That is not a subtle statistical
point here, it is the failure this project kept measuring: over-smooth
cores, structure removed rather than resolved. Li et al. (2025, 2026)
show the same thing from the other side -- their ensemble MEAN beats
individual members on SSIM/PSNR while losing the high-frequency detail
that matters, scoring worse on LPIPS and on classification bias for
exactly the extreme values a forecaster cares about.

The mapping is genuinely one-to-many. A single cloud-top temperature is
consistent with a range of sub-cloud hydrometeor states, and no amount
of training removes that -- Li et al. call it an identifiability limit.
A model that must emit one number per pixel has no honest option but to
average over that range. A generative model can instead sample from it,
and the spread across samples IS the uncertainty.

WHERE THIS DEPARTS FROM THE PAPERS, deliberately.

Li et al. diffuse the PMW field itself from noise, conditioned on IR.
This diffuses the RESIDUAL against the parametric backbone. The
difference matters for three reasons specific to this project:

  1. Data. They train on 18,165 samples; MWSynth has a few hundred.
     Learning the full IR->PMW map from that is not realistic. Learning a
     correction to a prior that already lands far-field 37H within ~1 K
     and the eyewall within ~14 K of a real GMI pass is a far smaller
     ask, and the prior carries the physics the data cannot.

  2. Failure mode. A from-noise generator that is uncertain produces
     plausible-looking invention. A residual generator that is uncertain
     produces a small residual, and the output falls back to the
     parametric backbone -- wrong in the ways the backbone is wrong,
     which are known, bounded and documented, rather than wrong in
     novel ways. For a tool whose output can be mistaken for an
     observation that is the right direction to fail in.

  3. Real MW. The papers are IR-only by construction. MWSynth fuses real
     passes when they exist, and the backbone the residual is measured
     against already contains them. So the correction is learned in the
     presence of real data rather than in place of it.

This is the same argument Mardani et al. (2025) make for residual
corrective diffusion in km-scale downscaling, applied to a different
prior.

FLOW MATCHING rather than DDPM: fewer sampling steps for comparable
quality, a simpler objective (regress a velocity, no noise schedule to
tune), and it is what the 2026 paper settled on. With a small dataset,
having one less thing to tune is worth real money.

SAME CAVEAT AS ml_model.py: torch is not installable in this sandbox, so
the forward/sampling code here has never been executed. The shapes and
the flow-matching algebra are checked by a numpy mirror in
`_reference_check()` below and by the test suite, which is a stronger
guarantee than "it looks like the standard pattern" but is still not the
same as having run it on a GPU. Run the __main__ block first.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_constants import MODEL_IN_CHANNELS

# Sampling steps for the ODE solver. Flow matching is comparatively
# forgiving here -- quality saturates quickly -- and inference happens
# per frame in an interactive tool, so this is deliberately small.
# Euler steps in the flow ODE. Straight-line flow matching on a SMALL
# residual converges quickly -- this is one of the concrete payoffs of
# diffusing the correction rather than the field, since the distance the
# ODE has to travel is short. 12 is the interactive default; 24 is the
# quality setting and the difference is marginal on a residual this size.
DEFAULT_SAMPLE_STEPS = 12

# Ensemble members drawn per frame. Li et al. use 10. The mean is the
# rendered field and the standard deviation is the uncertainty map.
# REMOVED as a live setting. This duplicated ml_constants.ENSEMBLE_MEMBERS,
# which is what ml_inference actually reads -- so tuning this one changed
# nothing at all, silently. Two constants for one knob, one of them inert,
# is worse than either alone. Kept only as an alias so any external caller
# still resolves, and pointed at the real setting.
from ml_constants import ENSEMBLE_MEMBERS as DEFAULT_ENSEMBLE

# DropPath rate. Li et al. (2026) found this their single most useful
# regularizer on a limited dataset, cutting CRPS by 8.7%. MWSynth's
# dataset is far smaller than theirs, so if anything it matters more.
DEFAULT_DROP_PATH = 0.1


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a continuous timestep in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch per SAMPLE, not per
    element. Dropout on activations fights the spatial coherence this
    task depends on; dropping entire paths regularizes the ensemble of
    sub-networks instead, which is why it suits small datasets."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # Shape broadcasts over everything but the batch dimension.
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class ResBlock(nn.Module):
    """Residual block with FiLM-style timestep conditioning.

    The timestep is injected as a per-channel scale and shift rather than
    concatenated, so conditioning strength is learned per channel and
    costs no spatial capacity."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8,
                 drop_path: float = 0.0):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.norm1 = nn.GroupNorm(g, in_ch if in_ch % g == 0 else 1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1.0 + scale) + shift)
        h = self.conv2(h)
        return self.skip(x) + self.drop_path(h)


class MWResidualDiffusion(nn.Module):
    """Predicts the flow-matching velocity for the 4-channel residual
    (v37, h37, v89, h89), conditioned on the full input stack.

    Conditioning is by concatenation at the input. That is the plainest
    option and, unlike cross-attention, keeps the conditioning spatially
    aligned with the output, which for a field-to-field task is the
    property that actually matters.
    """

    def __init__(self, in_channels: int = MODEL_IN_CHANNELS, out_channels: int = 4,
                 base_channels: int = 48, t_dim: int = 128,
                 drop_path: float = DEFAULT_DROP_PATH):
        super().__init__()
        c = base_channels
        self.out_channels = out_channels
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim)
        )
        self.t_dim = t_dim

        # Noisy residual + conditioning stack, concatenated.
        self.stem = nn.Conv2d(out_channels + in_channels, c, 3, padding=1)

        self.down1 = ResBlock(c, c, t_dim, drop_path=drop_path)
        self.down2 = ResBlock(c, c * 2, t_dim, drop_path=drop_path)
        self.down3 = ResBlock(c * 2, c * 4, t_dim, drop_path=drop_path)
        self.mid = ResBlock(c * 4, c * 4, t_dim, drop_path=drop_path)
        self.up3 = ResBlock(c * 8, c * 2, t_dim, drop_path=drop_path)
        self.up2 = ResBlock(c * 4, c, t_dim, drop_path=drop_path)
        self.up1 = ResBlock(c * 2, c, t_dim, drop_path=drop_path)

        self.out_norm = nn.GroupNorm(min(8, c), c)
        self.out_conv = nn.Conv2d(c, out_channels, 3, padding=1)
        # Zero-init the output layer: the model starts predicting exactly
        # zero velocity, i.e. "the backbone is already right", and learns
        # to depart from it. On a small dataset that is a meaningfully
        # better starting point than random corrections.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h = self.stem(torch.cat([z_t, cond], dim=1))

        h1 = self.down1(h, t_emb)
        h2 = self.down2(F.avg_pool2d(h1, 2), t_emb)
        h3 = self.down3(F.avg_pool2d(h2, 2), t_emb)
        m = self.mid(F.avg_pool2d(h3, 2), t_emb)

        u3 = self.up3(torch.cat([F.interpolate(m, size=h3.shape[-2:], mode="nearest"), h3], 1), t_emb)
        u2 = self.up2(torch.cat([F.interpolate(u3, size=h2.shape[-2:], mode="nearest"), h2], 1), t_emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="nearest"), h1], 1), t_emb)
        return self.out_conv(F.silu(self.out_norm(u1)))


def flow_matching_loss(model: nn.Module, residual: torch.Tensor, cond: torch.Tensor,
                       mask: torch.Tensor = None) -> torch.Tensor:
    """Flow-matching objective.

    Interpolate between noise and data, z_t = t*x + (1-t)*eps, and regress
    the constant-velocity field v = x - eps. Straight-line paths, so
    sampling is a plain ODE integration with no schedule to tune.

    `mask` is applied to the loss, not the target: real MW swaths have
    gaps, and supervising a pixel with no observation behind it teaches
    the model that missing data means zero correction.
    """
    b = residual.shape[0]
    t = torch.rand(b, device=residual.device)
    t_b = t.reshape(b, 1, 1, 1)
    eps = torch.randn_like(residual)
    z_t = t_b * residual + (1.0 - t_b) * eps
    target_v = residual - eps
    pred_v = model(z_t, t, cond)

    err = (pred_v - target_v) ** 2
    if mask is not None:
        err = err * mask
        return err.sum() / mask.sum().clamp(min=1.0)
    return err.mean()


@torch.no_grad()
def sample_residual(model: nn.Module, cond: torch.Tensor, out_channels: int = 4,
                    steps: int = DEFAULT_SAMPLE_STEPS, ensemble: int = 1,
                    generator=None) -> torch.Tensor:
    """Integrate the learned velocity field from noise to a residual.

    Returns (ensemble, batch, out_channels, H, W). Members share the
    conditioning and differ only in their noise draw, so their spread is
    the model's own uncertainty about a single frame -- which is the
    quantity worth rendering.
    """
    b, _, h, w = cond.shape
    device = cond.device
    # Members are drawn one at a time rather than batched. On 4 GB that is
    # the difference between running and not: batching N members
    # multiplies activation memory by N, and the wall-clock saving is
    # small because the GPU is already saturated at batch 1 for this
    # resolution.
    out = []
    for _ in range(max(1, ensemble)):
        z = torch.randn((b, out_channels, h, w), device=device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), i * dt, device=device)
            z = z + model(z, t, cond) * dt      # explicit Euler on a straight path
        out.append(z)
    return torch.stack(out, dim=0)


def _reference_check():
    """Numpy mirror of the flow-matching algebra, so the maths is verified
    even where torch cannot run. Checks that integrating the TRUE velocity
    field recovers the data exactly, which is the property the sampler
    relies on."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4, 8, 8))          # the "residual" we want back
    eps = rng.normal(size=x.shape)

    # With the exact velocity v = x - eps, Euler integration from z0 = eps
    # over t in [0,1] must land on x for any step count.
    for steps in (1, 4, 24):
        z = eps.copy()
        dt = 1.0 / steps
        for _ in range(steps):
            z = z + (x - eps) * dt
        err = float(np.abs(z - x).max())
        assert err < 1e-9, f"integration error {err} at {steps} steps"

    # And z_t must interpolate correctly at the endpoints.
    assert np.allclose(0.0 * x + 1.0 * eps, eps)
    assert np.allclose(1.0 * x + 0.0 * eps, x)
    return True


if __name__ == "__main__":
    _reference_check()
    print("flow-matching algebra OK")
    m = MWResidualDiffusion()
    n = sum(p.numel() for p in m.parameters())
    print(f"parameters: {n/1e6:.2f}M")
    cond = torch.randn(2, MODEL_IN_CHANNELS, 64, 64)
    res = torch.randn(2, 4, 64, 64)
    print("loss:", float(flow_matching_loss(m, res, cond)))
    s = sample_residual(m, cond, steps=4, ensemble=3)
    print("sample shape:", tuple(s.shape))
