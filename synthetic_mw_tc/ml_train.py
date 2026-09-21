"""
Training script for MWCorrectionUNet (ml_model.py) against the paired
examples accumulated by training_data_export.py / ml_data_mining.py.

HONEST CAVEAT, same as ml_model.py: PyTorch isn't installable in this
sandbox, so this has never actually been run -- no confirmed loss curve,
no confirmed checkpoint produced. Written carefully against standard
PyTorch training-loop patterns, but treat this as a starting point to
run and debug on your own machine, not as verified-working code the way
the rest of this project's Python has been.

DESIGN NOTES:
- Patches, not full scenes: each .npz example gets a fixed-size patch
  extracted (centered on the storm by default, since that's the region
  that actually matters and keeps memory bounded regardless of how big
  any individual saved scene happens to be).
- Masked loss: target pixels are NaN wherever there was no real MW
  coverage in the original scene (see training_data_export.py) -- the
  loss only accumulates over pixels that actually have a real
  supervision signal, not synthetic zeros.
- Split by STORM, not by individual example -- multiple examples from
  the same storm a few hours apart are highly correlated (same storm
  structure, similar intensity), so splitting by individual example
  would leak information between train and validation and overstate
  how well the model generalizes to a genuinely new storm.
- Residual target: (target - backbone), computed here, not stored
  pre-computed in the .npz files -- keeps the saved data format
  agnostic to exactly how the residual is framed, in case that changes.
"""
from __future__ import annotations

import glob
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml_model import MWCorrectionUNet
import ml_diffusion
from ml_constants import CORRECTION_ARCH, DIFFUSION_BASE_CHANNELS
from ml_constants import (normalize_channel, TB_MEAN, TB_STD, VMAX_MEAN, VMAX_STD, RMW_MEAN, RMW_STD, PATCH_SIZE,
                          PATCH_SAMPLE_STRIDE, PATCH_FOOTPRINT_PX, MODEL_EXTRA_IR_BANDS,
                          ELEV_MEAN, ELEV_STD, MODEL_IN_CHANNELS)
from mw_composites import COLOR_RED_THETA

def _wrap_lon_delta(dlon):
    """Shortest signed longitude difference, dateline-safe.

    A storm near 180 has grid longitudes on both sides of the wrap, so a
    naive (lon - center_lon) reads as up to 360 degrees where the true
    separation is a fraction of a degree. Every radius, distance and
    patch-centering calculation built on that would be wrong for west
    Pacific storms crossing the dateline -- the case that arrives the
    moment Himawari support is added.
    """
    return (dlon + 180.0) % 360.0 - 180.0

# Weight on the PCT-space loss term relative to the per-channel L1 term.
# See masked_pct_loss() for why this term exists at all.
PCT_LOSS_WEIGHT = 0.5

# Probability of zeroing the vmax/RMW conditioning layers for a training
# sample. See MWCorrectionDataset for the reasoning.
SCALAR_DROPOUT_P = 0.25

# Mirror (left-right flip) augmentation. OFF by default: it reverses a
# cyclone's rotational handedness, and since SH basins are excluded from
# mining, the mirrored handedness never occurs at inference. Set True
# only if SH storms are added to the training set, at which point
# mirroring becomes a legitimate way to share structure across
# hemispheres.
ALLOW_MIRROR_AUGMENTATION = False

# Intensity bands (kt, upper-exclusive) used to stratify the train/val
# split and to report validation loss per band. Boundaries are the
# operational TD / TS / cat1-2 / major breakpoints.
INTENSITY_BANDS = ((0, 34), (34, 64), (64, 96), (96, 999))


def band_label(vmax_kt: float) -> str:
    for lo, hi in INTENSITY_BANDS:
        if lo <= vmax_kt < hi:
            return f"{lo}-{hi if hi < 999 else '+'}kt"
    return "unknown"

DEFAULT_DATA_DIR = os.path.expanduser("~/.synthetic_mw_tc/training_data")
DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/.synthetic_mw_tc/ml_checkpoints")


def _normalize_tb(x: np.ndarray, channel: str = "ir") -> np.ndarray:
    """Normalize one named input channel.

    Delegates to ml_constants.normalize_channel so this and ml_inference
    share ONE implementation. They previously each carried their own
    arithmetic -- identical only by coincidence -- and a normalization
    mismatch between training and inference produces no error at all,
    just quietly wrong corrections.
    """
    return normalize_channel(x, channel)


def _denormalize_tb(x, channel: str = "ir"):
    from ml_constants import denormalize_channel
    return denormalize_channel(x, channel)


def _extract_centered_patch(arr: np.ndarray, center_rc: tuple, patch_size: int) -> np.ndarray:
    """Extract a patch_size x patch_size patch centered on (row, col),
    zero-padding if the patch would extend past the array's edges (can
    happen for a storm near the edge of its GOES crop)."""
    h, w = arr.shape[:2]
    r, c = center_rc
    half = patch_size // 2
    r0, r1 = r - half, r - half + patch_size
    c0, c1 = c - half, c - half + patch_size

    pad_top = max(0, -r0)
    pad_left = max(0, -c0)
    pad_bottom = max(0, r1 - h)
    pad_right = max(0, c1 - w)

    r0c, r1c = max(0, r0), min(h, r1)
    c0c, c1c = max(0, c0), min(w, c1)
    cropped = arr[r0c:r1c, c0c:c1c]

    if pad_top or pad_bottom or pad_left or pad_right:
        pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)] + [(0, 0)] * (cropped.ndim - 2)
        cropped = np.pad(cropped, pad_width, mode="constant", constant_values=np.nan)

    return cropped


class MWCorrectionDataset(Dataset):
    """Loads .npz examples from data_dir, extracts a fixed-size
    storm-centered patch from each, and returns
    (input_tensor[C,H,W], target_tensor[4,H,W], mask_tensor[4,H,W]).

    Splitting by storm should happen BEFORE constructing this (pass only
    the file list for the storms belonging to this split) -- see
    make_train_val_split() below.
    """

    def __init__(self, file_paths: list, patch_size: int = PATCH_SIZE, augment: bool = False,
                 scalar_dropout: float = 0.0):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.augment = augment
        # Probability of blanking the vmax/RMW conditioning layers.
        #
        # WHY: those two layers are constant across the entire patch,
        # which makes them by far the cheapest feature for a
        # fully-convolutional network to key on -- far cheaper than
        # learning structure from the imagery. On a small dataset that is
        # a trap: the model can drive its loss down by learning "weak,
        # small storm => weak signature" and then apply that prior even
        # when the imagery plainly shows an organised core. That is
        # exactly the failure seen on a 50 kt storm with a genuine
        # radar/MW core (vmax -0.75 sigma, RMW -1.26 sigma), where the
        # correction suppressed a signature that really existed.
        #
        # Blanking them on a fraction of samples forces the imagery to
        # carry the prediction on its own some of the time. Zero is the
        # right blank value because these layers are already normalized,
        # so 0 means "an average storm" -- the same imputation used when
        # best-track RMW is genuinely missing -- rather than an
        # out-of-distribution sentinel.
        self.scalar_dropout = scalar_dropout

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path, allow_pickle=True)

        ir = data["ir_band13"]
        wv = data["wv_band9"]
        swir = data["swir_band7"]
        lat = data["lat"]
        lon = data["lon"]
        storm_lat = float(data["storm_lat"])
        storm_lon = float(data["storm_lon"])
        # Intensity and RMW -- see ml_model.py's docstring for why these
        # matter for a CORRECTION model specifically: the parametric
        # algorithm being corrected already leans on both, so the
        # correction model needs the same context to learn anything
        # intensity/size-dependent, rather than only ever seeing raw
        # imagery. Both already saved in every .npz by
        # training_data_export.py -- this was previously a real gap
        # where the data existed but was never actually loaded here.
        storm_vmax_kt = float(data["storm_vmax_kt"])
        storm_rmw_nm = float(data["storm_rmw_nm"])
        if not np.isfinite(storm_rmw_nm):
            # RMW is genuinely missing for a meaningful fraction of
            # historical best-track entries (not every advisory
            # includes it) -- impute with the normalization mean rather
            # than propagate NaN into the model input, which would
            # otherwise poison the whole tensor via broadcasting.
            storm_rmw_nm = RMW_MEAN

        def _load_tb(arr):
            """Read a brightness-temperature array, packed or not.

            0.127 wired in the scaled-uint16 packing that had been dead
            since 0.104. Keying on DTYPE rather than on a format flag means
            a dataset can hold both -- which it will, since the packing
            landed mid-mine. Treating a packed array as float would train
            on values around 25,000 K and look like a physics failure
            rather than a format one.
            """
            a = np.asarray(arr)
            if a.dtype == np.uint16:
                from training_data_export import unpack_tb
                return unpack_tb(a)
            return a.astype(np.float32)

        backbone_v37 = _load_tb(data["backbone_v37"])
        backbone_h37 = _load_tb(data["backbone_h37"])
        backbone_v89 = _load_tb(data["backbone_v89"])
        backbone_h89 = _load_tb(data["backbone_h89"])
        target_v37 = _load_tb(data["target_v37"])
        target_h37 = _load_tb(data["target_h37"])
        target_v89 = _load_tb(data["target_v89"])
        target_h89 = _load_tb(data["target_h89"])

        # Find the pixel nearest the storm center to center the patch on --
        # simple nearest-neighbor via minimum distance, adequate at this
        # grid resolution (no need for anything fancier than brute-force
        # argmin over a storm-scale crop).
        dist2 = (lat - storm_lat) ** 2 + _wrap_lon_delta(lon - storm_lon) ** 2
        center_rc = np.unravel_index(np.argmin(dist2), dist2.shape)

        ps = self.patch_size
        fp = ps * PATCH_SAMPLE_STRIDE   # native pixels covered (see ml_constants)

        def _extract(arr):
            """Extract the native-resolution footprint and block-average it
            down to the model's patch size. Must mirror ml_inference's
            _downsample exactly: training on subsampled data while inferring
            on averaged data would be a silent input mismatch."""
            patch = _extract_centered_patch(arr, center_rc, fp)
            if PATCH_SAMPLE_STRIDE == 1:
                return patch
            k = PATCH_SAMPLE_STRIDE
            h, w = patch.shape[0] // k, patch.shape[1] // k
            return patch[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))

        ir_p = _extract(ir)
        wv_p = _extract(wv)
        swir_p = _extract(swir)
        bv37_p = _extract(backbone_v37)
        bh37_p = _extract(backbone_h37)
        bv89_p = _extract(backbone_v89)
        bh89_p = _extract(backbone_h89)
        tv37_p = _extract(target_v37)
        th37_p = _extract(target_h37)
        tv89_p = _extract(target_v89)
        th89_p = _extract(target_h89)

        if self.augment:
            # Horizontal flip is DISABLED by default (see
            # ALLOW_MIRROR_AUGMENTATION). Mirroring a cyclone reverses its
            # rotational handedness, turning a Northern-Hemisphere storm
            # into something that spirals the way only a Southern-
            # Hemisphere storm does. The mining config excludes SH
            # basins, so every real storm the model sees is NH -- with
            # mirroring on, half of all augmented samples had a
            # handedness that never occurs at inference, spending model
            # capacity on a case that cannot arise and denying it spiral
            # handedness as a usable cue. Rotation is kept: a
            # storm-centered patch has no privileged absolute
            # orientation in the same way.
            if ALLOW_MIRROR_AUGMENTATION and random.random() < 0.5:
                flip_fn = lambda a: np.fliplr(a).copy()
                ir_p, wv_p, swir_p = flip_fn(ir_p), flip_fn(wv_p), flip_fn(swir_p)
                bv37_p, bh37_p, bv89_p, bh89_p = flip_fn(bv37_p), flip_fn(bh37_p), flip_fn(bv89_p), flip_fn(bh89_p)
                tv37_p, th37_p, tv89_p, th89_p = flip_fn(tv37_p), flip_fn(th37_p), flip_fn(tv89_p), flip_fn(th89_p)
            k = random.choice([0, 1, 2, 3])
            if k:
                rot_fn = lambda a: np.rot90(a, k).copy()
                ir_p, wv_p, swir_p = rot_fn(ir_p), rot_fn(wv_p), rot_fn(swir_p)
                bv37_p, bh37_p, bv89_p, bh89_p = rot_fn(bv37_p), rot_fn(bh37_p), rot_fn(bv89_p), rot_fn(bh89_p)
                tv37_p, th37_p, tv89_p, th89_p = rot_fn(tv37_p), rot_fn(th37_p), rot_fn(tv89_p), rot_fn(th89_p)

        backbone_stack = np.stack([bv37_p, bh37_p, bv89_p, bh89_p], axis=0)
        target_stack = np.stack([tv37_p, th37_p, tv89_p, th89_p], axis=0)

        mask = np.isfinite(target_stack)
        # Residual target: model predicts what to ADD to the backbone to
        # match the real MW value. Undefined (NaN) wherever there's no
        # real coverage -- filled with 0 here only because the mask
        # (returned separately) is what actually excludes these pixels
        # from the loss; the 0 fill is just to keep the tensor free of
        # NaN for downstream arithmetic, not a claim that 0 is the right
        # target there.
        residual = np.where(mask, target_stack - backbone_stack, 0.0)

        # Broadcast the storm-state scalars to constant-value patch_size x
        # patch_size layers -- the standard way to inject global/scalar
        # context into an otherwise fully-convolutional architecture, so
        # every spatial location in this patch sees the same "this is a
        # 65kt storm" (or whatever) signal alongside the per-pixel imagery.
        vmax_norm = (storm_vmax_kt - VMAX_MEAN) / VMAX_STD
        rmw_norm = (storm_rmw_nm - RMW_MEAN) / RMW_STD
        if self.scalar_dropout > 0.0 and random.random() < self.scalar_dropout:
            vmax_norm = 0.0
            rmw_norm = 0.0
        vmax_layer = np.full((ps, ps), vmax_norm, dtype=np.float32)
        rmw_layer = np.full((ps, ps), rmw_norm, dtype=np.float32)

        # Fixed channel order -- ml_constants.INPUT_CHANNEL_LAYOUT. Bands
        # an example doesn't carry become a neutral plane rather than
        # being dropped, so older exports remain trainable against the
        # current architecture instead of silently shifting every channel
        # index by one.
        neutral = np.zeros((ps, ps), dtype=np.float32)

        def _opt(key):
            if key not in getattr(data, "files", ()):
                return neutral
            return _normalize_tb(_extract(np.asarray(data[key], dtype=np.float32)), key)

        extra_layers = [_opt(f"ir_band{b}") for b in MODEL_EXTRA_IR_BANDS]

        def _surface(key, mean, std):
            if key not in getattr(data, "files", ()):
                return neutral
            p_ = _extract(np.asarray(data[key], dtype=np.float32))
            return ((p_ - mean) / std).astype(np.float32)

        def _flash_channel(d, extract, neutral_plane):
            if "flash_density" not in getattr(d, "files", ()):
                return neutral_plane
            import glm_lightning
            return glm_lightning.normalize_for_model(
                extract(np.asarray(d["flash_density"], dtype=np.float32))
            ).astype(np.float32)

        input_stack = np.stack(
            [_normalize_tb(ir_p, "ir"), _normalize_tb(wv_p, "wv"), _normalize_tb(swir_p, "swir"),
             *extra_layers,
             _normalize_tb(bv37_p, "backbone_v37"), _normalize_tb(bh37_p, "backbone_h37"), _normalize_tb(bv89_p, "backbone_v89"), _normalize_tb(bh89_p, "backbone_h89"),
             vmax_layer, rmw_layer,
             _surface("land_fraction", 0.0, 1.0),
             _surface("elevation_m", ELEV_MEAN, ELEV_STD),
             _flash_channel(data, _extract, neutral)],
            axis=0,
        )
        # Any remaining NaN in the INPUT (e.g. backbone at a masked-out
        # edge pixel) gets zero-filled post-normalization -- inputs must
        # be fully finite for the network, unlike the target/mask pair.
        input_stack = np.nan_to_num(input_stack, nan=0.0)

        return (
            torch.from_numpy(input_stack.astype(np.float32)),
            torch.from_numpy((residual / TB_STD).astype(np.float32)),  # scale residual similarly to inputs
            torch.from_numpy(mask.astype(np.float32)),
            # The TRUE vmax, not the possibly-dropped-out conditioning
            # value -- used only to bucket validation loss by intensity
            # band, never fed to the model.
            torch.tensor(storm_vmax_kt, dtype=torch.float32),
        )


def learning_curve_storm_subsets(data_dir: str = DEFAULT_DATA_DIR,
                                 fractions=(0.25, 0.5, 0.75, 1.0),
                                 seed: int = 42):
    """Training-set subsets for a learning curve, taken BY STORM.

    Answers directly a question this project has only ever reasoned about
    indirectly: does more data still help? Bias sat at ~11.5 K across
    298 and then 799 examples while spread fell, which suggested the
    remaining error was systematic -- but that was an inference from two
    points, not a curve.

    It also prices `--max-per-storm`. That flag deliberately shrinks the
    dataset to cut mining time; if the curve is already flat at half the
    data, the cap costs nothing and future mines can be shorter still.

    Subsets are nested and drawn BY STORM, not by example, for the same
    reason make_train_val_split is: two overpasses of one hurricane hours
    apart are nearly the same picture, so counting them as independent
    training material overstates how much data a given fraction really
    represents.
    """
    train_files, val_files = make_train_val_split(data_dir, seed=seed)
    storms = sorted(set(os.path.basename(f).split("_")[0] for f in train_files))
    rng = random.Random(seed)
    rng.shuffle(storms)
    out = []
    for frac in fractions:
        k = max(1, int(round(len(storms) * float(frac))))
        keep = set(storms[:k])
        subset = [f for f in train_files
                  if os.path.basename(f).split("_")[0] in keep]
        out.append({"fraction": float(frac), "n_storms": k,
                    "n_examples": len(subset), "train_files": subset,
                    "val_files": val_files})
    return out


def make_train_val_split(data_dir: str = DEFAULT_DATA_DIR, val_fraction: float = 0.15, seed: int = 42) -> tuple:
    """Splits by STORM ID (parsed from each filename's leading segment,
    see training_data_export.py's naming convention), not by individual
    file -- so validation genuinely tests generalization to storms the
    model never saw during training, not just held-out times from
    storms it already learned from.

    STRATIFIED BY PEAK INTENSITY. Previously this shuffled the storm list
    and took the first 15%, which at this dataset size is a handful of
    storms drawn without regard to intensity -- so the validation set
    could easily contain no weak storm at all. That matters more than it
    sounds: the observed failure mode is a model that suppresses the
    signature on weak-but-organised storms, and an all-strong validation
    set would score that model as healthy while it deleted cores on
    exactly the frames a forecaster would care about. Sampling
    val_fraction from within each intensity band guarantees every band is
    represented in validation.

    Falls back to the old unstratified shuffle if intensities can't be
    read -- a split is better than no split.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    storms = sorted(set(os.path.basename(f).split("_")[0] for f in files))

    rng = random.Random(seed)
    peaks = _storm_peak_vmax(data_dir)

    if peaks:
        by_band = {}
        for storm in storms:
            v = peaks.get(storm)
            key = band_label(v) if v is not None else "unknown"
            by_band.setdefault(key, []).append(storm)
        val_storms = set()
        for band, band_storms in sorted(by_band.items()):
            rng.shuffle(band_storms)
            # At least one storm per band whenever the band has more than
            # one, so no band is silently absent from validation. A band
            # with a single storm goes to training -- holding it out would
            # mean the model never sees that intensity regime at all.
            n_val = max(1, int(round(len(band_storms) * val_fraction))) if len(band_storms) > 1 else 0
            val_storms.update(band_storms[:n_val])
    else:
        shuffled = list(storms)
        rng.shuffle(shuffled)
        n_val_storms = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
        val_storms = set(shuffled[:n_val_storms])

    train_files = [f for f in files if os.path.basename(f).split("_")[0] not in val_storms]
    val_files = [f for f in files if os.path.basename(f).split("_")[0] in val_storms]
    return train_files, val_files


def masked_pixel_count(mask: torch.Tensor) -> float:
    """How many pixels in this batch actually carry supervision. Used to
    detect the silent-failure case where NOTHING does -- see train()."""
    return float(mask.sum().item())


def masked_pct_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss on the polarization-corrected combination that the colour
    composites actually render, rather than on V and H independently.

    WHY THIS TERM EXISTS: the per-channel loss above has no view of the
    combination
        PCT37 = 2.181*V37 - 1.181*H37   (rendered over a 20 K window)
        PCT89 = 1.818*V89 - 0.818*H89   (rendered over a 90 K window)
    so nothing in training ever penalised a correction that moved V and H
    in opposite directions. That combination is amplified up to 3.36x at
    37 GHz, which means a per-channel error small enough to look
    negligible in the L1 term can traverse the ENTIRE visible colour
    range. Worse, `best_val_loss` selects the saved checkpoint using that
    same blind metric, so the model being kept was chosen on a quantity
    only loosely related to what the output looks like.

    Channel order is (v37, h37, v89, h89), matching the dataset's
    target_stack. A pixel is supervised only where BOTH channels of a
    frequency pair have coverage -- the combination is undefined
    otherwise, and taking one channel alone would invent a polarization
    difference the data does not contain.
    """
    total = None
    for idx_v, idx_h, freq in ((0, 1, 37), (2, 3, 89)):
        theta = COLOR_RED_THETA[freq]
        pair_mask = mask[:, idx_v] * mask[:, idx_h]
        pred_pct = (1.0 + theta) * pred[:, idx_v] - theta * pred[:, idx_h]
        targ_pct = (1.0 + theta) * target[:, idx_v] - theta * target[:, idx_h]
        diff = torch.abs(pred_pct - targ_pct) * pair_mask
        term = diff.sum() / pair_mask.sum().clamp(min=1.0)
        total = term if total is None else total + term
    return total / 2.0


def combined_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  pct_weight: float = None) -> tuple:
    """Returns (total, l1_component, pct_component). Reported separately
    so a run shows whether the two terms are actually trading off or one
    is dominating.

    pct_weight defaults to None and is resolved from the module global at
    CALL time, not bound as a default at def time -- otherwise
    run_ml_pipeline's `ml_train.PCT_LOSS_WEIGHT = ...` override would be
    silently ignored, which is the kind of knob that appears to work and
    does nothing."""
    if pct_weight is None:
        pct_weight = PCT_LOSS_WEIGHT
    l1 = masked_loss(pred, target, mask)
    pct = masked_pct_loss(pred, target, mask)
    return l1 + pct_weight * pct, l1, pct


def _storm_peak_vmax(data_dir: str) -> dict:
    """Map storm ID -> peak vmax across its examples, by reading only the
    storm_vmax_kt scalar from each .npz. npz is a zip archive, so pulling
    one small array does not decompress the image stacks."""
    peaks = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        storm = os.path.basename(f).split("_")[0]
        try:
            with np.load(f, allow_pickle=True) as d:
                v = float(d["storm_vmax_kt"])
        except Exception:
            continue
        if np.isfinite(v):
            peaks[storm] = max(peaks.get(storm, -np.inf), v)
    return peaks


def _vh_physics_id():
    """Identifier for the backbone radiative physics in force at training
    time. Imported lazily so ml_train stays importable without the full
    generation stack."""
    try:
        from synthetic_algorithm import VH_PHYSICS_ID
        return VH_PHYSICS_ID
    except Exception:
        return None


# Lives in training_data_export (which has no torch dependency) so it can
# be tested and used without importing the training stack. Re-exported here
# because train() is its main caller.
from training_data_export import check_physics_consistency  # noqa: E402


def crps_ensemble(members_k, truth_k, mask=None):
    """Continuous Ranked Probability Score, the fair-weather estimator.

        CRPS = mean|x_i - y|  -  0.5 * mean|x_i - x_j|

    WHY ADD IT. Li et al. (2026) score their synthesis with CRPS
    throughout, and for good reason: it is a PROPER scoring rule, so it
    rewards a sharp forecast only when that sharpness is justified.

    This project selects on ensemble-mean RMSE skill, which cannot see
    calibration at all -- a model could collapse to a single draw and
    score identically. And the ensemble here is measurably overconfident,
    at 1.95x, so the metric was blind to a known defect. CRPS penalises
    exactly that: the first term is accuracy, the second rewards spread,
    and an under-dispersed ensemble loses on the second.

    Reported in Kelvin, lower is better. Not yet the selection metric --
    changing what a run optimises deserves a measured comparison first,
    not a swap on the strength of an argument.
    """
    m = np.asarray(members_k, dtype=np.float64)     # (members, ...)
    y = np.asarray(truth_k, dtype=np.float64)
    if mask is not None:
        sel = np.asarray(mask, dtype=bool)
        if not sel.any():
            return float("nan")
        m = m[:, sel]
        y = y[sel]
    else:
        m = m.reshape(m.shape[0], -1)
        y = y.reshape(-1)
    n = m.shape[0]
    if n == 0 or y.size == 0:
        return float("nan")
    accuracy = float(np.mean(np.abs(m - y[None, :])))
    if n == 1:
        return accuracy          # no spread term with a single member
    # Mean pairwise spread, over ordered pairs including i == j, which is
    # the standard fair-weather form.
    spread = float(np.mean(np.abs(m[:, None, :] - m[None, :, :])))
    return accuracy - 0.5 * spread


def sampled_skill(model, loader, device, steps=16, members=4, max_batches=4):
    """Sample residuals and score them against the DO-NOTHING baseline.

    WHY THIS IS NECESSARY: the flow-matching loss is not interpretable on
    its own. For z_t = t*x1 + (1-t)*x0 the best achievable loss is
    E_t[s^2 / (t^2 s^2 + (1-t)^2)], which depends entirely on s, the
    residual SCALE -- so the number mostly reports how big the residuals
    happened to be, not whether the model learned anything useful. A
    val_loss of 0.075 is the optimum for s ~ 0.05 (about 2 K) and is
    simply unreachable if the true residuals are larger. Watching it fall
    tells you the optimizer is working; it does not tell you the
    correction is worth applying.

    So: draw a sample, and compare its error against the error of
    predicting zero everywhere. Skill = 1 - rmse_model / rmse_zero.

        skill > 0   the correction beats leaving the backbone alone
        skill ~ 0   the model has learned that the residual is small,
                    which is a real thing to learn and useless to apply
        skill < 0   applying it makes the frame worse

    Reported in Kelvin, masked to pixels with real MW behind them. This
    is the number to judge the model on.
    """
    import torch

    model.eval()
    se_model = se_zero = n_px = spread_sum = se_member = 0.0
    crps_sum = crps_px = 0.0
    se_offset = 0.0
    with torch.no_grad():
        for bi, (inputs, targets, masks, _v) in enumerate(loader):
            if bi >= max_batches:
                break
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            # Score the ENSEMBLE MEAN, not a single draw. This was
            # ensemble=1, which quietly penalised the model for being
            # generative: RMSE is minimised by the conditional MEAN, and a
            # single sample from a conditional distribution has expected
            # squared error of bias^2 + 2*sigma^2 against the mean's
            # bias^2 + sigma^2/N. Scoring one draw measures the model's
            # spread as if it were error. Li et al. make exactly this
            # point from the other side -- their ensemble mean wins on
            # pixel metrics while individual members win on perceptual
            # ones -- and I wrote that in this module's docstring and then
            # scored with one member anyway.
            ens = ml_diffusion.sample_residual(
                model, inputs, out_channels=targets.shape[1],
                steps=steps, ensemble=members
            )
            drawn = ens.mean(dim=0)
            spread_sum += float((ens.std(dim=0, unbiased=(members > 1)) * masks).sum())
            # Per-member error too. With both, bias and spread separate:
            #   single draw   MSE = bias^2 + sigma^2
            #   mean of N     MSE = bias^2 + sigma^2/N
            # which says how much of the remaining error more sampling
            # could ever remove, and how much is simply the model being
            # wrong.
            se_member += float(sum(
                (((ens[m] - targets) ** 2) * masks).sum() for m in range(ens.shape[0])
            )) / ens.shape[0]
            # CRPS over the real-MW pixels of this batch. Accumulated as a
            # pixel-weighted mean so batches of differing coverage combine
            # correctly.
            try:
                _m = masks.detach().cpu().numpy().astype(bool)
                if _m.any():
                    _ens_np = ens.detach().cpu().numpy()
                    _tgt_np = targets.detach().cpu().numpy()
                    # x TB_STD: this loop works in NORMALIZED units and
                    # multiplies by TB_STD only when reporting rmse. CRPS
                    # was handed the same normalized arrays and reported
                    # them raw, so it printed 0.26-0.40 "K" beside a bias
                    # of 17-22 K -- a proper scoring rule reading forty
                    # times better than the error it scores.
                    _c = crps_ensemble(_ens_np, _tgt_np, mask=_m) * TB_STD
                    if np.isfinite(_c):
                        crps_sum += _c * float(_m.sum())
                        crps_px += float(_m.sum())
            except Exception:
                pass    # diagnostic only; must never fail the epoch

            # How much of the correction is just a CONSTANT SHIFT?
            #
            # A near-constant offset is the easiest thing a network can
            # learn, and since 0.128 removed the baseline-shift leak the
            # residual target contains exactly that -- which is why epoch
            # 1 now starts positive where it used to start at -0.20.
            #
            # That is the fix working, but it is also a warning. A model
            # whose skill comes mostly from a spatially flat shift is
            # doing the CALIBRATION CONSTANTS' job, badly and expensively.
            # Scoring the offset alone against the full correction says
            # which it is, and therefore how much step 3 should be
            # expected to reclaim.
            try:
                _flat = drawn.mean(dim=(2, 3), keepdim=True).expand_as(drawn)
                se_offset += float((((_flat - targets) ** 2) * masks).sum())
            except Exception:
                pass

            se_model += float((((drawn - targets) ** 2) * masks).sum())
            se_zero += float(((targets ** 2) * masks).sum())
            n_px += float(masks.sum())
    if n_px == 0:
        return None
    rmse_model = (se_model / n_px) ** 0.5 * TB_STD
    rmse_zero = (se_zero / n_px) ** 0.5 * TB_STD
    skill = 1.0 - (rmse_model / rmse_zero) if rmse_zero > 0 else 0.0
    rmse_member = (se_member / n_px) ** 0.5 * TB_STD
    # Solve the two equations above for bias and sigma.
    var_sigma = max((rmse_member ** 2 - rmse_model ** 2) / (1.0 - 1.0 / members), 0.0) \
        if members > 1 else 0.0
    var_bias = max(rmse_member ** 2 - var_sigma, 0.0)
    bias_k, sigma_k = var_bias ** 0.5, var_sigma ** 0.5
    spread_k = (spread_sum / n_px) * TB_STD
    return {"rmse_model_k": rmse_model, "rmse_zero_k": rmse_zero, "skill": skill,
            "mean_spread_k": spread_k, "rmse_member_k": rmse_member,
            "crps_k": (crps_sum / crps_px) if crps_px else float("nan"),
            # Skill achievable with a spatially FLAT correction alone.
            "skill_offset_only": (1.0 - ((se_offset / n_px) ** 0.5 * TB_STD) / rmse_zero)
                                 if (n_px and se_offset and rmse_zero) else None,
            "bias_k": bias_k, "sigma_k": sigma_k,
            # Ratio of actual error to claimed spread. 1.0 is calibrated;
            # above 1.0 means the ensemble is OVERCONFIDENT.
            "overconfidence": (rmse_model / spread_k) if spread_k > 0 else float("nan"),
            # The floor an infinite ensemble would reach -- how much of
            # the residual is bias that no amount of sampling removes.
            "skill_ceiling": 1.0 - (bias_k / rmse_zero) if rmse_zero > 0 else 0.0}


def training_scalar_stats(file_paths: list) -> dict:
    """Mean/std of storm vmax and RMW across the TRAINING files.

    Recorded into the checkpoint so inference can measure how far a given
    storm sits from the distribution the model actually saw, rather than
    from the ml_constants normalization values -- those are round numbers
    chosen for normalization, not measurements of this dataset, and the
    gap between them matters precisely for the outlier storms where the
    novelty taper is supposed to act."""
    vmaxes, rmws = [], []
    for f in file_paths:
        try:
            with np.load(f, allow_pickle=True) as d:
                v = float(d["storm_vmax_kt"]); r = float(d["storm_rmw_nm"])
        except Exception:
            continue
        if np.isfinite(v):
            vmaxes.append(v)
        if np.isfinite(r):
            rmws.append(r)
    if not vmaxes:
        return {}
    out = {"vmax_mean": float(np.mean(vmaxes)), "vmax_std": float(np.std(vmaxes)) or VMAX_STD,
           "n_examples": len(vmaxes)}
    if rmws:
        out["rmw_mean"] = float(np.mean(rmws))
        out["rmw_std"] = float(np.std(rmws)) or RMW_STD
        out["n_rmw"] = len(rmws)
    return out


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss (more robust to outlier pixels than MSE for this kind of
    brightness-temperature data, which can have sharp real gradients at
    the eyewall), averaged only over pixels where the mask is 1."""
    diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def train(
    data_dir: str = DEFAULT_DATA_DIR,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    patch_size: int = PATCH_SIZE,
    batch_size: int = 2,
    epochs: int = 50,
    lr: float = 1e-4,
    base_channels: int = 32,
    device: str = None,
    num_workers: int = 4,
    compile_model: bool = False,
    early_stop_patience: int = 10,
):
    """batch_size default lowered from an earlier 8 to 2, given PATCH_SIZE
    is now 256 (up from 128) -- 4x the pixels per sample, and full
    training (forward + backward + optimizer state) uses meaningfully
    more memory than a forward-pass-only shape check confirms. This is
    still an ESTIMATE, not a measurement on your actual hardware: watch
    nvidia-smi (or just whether this errors with a CUDA out-of-memory
    message) on the first run and adjust from there -- if 2 fits with
    room to spare, there's likely headroom to go higher; if it doesn't
    fit, dropping base_channels (fewer feature maps per layer, at some
    cost to model capacity) is the other lever before shrinking
    patch_size back down.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA device found -- training on CPU will be very slow for a U-Net. "
              "If you have the RTX 3050 available, check `torch.cuda.is_available()` and your "
              "PyTorch/CUDA install (see requirements.txt's note on installing the CUDA build).")

    train_files, val_files = make_train_val_split(data_dir)
    print(f"Train examples: {len(train_files)}, Val examples: {len(val_files)}")
    if len(train_files) < 20:
        print("WARNING: very little training data so far. This is expected early on -- "
              "keep accumulating via normal app use (with the training-data-export checkbox "
              "checked) and/or ml_data_mining.py before expecting a genuinely useful model. "
              "Training will still run, but don't trust the result yet with this few examples.")

    # Fail fast on a mixed-vintage dataset, before spending a training run.
    physics_groups = check_physics_consistency(train_files + val_files)
    if physics_groups:
        print("Backbone physics vintage: "
              + ", ".join(f"{k} ({len(v)})" for k, v in sorted(physics_groups.items())))

    train_scalar_stats = training_scalar_stats(train_files)
    if train_scalar_stats:
        print(f"Training distribution: vmax {train_scalar_stats['vmax_mean']:.0f}"
              f"+/-{train_scalar_stats['vmax_std']:.0f} kt"
              + (f", RMW {train_scalar_stats.get('rmw_mean', float('nan')):.0f}"
                 f"+/-{train_scalar_stats.get('rmw_std', float('nan')):.0f} nm"
                 if 'rmw_mean' in train_scalar_stats else ""))

    train_ds = MWCorrectionDataset(train_files, patch_size=patch_size, augment=True,
                                   scalar_dropout=SCALAR_DROPOUT_P)
    val_ds = MWCorrectionDataset(val_files, patch_size=patch_size, augment=False)
    # pin_memory speeds host->GPU copies; persistent_workers avoids
    # respawning worker processes every epoch, which is especially costly
    # on Windows where workers are spawned rather than forked.
    loader_kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)

    # Architecture switch (ml_constants.CORRECTION_ARCH). Both are trained
    # by the same loop; only the model and the loss differ.
    #
    # NOTE: in_channels was hardcoded to 9 here, which went stale the
    # moment 0.98 took the input stack to 18. ml_model's DEFAULT was
    # updated but this explicit argument overrode it, so training would
    # have built a 9-channel model and then been handed 18-channel input.
    # Always take the width from the shared constant.
    if CORRECTION_ARCH == "diffusion":
        from ml_diffusion import MWResidualDiffusion
        model = MWResidualDiffusion(
            in_channels=MODEL_IN_CHANNELS, out_channels=4,
            base_channels=DIFFUSION_BASE_CHANNELS,
        ).to(device)
        print(f"Architecture: residual flow-matching diffusion "
              f"(base_channels={DIFFUSION_BASE_CHANNELS})")
    else:
        model = MWCorrectionUNet(
            in_channels=MODEL_IN_CHANNELS, out_channels=4, base_channels=base_channels).to(device)
        print("Architecture: deterministic U-Net baseline")

    # torch.compile is LAZY: it returns a wrapped model immediately and
    # only actually compiles during the first forward pass. An earlier
    # version wrapped just the torch.compile() call in try/except, which
    # therefore caught nothing -- the real failure (TritonMissing) surfaced
    # later inside the training loop and killed the run. The fallback has
    # to force a real forward pass inside the guard to be worth anything.
    #
    # Defaults to OFF because it was confirmed broken on this setup:
    # Windows needs a separate version-matched triton-windows package, and
    # the log also reported "Not enough SMs to use max_autotune_gemm mode"
    # -- a 3050 is too small for compile's better optimizations, so the
    # realistic upside here is small relative to the setup pain.
    uncompiled_model = model
    if compile_model and hasattr(torch, "compile"):
        try:
            candidate = torch.compile(model)
            # Force compilation NOW, with a throwaway batch matching the
            # real input shape, so any failure happens here rather than
            # mid-training.
            probe = torch.zeros(1, 9, patch_size, patch_size, device=device)
            with torch.no_grad():
                candidate(probe)
            model = candidate
            print("torch.compile: enabled and verified with a warmup pass")
        except Exception as e:
            model = uncompiled_model
            print(f"torch.compile unavailable ({type(e).__name__}) -- continuing "
                  "uncompiled. This is not a problem; it is a speed optimization "
                  "only, and the model trains identically without it.")

    # TF32 matmul/conv on Ampere and newer (the RTX 3050 qualifies) is a
    # free accuracy-for-speed trade that is well within tolerance for
    # brightness-temperature regression.
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # fixed input size -> good autotune
        except Exception:
            pass    # TF32/cudnn tuning is a speed hint; older torch or a
                    # non-TF32 GPU just means the default path is used
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Validation flattened around epoch 29 in a real run while training loss
    # kept falling -- the classic point where further epochs fit the
    # training storms rather than generalising. Halving the LR when
    # validation stops improving usually extracts a little more, and
    # stopping early avoids burning epochs that cannot help.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)
    epochs_since_improvement = 0
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        epoch_supervised_px = 0.0
        for inputs, targets, masks, _vmax in train_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            epoch_supervised_px += masked_pixel_count(masks)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                if CORRECTION_ARCH == "diffusion":
                    # Flow matching regresses a velocity, so there is no
                    # per-pixel prediction to score with combined_loss --
                    # the PCT term has no meaning against a velocity
                    # field. Structural quality is judged by sampling at
                    # validation time instead.
                    loss = ml_diffusion.flow_matching_loss(model, targets, inputs, masks)
                else:
                    pred = model(inputs)
                    loss, _l1, _pct = combined_loss(pred, targets, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
            train_batches += 1

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        val_l1_sum, val_pct_sum = 0.0, 0.0
        band_sums, band_counts = {}, {}
        with torch.no_grad():
            for inputs, targets, masks, vmax_b in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    if CORRECTION_ARCH == "diffusion":
                        loss = ml_diffusion.flow_matching_loss(model, targets, inputs, masks)
                        l1 = pct = loss
                        pred = None
                    else:
                        pred = model(inputs)
                        loss, l1, pct = combined_loss(pred, targets, masks)
                val_loss_sum += loss.item()
                val_l1_sum += l1.item()
                val_pct_sum += pct.item()
                val_batches += 1
                # Per-sample band attribution. Done per sample rather than
                # per batch because a batch can straddle intensity bands,
                # and averaging across that would hide exactly the
                # weak-storm degradation this reporting exists to catch.
                for i in range(inputs.shape[0]):
                    sl = slice(i, i + 1)
                    if CORRECTION_ARCH == "diffusion":
                        s_loss = ml_diffusion.flow_matching_loss(
                            model, targets[sl], inputs[sl], masks[sl])
                    else:
                        s_loss, _, _ = combined_loss(pred[sl], targets[sl], masks[sl])
                    key = band_label(float(vmax_b[i]))
                    band_sums[key] = band_sums.get(key, 0.0) + s_loss.item()
                    band_counts[key] = band_counts.get(key, 0) + 1

        train_loss = train_loss_sum / max(1, train_batches)
        val_loss = val_loss_sum / max(1, val_batches) if val_batches else float("nan")
        val_l1 = val_l1_sum / max(1, val_batches) if val_batches else float("nan")
        val_pct = val_pct_sum / max(1, val_batches) if val_batches else float("nan")
        band_str = " ".join(
            f"{k}={band_sums[k]/band_counts[k]:.3f}(n{band_counts[k]})"
            for k in sorted(band_sums)
        )
        # For diffusion, L1/PCT are copies of the flow-matching loss --
        # combined_loss has no meaning against a velocity field -- so
        # printing them implies a decomposition that does not exist.
        if CORRECTION_ARCH == "diffusion":
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} (flow-matching MSE) "
                  f"supervised_px={epoch_supervised_px:,.0f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} (L1={val_l1:.4f} PCT={val_pct:.4f}) "
                  f"supervised_px={epoch_supervised_px:,.0f}")

        # Periodic sampling check. Costs a few seconds; it is the only
        # number here that can distinguish a useful model from one that
        # learned the residual is small.
        # Computed EVERY epoch for diffusion, because it is now the
        # selection metric. Costs a few seconds; selecting on the wrong
        # number costs the whole run.
        sk = None
        if CORRECTION_ARCH == "diffusion":
            sk = sampled_skill(model, val_loader, device)
            if sk:
                verdict = ("beats doing nothing" if sk["skill"] > 0.02
                           else "NO BETTER than leaving the backbone alone"
                           if sk["skill"] > -0.02 else "WORSE than doing nothing")
                print(f"    sampled: rmse {sk['rmse_model_k']:.2f}K vs "
                      f"zero-correction {sk['rmse_zero_k']:.2f}K -> "
                      f"skill {sk['skill']:+.3f} ({verdict}); "
                      f"ensemble spread {sk['mean_spread_k']:.2f}K")
                _off = sk.get("skill_offset_only")
                if _off is not None and sk.get("skill") is not None:
                    _share = _off / sk["skill"] if sk["skill"] > 0 else float("nan")
                    print(f"      of that skill, {_share*100:.0f}% is reachable with a "
                          f"FLAT offset alone (skill {_off:+.3f}) -- the rest is "
                          f"structure the constants cannot supply")
                if np.isfinite(sk.get("crps_k", float("nan"))):
                    # CRPS is a PROPER scoring rule: it rewards sharpness
                    # only when justified, so unlike the skill number it
                    # can see the ensemble's 1.95x overconfidence.
                    print(f"      CRPS {sk['crps_k']:.2f}K "
                          f"(proper score; penalises over- and under-spread)")
                print(f"      bias {sk['bias_k']:.2f}K + spread {sk['sigma_k']:.2f}K "
                      f"-> ceiling at infinite members skill {sk['skill_ceiling']:+.3f}; "
                      f"overconfidence {sk['overconfidence']:.2f}x")

        # SELECTION METRIC. The flow-matching loss correlates only weakly
        # with usefulness -- measured r = -0.41 across a real run, where a
        # perfect selection metric would be -1.0. On that run the saved
        # checkpoint (best val_loss, epoch 36) was NOT the best model:
        # epoch 40 had a worse loss and clearly better skill (+0.139 vs
        # the +0.085 seen at the last comparable point). Select on the
        # thing being optimised for.
        if sk is not None:
            selection_metric = -sk["skill"]      # lower is better, like a loss
        else:
            selection_metric = val_loss
        if band_str:
            print(f"    val by intensity: {band_str}")

        # Hard stop on the silent-failure case. masked_loss() divides by a
        # denominator clamped to 1, so with an all-zero mask it returns
        # exactly 0.0 forever and the model receives no gradient at all --
        # which looks like "converged instantly" but is really "trained on
        # nothing." A real run burned 50 epochs and wrote a checkpoint this
        # way. Refuse to continue rather than produce a useless checkpoint.
        if epoch == 0 and epoch_supervised_px == 0:
            raise RuntimeError(
                "No supervised pixels in ANY training patch -- every target is NaN "
                "inside the storm-centered patch, so the loss is identically zero "
                "and nothing can be learned. This is a data problem, not a training "
                "one. Run diagnose_training_data.py to see which files are empty "
                "and why."
            )

        if val_batches:
            scheduler.step(val_loss)

        if val_batches and selection_metric < best_val_loss:
            best_val_loss = selection_metric
            epochs_since_improvement = 0
            ckpt_path = os.path.join(checkpoint_dir, "mw_correction_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_l1": val_l1,
                "val_pct": val_pct,
                # Per-band validation loss at the moment this checkpoint
                # was selected. Recorded so a checkpoint can be judged on
                # whether it is uniformly decent or merely good on the
                # intensity band that happened to dominate the val set --
                # a single scalar val_loss cannot distinguish those, and
                # the difference is the whole reason this stratification
                # exists.
                "val_by_band": {k: band_sums[k] / band_counts[k] for k in band_sums},
                "val_band_counts": dict(band_counts),
                "pct_loss_weight": PCT_LOSS_WEIGHT,
                "mirror_augmentation": ALLOW_MIRROR_AUGMENTATION,
                # Backbone physics this model's residuals are measured
                # against. ml_inference refuses to trust a checkpoint
                # whose value differs from the live backbone.
                "vh_physics_id": _vh_physics_id(),
                "in_channels": MODEL_IN_CHANNELS,
                "arch": CORRECTION_ARCH,
                "patch_sample_stride": PATCH_SAMPLE_STRIDE,
                # Consumed by ml_inference's novelty taper.
                "train_scalar_stats": train_scalar_stats,
                # MEASURED, so inference stops relying on a constant that
                # was right for one run. Overconfidence climbed 1.43x ->
                # 3.85x across a single training run as the ensemble
                # collapsed, so no single hardcoded figure can be correct
                # for both ends of it.
                "ensemble_overconfidence": (
                    float(sk["mean_spread_k"] and
                          (sk["rmse_model_k"] / sk["mean_spread_k"]))
                    if sk and sk.get("mean_spread_k") else None),
                "scalar_dropout_p": SCALAR_DROPOUT_P,
                "base_channels": base_channels,
                "tb_mean": TB_MEAN,
                "tb_std": TB_STD,
                "saved_at": datetime.now().isoformat(),
            }, ckpt_path)
            print(f"  saved new best checkpoint: {ckpt_path} (val_loss={val_loss:.4f}{" skill=%+.3f" % sk["skill"] if sk else ""})")
        elif val_batches:
            epochs_since_improvement += 1
            if early_stop_patience and epochs_since_improvement >= early_stop_patience:
                print(f"\nNo validation improvement for {early_stop_patience} epochs "
                      f"(best {best_val_loss:.4f}"
                      + (" = -skill" if CORRECTION_ARCH == "diffusion" else "")
                      + f"). Stopping early -- the saved "
                      "checkpoint is already the best one seen.")
                break

    print("Training complete.")


if __name__ == "__main__":
    train()
