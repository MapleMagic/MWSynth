"""
Applies a trained MWCorrectionUNet checkpoint to the parametric
algorithm's backbone output during real image generation -- the
inference-side counterpart to ml_train.py.

DESIGN, per direct guidance: the correction is applied to the BACKBONE
(the GOES-only parametric output), BEFORE the existing real-MW fusion
blend runs -- not to the final output directly. This is deliberate and
does real work: the existing fusion already weights backbone-vs-real-MW
by confidence (fresh/high-confidence real MW dominates; aging or absent
real MW lets the backbone carry more of the final result). Correcting
the backbone itself means that exact same confidence weighting
automatically makes the ML correction's influence on the FINAL output
inversely related to real-MW confidence too, with no new blending logic
needed: when a fresh real pass is available, it still dominates and the
correction's visible effect shrinks; when confidence is low or there's
no real MW at all, the (now ML-corrected) backbone carries the output,
which is exactly when the correction should matter most.

Applied on every generate by default, but the strength IS adjustable at
runtime (see DEFAULT_CORRECTION_STRENGTH and the GUI's ML correction box);
strength 0 makes it an exact no-op without deleting the checkpoint. The
docstring previously said "ALWAYS APPLIED -- not a toggle", which stopped
being true when the strength control was added. This module is
written to degrade completely gracefully to a no-op (return the
backbone completely unchanged) whenever torch isn't installed, no
checkpoint exists yet, or anything about applying the model fails --
critical given this is wired into the main generation
path. A
correction step that can silently break generation entirely would be a
regression against the whole rest of this project's reliability work.

PATCH-BASED, not whole-image: the model is trained on storm-centered
PATCH_SIZE x PATCH_SIZE patches (ml_constants.py), and running it on a
much larger full scene at inference would be feeding it a genuinely
different input distribution than anything it was trained on, even
though the architecture itself has no size-dependent layers that would
technically prevent it. Applied within a patch matching the training
size, blended back into the full backbone with a smooth taper (not a
hard edge) so the patch boundary isn't a visible seam.

HONEST CAVEAT, consistent with the rest of the ML work: this has never
been run against a real trained checkpoint (none exists yet) or with
torch actually installed in this sandbox. The graceful-fallback paths
(no torch, no checkpoint) are tested directly; the actual model-forward-pass
path can only be confirmed once a real checkpoint exists to load.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


def _flash_layer(flash_density):
    """log1p-scaled flash density, or None. Scaling lives in
    glm_lightning so training and inference cannot diverge."""
    if flash_density is None:
        return None
    try:
        import glm_lightning
        return glm_lightning.normalize_for_model(flash_density)
    except Exception:
        return None


def _downsample(patch):
    """Block-average a PATCH_FOOTPRINT_PX patch down to PATCH_SIZE.

    Averaging rather than subsampling: the footprint exists to give the
    model more context, and throwing away 3 of every 4 native pixels
    would discard exactly the fine texture that context is meant to
    summarize."""
    if PATCH_SAMPLE_STRIDE == 1:
        return patch
    k = PATCH_SAMPLE_STRIDE
    h = patch.shape[0] // k
    w = patch.shape[1] // k
    return patch[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))

from ml_constants import (normalize_channel, CORRECTION_ARCH, MODEL_IN_CHANNELS, ENSEMBLE_MEMBERS, DIFFUSION_SAMPLE_STEPS,
                          TB_MEAN, TB_STD, VMAX_MEAN, VMAX_STD, RMW_MEAN, RMW_STD, PATCH_SIZE,
                          PATCH_BLEND_TAPER_PX, PATCH_SAMPLE_STRIDE, PATCH_FOOTPRINT_PX,
                          DIFFUSION_BASE_CHANNELS,
                          MODEL_EXTRA_IR_BANDS, ELEV_MEAN, ELEV_STD,
                          # Dropped from this list by an earlier edit to it
                          # while still being USED below -- a NameError on
                          # the first diffusion inference, in code with no
                          # try block around it. Nothing caught it because
                          # no diffusion frame has been generated since.
                          ENSEMBLE_SPREAD_CALIBRATION)

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

DEFAULT_CHECKPOINT_PATH = os.path.expanduser("~/.synthetic_mw_tc/ml_checkpoints/mw_correction_best.pt")

# Scale applied to the model's predicted correction. 1.0 = use it as
# trained; 0.0 = disable entirely (equivalent to having no checkpoint,
# but without deleting the file). Values in between let the correction
# be dialled back if it proves too aggressive -- useful because a model
# trained on a small dataset with a pixel-wise L1 loss tends toward
# smooth, confident-looking corrections that may overstate what the data
# actually supports.
#
# These two module-level values are DEFAULTS ONLY, read from the
# environment once at import. apply_ml_correction() accepts explicit
# `strength` / `max_correction_k` arguments that override them per call,
# which is what the GUI's ML-strength box uses -- an env var read at
# import time cannot be changed without restarting the app, which made
# A/B comparison within a single session impossible.
DEFAULT_CORRECTION_STRENGTH = float(os.environ.get("MWSYNTH_ML_STRENGTH", "1.0"))

# Hard clamp on the per-pixel correction magnitude, in Kelvin. A
# correction larger than this is not a plausible refinement of the
# parametric backbone -- it is the model asserting something the
# training data cannot justify. Clamping bounds the worst case rather
# than trusting the model to stay reasonable everywhere.
#
# IMPORTANT CAVEAT on what this clamp does and does not bound: it is a
# per-channel bound in RAW Tb space, but the composites that are actually
# looked at are polarization-corrected combinations --
#   PCT37 = 2.181*V37 - 1.181*H37   (rendered over a 20 K window)
#   PCT89 = 1.818*V89 - 0.818*H89   (rendered over a 90 K window)
# so a correction that moves V and H in OPPOSITE directions is amplified
# by up to 3.36x at 37 GHz. A per-channel correction of under 6 K can
# therefore traverse the entire visible 37 GHz colour range. MAX_PCT_K
# below bounds that combination directly; MAX_CORRECTION_K alone does not.
DEFAULT_MAX_CORRECTION_K = float(os.environ.get("MWSYNTH_ML_MAX_K", "25.0"))

# Clamp on the correction's effect in polarization-corrected space (see
# above). Applied after the per-channel clamp by scaling both channels of
# a frequency pair together, which preserves the correction's V/H
# structure instead of clipping the two channels independently into a
# combination neither of them implied. 0 disables.
DEFAULT_MAX_PCT_K = float(os.environ.get("MWSYNTH_ML_MAX_PCT_K", "12.0"))

# PCT coefficients per frequency, matching mw_composites.py.
_PCT_COEFFS = {37: (2.181, 1.181), 89: (1.818, 0.818)}

# --- Novelty-scaled strength -----------------------------------------
#
# The model conditions on vmax and RMW as constant layers, which makes
# them the cheapest features for it to key on, and it demonstrably leans
# on them: a 50 kt storm with a genuine radar/MW core (vmax -0.75 sigma,
# RMW -1.26 sigma) had that core suppressed, apparently because the
# scalars said "weak, small" and the model trusted them over the imagery.
#
# A single global strength dial cannot fix that, because the problem is
# not that the correction is too large everywhere -- it is that the
# correction is least trustworthy exactly where the storm is unusual. On
# that frame, preserving any signature at all would have required
# strength <= 0.17, which is close enough to off to be useless on the
# frames where the correction genuinely helps.
#
# So instead: measure how far this storm's (vmax, RMW) sits from the
# training distribution, and taper strength as that distance grows. Inside
# NOVELTY_FULL_SIGMA the model is interpolating and gets full strength;
# beyond NOVELTY_ZERO_SIGMA it is extrapolating and is held at
# NOVELTY_MIN_SCALE. Linear in between.
#
# This is deliberately a crude 2-D distance, not a density model. With a
# few hundred training examples anything more sophisticated would be
# fitting noise, and the point is only to distinguish "ordinary storm"
# from "nothing like this in training".
NOVELTY_SCALING_ENABLED = os.environ.get("MWSYNTH_ML_NOVELTY_SCALING", "1") != "0"
NOVELTY_FULL_SIGMA = 1.0
NOVELTY_ZERO_SIGMA = 2.5
NOVELTY_MIN_SCALE = 0.25


def novelty_sigma(vmax_kt: float, rmw_nm: float, stats: Optional[dict] = None) -> float:
    """Distance of this storm's (vmax, RMW) from the training
    distribution, in sigma. `stats` may carry per-field means/stds
    recorded from the actual training set at checkpoint time; falls back
    to the ml_constants normalization values, which approximate it."""
    vm = (stats or {}).get("vmax_mean", VMAX_MEAN)
    vs = (stats or {}).get("vmax_std", VMAX_STD) or VMAX_STD
    rm = (stats or {}).get("rmw_mean", RMW_MEAN)
    rs = (stats or {}).get("rmw_std", RMW_STD) or RMW_STD
    dv = (float(vmax_kt) - vm) / vs
    dr = (float(rmw_nm) - rm) / rs
    return float(np.sqrt(dv * dv + dr * dr))


def novelty_scale(sigma: float) -> float:
    """Strength multiplier for a given novelty distance."""
    if sigma <= NOVELTY_FULL_SIGMA:
        return 1.0
    if sigma >= NOVELTY_ZERO_SIGMA:
        return NOVELTY_MIN_SCALE
    span = NOVELTY_ZERO_SIGMA - NOVELTY_FULL_SIGMA
    frac = (sigma - NOVELTY_FULL_SIGMA) / span
    return 1.0 - frac * (1.0 - NOVELTY_MIN_SCALE)


_model_cache = {"key": None, "model": None, "device": None}

# Training-set (vmax, RMW) statistics read from the checkpoint, when it
# records them. None means fall back to the ml_constants normalization
# values inside novelty_sigma().
_ckpt_train_stats = None

# V/H physics identifier recorded in the checkpoint at training time, or
# None for checkpoints predating it. Compared against the live backbone's
# synthetic_algorithm.VH_PHYSICS_ID -- see _check_physics_match().
_ckpt_vh_physics_id = None

# Architecture recorded in the checkpoint ("diffusion" | "unet"). Defaults
# to "unet" for checkpoints predating the field.
_ckpt_arch = "unet"

# Overconfidence measured during the run that produced the loaded
# weights, or None for checkpoints predating it being recorded.
_ckpt_overconfidence = None


def _spread_calibration() -> float:
    """How much to widen the raw ensemble spread.

    Prefers the MEASURED value from the run that produced these weights,
    falling back to the constant only for older checkpoints.
    ENSEMBLE_SPREAD_CALIBRATION was fitted to one run; the real figure
    moved 1.43x -> 3.85x WITHIN a single run as the ensemble collapsed,
    so one number cannot describe both ends of it, let alone two runs.
    """
    if _ckpt_overconfidence and _ckpt_overconfidence > 0:
        return float(np.clip(_ckpt_overconfidence, 1.0, 6.0))
    return float(ENSEMBLE_SPREAD_CALIBRATION)


def _check_physics_match():
    """Return a warning string if the checkpoint was trained against a
    different backbone physics than the one now in use, else None.

    A correction model predicts a residual against a specific backbone.
    Change the backbone's radiative mapping and every residual it learned
    is measured from the wrong place. This is invisible from inside the
    model -- it will happily emit confident corrections -- so the check
    has to be external.
    """
    try:
        from synthetic_algorithm import VH_PHYSICS_ID as live_id
    except Exception:
        return None
    if _ckpt_vh_physics_id == live_id:
        return None
    was = _ckpt_vh_physics_id or "pre-0.95 (unversioned)"
    return (f"checkpoint was trained against backbone physics '{was}' but the live "
            f"backbone is '{live_id}' -- the residuals it learned are measured from a "
            f"backbone that no longer exists. RETRAIN before trusting this; set ML "
            f"strength to 0 in the meantime.")


def _load_model_cached(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
    """Lazily load and cache a trained checkpoint -- loading a PyTorch
    model from disk has real overhead, not worth repeating on every
    single generate_synthetic_mw call. Cached by path, so switching
    checkpoints (e.g. a newly retrained one) is picked up automatically
    without needing to restart the app -- the cache just gets replaced
    if a different path is requested.

    Returns None (not an exception) if torch isn't installed, the
    checkpoint file doesn't exist, or loading fails for any reason --
    every caller in this module treats None as "no correction available
    right now," not an error.

    The cache key includes the file's mtime and size, not just its path.
    Keying on path alone was a real bug: retraining writes a new model to
    the SAME default path (mw_correction_best.pt), so a long-running GUI
    session would keep serving the stale in-memory model indefinitely
    while appearing to use the new one. This docstring previously claimed
    retrained checkpoints were picked up automatically; now they actually
    are.
    """
    try:
        import torch
        from ml_model import MWCorrectionUNet
        from ml_diffusion import MWResidualDiffusion
    except ImportError:
        return None, None

    if not os.path.exists(checkpoint_path):
        return None, None

    try:
        st = os.stat(checkpoint_path)
        cache_key = (checkpoint_path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None, None

    if _model_cache["key"] == cache_key and _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["device"]

    global _ckpt_train_stats, _ckpt_vh_physics_id, _ckpt_arch
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Default matches what training actually uses. 32 was left over
        # from an earlier architecture, so a checkpoint missing the key
        # would have built the wrong width and failed on the state dict
        # with a shape error rather than anything readable.
        base_channels = checkpoint.get("base_channels", DIFFUSION_BASE_CHANNELS)

        # Read the architecture BEFORE constructing anything. Building the
        # wrong class and then loading weights into it fails as a wall of
        # shape errors that says nothing about the real cause, and reading
        # a stale module-level value here would pick whatever the LAST
        # checkpoint was -- so a session that loaded a U-Net first would
        # then mis-build a diffusion checkpoint.
        arch = (checkpoint.get("arch") if isinstance(checkpoint, dict) else None) or "unet"

        if arch == "diffusion":
            model = MWResidualDiffusion(
                in_channels=MODEL_IN_CHANNELS, out_channels=4, base_channels=base_channels
            ).to(device)
        else:
            model = MWCorrectionUNet(
                in_channels=MODEL_IN_CHANNELS, out_channels=4, base_channels=base_channels
            ).to(device)

        state = checkpoint["model_state_dict"]
        # If the checkpoint was saved from a torch.compile()'d model, every
        # key is prefixed with "_orig_mod." and load_state_dict would fail
        # with a wall of unexpected-key errors. Strip it so a checkpoint
        # trained with COMPILE_MODEL=True still loads here, where the model
        # is built uncompiled.
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
    except Exception:
        return None, None

    _ckpt_arch = arch
    # Verify the CONTRACT the checkpoint records, before loading weights.
    #
    # Training saves in_channels and patch_sample_stride; inference read
    # neither, building the model from whatever the current constants say.
    # A channel-count change then surfaced as a torch shape error deep in
    # load_state_dict, and a STRIDE change was worse -- the shapes match,
    # so it loads cleanly and silently applies the correction over the
    # wrong ground area. The layout narrowed 19 -> 15 in 0.142, so this is
    # live right now.
    if isinstance(checkpoint, dict):
        _ck_in = checkpoint.get("in_channels")
        if _ck_in is not None and int(_ck_in) != int(MODEL_IN_CHANNELS):
            raise RuntimeError(
                f"checkpoint was trained with {_ck_in} input channels but this "
                f"build uses {MODEL_IN_CHANNELS}. Retrain, or restore the "
                f"channel layout it was trained against.")
        _ck_stride = checkpoint.get("patch_sample_stride")
        if _ck_stride is not None and int(_ck_stride) != int(PATCH_SAMPLE_STRIDE):
            raise RuntimeError(
                f"checkpoint was trained at patch stride {_ck_stride} but this "
                f"build uses {PATCH_SAMPLE_STRIDE}. The shapes still match, so "
                f"this would load cleanly and correct the wrong ground area.")

    _ckpt_train_stats = checkpoint.get("train_scalar_stats") if isinstance(checkpoint, dict) else None
    _ckpt_vh_physics_id = checkpoint.get("vh_physics_id") if isinstance(checkpoint, dict) else None
    # Prefer the value MEASURED during the run that produced these
    # weights. ENSEMBLE_SPREAD_CALIBRATION was fitted to one training run
    # and is a constant; the real figure moved 1.43x -> 3.85x within a
    # single run as the ensemble collapsed, so the checkpoint's own
    # measurement is the only one that describes these weights.
    global _ckpt_overconfidence
    _ckpt_overconfidence = (checkpoint.get("ensemble_overconfidence")
                            if isinstance(checkpoint, dict) else None)

    _model_cache["key"] = cache_key
    _model_cache["model"] = model
    _model_cache["device"] = device
    return model, device


def _make_taper(patch_size: int, taper_px: int) -> np.ndarray:
    """2D taper: 1.0 in the patch interior, smoothly ramping to 0.0 over
    the last `taper_px` pixels toward each edge -- used to blend a
    corrected patch back into the full backbone without a visible seam
    at the patch boundary. Separable (built as an outer product of two
    1D ramps), which is both simpler and correct for a rectangular
    patch with the same taper width on all four sides.
    """
    taper_px = max(1, min(taper_px, patch_size // 2))
    ramp = np.ones(patch_size, dtype=np.float64)
    for i in range(taper_px):
        val = (i + 1) / (taper_px + 1)
        ramp[i] = val
        ramp[patch_size - 1 - i] = val
    return np.outer(ramp, ramp)


def _patch_bounds(center_rc: tuple, patch_size: int, array_shape: tuple) -> tuple:
    """Compute the (read_slice, write_slice, patch_local_slice) needed to
    extract a patch_size x patch_size region centered on center_rc from
    an array of array_shape, clipped to the array's actual bounds (a
    storm near the edge of its GOES crop won't have a full patch
    available on all sides). Returns:
        row0, row1, col0, col1: the CLIPPED region in the full array
        local_row0, local_col0: where that clipped region starts within
            an otherwise-full patch_size x patch_size patch (nonzero
            only when the storm is near an edge)
    """
    h, w = array_shape
    r, c = center_rc
    half = patch_size // 2
    row0, row1 = r - half, r - half + patch_size
    col0, col1 = c - half, c - half + patch_size

    local_row0 = max(0, -row0)
    local_col0 = max(0, -col0)
    row0_clipped, row1_clipped = max(0, row0), min(h, row1)
    col0_clipped, col1_clipped = max(0, col0), min(w, col1)

    return row0_clipped, row1_clipped, col0_clipped, col1_clipped, local_row0, local_col0


def apply_ml_correction(
    ir_tb: np.ndarray,
    wv_tb: np.ndarray,
    swir_tb: np.ndarray,
    v37_backbone: np.ndarray,
    h37_backbone: np.ndarray,
    v89_backbone: np.ndarray,
    h89_backbone: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    storm_lat: float,
    storm_lon: float,
    storm_vmax_kt: float,
    storm_rmw_nm: Optional[float],
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    progress_callback=None,
    strength: Optional[float] = None,
    max_correction_k: Optional[float] = None,
    max_pct_k: Optional[float] = None,
    fallback_rmw_nm: Optional[float] = None,
    extra_ir: Optional[dict] = None,
    land_fraction: Optional[np.ndarray] = None,
    elevation_m: Optional[np.ndarray] = None,
    flash_density: Optional[np.ndarray] = None,
    stats_out: Optional[dict] = None,
) -> tuple:
    """Apply a trained correction model to the backbone V/H fields,
    within a storm-centered patch, blended back with a smooth taper.
    Returns (v37, h37, v89, h89) -- the corrected fields if a checkpoint
    was available and applying it succeeded, or the EXACT SAME arrays
    passed in (unchanged) if not. Never raises: any failure along the
    way (no torch, no checkpoint, shape mismatch, a bad forward pass)
    falls back to returning the original backbone untouched, logged via
    progress_callback if given but not treated as fatal to generation.

    strength / max_correction_k / max_pct_k: per-call overrides for the
        module defaults. Passed explicitly by the caller (the GUI's ML
        strength box) so these can be changed within a running session.
    fallback_rmw_nm: RMW to use when best-track `storm_rmw_nm` is None.
        The caller should pass the SAME estimate the parametric backbone
        was built with (synthetic_algorithm's Willoughby fallback,
        converted to nm). Previously this fell back to RMW_MEAN, meaning
        that on any frame without a best-track RMW the model was told
        "average storm" while the backbone underneath it had been shaped
        by a completely different radius -- a silent inconsistency
        between the two halves of the same frame.
    stats_out: optional dict, populated in place with correction
        diagnostics (see below). Kept as an out-parameter rather than an
        extra return value so existing 4-tuple callers keep working.
    """
    strength = DEFAULT_CORRECTION_STRENGTH if strength is None else float(strength)
    max_correction_k = DEFAULT_MAX_CORRECTION_K if max_correction_k is None else float(max_correction_k)
    max_pct_k = DEFAULT_MAX_PCT_K if max_pct_k is None else float(max_pct_k)

    if strength == 0.0:
        if stats_out is not None:
            stats_out.update({"applied": False, "reason": "strength 0"})
        return v37_backbone, h37_backbone, v89_backbone, h89_backbone

    model, device = _load_model_cached(checkpoint_path)
    if model is None:
        return v37_backbone, h37_backbone, v89_backbone, h89_backbone

    try:
        import torch

        dist2 = (lat - storm_lat) ** 2 + _wrap_lon_delta(lon - storm_lon) ** 2
        center_rc = np.unravel_index(np.argmin(dist2), dist2.shape)

        row0, row1, col0, col1, local_row0, local_col0 = _patch_bounds(center_rc, PATCH_FOOTPRINT_PX, ir_tb.shape)
        n_rows, n_cols = row1 - row0, col1 - col0
        if n_rows <= 0 or n_cols <= 0:
            return v37_backbone, h37_backbone, v89_backbone, h89_backbone

        def _norm_tb(full_arr, channel="ir"):
            # Pad with TB_MEAN, not zeros. A storm near the edge of its
            # GOES crop doesn't have a full patch available, and the old
            # zero-fill normalized to (0 - TB_MEAN)/TB_STD, i.e. about
            # -6 sigma -- an input value the model never saw once during
            # training, fed to it as a hard-edged block adjacent to the
            # real data. Padding with the mean normalizes to exactly 0,
            # which is at least in-distribution and edge-neutral.
            patch = np.full((PATCH_FOOTPRINT_PX, PATCH_FOOTPRINT_PX), TB_MEAN, dtype=np.float32)
            patch[local_row0:local_row0 + n_rows, local_col0:local_col0 + n_cols] = full_arr[row0:row1, col0:col1]
            # Per-channel via the shared implementation in ml_constants,
            # so training and inference cannot drift apart.
            return normalize_channel(_downsample(patch), channel)

        def _norm_plain(full_arr, mean, std, fill):
            """Same extraction for non-Tb conditioning fields."""
            patch = np.full((PATCH_FOOTPRINT_PX, PATCH_FOOTPRINT_PX), fill, dtype=np.float32)
            if full_arr is not None:
                patch[local_row0:local_row0 + n_rows, local_col0:local_col0 + n_cols] = np.asarray(
                    full_arr, dtype=np.float32)[row0:row1, col0:col1]
            return (_downsample(patch) - mean) / std

        if storm_rmw_nm is not None and np.isfinite(storm_rmw_nm):
            rmw_value = storm_rmw_nm
        elif fallback_rmw_nm is not None and np.isfinite(fallback_rmw_nm):
            rmw_value = fallback_rmw_nm
        else:
            rmw_value = RMW_MEAN
        vmax_layer = np.full((PATCH_SIZE, PATCH_SIZE), (storm_vmax_kt - VMAX_MEAN) / VMAX_STD, dtype=np.float32)
        rmw_layer = np.full((PATCH_SIZE, PATCH_SIZE), (rmw_value - RMW_MEAN) / RMW_STD, dtype=np.float32)

        # Fixed channel order -- see ml_constants.INPUT_CHANNEL_LAYOUT.
        # Missing extra bands become a neutral TB_MEAN plane rather than
        # being dropped, so the channel count is constant and a checkpoint
        # stays loadable regardless of which bands a frame actually has.
        extra = extra_ir or {}
        neutral = np.full((PATCH_SIZE, PATCH_SIZE), 0.0, dtype=np.float32)
        extra_layers = [
            _norm_tb(extra[b], f"ir_band{b}") if extra.get(b) is not None else neutral
            for b in MODEL_EXTRA_IR_BANDS
        ]
        input_stack = np.stack([
            _norm_tb(ir_tb, "ir"), _norm_tb(wv_tb, "wv"), _norm_tb(swir_tb, "swir"),
            *extra_layers,
            _norm_tb(v37_backbone, "backbone_v37"), _norm_tb(h37_backbone, "backbone_h37"), _norm_tb(v89_backbone, "backbone_v89"), _norm_tb(h89_backbone, "backbone_h89"),
            vmax_layer, rmw_layer,
            _norm_plain(land_fraction, 0.0, 1.0, 0.0),
            _norm_plain(elevation_m, ELEV_MEAN, ELEV_STD, 0.0),
            # Lightning. Absent GLM gives all zeros, which is also what a
            # genuinely lightning-free scene gives -- deliberately, so the
            # input distribution does not shift with product availability.
            _norm_plain(_flash_layer(flash_density), 0.0, 1.0, 0.0),
        ], axis=0)
        input_stack = np.nan_to_num(input_stack, nan=0.0)

        input_tensor = torch.from_numpy(input_stack.astype(np.float32)).unsqueeze(0).to(device)
        ensemble_spread_k = None
        with torch.no_grad():
            if _ckpt_arch == "diffusion":
                import ml_diffusion
                # Draw an ensemble and take its mean as the correction.
                #
                # The MEAN is deliberate and worth being explicit about:
                # Li et al. show the ensemble mean scores better on
                # pixel-wise metrics but loses high-frequency detail
                # relative to individual members. Rendering a single
                # member would look sharper and be, frame to frame,
                # arbitrary -- two runs of the same frame would disagree
                # on where a rainband sits. For a monitoring tool that
                # instability is worse than smoothness. The detail is not
                # discarded, it is reported: the member SPREAD becomes the
                # uncertainty field, which is information the
                # deterministic model could not produce at all.
                members = ml_diffusion.sample_residual(
                    model, input_tensor, out_channels=4,
                    steps=DIFFUSION_SAMPLE_STEPS, ensemble=ENSEMBLE_MEMBERS,
                )
                arr = members[:, 0].cpu().numpy()          # (E, 4, P, P)
                correction = arr.mean(axis=0)
                # Calibrated before it leaves this function. The raw
                # ensemble spread was measured to understate the true
                # error by about 1.7x (see ENSEMBLE_SPREAD_CALIBRATION),
                # and a confidence field that is confidently too small is
                # worse than no confidence field at all.
                ensemble_spread_k = (arr.std(axis=0, ddof=1 if arr.shape[0] > 1 else 0)
                                     * TB_STD * _spread_calibration())
            else:
                correction = model(input_tensor)[0].cpu().numpy()  # (4, P, P), TB_STD-scaled

        correction_k = correction * TB_STD  # de-normalize back to real Kelvin correction magnitude

        # Clamp FIRST, then scale by strength. The old order (scale, then
        # clamp) made `strength` a non-linear dial wherever the clamp was
        # binding: a raw 60 K prediction became 30 K at strength 0.5 and
        # was then clipped back to 25 K -- the same value strength 1.0
        # produced. Half strength gave identical output to full strength
        # exactly in the core, which is where the clamp binds and where
        # the dial matters most. Clamping first makes strength a true
        # linear scale on a bounded correction.
        raw_peak_k = float(np.nanmax(np.abs(correction_k))) if correction_k.size else 0.0
        if max_correction_k > 0:
            correction_k = np.clip(correction_k, -max_correction_k, max_correction_k)
        n_at_channel_clamp = (
            int(np.sum(np.abs(correction_k) >= max_correction_k - 1e-6)) if max_correction_k > 0 else 0
        )

        # Bound the correction in polarization-corrected space, where the
        # composites actually live. Scale the V/H pair of a frequency
        # together by a single per-pixel factor so the correction's
        # polarization structure survives -- clipping V and H
        # independently would synthesize a V-H difference the model never
        # predicted.
        n_at_pct_clamp = 0
        if max_pct_k > 0:
            for idx_v, idx_h, freq in ((0, 1, 37), (2, 3, 89)):
                a, b = _PCT_COEFFS[freq]
                d_pct = a * correction_k[idx_v] - b * correction_k[idx_h]
                over = np.abs(d_pct) > max_pct_k
                n_at_pct_clamp += int(np.sum(over))
                if np.any(over):
                    scale = np.ones_like(d_pct)
                    denom = np.abs(d_pct)
                    scale[over] = max_pct_k / np.maximum(denom[over], 1e-6)
                    correction_k[idx_v] = correction_k[idx_v] * scale
                    correction_k[idx_h] = correction_k[idx_h] * scale

        # Novelty taper, applied on top of the user's strength. Reported
        # separately in the stats so it is never mistaken for the user's
        # own setting -- a correction quietly scaled to 0.25 while the GUI
        # reads 1.00 would be genuinely confusing.
        nov_sigma = novelty_sigma(storm_vmax_kt, rmw_value, _ckpt_train_stats)
        nov_scale = novelty_scale(nov_sigma) if NOVELTY_SCALING_ENABLED else 1.0
        effective_strength = strength * nov_scale

        if effective_strength != 1.0:
            correction_k = correction_k * effective_strength

        # The model works at PATCH_SIZE; the field it is blended into is
        # PATCH_FOOTPRINT_PX. Upsample by simple pixel repetition, then
        # smooth just enough that the stride does not show up as blocking.
        if PATCH_SAMPLE_STRIDE > 1:
            correction_k = np.repeat(np.repeat(correction_k, PATCH_SAMPLE_STRIDE, axis=1),
                                     PATCH_SAMPLE_STRIDE, axis=2)
            correction_k = np.stack([
                gaussian_filter(c, sigma=PATCH_SAMPLE_STRIDE * 0.6) for c in correction_k
            ], axis=0)
            # The uncertainty field takes the SAME path as the correction
            # it describes. It was computed per pixel and then reduced to
            # a mean and a max, so the one thing it was good for -- saying
            # WHERE the model is guessing -- was thrown away, leaving two
            # scalars that say only how much on average.
            if ensemble_spread_k is not None:
                ensemble_spread_k = np.repeat(
                    np.repeat(ensemble_spread_k, PATCH_SAMPLE_STRIDE, axis=1),
                    PATCH_SAMPLE_STRIDE, axis=2)
                ensemble_spread_k = np.stack([
                    gaussian_filter(c, sigma=PATCH_SAMPLE_STRIDE * 0.6)
                    for c in ensemble_spread_k
                ], axis=0)

        taper = _make_taper(PATCH_FOOTPRINT_PX, PATCH_BLEND_TAPER_PX * PATCH_SAMPLE_STRIDE)

        def _blend_spread(spread_stack):
            """Place the per-pixel spread on the full grid.

            Outside the patch there is no ensemble and therefore no
            uncertainty estimate -- NaN rather than zero, because zero
            would read as "certain" in exactly the region the model never
            looked at.
            """
            out = np.full(v37_backbone.shape, np.nan, dtype=np.float32)
            patch_taper = taper[local_row0:local_row0 + n_rows,
                                local_col0:local_col0 + n_cols]
            # Mean across the four output channels: one uncertainty map,
            # not four, since the panel it feeds shows a single field.
            mean_spread = np.nanmean(spread_stack, axis=0)
            patch = mean_spread[local_row0:local_row0 + n_rows,
                                local_col0:local_col0 + n_cols]
            # Weighted by the same taper, so the edge where the correction
            # fades out does not read as confident.
            out[row0:row1, col0:col1] = np.where(patch_taper > 0.05, patch, np.nan)
            return out

        def _blend(full_arr, correction_channel):
            result = full_arr.copy()
            patch_taper = taper[local_row0:local_row0 + n_rows, local_col0:local_col0 + n_cols]
            patch_correction = correction_channel[local_row0:local_row0 + n_rows, local_col0:local_col0 + n_cols]
            region = result[row0:row1, col0:col1]
            result[row0:row1, col0:col1] = region + patch_correction * patch_taper
            return result

        v37_out = _blend(v37_backbone, correction_k[0])
        h37_out = _blend(h37_backbone, correction_k[1])
        v89_out = _blend(v89_backbone, correction_k[2])
        h89_out = _blend(h89_backbone, correction_k[3])

        # Diagnostics are measured on what was ACTUALLY added to the
        # backbone (post-taper, over the region that really received a
        # correction), not on the model's raw output over the whole
        # padded patch. The previous version reported the latter, which
        # both included the zero-padded margin and ignored the taper, so
        # the numbers didn't describe the change the image received.
        eff_taper = taper[local_row0:local_row0 + n_rows, local_col0:local_col0 + n_cols]
        eff = correction_k[:, local_row0:local_row0 + n_rows, local_col0:local_col0 + n_cols] * eff_taper
        abs_eff = np.abs(eff)
        peak = float(np.nanmax(abs_eff)) if abs_eff.size else 0.0
        mean = float(np.nanmean(abs_eff)) if abs_eff.size else 0.0

        # Peak excursion in the space the composites are rendered in, as
        # a percentage of each colour window. This is the number that
        # actually predicts whether the correction is visible: a modest
        # per-channel delta can still saturate the 20 K 37 GHz window.
        pct_peak = {}
        for idx_v, idx_h, freq, window in ((0, 1, 37, 20.0), (2, 3, 89, 90.0)):
            a, b = _PCT_COEFFS[freq]
            d_pct = a * eff[idx_v] - b * eff[idx_h]
            pk = float(np.nanmax(np.abs(d_pct))) if d_pct.size else 0.0
            pct_peak[freq] = (pk, 100.0 * pk / window)

        n_px = max(1, int(eff[0].size))
        stats = {
            "applied": True,
            "strength": strength,
            "novelty_sigma": nov_sigma,
            "novelty_scale": nov_scale,
            "effective_strength": effective_strength,
            "mean_abs_k": mean,
            "peak_abs_k": peak,
            "raw_peak_abs_k": raw_peak_k,
            "frac_at_channel_clamp": n_at_channel_clamp / float(max(1, correction_k.size)),
            "frac_at_pct_clamp": n_at_pct_clamp / float(2 * n_px),
            "pct_peak_37_k": pct_peak[37][0],
            "pct_peak_37_pctwindow": pct_peak[37][1],
            "pct_peak_89_k": pct_peak[89][0],
            "pct_peak_89_pctwindow": pct_peak[89][1],
            "rmw_nm_used": float(rmw_value),
            "rmw_source": ("best-track" if (storm_rmw_nm is not None and np.isfinite(storm_rmw_nm))
                           else ("backbone-fallback" if fallback_rmw_nm is not None else "RMW_MEAN")),
            "checkpoint": os.path.basename(checkpoint_path),
            "physics_mismatch": _check_physics_match(),
            "arch": _ckpt_arch,
            "ensemble_members": ENSEMBLE_MEMBERS if _ckpt_arch == "diffusion" else 1,
            "ensemble_spread_mean_k": (float(np.nanmean(ensemble_spread_k))
                                       if ensemble_spread_k is not None else None),
            "ensemble_spread_max_k": (float(np.nanmax(ensemble_spread_k))
                                      if ensemble_spread_k is not None else None),
            # The full per-pixel field, blended onto the output grid the
            # same way the correction is, so it aligns with what it
            # describes. Already calibrated by
            # ENSEMBLE_SPREAD_CALIBRATION.
            "ensemble_spread_field_k": (
                _blend_spread(ensemble_spread_k)
                if ensemble_spread_k is not None else None),
        }
        if stats_out is not None:
            stats_out.update(stats)

        if progress_callback:
            warn = ""
            mismatch = stats.get("physics_mismatch")
            if mismatch:
                warn = f"  [WARNING: {mismatch}]"
            elif stats["frac_at_channel_clamp"] > 0.10:
                warn = (f"  [WARNING: {stats['frac_at_channel_clamp']*100:.0f}% of the patch is pinned at "
                        f"the {max_correction_k:g}K clamp -- the model is asserting more than the "
                        f"clamp allows; consider lowering ML strength]")
            progress_callback(
                f"ML correction applied (strength {strength:g}"
                + (f" x novelty {nov_scale:.2f} [{nov_sigma:.1f} sigma from training] "
                   f"= {effective_strength:.2f}" if nov_scale < 1.0 else "") + ", "
                f"mean |delta| {mean:.1f}K, peak |delta| {peak:.1f}K "
                f"[raw {raw_peak_k:.1f}K], clamp {max_correction_k:g}K; "
                f"PCT peak 37GHz {pct_peak[37][0]:.1f}K ({pct_peak[37][1]:.0f}% of window), "
                f"89GHz {pct_peak[89][0]:.1f}K ({pct_peak[89][1]:.0f}%); "
                f"RMW {rmw_value:.1f}nm ({stats['rmw_source']}); "
                f"checkpoint {os.path.basename(checkpoint_path)}).{warn}"
            )

        return v37_out, h37_out, v89_out, h89_out

    except Exception as e:
        if stats_out is not None:
            stats_out.update({"applied": False, "reason": f"{type(e).__name__}: {e}"})
        if progress_callback:
            progress_callback(f"ML correction skipped this frame ({type(e).__name__}: {e}) -- using uncorrected backbone.")
        return v37_backbone, h37_backbone, v89_backbone, h89_backbone
