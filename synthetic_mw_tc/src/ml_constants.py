"""
Normalization constants shared between ml_train.py (training) and
ml_inference.py (applying a trained model during real generation).

Pulled into their own module specifically so training and inference
can't drift out of sync -- if these lived separately in each file and
someone updated one without the other, a trained checkpoint would
silently be fed differently-scaled inputs at inference time than it was
trained on, which is exactly the kind of bug that doesn't throw an
error, just quietly produces wrong corrections.

Same "reasonable default, not yet confirmed against the real
accumulated dataset's actual distribution" caveat as when these first
appeared in ml_train.py -- worth revisiting (e.g. computing real
per-channel mean/std) once there's enough real training data to check
against.
"""

# Brightness temperature channels (IR/WV/SWIR/backbone V-H) -- roughly
# 150-320K for the channels this project uses.
TB_MEAN = 260.0
TB_STD = 40.0

# --- Per-channel normalization (0.118) -------------------------------
# One shared (260, 40) for every brightness-temperature channel left the
# inputs badly conditioned, because the channels do not share a range:
#
#     channel              normalized span with (260, 40)
#     IR band 13           -1.75 ..  1.00
#     37V                  -0.50 ..  0.62      <- ~1 sigma of range used
#     37H                  -2.75 ..  0.50
#     WV bands 8/9/10      -1.50 ..  0.00      <- never positive
#
# The water-vapour channels sit entirely on one side of zero and 37V
# barely varies, so the network had to spend capacity undoing a constant
# offset per channel before it could learn anything. That is not a
# crash-class bug; it is a quiet accuracy tax on every example.
#
# These are physically-motivated centres and spreads per channel, still
# defaults rather than measured statistics -- but per-channel defaults are
# strictly better conditioned than one global pair. Computing true
# statistics from the mined dataset and persisting them in the checkpoint
# (as train_scalar_stats already does for vmax/RMW) is the proper fix and
# the obvious follow-up.
CHANNEL_NORM = {
    # Window IR: wide range, cold tops to warm surface.
    "ir": (260.0, 40.0),
    "ir_band13": (260.0, 40.0),
    "ir_band14": (258.0, 40.0),
    "ir_band15": (255.0, 40.0),
    "ir_band11": (255.0, 40.0),
    "ir_band12": (245.0, 35.0),
    # Water vapour: cold and narrow, never sees the surface.
    "wv": (235.0, 20.0),
    "ir_band8": (232.0, 15.0),
    "ir_band9": (238.0, 18.0),
    "ir_band10": (242.0, 20.0),
    "ir_band16": (248.0, 25.0),
    # Shortwave IR.
    "swir": (265.0, 35.0),
    # Microwave backbone and targets, per frequency and polarization.
    "backbone_v37": (262.0, 15.0),
    "backbone_h37": (215.0, 35.0),
    "backbone_v89": (265.0, 30.0),
    "backbone_h89": (255.0, 40.0),
}

# Fallback for any channel not listed, so an added channel degrades to
# the old behaviour rather than raising.
CHANNEL_NORM_DEFAULT = (TB_MEAN, TB_STD)


def normalize_channel(arr, channel: str):
    """Normalize one named channel. THE single implementation.

    Both ml_train and ml_inference call this. They previously each had
    their own arithmetic -- identical, but only by coincidence -- and a
    normalization mismatch between training and inference is the archetype
    of a bug that throws no error and quietly produces wrong output.
    """
    import numpy as _np
    mean, std = CHANNEL_NORM.get(channel, CHANNEL_NORM_DEFAULT)
    return (_np.asarray(arr, dtype=_np.float32) - mean) / std


def denormalize_channel(arr, channel: str):
    """Inverse of normalize_channel."""
    import numpy as _np
    mean, std = CHANNEL_NORM.get(channel, CHANNEL_NORM_DEFAULT)
    return _np.asarray(arr, dtype=_np.float32) * std + mean

# Storm intensity in knots -- typically ~20-180kt for the systems this
# project cares about.
VMAX_MEAN, VMAX_STD = 80.0, 40.0

# RMW in nautical miles -- typically ~10-100nm.
RMW_MEAN, RMW_STD = 30.0, 20.0

# Patch size (pixels) the correction model is trained/applied on --
# storm-centered, not the full scene. Must match between training and
# inference: a model trained on one patch size hasn't been shown
# anything about how its own features behave at a different input size,
# even though the architecture itself (no size-dependent layers) would
# technically accept one.
#
# Set to 256 (up from an initial 128) per direct feedback after
# reviewing real output: at typical GOES resolution (~2km/pixel), 256px
# covers roughly 512x512km -- comfortably covers even large, mature systems
# (ROCI 450-500km isn't unusual) with margin on all sides, where 128px
# (~256x256km) was cutting off exactly the outer spiral banding that
# makes a GOES-heavy frame look visibly different from a real MW pass.
PATCH_SIZE = 256

# --- Wider footprint at coarser sampling (0.98) ----------------------
# The model still sees PATCH_SIZE x PATCH_SIZE, but each model pixel now
# spans PATCH_SAMPLE_STRIDE native GOES pixels, so the patch COVERS
# PATCH_FOOTPRINT_PX native pixels -- about 1024 km at ~2 km GOES
# resolution instead of ~512 km.
#
# WHY: on Lowell the ML patch boundary was plainly visible partway across
# the image. Differencing the ML-on and ML-off panels showed the change
# confined to a hard-edged block of about 613 km, so the inner region was
# treated and the outer was not, on a storm whose field spanned well over
# 1000 km. Li et al. (2025) use 256 x 256 at 4 km (1024 km) for the same
# task, which is the same conclusion reached from the other direction.
#
# Trading resolution for coverage is the right way round here: the
# correction exists to fix large-scale structure the parametric backbone
# gets wrong, and the backbone is already smoothed at roughly this scale.
PATCH_SAMPLE_STRIDE = 2
PATCH_FOOTPRINT_PX = PATCH_SIZE * PATCH_SAMPLE_STRIDE

# --- Input channel layout (0.98) -------------------------------------
# Fixed and shared so training and inference cannot drift. Bands that a
# given frame doesn't have are filled with the neutral normalized value
# (0.0, i.e. TB_MEAN) rather than dropped, so the channel COUNT never
# varies -- a checkpoint stays loadable whether or not every band was
# fetched.
#
# Li et al. (2025) found the largest single improvement in their whole
# ablation came from moving beyond one IR band: their band-13-only
# experiment was worst on every metric, and adding bands 8-16 fixed it.
# These are the ABI bands they used that this project wasn't already
# ingesting (13, 9 and 7 are the existing ir/wv/swir channels).
# ORDERED BY MEASURED IMPORTANCE, most useful first. Li et al. (2026)
# published gradient saliency per IR channel (their Figure 5), which
# turns "add more bands" into a ranked shopping list:
#
#   ch11 (8.5 um, cloud-top phase)   most critical single input
#   ch10 (7.4 um, low-level WV)      dominant predictor, feeds convection
#   ch15 (12.3 um, dirty window)     split-window pair with 13
#   ch16 (13.3 um, CO2)              minimal -- stratospheric, decoupled
#   ch14 (11.2 um, longwave window)  low -- spectrally redundant with 13
#   ch08 (6.15 um, upper WV)         low -- saturates near the tropopause
#   ch12 (9.7 um, O3)                minimal -- decoupled from precip
#
# Bands 13, 9 and 7 are already the ir/wv/swir channels. The order here
# matters operationally: EXTRA_IR_FETCH_LIMIT trims from the END, so a
# reduced fetch drops the channels the paper found least informative
# rather than an arbitrary subset.
EXTRA_IR_BANDS = (11, 10, 15, 16, 14, 8, 12)

# How many of the above to actually fetch per frame. Each band is another
# S3 download, so this trades latency against input richness; 0 disables
# extra bands entirely and reproduces pre-0.98 behaviour. Channels not
# fetched are filled with a neutral plane, so the model input shape never
# changes -- only how much real information is in it.
EXTRA_IR_FETCH_LIMIT = 3

# Elevation normalization. Coarse on purpose: the meteorologically
# important part is land-versus-ocean and gross terrain, not metres.
ELEV_MEAN, ELEV_STD = 0.0, 500.0

# Channel order. Documented explicitly because a silent reordering
# between training and inference is exactly the class of bug that
# produces confident nonsense rather than an error.
INPUT_CHANNEL_LAYOUT = (
    ["ir", "wv", "swir"]
    # ONLY the bands actually fetched. The layout previously spanned all
    # of EXTRA_IR_BANDS (7) while EXTRA_IR_FETCH_LIMIT is 3, so FOUR
    # channels were permanently neutral planes -- costing model capacity
    # and teaching it that those inputs carry nothing.
    #
    # Derived from the fetch limit rather than declared separately, so
    # the two cannot drift apart again. Raising the limit widens the
    # model automatically; the bands are already ordered by Li et al.'s
    # saliency, so the first three are the informative ones and the rest
    # are the bands that analysis found contribute minimally.
    + [f"ir_band{b}" for b in EXTRA_IR_BANDS[:EXTRA_IR_FETCH_LIMIT]]
    + ["backbone_v37", "backbone_h37", "backbone_v89", "backbone_h89"]
    + ["vmax", "rmw", "land_fraction", "elevation"]
    # Lightning (0.112). Appended at the END so every earlier channel
    # keeps its index -- a mid-list insertion would silently reinterpret
    # every existing export.
    + ["flash_density"]
)
MODEL_IN_CHANNELS = len(INPUT_CHANNEL_LAYOUT)

# The extra IR bands that actually occupy channels, in channel order.
# Both ml_train and ml_inference build their stacks from THIS, not from
# EXTRA_IR_BANDS -- iterating the full band list produced a stack of 19
# planes for a model declared with 15, which is a crash on the first
# batch rather than a silent error, but only after a full mine.
MODEL_EXTRA_IR_BANDS = tuple(
    int(c[len("ir_band"):]) for c in INPUT_CHANNEL_LAYOUT if c.startswith("ir_band")
)

# Blend taper width (pixels) for smoothly merging a corrected patch back
# into the full backbone at inference time, so the patch boundary isn't
# a visible seam -- see ml_inference.py. Scaled up proportionally with
# PATCH_SIZE to keep a similar relative taper fraction, not just left at
# the old absolute pixel count.
PATCH_BLEND_TAPER_PX = 32


# --- Correction model architecture (0.100) ---------------------------
# "diffusion" -> ml_diffusion.MWResidualDiffusion, a conditional flow-
#   matching model over the RESIDUAL against the parametric backbone.
#   Samples an ensemble, so it yields an uncertainty field for free and
#   does not collapse ambiguity to a conditional mean.
# "unet"      -> ml_model.MWCorrectionUNet, the original deterministic
#   regressor. Retained as a baseline: it is cheaper, and having the two
#   trained on the same data is the only way to tell whether the extra
#   machinery is actually earning its place on THIS dataset rather than
#   on the paper's.
CORRECTION_ARCH = "diffusion"

# Ensemble members drawn per frame at inference. The mean is rendered;
# the spread is reported as uncertainty. More members cost linear time.
# --- Sizing for a 4 GB laptop GPU (RTX 3050) -------------------------
# Li et al. trained on four RTX 4090s -- 96 GB against 4, a 24x gap -- so
# their settings are not a target to aim at, and shrinking their DiT until
# it fits would give up the global attention that made it worth choosing.
# These numbers are picked so the thing runs at all on the intended
# machine, and are the first place to look if it OOMs.
#
# base_channels 48 with the 4-level U-Net is roughly 3.4M parameters; the
# activation peak at 256x256, batch 2, fp16 is around 1.3 GB, which leaves
# headroom on a 4 GB card for optimizer state and fragmentation. Drop to 32
# if training OOMs; that costs capacity but not correctness.
DIFFUSION_BASE_CHANNELS = 48

# Members are sampled SEQUENTIALLY, so cost is linear in members x steps.
# 8 x 24 = 192 forward passes per frame, which is fine for a batch export
# and too slow to sit and watch. 4 x 12 = 48 passes is the interactive
# setting and still gives a usable spread estimate.
#
# Statistical caveat worth knowing: the spread from 4 members is noisy,
# and the sample standard deviation is biased low at small N even with
# the ddof correction. Read it as "where is the model unsure", not as a
# calibrated error bar.
# Ratio by which the raw ensemble spread UNDERSTATES the true error.
# Measured on the first converged run: 4-member spread 6.85 K against an
# actual RMSE of 11.78 K, so the ensemble is overconfident by about 1.7x.
#
# This is unsurprising -- a model trained on 264 examples has no way to
# represent the uncertainty arising from everything it has never seen --
# but it matters because the spread is DISPLAYED as a confidence field.
# An uncertainty estimate that is confidently too small is worse than
# none, so it is scaled before display and labelled as calibrated.
#
# Re-measure with `sampled_skill`'s "overconfidence" figure whenever the
# model or the dataset changes; it is an empirical constant, not a
# derived one.
# RE-MEASURED on the 799-example run: overconfidence settled at 1.9-2.0x
# across the converged epochs (was 1.7x on 298 examples). More data made
# the ensemble TIGHTER without making it proportionally more accurate, so
# it became more overconfident, not less -- which is what a model that has
# seen more of the distribution but still cannot represent what it has
# never seen should do.
ENSEMBLE_SPREAD_CALIBRATION = 1.95

ENSEMBLE_MEMBERS = 4

# ODE steps per member.
DIFFUSION_SAMPLE_STEPS = 24
