"""
Non-ML synthetic passive-microwave (37 GHz / 89 GHz) generator for tropical
cyclones, driven by GOES ABI bands 2/7/9/13 and NHC best-track intensity.

PHYSICAL BASIS (why this isn't just a colorized IR image):

  - At 85-92 GHz, brightness temperature over a tropical cyclone is
    dominated by scattering from precipitation-sized ice hydrometeors
    aloft (graupel/snow in convective towers). Deep, cold-topped convection
    (low IR Tb) => strong ice scattering => strongly DEPRESSED 89 GHz Tb.
    This is the physical basis of the classic "donut"/eyewall signature
    seen in 85-91 GHz color composites.

  - At 37 GHz, the same ice-scattering depression exists but is weaker
    (longer wavelength interacts less with small ice particles), while
    liquid-phase rain below the freezing level actually EMITS at 37 GHz
    over a radiometrically-cold ocean background, raising Tb in moderate
    rain before the heaviest convective cores depress it again. This
    produces the emission-then-scattering shape the algorithm reproduces.

  - IR alone (band 13) cannot distinguish "cold cirrus anvil with no
    underlying scatterers" from "cold overshooting convective top with
    heavy underlying ice" -- which is exactly why real MW imagery is more
    diagnostic for intensity/structure than IR. We partially compensate
    for that ambiguity using:
      * band 9 (6.9 um WV) to identify deep moist convective cores vs.
        thin/detached cirrus (WV stays cold only over sustained deep
        convection, decays faster over thin anvil)
      * band 7 (3.9 um) minus band 13 texture as a coarse proxy for
        convective-scale (small, bumpy) vs. stratiform (smooth) cloud tops
      * best-track intensity + a Willoughby (2006) RMW estimate to impose
        a physically-plausible radial vortex structure (tight eyewall
        ring at high intensity, broad/diffuse at low intensity) rather
        than just redrawing whatever blob shape the IR happens to show

  This is a calibrated HEURISTIC, not a radiative transfer model. It is
  meant to be tuned/calibrated against real coincident MW passes once
  that ingestion module exists (see mw_ingest.py, WIP). Treat the
  constants in CALIBRATION below as a first-guess starting point pulled
  from general published TB ranges for TC eyewalls/rainbands, not as
  measured fits.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import mw_surface as mw_surface_mod
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import griddata

from data_types import BandImage, StormFix, SyntheticMWResult
from qc_utils import sanitize_field, assert_finite

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


# ---------------------------------------------------------------------------
# Calibration constants (first-guess; TODO recalibrate against real MW data)
# ---------------------------------------------------------------------------
# Identifier for the V/H radiative physics. Bump whenever the mapping
# from response to V/H brightness temperature changes in a way that moves
# the mean level or the polarization structure -- i.e. the CALIBRATION
# V/H constants, or the emission-versus-scattering split.
#
# WHY: the ML correction is trained to predict a residual against a
# SPECIFIC backbone. 0.95 inverted the 37 GHz eyewall from a scattering
# depression to an emission signature, a ~120 K change at the core. Any
# checkpoint trained before that learned to correct a backbone that no
# longer exists, so applying it now is a train/inference mismatch of the
# worst kind -- and it showed exactly that way on the first real frame,
# dragging the fused 37 GHz eyewall radius from 26 km to 2 km. A stale
# checkpoint cannot detect this about itself, so the backbone has to
# announce its own identity and ml_inference has to check it.
_VH_PHYSICS_BASE = "0.136"


def _compute_vh_physics_id() -> str:
    """Identifier DERIVED from the constants, not maintained by hand.

    A hand-written string goes stale the moment someone tunes a value and
    forgets to touch it -- which happened immediately. 0.135 changed
    SATURATION_K from 3.0 to 0.75, a genuine backbone change, while the
    identifier still read "0.133-...". Two datasets with different physics
    would have carried the same label, resumability would have skipped the
    old files as current, and check_physics_consistency would have seen one
    vintage and approved. The guard defeated by the thing it guards.

    Hashing the physics-affecting constants makes that impossible: any
    change to any of them produces a new id automatically, so a mixed
    dataset is refused without anyone having to remember.
    """
    import hashlib

    parts = [_VH_PHYSICS_BASE]
    for k in sorted(CALIBRATION):
        v = CALIBRATION[k]
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v!r}")
    try:
        import mw_surface as _ms
        for name in ("SATURATION_K", "RESPONSE_GAMMA", "LIQUID_SMOOTH_KM",
                     "WIND_SENSITIVITY", "MAX_ROUGHENING_K", "AMBIENT_WIND_MS",
                     "ATMOS_SWING_K", "WV_DRY_K", "WV_MOIST_K",
                     "FREEZING_LEVEL_KM", "RANKINE_ALPHA"):
            parts.append(f"{name}={getattr(_ms, name, None)!r}")
    except Exception:
        parts.append("mw_surface=unavailable")
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return f"{_VH_PHYSICS_BASE}-{digest}"
#
# Bumped for the accumulated backbone changes since 0.99, and above all
# for 0.128: the stored backbone went from (parametric + per-frame
# baseline shift) to pure parametric, so the residual target
# real_MW - backbone changed MEANING. That is precisely what this
# identifier exists to catch, and 0.128 failed to bump it -- an
# oversight that would have let ml_train mix vintages silently and let
# calibrate_constants fit across the boundary, baking the very leak
# 0.128 removed back into the constants.
#
# Also covers 0.112 (sensor PSF), 0.114 (per-frequency response and wind
# roughening) and 0.113 (parallax), all of which move the backbone.

CALIBRATION = {
    # Clear-sky / background brightness temps over open tropical ocean.
    #
    # CORRECTED IN 0.89. These were 165.0 and 250.0, which are not
    # clear-ocean values at all: real AMSR-class clear tropical ocean is
    # roughly 200-205 K at 37 GHz V-pol and 265-270 K at 89 GHz V-pol.
    # (165 K sits between ocean H-pol and V-pol at 37 GHz and matches
    # neither -- most likely an H-pol-ish figure used where a V-pol one
    # was needed, since mw_compare measures freq_37ghz against real
    # V-pol.)
    #
    # The evidence that this was the bug is unusually direct: the
    # persisted auto-calibration EMA, which learns (synthetic - real) over
    # many passes and knows nothing about these constants, converged to
    # -36.8 K at 37 GHz and -16.5 K at 89 GHz over 54 observations. Those
    # are the baseline errors above, to within about a Kelvin on both
    # channels. The EMA had independently rediscovered this, and was
    # spending its entire clamp budget (40 K) papering over it -- at
    # -36.8 K it was within 8% of saturating, after which it would have
    # silently stopped tracking anything real.
    #
    # SCOPE: these constants drive the SCALAR tb37/tb89 fields only. The
    # V/H fields that produce the colour composites use the separate
    # bg_v_37/bg_h_37/... constants below and are unaffected, so this
    # changes the quantitative comparison numbers without changing the
    # rendered imagery.
    "bg_tb_37": 202.0,
    "bg_tb_89": 266.0,
    # Peak emission bump at 37 GHz from moderate rain before scattering wins.
    "emission_boost_37": 70.0,
    # Max scattering depression at each frequency in the coldest/most
    # convective pixels (applied on top of / instead of the emission term).
    #
    # RETUNED IN 0.89 alongside the background temps above, because these
    # were tuned as offsets from baselines that were ~37 K and ~16 K too
    # cold. Held at their old values (90 / 140) they now imply floors of
    # 202-90 = 112 K at 37 GHz and 266-140 = 126 K at 89 GHz, which no
    # real observation approaches -- 37 GHz scattering is weak, with deep
    # convection reaching roughly 180-200 K, and 89 GHz minima in intense
    # convection sit around 150-180 K. The old constants were only
    # plausible because the baseline they were subtracted from was wrong;
    # correcting one without the other would trade a bias error for an
    # unphysical dynamic range.
    #
    # Same "first-guess, needs real-data calibration" caveat as the rest
    # of this dict -- these are chosen to put the extremes in the right
    # neighbourhood, not fitted. The persisted EMA will refine the
    # residual bias from here, which is what it is actually for.
    "max_depression_37": 35.0,
    "max_depression_89": 105.0,
    # IR Tb (K) considered "no meaningful convection" vs. "extreme overshoot".
    "ir_warm_bound": 273.0,
    "ir_cold_bound": 185.0,
    # Separate, MUCH colder-onset threshold used only for the V/H
    # emission-vs-depression split (not for convective_signal/response,
    # which keeps using ir_warm_bound above). See scat_pot_vh's comment
    # in generate_synthetic_mw for the full reasoning -- this decouples
    # "how much rain/response is happening" from "how much of that is
    # ice-scattering vs. warm-rain emission," which a single shared
    # threshold couldn't represent (real storms have extensive warm-rain
    # regions with elevated response but little ice aloft; the coupled
    # version had no way to show that, collapsing straight from green to
    # red with no real cyan/pink in between -- confirmed both
    # mathematically and against a real-data comparison).
    # Per-frequency scattering onset (see scat_pot_vh_37 / _89 below).
    # 89 GHz keeps the previously tuned 230 K: ice scattering genuinely
    # dominates there. 37 GHz uses 190 K, only just above ir_cold_bound,
    # so that ONLY genuine overshooting tops scatter and ordinary eyewall
    # convection emits -- matching real GMI 37H, which rises ~80 K from
    # ocean to eyewall rather than falling.
    #
    # 200 K was tried first and was still too warm: a normal deep eyewall
    # top near 195 K came out 35% scattering, which held H37 about 70 K
    # below the observed value. Worth being explicit that using CLOUD-TOP
    # IR to decide emission-versus-scattering is physically weak at
    # 37 GHz in the first place -- the emission comes from liquid water
    # below the freezing level, which cloud-top temperature only loosely
    # constrains. This threshold is a pragmatic stand-in, not a
    # derivation, and it is the obvious place a radiative-transfer
    # treatment would replace guesswork.
    "vh_scatter_warm_bound_37": 190.0,
    "vh_scatter_warm_bound_89": 230.0,
    "vh_scatter_warm_bound": 230.0,

    # --- Separate V/H baselines, added so real color-composite techniques
    # (mw_composites.py) can be reused on synthetic output instead of a
    # plain scalar colormap. The scalar bg_tb_*/emission_boost_*/
    # max_depression_* above are UNCHANGED and still drive freq_37ghz/
    # freq_89ghz (kept for mw_compare.py's stats, which need one scalar
    # to compare against real V-pol/PCT) -- these V/H constants drive a
    # separate, parallel computation of v37/h37/v89/h89 for coloring only.
    #
    # Clear-ocean V is warmer than H (ocean's polarization difference);
    # rain emission raises both, with H rising faster/more so the two
    # converge in heavier rain. Ice scattering depresses both similarly
    # (largely unpolarized scattering). These are first-guess estimates,
    # not measured -- same "needs real-data calibration" caveat as the
    # rest of this dict.
    #
    # IMPORTANT: these baselines were retuned after porting the exact NRL
    # GeoIPS color-composite algorithm into mw_composites.py. The composite
    # expects clear-ocean V/H to fall within specific ranges to render as
    # green (37 GHz)/gray (89 GHz) rather than red -- the earlier baselines
    # (190/150 and 245/190) predated having the exact NRL formula and
    # rendered as a jarring red/dark-red "ocean" once the exact algorithm
    # was in place.
    #
    # RETUNED AGAIN after a real run's auto-calibration revealed the
    # problem with the previous fix: V=208,H=125 was tuned to match a
    # reference IMAGE's look, but real ingested AMSR2 data for an actual
    # storm consistently measured a ~+71K gap (37 GHz) between that
    # baseline and reality across 7 separate comparisons -- meaning the
    # aesthetic target was fighting against what the data actually shows.
    # Each frame's one-shot correction was closing that entire gap
    # unclamped (see mw_compare.apply_bias_calibration's new max_shift_k
    # cap, added at the same time this was fixed), pushing V/H into the
    # saturating tail of the NRL color ranges -- everything rendered as
    # the same maxed-out green/cyan with no spatial texture, exactly what
    # was reported.
    #
    # Moved the raw baseline partway toward what that calibration data
    # suggests (not all the way -- this is one storm's history, not a
    # validated dataset) and verified the result isn't saturated: V=250,
    # H=170 -> RGB(0,148,18), a moderate, non-maxed green (compare to the
    # over-corrected RGB(0,210,65) that prompted this fix). Expect this to
    # keep moving as more real comparisons accumulate -- that's the
    # persisted offset's job, now safely clamped either way.
    #
    # IF YOU'VE RUN AN EARLIER VERSION: click "Reset persisted calibration"
    # on the Generate tab once after updating -- the old learned offset was
    # calibrated against the previous (now-changed) baseline and is no
    # longer meaningful relative to this one.
    "bg_v_37": 250.0,
    # Raised from 170 to 182: measured GMI 37H far-field ocean around this
    # storm was 182-193 K. (Calm-ocean 37H is nearer 150-160 K, but a
    # hurricane's outer wind field roughens the surface and raises H-pol
    # emissivity substantially, and that is the regime this tool renders.)
    # AMBIENT tropical background, not clear calm ocean. Kept at the
    # measured 182 K deliberately, after a first attempt lowered it to 168
    # on the theory that wind roughening explained the gap to a
    # calm-ocean 150-155 K. It does not: a 0.75 K per m/s slope would
    # need ~40 m/s to bridge 30 K, and the region measured had maybe
    # 15 m/s. The rest is ATMOSPHERIC -- water vapour and cloud liquid
    # over a humid tropical ocean add roughly 20-25 K at 37H, and this
    # project has no atmospheric term at all.
    #
    # So this constant means "tropical ocean background including the
    # mean atmospheric contribution, at ambient wind", which is what the
    # measurement actually sampled. mw_surface adds only the
    # storm-relative EXCESS above that ambient, which is the part that
    # genuinely varies with radius. A real atmospheric term would be the
    # next improvement and would let this be decomposed properly.
    "bg_h_37": 182.0,
    # Retuned so a saturated eyewall reaches the measured real values:
    # H37 182 -> ~275 K and V37 250 -> ~278 K, which collapses the
    # polarization difference from 68 K to about 3 K. That collapse is
    # the actual 37 GHz eyewall signature -- an optically thick, nearly
    # unpolarized emitting layer -- and it is what turns the core white
    # in the NRL composite. The old 55/85 pair could only close V-H by
    # 30 K even at full emission, and the shared scattering fraction
    # then throttled that to about 6 K.
    "emission_boost_v37": 28.0,
    "emission_boost_h37": 93.0,
    # ASYMMETRIC depression -- V drops much more than H (physically:
    # ice scattering pulls both toward a similar cold floor, and V starts
    # higher so it has further to fall; H is already closer to that floor).
    # This was a genuine bug, not a calibration issue: with the previous
    # EQUAL depression (90,90), the quantity that drives the red channel
    # (2.181*V-1.181*H, weighted ~2x toward V) only dropped ~58K at
    # realistic peak response (~0.64), but needs to drop ~64K just to
    # start showing ANY red and ~84K for full red -- meaning 37 GHz could
    # never show pink/red at all, regardless of storm intensity, which is
    # exactly what two real test runs showed (flat green with a slightly
    # darker blob, no red/pink, in both a calibrated and an uncalibrated
    # run -- ruling out calibration as the cause and pointing at this).
    # Verified with a realistic multi-degree storm scene: (110,70) gives a
    # gradual green -> partial-red -> full-red progression across the
    # realistic response range (~0.4 to ~0.64 peak), not an abrupt jump.
    # Small at 37 GHz: ice scattering is weak there, and these now only
    # bite in the coldest overshooting tops (see vh_scatter_warm_bound_37).
    "max_depression_v37": 20.0,
    "max_depression_h37": 12.0,

# --- Land background (see surface_type.py) --------------------------
    # Land at 37/89 GHz is a near-blackbody in BOTH polarizations
    # (emissivity ~0.9-0.95), so it sits close to its physical
    # temperature with a polarization difference of only a few K, where
    # ocean runs 55-80 K. Applying ocean constants over land does not just
    # mis-level the field -- a warm, unpolarized surface is precisely the
    # signature heavy precipitation produces over ocean, so unmasked land
    # actively imitates the feature the composites exist to show, right
    # where a landfalling storm matters most.
    #
    # These are clear-sky land values for a warm tropical land surface.
    # Deliberately NOT split by land-cover type: without a vegetation or
    # soil-moisture input that would be false precision, and the point
    # here is to stop rendering coastlines as convection, not to retrieve
    # land emissivity.
    "bg_v_37_land": 288.0,
    "bg_h_37_land": 282.0,
    "bg_v_89_land": 290.0,
    "bg_h_89_land": 285.0,
    "bg_v_89": 280.0,
    "bg_h_89": 260.0,
# 89 GHz EMISSION (0.133). Until now 89 GHz had no emission term at
    # all -- it was pure depression from a fixed background, monotonically
    # decreasing in response. That is wrong in the same way 37 GHz was
    # before 0.95: over ocean, cloud and rain liquid EMIT at 89 GHz and
    # raise Tb before ice scattering takes over and drops it. The real
    # response is a hook, not a ramp, and light-to-moderate convection
    # sits on the rising part of it.
    #
    # Smaller than at 37 GHz because 89 GHz saturates in less liquid and
    # the ocean background is already much warmer, so there is less room
    # to rise. H-pol gains more than V-pol, as everywhere else here.
    # SMALL, and checked against the physical temperature limit. A first
    # attempt used 12/22, which put V-pol at 280 + 12 + 8 (atmospheric) =
    # 300 K -- at the sea-surface physical temperature, which an emitting
    # layer cannot exceed.
    #
    # The real Norbert 89H frame settles it: the ocean background is
    # ~280 K and EVERY convective feature is colder than it. At 89 GHz
    # over ocean the background already sits close to saturation, so
    # there is very little room to rise. The term is real but weak, and
    # the resulting hook is a few K rather than a prominent bump.
    "emission_boost_v89": 5.0,
    "emission_boost_h89": 12.0,
    "max_depression_v89": 130.0,
    "max_depression_h89": 150.0,
}


def _scattering_potential(ir_tb: np.ndarray) -> np.ndarray:
    """Map IR brightness temp to a 0-1 'how much ice-scattering signal
    should be here' potential. Colder IR -> higher potential."""
    warm = CALIBRATION["ir_warm_bound"]
    cold = CALIBRATION["ir_cold_bound"]
    pot = (warm - ir_tb) / (warm - cold)
    return np.clip(pot, 0.0, 1.0)


def _wv_depth_mask(wv_tb: np.ndarray, ir_tb: np.ndarray) -> np.ndarray:
    """Distinguish sustained deep convection (WV stays nearly as cold as IR)
    from thin/detached cirrus (WV noticeably warmer than IR, since WV
    channel senses a higher, often moister/colder-appearing layer only when
    convection is actively lofting moisture -- thin cirrus decouples from
    the WV signal faster than from the IR signal). Returns 0-1, 1 = deep."""
    diff = wv_tb - ir_tb
    # diff near 0 (or negative) => deep sustained convection
    # diff large positive (WV much warmer than IR) => thin cirrus, discount it
    mask = 1.0 - np.clip(diff / 25.0, 0.0, 1.0)
    return mask


def _texture_signal(swir_tb: np.ndarray, ir_tb: np.ndarray) -> np.ndarray:
    """Coarse convective-texture proxy from band7-band13 gradient magnitude.
    Bumpy/granular convective towers produce more local gradient than
    smooth stratiform anvil."""
    diff = swir_tb - ir_tb
    gx = sobel(diff, axis=0, mode="nearest")
    gy = sobel(diff, axis=1, mode="nearest")
    grad = np.hypot(gx, gy)
    if grad.max() > 0:
        grad = grad / np.percentile(grad, 99.0).clip(min=1e-6)
    return np.clip(grad, 0, 1)


def _normalize_texture(pattern: np.ndarray) -> np.ndarray:
    """Normalize a texture pattern to unit standard deviation, so its
    amplitude can be controlled independently of the source band's own
    natural Kelvin-scale variability."""
    std = np.std(pattern)
    return pattern / std if std > 1e-6 else pattern


def _measure_real_texture_amplitude(real_field: np.ndarray, highpass_sigma: float = 2.0):
    """Measure the ACTUAL local texture amplitude (Kelvin std of a high-
    pass-filtered version) of a real MW field, on its own native swath
    grid. Used to CALIBRATE how much synthetic texture to inject into the
    backbone, instead of a fixed guessed Kelvin constant -- added after a
    three-way real-data comparison (GOES-only loop vs. GOES+real-MW
    single frame vs. GOES+real-MW loop) showed the fixed-amplitude
    injection had fully solved the "flat region" bug (GOES-only loop:
    consistent grain everywhere, confirmed) but left a SUBTLER residual
    seam specifically where real MW was blended in -- not a
    flat-vs-textured mismatch anymore, but a texture-STYLE mismatch: the
    synthetic backbone's guessed amplitude doesn't necessarily match
    what THIS particular real pass's actual texture looks like. Measuring
    it directly from the real data closes that gap on a per-storm,
    per-pass basis rather than relying on one fixed guess to fit every
    real MW pass's actual character.

    Returns None if the field has no finite data (nothing to measure).
    """
    valid = np.isfinite(real_field)
    if not valid.any():
        return None
    filled = np.where(valid, real_field, np.nanmean(real_field[valid]))
    highpassed = filled - gaussian_filter(filled, sigma=highpass_sigma)
    return float(np.std(highpassed[valid]))


def _measure_real_baseline_value(real_field: np.ndarray, percentile: float = 50.0):
    """Robust estimate of real MW data's TYPICAL/BACKGROUND Kelvin level
    (median by default) -- used to calibrate the synthetic backbone's
    baseline to match what real data actually shows for THIS pass,
    instead of a fixed guessed CALIBRATION constant (bg_v_37=250 etc).

    Median (not mean) is deliberate: robust to the storm's extreme core
    values being a small fraction of a wide-coverage swath's total pixel
    count, so this approximates "typical background condition" without
    needing to explicitly mask out the storm core first.

    Added after a direct real-storm comparison showed a genuine color/
    saturation mismatch specifically at 37 GHz: GOES-only regions
    rendered a noticeably more saturated, vivid green than the muted,
    textured green seen where real MW data was actually blended in --
    even after the texture-amplitude fix (a separate problem: that one
    was about local variance, this one is about the baseline Kelvin
    LEVEL the color composite starts from). Returns None if the field
    has no finite data.
    """
    valid = np.isfinite(real_field)
    if not valid.any():
        return None
    return float(np.percentile(real_field[valid], percentile))


def _willoughby_rmw_km(vmax_kt: float, lat_deg: float) -> float:
    """Willoughby et al. (2006) empirical RMW estimate from intensity and
    latitude. Used only when best-track RMW is not provided."""
    return 46.29 * np.exp(-0.0153 * vmax_kt + 0.0166 * abs(lat_deg))


def _distance_km_from(lat, lon, clat, clon):
    """Great-circle-ish distance from a storm centre, dateline-safe."""
    dlat = np.asarray(lat) - clat
    dlon = _wrap_lon_delta(np.asarray(lon) - clon) * np.cos(np.radians(clat))
    return np.sqrt(dlat ** 2 + dlon ** 2) * 111.32


def _radial_weight(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    rmw_km: float,
    vmax_kt: float,
    roci_km: Optional[float] = None,
    return_geometry: bool = False,
) -> np.ndarray:
    """Build a radially-varying weight (0-1) that concentrates the
    algorithm's response into a plausible eyewall ring + broader rainband
    envelope, scaled by intensity. Stronger storms get a tighter, sharper
    ring; weaker/disorganized storms get a broad, gentle bump.

    roci_km, when available from best track, directly bounds how far out
    the rainband/CDO envelope should extend -- a large, sprawling storm
    (big ROCI) and a small, compact one (small ROCI) at the *same* Vmax
    can have very different real-world MW footprints, which a Vmax-only
    envelope estimate can't capture. When ROCI isn't available (common --
    it's not populated on every best-track row), we fall back to the
    Vmax-based interpolation as before.
    """
    # Approximate great-circle distance in km (equirectangular is fine at
    # mesoscale-sector spatial scales, ~1000km).
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    dy = (lat - center_lat) * km_per_deg_lat
    dx = _wrap_lon_delta(lon - center_lon) * km_per_deg_lon
    r = np.hypot(dx, dy)

    # Eyewall ring: Gaussian centered at RMW.
    #
    # Width now scales with RMW as well as intensity. It previously
    # depended on Vmax ALONE, via
    #     np.interp(vmax_kt, [20,60,100,140,180], [80,60,40,25,18])
    # which is an absolute width in km and takes no account of the radius
    # it is centered on. For a compact storm that is incoherent: Lowell
    # (EP12) at ~110 kt with a 19 km RMW got a 36 km ring width -- a
    # Gaussian whose sigma is nearly twice its own centre radius is not a
    # ring, it is a blob, and once the monotonically-decaying envelope
    # below is added the combined maximum migrates outward. Measured
    # against the old constants the weight peaked at 41.5 km for a 19 km
    # RMW (2.18x), matching the 2.27 eyewall/RMW ratio the structural
    # metrics reported on real Lowell frames.
    #
    # The error scaled with compactness: ratios came out near 1.0 for
    # RMW 55-80 km (where the fixed width happened to suit), 1.7 at
    # 28 km, and 2.2 at 19 km. It also explains the filled, ringless core
    # on Edouard (RMW 9 km, 50 kt): a 65 km width around a 9 km radius,
    # with no eye suppression below 65 kt, peaks at r=0.
    #
    # A real eyewall's width scales roughly with the eye it surrounds, so
    # the width is now a FRACTION of RMW, with the intensity dependence
    # kept as a multiplier on that fraction (stronger storms have
    # proportionally tighter eyewalls). Bounds keep it physical at both
    # extremes: no narrower than 8 km (below the effective resolution
    # this is rendered at, so a tighter ring would just alias) and no
    # wider than 60 km (beyond which it stops being a ring at all).
    ring_width_frac = np.interp(vmax_kt, [20, 60, 100, 140, 180], [1.30, 1.00, 0.75, 0.55, 0.45])
    ring_width_km = float(np.clip(ring_width_frac * rmw_km, 8.0, 60.0))
    # Centred OUTWARD of the RMW. Best-track RMW is a surface wind
    # radius; eyewall convection tilts outward with height, so the
    # cloud-top and hydrometeor signature this synthesises sits further
    # out than the wind maximum. See mw_surface.EYEWALL_TILT_KM.
    _ring_r = mw_surface_mod.tilted_ring_radius_km(rmw_km)
    ring = np.exp(-0.5 * ((r - _ring_r) / ring_width_km) ** 2)

    # Broad rainband/CDO envelope: slowly decaying with radius. Prefer the
    # best-track ROCI when available (actual observed storm size) over the
    # Vmax-only guess, since intensity and size are only loosely correlated
    # in reality (e.g. a compact 100kt storm vs. a sprawling 100kt storm).
    if roci_km is not None and roci_km > 0:
        # ROCI marks the outermost closed isobar, not a hard rain edge, so
        # use it as the e-folding scale directly rather than a hard cutoff --
        # this keeps a smooth taper instead of an unrealistic cliff edge.
        envelope_scale_km = np.clip(roci_km * 0.55, 80, 600)
    else:
        envelope_scale_km = np.interp(vmax_kt, [20, 60, 100, 140, 180], [150, 220, 300, 380, 420])
    envelope = np.exp(-r / envelope_scale_km)

    # Eye suppression: inside ~0.5*RMW, response drops toward the (warm,
    # clear) eye signature for organized/intense storms.
    #
    # The 0.35 scale was tested against a hypothesis in 0.119 and KEPT.
    # On controlled synthetic scenes the measured eyewall/RMW ratio drifts
    # 0.77 (35 kt) to 1.17 (140 kt), and eye suppression looked like the
    # cause since eye_frac scales with intensity and the kernel removes
    # 13.5% of the response at 0.7*RMW. Narrowing it to 0.22 changed the
    # spread by nothing at all (0.40 before and after), so that hypothesis
    # is wrong and the constant is left alone.
    #
    # Also ruled out: the sensor PSF (drift is identical with it disabled)
    # and grid resolution (identical from 3 to 27 pixels across the RMW).
    # The remaining candidate is the interaction between the IR-driven
    # convective signal and the radial prior -- but the IR scenes used
    # here are synthetic, and their cloud-shield geometry relative to RMW
    # is an assumption, so this may be measuring the test rather than the
    # algorithm. Worth re-checking against real frames across intensity
    # once the mined dataset exists; not worth changing a tuned constant
    # on synthetic evidence.
    eye_frac = np.clip((vmax_kt - 50) / 80, 0, 1)  # eyes mostly form >~65kt
    eye_suppress = 1.0 - eye_frac * np.exp(-0.5 * (r / (0.35 * rmw_km + 1e-6)) ** 2)

    # Combine. NORMALIZED by the coefficient sum, not clipped.
    #
    # This was `np.clip(0.55*ring + 0.65*envelope, 0, 1)`, and the two
    # coefficients sum to 1.20 -- so wherever ring and envelope were both
    # strong the sum exceeded 1 and was flattened. For Lowell that clipped
    # across 36% of the inner 60 km, turning the eyewall ring into a
    # plateau of exactly 1.0. The argmax was then decided by eye_suppress
    # shaving the inner side, which pushed the apparent eyewall to the
    # OUTER edge of the clipped region. The ring term was being erased
    # precisely in the storms it exists to represent -- the more intense
    # and compact the storm, the more of its core saturated.
    #
    # Dividing by the coefficient sum keeps the result in [0,1] by
    # construction while preserving the ring's shape. Combined with the
    # RMW-scaled width above, the weight now peaks within a few percent
    # of RMW (1.02-1.04x across the intense/compact range, versus
    # 1.6-2.2x before), and Edouard's 9 km RMW produces an actual ring at
    # 8 km instead of a filled disc peaking at r=0.
    #
    # Side effect, deliberate and accepted: peak weight is now ~0.82-0.95
    # rather than a saturated 1.0, so the overall response amplitude drops
    # slightly. That is the correct behaviour -- the old 1.0 was an
    # artifact of clipping, not a real maximum -- and the persisted
    # calibration EMA absorbs the residual level change.
    weight = ((0.55 * ring + 0.65 * envelope) / 1.20) * eye_suppress
    weight = np.clip(weight, 0, 1)
    if return_geometry:
        # The radius field and envelope scale, so a caller can re-weight
        # by radius per frequency without recomputing either or having to
        # duplicate the ROCI/vmax logic that produced the scale.
        return weight, r, float(envelope_scale_km)
    return weight


def _regrid_to(target_lat, target_lon, src: BandImage) -> np.ndarray:
    """Nearest-neighbor regrid src.values onto the target lat/lon grid.
    Used to bring band 2 (0.5km) or mismatched grids onto the band13 grid."""
    if src.values.shape == target_lat.shape:
        out = src.values
    else:
        points = np.column_stack([src.lat.ravel(), src.lon.ravel()])
        values = src.values.ravel()
        out = griddata(points, values, (target_lat, target_lon), method="nearest")
    # griddata's "nearest" extrapolates everywhere so this is normally a
    # no-op, but guard anyway (e.g. if the source itself still had NaNs).
    out, _ = sanitize_field(out)
    return out


# ---------------------------------------------------------------------------
# Multi-source fusion: combine GOES-derived, real-MW-derived, and radar-
# derived "response" fields as weighted equals (per pixel, priority-
# weighted by data availability) instead of sequentially layering a
# correction on top of a correction -- see generate_synthetic_mw's
# real_swath/radar_* parameters and DEFAULT_FUSION_WEIGHTS below.
# ---------------------------------------------------------------------------
DEFAULT_FUSION_WEIGHTS = {"goes": 0.1, "mw": 0.3, "radar": 0.6}
MW_FUSION_MAX_DISTANCE_KM = 300.0
RADAR_FUSION_MAX_DISTANCE_KM = 150.0


def _regrid_external_with_mask(src_lat, src_lon, src_values, dst_lat, dst_lon, max_distance_km, method="nearest"):
    """Regrid a SPARSE external source (a real MW swath or radar gates --
    doesn't cover the whole destination grid, unlike band9/band7 which
    always cover the full GOES sector) onto the destination grid, with
    destination pixels farther than max_distance_km from the nearest
    source point masked out (NaN) -- scipy's "nearest" griddata mode
    extrapolates everywhere by default, which would incorrectly claim
    coverage across the whole GOES sector from a swath that only actually
    covers a fraction of it.

    method: "nearest" (default) or "linear". Matters for radar
    specifically -- NEXRAD gates are sampled along discrete angular rays,
    and at longer range (order ~100km+) the gap between adjacent rays
    exceeds typical GOES pixel spacing, so "nearest" leaves visible
    unfilled gaps between rays (a radial "spoke" pattern, confirmed
    directly against a real screenshot showing a storm 91mi from its
    radar -- computed gap ~2.5km at that range, bigger than GOES pixel
    spacing). "linear" smoothly interpolates between neighboring rays
    instead of creating hard nearest-neighbor cell boundaries, which
    should eliminate that artifact. Real MW swaths don't have this
    problem (denser, more uniform native sampling), so they stay on
    "nearest" by default.

    "linear" returns NaN outside the source points' convex hull -- this
    is intentional and NOT filled with a nearest-neighbor fallback (an
    earlier version did this, which reintroduced the exact spoke pattern
    this method exists to avoid, specifically in the boundary zone between
    good linear coverage and the max_distance cutoff -- confirmed against
    a real screenshot showing a pronounced wedge/fan pattern from a radar
    site with known beam-blockage sectors, where that boundary zone was
    large). A destination pixel genuinely outside the convex hull of real
    data points should be treated as "no radar data here" and fall back
    to whatever other sources are available, not have a value invented
    for it via crude nearest-neighbor.

    Returns an array shaped like dst_lat, with NaN where no source data
    was close enough to trust.
    """
    from scipy.spatial import cKDTree

    src_points = np.column_stack([np.ravel(src_lat), np.ravel(src_lon)])
    values = np.ravel(src_values).astype(np.float64)
    valid_src = np.isfinite(values) & np.isfinite(src_points).all(axis=1)
    src_points = src_points[valid_src]
    values = values[valid_src]

    if len(values) == 0:
        return np.full(dst_lat.shape, np.nan)

    dst_points = np.column_stack([dst_lat.ravel(), dst_lon.ravel()])
    regridded = griddata(src_points, values, dst_points, method=method).reshape(dst_lat.shape)

    tree = cKDTree(src_points)
    dist_deg, _ = tree.query(dst_points)
    # Approximate degree->km conversion for a coarse validity mask -- not
    # precise (1 deg longitude != 1 deg latitude in km except at the
    # equator), but this only needs to be roughly right to decide "is
    # there real data anywhere near this pixel," not to position anything.
    dist_km_approx = (dist_deg * 111.0).reshape(dst_lat.shape)

    regridded[dist_km_approx > max_distance_km] = np.nan
    return regridded


def _regrid_confidence_taper(site_lat, site_lon, dst_lat, dst_lon, taper_start_km, taper_end_km):
    """For each destination pixel, a 0-1 confidence weight based on
    great-circle distance from the radar SITE (not from the sparse gate
    data itself, which would be circular -- a pixel with valid regridded
    data is by definition near ITS nearest gate): 1.0 within
    taper_start_km, linearly falling to 0.0 by taper_end_km. Added
    specifically for radar: real NEXRAD reliability genuinely degrades
    with range (beam widens -- confirmed directly: ~2.2km beam diameter
    at 143km range, vs ~1km close in -- so a single far-range gate
    represents a much coarser, blurrier sample), and a real run showed a
    small isolated patch of long-range (89mi, right at our 150km cutoff)
    radar data producing a visible, disconnected color anomaly unrelated
    to the storm's actual structure. Rather than a hard cutoff
    (full-strength data right up to the boundary, then nothing), this
    tapers radar's influence down gracefully as confidence genuinely
    decreases near the edge of its useful range.
    """
    R = 6371.0
    lat1 = np.radians(site_lat)
    lon1 = np.radians(site_lon)
    lat2 = np.radians(dst_lat)
    lon2 = np.radians(dst_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    taper = np.clip((taper_end_km - dist_km) / (taper_end_km - taper_start_km), 0.0, 1.0)
    return taper


def _multispectral_texture_field(ir_tb: np.ndarray, wv_tb: np.ndarray, swir_tb: np.ndarray, vis: Optional[np.ndarray] = None, highpass_sigma: float = 3.0) -> np.ndarray:
    """Extract genuine fine-scale spatial texture from the ACTUAL GOES
    multispectral imagery (not synthesized/fake noise) by high-pass
    filtering each band -- original minus a heavily-smoothed version of
    itself -- and combining them. Captures real small-scale cloud
    structure (individual convective towers, banding gradients, fine
    cirrus streaking) that a purely smooth radial-envelope model
    discards entirely.

    THIS IS THE FIX for a real, well-articulated problem: even with
    edge-feathering, blending real MW's genuinely grainy texture into a
    perfectly smooth GOES-only backbone always shows a seam, because a
    smooth region meeting a textured region is visually obvious no matter
    how gradual the weight transition is. The real fix isn't a better
    transition, it's giving the backbone ITSELF comparable fine-scale
    character everywhere -- using the OTHER GOES bands (WV, SWIR, and VIS
    when available), not just band 13, which is what "all 4 bands" was
    actually for beyond the existing coarse convective_signal blend.

    Returns a roughly unit-scaled (not literally 0-1, but robustly
    normalized so its 95th-percentile magnitude is ~1) texture field --
    scale it by however many Kelvin of injected texture is wanted at the
    call site, per frequency/channel.
    """
    def highpass(field):
        smoothed = gaussian_filter(field, sigma=highpass_sigma)
        return field - smoothed

    hp_ir = highpass(ir_tb)
    hp_wv = highpass(wv_tb)
    hp_swir = highpass(swir_tb)

    combined = 0.5 * hp_ir + 0.3 * hp_wv + 0.2 * hp_swir
    if vis is not None:
        hp_vis = highpass(vis)
        combined = combined * 0.8 + 0.2 * hp_vis

    scale = np.percentile(np.abs(combined), 95)
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros_like(combined)
    result = combined / scale

    # Hard clip -- a REAL bug, found while investigating an unrelated
    # complaint: this function's normalization uses the 95th-PERCENTILE
    # magnitude as its scale (by design, since that's more robust to a
    # few extreme pixels than the max would be) -- but with no clip
    # anywhere afterward, that's exactly backwards for the pixels that
    # ARE in that top 5%: an isolated sharp gradient (confirmed directly:
    # a storm's own eye/eyewall boundary, precisely where structure is
    # sharpest) produced a texture value of -95 in one test -- 95x the
    # intended "roughly unit-scaled" output. Multiplied by the Kelvin
    # injection amplitudes below, this silently produced PHYSICALLY
    # IMPOSSIBLE negative-Kelvin V/H values (confirmed: -386K), hidden
    # from view because final RGB rendering clips to 0-255 and an
    # extreme-but-clipped pixel just looks like ordinary saturated red,
    # not obviously broken. Clipping here (a few "sigma" above the
    # reference percentile, not the percentile itself) preserves genuine
    # texture character while making a repeat of that impossible.
    return np.clip(result, -TEXTURE_FIELD_CLIP, TEXTURE_FIELD_CLIP)


TEXTURE_FIELD_CLIP = 3.0


# Backbone texture injection magnitudes (Kelvin) -- how much of the
# multispectral texture field to add to the GOES-only backbone's V/H,
# AFTER the main parametric-formula smoothing (so it isn't immediately
# blurred back away). 89 GHz gets a larger injection than 37 GHz,
# matching real passive microwave behavior: 89 GHz is more sensitive to
# fine-scale ice-scattering variability than 37 GHz, so real 89 GHz
# imagery is naturally grainier. H-channel gets a modestly larger
# injection than V within each frequency too, consistent with H's
# greater sensitivity to surface/scattering variability elsewhere in
# this codebase (see the CALIBRATION comments on asymmetric depression).
# These are first-guess magnitudes, not measured -- same "needs real-
# data calibration" caveat as the rest of CALIBRATION; tune here if the
# backbone still reads as too smooth or, in the other direction, too
# noisy once compared against more real passes.
TEXTURE_INJECTION_V37_K = 4.0
TEXTURE_INJECTION_H37_K = 5.0
TEXTURE_INJECTION_V89_K = 7.0
TEXTURE_INJECTION_H89_K = 9.0


def _nan_aware_light_smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a field that may contain NaN (e.g. real MW data
    regridded outside its coverage area) without letting the NaN region
    corrupt values near the boundary -- a naive
    gaussian_filter(np.nan_to_num(field)) would blur zeros in from just
    outside the coverage edge, artificially cooling/warming pixels near
    the boundary rather than just smoothing them. Standard normalize-by-
    smoothed-mask trick: smooth the (NaN->0) data and a validity mask
    separately, then divide -- keeps the same coverage boundary, just
    smooths the values within/near it correctly.
    """
    nan_mask = ~np.isfinite(field)
    if not nan_mask.any():
        return gaussian_filter(field, sigma=sigma)
    filled = np.where(nan_mask, 0.0, field)
    weight = np.where(nan_mask, 0.0, 1.0)
    smoothed_filled = gaussian_filter(filled, sigma=sigma)
    smoothed_weight = gaussian_filter(weight, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed_filled / smoothed_weight
    result[nan_mask] = np.nan
    return result


# Light smoothing applied specifically to real-MW-regridded value fields --
# NOT the same as the GOES backbone's smoothing_sigma (1.2, much heavier).
# Added after a real test caught a moire/stripe resampling artifact when
# a swath's native grid and the GOES analysis grid are both regular but at
# different resolutions (nearest-neighbor regrid between two regular
# grids can alias into axis-aligned stripes). Real satellite swaths are
# naturally irregular/sheared (not perfectly regular), so this specific
# artifact is unlikely with real data, but this guards against it (and
# against implausibly sharp pixel-to-pixel sensor noise) while removing
# only a small fraction of real MW's genuine spatial texture -- verified
# directly: local-texture measurements after this fix are still ~4x
# higher than GOES-only, the fused output just isn't literally raw-pixel
# noise anymore.
MW_VALUE_SMOOTHING_SIGMA = 0.6

# Physically-plausible Kelvin bounds per channel -- values outside these
# are essentially certain to be a data-quality problem (uncaught fill
# values, calibration glitches, or the more subtle limb/edge-of-scan
# degradation known to affect conically-scanning radiometers' outermost
# cross-track pixels -- viewing geometry changes near a swath's edge,
# which is a real, documented effect) rather than a genuine Earth-
# observed brightness temperature. Added after a real run (Tropical
# Storm Chantal 2025, no radar involved at all -- isolating this from any
# radar-specific fix) showed a sharp, geometrically coherent wedge of
# anomalous color in a real swath's corner, most consistent with exactly
# this kind of edge-of-scan data issue rather than an isolated single-
# point spike (which _despeckle_gates in radar_ingest.py already handles
# for radar, but real_swath data had no equivalent check at all).
MW_PHYSICAL_BOUNDS = {
    "v37": (130.0, 310.0),
    "h37": (80.0, 300.0),
    "v89": (150.0, 310.0),
    "h89": (90.0, 300.0),
}


def _apply_physical_bounds(values: np.ndarray, bounds: tuple) -> np.ndarray:
    """Mask (to NaN) any value outside a physically-plausible range for
    that channel. NaN'd values are then naturally excluded by
    _regrid_external_with_mask's existing valid-source filtering -- no
    other code needs to change to benefit from this."""
    lo, hi = bounds
    result = np.asarray(values, dtype=np.float64).copy()
    result[(result < lo) | (result > hi)] = np.nan
    return result


def _despeckle_mw_field(values: np.ndarray, window: int = 5, threshold_k: float = 15.0) -> np.ndarray:
    """Mask (to NaN) isolated, spatially-unsupported pixels in a real MW
    swath field -- ones that jump sharply from their LOCAL neighborhood
    without any nearby pixels agreeing, which real Tb fields (smoothly
    varying at the sensor's native resolution) shouldn't do. This is the
    direct analog of radar_ingest.py's _despeckle_gates (isolated radar
    gate spikes -- ground clutter, AP, biological scatterers), applied
    here to real MW pixel data, which had NO equivalent check at all:
    _apply_physical_bounds only catches values outside a physically
    POSSIBLE range, not values that are individually plausible but
    inconsistent with every pixel around them (sensor noise, RFI,
    footprint contamination). Added after a real run showed a genuine
    artifact -- a thin isolated spike extending from the storm core
    across an otherwise smooth 89 GHz field, and a separate desaturated
    gray/white patch next to the core (most consistent with a NaN or
    wildly-inconsistent pixel surviving into the final color composite,
    where it renders as neither a valid color nor an obviously-excluded
    one).

    Uses a local MEDIAN filter (robust to a single bad pixel skewing a
    mean) as the "what should this pixel roughly look like" reference;
    anything more than threshold_k away from its own local median gets
    masked. window=5 and threshold_k=15K are deliberately conservative --
    real MW imagery has genuine fine-scale texture (confirmed and
    preserved elsewhere in this project), so this should only catch
    genuinely isolated, unsupported spikes, not legitimate local gradient.
    """
    from scipy.ndimage import median_filter

    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return arr

    filled = np.where(valid, arr, np.nanmedian(arr[valid]))
    local_median = median_filter(filled, size=window, mode="nearest")
    deviation = np.abs(filled - local_median)

    result = arr.copy()
    result[valid & (deviation > threshold_k)] = np.nan
    return result


def _external_scattering_index(pct: np.ndarray, warm_ref: float, cold_ref: float) -> np.ndarray:
    """Same shape of transform as _scattering_index_37/89 (0 at/above
    warm_ref, 1 at/below cold_ref) but for a REAL PCT field (from actual
    V/H data), giving a response-field-comparable 0-1 index instead of a
    GOES-IR-derived one. Reused for both real MW and could be reused for
    other PCT-like sources."""
    x = np.clip((warm_ref - pct) / (warm_ref - cold_ref), 0.0, 1.0)
    return x


def _morph_age_confidence(pass_age_hours: float, full_confidence_hours: float = 1.5, floor_hours: float = 6.0, floor_value: float = 0.15) -> float:
    """0-1 confidence multiplier based on how old a MIMIC-TC-style
    advected (morphed) real MW pass is, for the "combine synthetic
    tracking with a morphed real pass" fusion mode. Full confidence for
    very recent passes (<full_confidence_hours old), linearly decaying to
    floor_value by floor_hours, then flat at the floor beyond that.

    Advection captures storm TRANSLATION (moving the old pattern to the
    storm's current position) but not genuine structural EVOLUTION
    (intensification, eyewall replacement, weakening) -- so trust in the
    advected pattern should erode with age even though the pattern
    itself doesn't change until a newer real pass arrives. Not zero at
    the floor: an aging real pass is still real observed structure, just
    increasingly less trustworthy relative to what GOES/radar currently
    show -- matches this project's general philosophy of graceful
    tapering over hard cutoffs (see _regrid_confidence_taper,
    _edge_feather_taper).
    """
    if pass_age_hours <= full_confidence_hours:
        return 1.0
    if pass_age_hours >= floor_hours:
        return floor_value
    frac = (floor_hours - pass_age_hours) / (floor_hours - full_confidence_hours)
    return floor_value + (1.0 - floor_value) * frac


def _crossfade_confidence_toward_after(minutes_until_after: float, crossfade_minutes: float = 60.0) -> float:
    """0-1 confidence for an "after" pass (one that occurred AFTER
    target_time, morphed BACKWARD to target_time's position -- valid for
    historical/archived generation, where a later pass has already
    happened and its data already exists) as part of a MIMIC-TC-style
    crossfade between a "before" pass and an "after" pass.

    0 confidence when target_time is more than crossfade_minutes (default
    60) before the after-pass's actual time; ramping LINEARLY up to 1.0
    exactly AT the after-pass's time. This is the mirror-image
    complement of _morph_age_confidence (which fades an aging "before"
    pass OUT as time moves away from it) -- here the "after" pass fades
    IN as target_time approaches it, so the transition between two real
    passes happens gradually across the crossfade window instead of
    jumping abruptly the instant a new pass becomes the "most recent" one.

    minutes_until_after: how many minutes from target_time until the
        after-pass's actual scene_time (should be >= 0 -- this function
        assumes the after-pass is indeed still in target_time's future).
    """
    if minutes_until_after <= 0:
        return 1.0  # target_time is at or past the after-pass's own time
    if minutes_until_after >= crossfade_minutes:
        return 0.0
    return 1.0 - (minutes_until_after / crossfade_minutes)


def _edge_feather_taper(src_lat, src_lon, dst_lat, dst_lon, taper_start_km, taper_end_km):
    """For each destination pixel, a 0-1 confidence weight based on
    distance to the NEAREST valid source point: 1.0 close to real data,
    linearly falling to 0.0 by taper_end_km. Used to feather a SWATH's
    coverage edge (real MW) rather than leaving a hard, visible seam
    where MW-textured data meets the pure-GOES-backbone fallback --
    confirmed directly in a real run: a crisp straight-line boundary
    matching the swath edge, not a gradual transition.

    Distance-to-nearest-point is the RIGHT metric here, unlike radar
    (_regrid_confidence_taper, which uses distance from the fixed radar
    SITE instead): a swath doesn't have a single site to measure range-
    dependent reliability from -- what matters for a swath is "how much
    real coverage is actually near me," not "how far from a transmitter."
    """
    from scipy.spatial import cKDTree

    src_points = np.column_stack([np.ravel(src_lat), np.ravel(src_lon)])
    valid = np.isfinite(src_points).all(axis=1)
    src_points = src_points[valid]
    if len(src_points) == 0:
        return np.zeros(dst_lat.shape)

    dst_points = np.column_stack([dst_lat.ravel(), dst_lon.ravel()])
    tree = cKDTree(src_points)
    dist_deg, _ = tree.query(dst_points)
    dist_km = (dist_deg * 111.0).reshape(dst_lat.shape)

    taper = np.clip((taper_end_km - dist_km) / (taper_end_km - taper_start_km), 0.0, 1.0)
    return taper


def _weighted_fuse(sources: dict, base_weights: dict) -> tuple[np.ndarray, dict]:
    """Per-pixel weighted combination of however many of `sources` are
    actually valid (finite) at that pixel, using base_weights as PRIORITY
    weights renormalized to the sources present -- e.g. if radar has the
    highest base weight but is NaN (no coverage) at a pixel, that pixel's
    weight redistributes to whichever of mw/goes are present there,
    rather than that priority being silently lost or, worse, radar's
    absence being treated as radar=0 (very different from "no data").

    sources: {key: array_or_None, ...} (all NaN-masked where invalid).
        Generic over key names -- used both for the original 3-way
        {"goes","mw","radar"} response-level fusion and the 2-way
        {"backbone","mw"} value-level fusion. At least one entry must be
        a fully-valid (no-NaN) array to establish the output shape and
        guarantee every pixel gets SOME value.
    base_weights: {key:.., ...} priority weights, same key names as sources.
        Each value can be a scalar (uniform priority everywhere) OR an
        array the same shape as the sources (a spatially-varying
        confidence, e.g. tapering a source's influence near the edge of
        its coverage instead of a hard cutoff -- see
        _regrid_confidence_taper / _edge_feather_taper).

    Returns (fused_array, coverage_fractions) where coverage_fractions
    reports what fraction of the grid each source actually contributed to
    (for diagnostics/transparency in the GUI, not used in the math).
    """
    shape = next(arr.shape for arr in sources.values() if arr is not None)
    total_weight = np.zeros(shape)
    weighted_sum = np.zeros(shape)
    coverage = {}

    for key, arr in sources.items():
        if arr is None:
            coverage[key] = 0.0
            continue
        w = np.broadcast_to(base_weights.get(key, 0.0), shape)
        valid = np.isfinite(arr)
        coverage[key] = float(valid.mean())
        weighted_sum[valid] += w[valid] * arr[valid]
        total_weight[valid] += w[valid]

    # If NO source was valid at a pixel there is no defensible value to
    # emit. This previously substituted a weight of 1.0 against a
    # weighted_sum of 0, producing 0.0 -- which is finite, so the
    # assert_finite guard at the end of generation would not catch it,
    # and 0 K renders as fully saturated red in both colour tables. A
    # silent, confident, physically impossible blob is the worst
    # available failure mode. NaN instead, so the existing guard fires
    # with a clear message if this ever actually happens. ("goes" and
    # "backbone" are each meant to be fully valid everywhere in their
    # respective fusions, so this should stay unreachable.)
    no_source = total_weight == 0
    safe_total = np.where(no_source, 1.0, total_weight)
    fused = weighted_sum / safe_total
    if np.any(no_source):
        fused = np.where(no_source, np.nan, fused)
    return fused, coverage


_WVIR_HISTORY = {}
_WVIR_HISTORY_MAX = 12


def _wvir_history_for(storm_id):
    """Recent WVIR results for one storm, oldest first.

    Several of Sanabia et al.'s stages are defined by CHANGE rather than
    by a snapshot -- outer erosion, inner decay and contraction all need
    a before -- so a single frame cannot distinguish them. Held in
    process rather than on disk: a GUI session walks frames in order,
    which is what this needs, and persisting it would mean reasoning
    about staleness across runs for little gain.
    """
    return _WVIR_HISTORY.get(storm_id)


def _wvir_remember(storm_id, result):
    h = _WVIR_HISTORY.setdefault(storm_id, [])
    h.append(result)
    if len(h) > _WVIR_HISTORY_MAX:
        del h[0]


def generate_synthetic_mw(
    band13: BandImage,
    band9: BandImage,
    band7: BandImage,
    storm_fix: StormFix,
    band2: Optional[BandImage] = None,
    real_swath=None,
    radar_lat: Optional[np.ndarray] = None,
    radar_lon: Optional[np.ndarray] = None,
    radar_dbz: Optional[np.ndarray] = None,
    radar_site_lat: Optional[float] = None,
    radar_site_lon: Optional[float] = None,
    echo_top_lat: Optional[np.ndarray] = None,
    echo_top_lon: Optional[np.ndarray] = None,
    echo_top_km: Optional[np.ndarray] = None,
    fusion_weights: Optional[dict] = None,
    mw_age_confidence: float = 1.0,
    real_swath_after=None,
    mw_confidence_after: float = 0.0,
    smoothing_sigma: float = 1.2,
    ml_strength: Optional[float] = None,
    extra_ir: Optional[dict] = None,
    flash_density: Optional[np.ndarray] = None,
    progress_callback=None,
) -> SyntheticMWResult:
    """Produce synthetic 37 GHz and 89 GHz Tb fields on band13's grid.

    band13, band9, band7: co-temporal GOES ABI BandImage objects (band13's
        grid is used as the output grid; band9/band7 are regridded onto it).
    storm_fix: best-track fix (ideally time-interpolated) giving the storm
        center used for the radial vortex structure model.
    band2: optional visible-band image for a modest daytime sharpening term.
        Ignored (with a diagnostics note) if it's nighttime at the storm's
        location/time, even if provided -- night-side Band 2 is just
        sensor noise near zero reflectance, not signal.

    real_swath: optional MWSwath (mw_ingest.py) -- if given, its V/H data
        is converted to a comparable 0-1 "response" index and FUSED with
        the GOES-derived response (not applied as a bias correction after
        the fact). Per-frequency: 37 GHz real data informs the 37 GHz
        fusion, 89 GHz real data informs 89 GHz, separately.
    radar_lat, radar_lon, radar_dbz: optional gridded NEXRAD reflectivity
        (radar_ingest.py) -- if given, also fused in, informing BOTH
        frequencies (reflectivity doesn't carry a frequency distinction
        the way V/H brightness temperature does).
    radar_site_lat, radar_site_lon: the radar station's own location (not
        the gate data) -- used to taper radar's confidence down near the
        edge of its useful range (real NEXRAD reliability genuinely
        degrades with range; see _regrid_confidence_taper). If omitted
        while radar data is provided, radar gets full weight everywhere
        it has coverage regardless of range -- fine close to the radar,
        risks exactly the isolated-anomaly issue this taper exists to fix
        for long-range data, so pass this whenever you have it.
    echo_top_lat, echo_top_lon, echo_top_km: optional gridded NEXRAD echo-
        top height (radar_ingest.get_echo_top_gates_near_storm) -- a
        DIFFERENT convective-intensity signal from base reflectivity
        (spots overshooting tops/VHTs: strong vertical extent even where
        low-level reflectivity looks unremarkable, or vice versa). Not
        additive with reflectivity's index -- combined via max() per
        pixel, since either signal independently indicates genuine
        intense convection and averaging them would dilute whichever one
        is actually the more diagnostic at that location.
    fusion_weights: override DEFAULT_FUSION_WEIGHTS ({"goes":0.1,"mw":0.3,
        "radar":0.6}) if you want different priority weighting. These are
        PRIORITY weights, renormalized per-pixel to whichever sources
        actually have data there -- see _weighted_fuse's docstring for why
        this isn't the same as "GOES only contributes 10% everywhere."
    mw_age_confidence: 0-1 multiplier on real_swath's weight in the final
        blend, for a MIMIC-TC-style "advected/morphed" real pass (see
        mw_ingest.morph_swath_to_time, _morph_age_confidence) -- 1.0 for
        a fresh, un-morphed pass (the default; most callers don't need to
        touch this). When real_swath has been repositioned from an older
        observation to align with the storm's current location, pass the
        age-based confidence here so an aging advected pass gradually
        cedes weight back to the GOES/radar backbone rather than being
        trusted exactly as much as a live pass forever.
    real_swath_after, mw_confidence_after: a SECOND real MW pass, one
        that occurred AFTER target_time (already morphed backward to
        target_time's position by the caller, same as real_swath is
        morphed forward) -- for MIMIC-TC-style crossfading between a
        "before" pass and an "after" pass, rather than only ever
        extrapolating forward from the most recent past pass. Valid only
        for historical/archived generation, where a later pass has
        already happened and its data already exists to fetch.
        mw_confidence_after should come from _crossfade_confidence_toward_after
        (ramping 0->1 over the 60 minutes leading up to the after-pass's
        own time), NOT _morph_age_confidence (which is for aging passes
        fading OUT, not upcoming passes fading IN). Both real_swath and
        real_swath_after can be given at once -- the fusion blends
        backbone/before-pass/after-pass as three weighted sources, not
        just two. Deliberately scoped down relative to real_swath: the
        after-pass only feeds the direct-value V/H blend, not the
        response-index/backbone-shaping computation, so it smooths the
        handoff to the next real pass without reshaping the overall
        storm structure model.
    """
    import solar

    is_daytime = solar.is_daytime(storm_fix.lat, storm_fix.lon, band13.scene_time)
    band2_used = band2 is not None and is_daytime
    if band2 is not None and not is_daytime:
        band2 = None  # discard rather than silently use night-side noise

    # QC step: sanitize band13 (the output grid) up front even though
    # goes_fetch already does this on ingestion -- cheap insurance against
    # any non-finite values slipping through (e.g. if BandImage objects
    # were constructed some other way than get_band_image()).
    ir_tb, _ = sanitize_field(band13.values)
    lat, _ = sanitize_field(band13.lat)
    lon, _ = sanitize_field(band13.lon)

    wv_tb = _regrid_to(lat, lon, band9)
    swir_tb = _regrid_to(lat, lon, band7)

    scat_pot = _scattering_potential(ir_tb)
    wv_mask = _wv_depth_mask(wv_tb, ir_tb)
    texture = _texture_signal(swir_tb, ir_tb)

    # SEPARATE, narrower/colder-onset scattering indicator used ONLY for
    # the V/H emission-vs-depression split below (NOT for convective_signal/
    # response, which keeps using the original wider-range scat_pot above,
    # unchanged). This exists to fix a real, mathematically-confirmed bug:
    # with a single shared scat_pot driving BOTH the overall response
    # magnitude AND the emission/depression split, the two are ~85%
    # correlated by construction (convective_signal is itself dominated by
    # scat_pot terms), so there is no regime where "moderate rain/response
    # is happening" without scattering ALSO being elevated -- meaning the
    # emission term's (1-scat_pot) factor gets suppressed almost exactly
    # where it would otherwise matter. Traced through the actual formula:
    # H37's B-channel contribution (cyan) never exceeded ~39/255 anywhere
    # across the ENTIRE response range with the coupled version -- cyan
    # was structurally unreachable, not just poorly tuned, matching a
    # direct real-data complaint (comparing the program's own GOES-only
    # output against real AMSR2 color37/color89 for the same storm: real
    # imagery shows a rich green->cyan->pink->red progression, GOES-only
    # output showed almost only green and red).
    #
    # Using a MUCH colder warm_bound (230K vs. the 273K used for overall
    # response) means moderate response levels (warm rain, general
    # convection -- physically real and common, not exotic) correspond to
    # scat_pot_vh near 0, letting the emission term fully warm H/V and
    # sustain a genuine cyan-producing plateau, while only the most
    # extreme, coldest pixels (genuine overshooting tops) trigger real
    # depression. Verified directly: this raises the sustained peak
    # B-channel contribution from ~39/255 to ~92/255 across a wide
    # (not narrow/instantaneous) portion of the response range.
    # PER-FREQUENCY as of 0.95. This used to be a single shared field, and
    # sharing it was wrong physics for 37 GHz.
    #
    # Validated against a real GPM GMI pass over Lowell (EP12) 44 minutes
    # before a GOES-only frame, with the IR centre check confirming the
    # synthetic centre to 2 km. Reading 37H off the GMI colour bar:
    #
    #     far-field ocean   182-193 K
    #     eyewall / core    269 K  (p10 257, p90 276)
    #
    # Real 37H RISES about 80 K from ocean to eyewall. The synthetic, with
    # the shared scattering fraction, produced 170 K -> 135 K: a FALL of
    # 35 K. The sign was inverted and the eyewall was wrong by ~134 K.
    #
    # The cause is physical, not a tuning error. At 89 GHz, ice scattering
    # dominates deep convection and depresses Tb hard -- the shared
    # formulation is right there. At 37 GHz, ice scattering is weak; the
    # eyewall signature is EMISSION from liquid precipitation, which warms
    # both polarizations toward the physical temperature and collapses the
    # polarization difference. Driving scat_pot toward 1 in cold IR made
    # the 37 GHz eyewall a scattering feature, which it is not.
    #
    # So 37 GHz now uses a much colder bound: only genuine overshooting
    # tops scatter, and everything short of that emits.
    def _scat_pot(warm_bound):
        return np.clip(
            (warm_bound - ir_tb) / (warm_bound - CALIBRATION["ir_cold_bound"]),
            0.0, 1.0,
        )

    scat_pot_vh_37 = _scat_pot(CALIBRATION["vh_scatter_warm_bound_37"])
    scat_pot_vh_89 = _scat_pot(CALIBRATION["vh_scatter_warm_bound_89"])
    scat_pot_vh = scat_pot_vh_89  # retained for any legacy reference

    # Combined convective-likelihood field.
    #
    # REWEIGHTED IN 0.99 so the WV mask GATES the response rather than
    # merely adding to it. The old form was
    #     0.55*scat_pot + 0.30*wv_mask*scat_pot + 0.15*texture
    # in which the 0.55 base term was ungated: a thick cirrus canopy, cold
    # in IR but with wv_mask near 0, still scored about 0.5. Cirrus is
    # largely TRANSPARENT at 37/89 GHz -- Li et al. (2026) find its PMW
    # signal is dominated by background emission, essentially
    # indistinguishable from clear sky -- so rendering it at half response
    # puts convection where the microwave sees none.
    #
    # That is very likely the mechanism behind a measured discrepancy:
    # against a real GMI pass the synthetic 89 GHz core came out about
    # 3.4x the observed area and contiguous, where the real signature was
    # a compact eyewall ring plus discrete spiral bands. A cold CDO is
    # mostly cirrus canopy; treating the whole canopy as convective
    # inflates exactly that way.
    #
    # Now (0.25 + 0.60*wv_mask) * scat_pot + 0.15*texture. Deep convection
    # (wv_mask ~ 1) is essentially unchanged at 0.85*scat_pot, while cirrus
    # (wv_mask ~ 0) drops from ~0.50 to ~0.22. Coefficients still sum to
    # exactly 1.0, so nothing clips.
    convective_signal = np.clip(
        (0.25 + 0.60 * wv_mask) * scat_pot + 0.15 * texture, 0, 1
    )

    if band2 is not None:
        vis = _regrid_to(lat, lon, band2)
        vis_grad = np.hypot(*np.gradient(vis))
        if vis_grad.max() > 0:
            vis_grad = vis_grad / np.percentile(vis_grad, 99.0).clip(min=1e-6)
        # Renormalized by the new coefficient total rather than added on
        # top and clipped. The three IR-family coefficients above already
        # sum to exactly 1.0, so adding a further 0.08 and clipping meant
        # every pixel that was already above 0.92 was flattened to 1.0 --
        # the same saturation pattern that was destroying the eyewall
        # ring in _radial_weight, in miniature. Small in practice (this
        # field peaks around 0.8 on real frames), but it silently removed
        # contrast from the most convective pixels, which are the ones
        # the field exists to identify.
        convective_signal = np.clip(
            (convective_signal + 0.08 * np.clip(vis_grad, 0, 1)) / 1.08, 0, 1
        )
    else:
        vis = None

    rmw_km = storm_fix.rmw_nm * 1.852 if storm_fix.rmw_nm else _willoughby_rmw_km(
        storm_fix.vmax_kt, storm_fix.lat
    )
    roci_km = storm_fix.roci_nm * 1.852 if storm_fix.roci_nm else None
    radial_w, radial_dist_km, _envelope_scale_km = _radial_weight(
        lat, lon, storm_fix.lat, storm_fix.lon, rmw_km, storm_fix.vmax_kt, roci_km=roci_km,
        return_geometry=True,
    )

    # --- Parallax (0.113) ---------------------------------------------
    # GOES sees a 12-16 km cloud top displaced away from the subsatellite
    # point by h*tan(zenith) -- 4-6 km for the east Pacific cases here,
    # 8-13 km in the western Atlantic and toward the limb. The MW target
    # does not share that displacement, so without this the correction
    # model is being asked to learn a coordinate transform on top of the
    # physics, from a few hundred examples.
    #
    # Corrected on the convective signal rather than the raw IR, so a
    # single resample carries every IR-derived input, and per-pixel by
    # height -- a warm eye and the cold tops around it are displaced by
    # very different amounts, which is precisely the structure at stake.
    try:
        import parallax as _px
        import goes_fetch as _gf
        _sat_lon = _gf.SUBSATELLITE_LON.get(getattr(band13, "satellite", None))
        if _sat_lon is not None:
            convective_signal = np.clip(
                _px.correct_field(convective_signal, lat, lon, ir_tb, _sat_lon), 0.0, 1.0)
            if progress_callback:
                progress_callback(_px.describe(lat, lon, ir_tb, _sat_lon))
    except Exception as e:
        if progress_callback:
            progress_callback(f"Parallax correction skipped ({type(e).__name__}: {e}).")

    goes_response = convective_signal * radial_w

    # --- Multi-source fusion: GOES + real MW + radar as weighted equals ---
    weights = fusion_weights or DEFAULT_FUSION_WEIGHTS

    mw_response_37 = mw_response_89 = None
    mw_v37_clean = mw_h37_clean = mw_v89_clean = mw_h89_clean = None
    if real_swath is not None:
        import mw_composites

        # QC pass BEFORE any downstream use -- see MW_PHYSICAL_BOUNDS'
        # comment for why. Every use of real_swath's V/H below reads from
        # these cleaned copies, not the raw swath, so both the response-
        # index computation and the direct-value fusion benefit uniformly.
        mw_v37_clean = _apply_physical_bounds(real_swath.v37, MW_PHYSICAL_BOUNDS["v37"])
        mw_h37_clean = _apply_physical_bounds(real_swath.h37, MW_PHYSICAL_BOUNDS["h37"])
        mw_v89_clean = _apply_physical_bounds(real_swath.v89, MW_PHYSICAL_BOUNDS["v89"])
        mw_h89_clean = _apply_physical_bounds(real_swath.h89, MW_PHYSICAL_BOUNDS["h89"])
        # Despeckle AFTER physical-bounds (catches a DIFFERENT class of
        # bad pixel -- individually plausible values that are still
        # spatially inconsistent with their neighbors -- see
        # _despeckle_mw_field's docstring for the real artifact that
        # motivated this).
        mw_v37_clean = _despeckle_mw_field(mw_v37_clean)
        mw_h37_clean = _despeckle_mw_field(mw_h37_clean)
        mw_v89_clean = _despeckle_mw_field(mw_v89_clean)
        mw_h89_clean = _despeckle_mw_field(mw_h89_clean)

        pct37_real = mw_composites.compute_pct(mw_v37_clean, mw_h37_clean, 37)
        idx37_real = _external_scattering_index(pct37_real, warm_ref=245.0, cold_ref=190.0)
        lat37_src, lon37_src = real_swath.grid_for(37)
        mw_response_37 = _regrid_external_with_mask(
            lat37_src, lon37_src, idx37_real, lat, lon, MW_FUSION_MAX_DISTANCE_KM
        )

        pct89_real = mw_composites.compute_pct(mw_v89_clean, mw_h89_clean, 89)
        idx89_real = _external_scattering_index(pct89_real, warm_ref=265.0, cold_ref=170.0)
        lat89_src, lon89_src = real_swath.grid_for(89)
        mw_response_89 = _regrid_external_with_mask(
            lat89_src, lon89_src, idx89_real, lat, lon, MW_FUSION_MAX_DISTANCE_KM
        )

    radar_response = None
    if radar_lat is not None and radar_dbz is not None:
        idx_radar = np.clip((np.asarray(radar_dbz) - 20.0) / 35.0, 0.0, 1.0)  # 20dBZ->0, 55dBZ->1
        radar_response = _regrid_external_with_mask(
            radar_lat, radar_lon, idx_radar, lat, lon, RADAR_FUSION_MAX_DISTANCE_KM, method="linear"
        )

    if echo_top_lat is not None and echo_top_km is not None:
        # 10km -> no signal, 18km -> max signal: roughly where echo tops
        # start meaningfully approaching/penetrating the tropopause,
        # indicating genuinely vigorous updrafts (overshooting tops/VHTs)
        # rather than just an ordinary deep-but-unremarkable convective cell.
        idx_echo_top = np.clip((np.asarray(echo_top_km) - 10.0) / 8.0, 0.0, 1.0)
        echo_top_response = _regrid_external_with_mask(
            echo_top_lat, echo_top_lon, idx_echo_top, lat, lon, RADAR_FUSION_MAX_DISTANCE_KM, method="linear"
        )
        if radar_response is None:
            radar_response = echo_top_response
        else:
            # Combine via max, not average: either signal alone (strong
            # low-level reflectivity OR a genuinely tall echo top)
            # independently indicates intense convection -- averaging
            # would dilute whichever one is actually the more diagnostic
            # signal at a given pixel. NaN-aware: a pixel with only one of
            # the two sources present should just use that one, not NaN.
            radar_response = np.fmax(radar_response, echo_top_response)

    if radar_response is not None and radar_site_lat is not None and radar_site_lon is not None:
        # Taper radar's confidence down near the edge of its useful range
        # -- see _regrid_confidence_taper's docstring. Full weight out to
        # 60% of the fusion radius, tapering to zero at the radius itself
        # (100km / 150km with the default RADAR_FUSION_MAX_DISTANCE_KM).
        confidence = _regrid_confidence_taper(
            radar_site_lat, radar_site_lon, lat, lon,
            taper_start_km=RADAR_FUSION_MAX_DISTANCE_KM * 0.6,
            taper_end_km=RADAR_FUSION_MAX_DISTANCE_KM,
        )
        radar_response = radar_response * confidence

    fused_response_37, coverage_37 = _weighted_fuse(
        {"goes": goes_response, "mw": mw_response_37, "radar": radar_response}, weights
    )
    fused_response_89, coverage_89 = _weighted_fuse(
        {"goes": goes_response, "mw": mw_response_89, "radar": radar_response}, weights
    )

    cal = CALIBRATION
    # Scalar (tb37/tb89): UNCHANGED index-blend-then-reconstruct approach,
    # kept simple/consistent since mw_compare.py's real-vs-synthetic stats
    # compare against this scalar specifically -- see the V/H section below
    # for the actual fix to the "looks like colorized IR" complaint.
    emission_term = cal["emission_boost_37"] * fused_response_37 * (1 - scat_pot)
    depression_37 = cal["max_depression_37"] * fused_response_37 * scat_pot
    tb37 = cal["bg_tb_37"] + emission_term - depression_37

    depression_89 = cal["max_depression_89"] * fused_response_89
    tb89 = cal["bg_tb_89"] - depression_89

    tb37 = gaussian_filter(tb37, sigma=smoothing_sigma)
    tb89 = gaussian_filter(tb89, sigma=smoothing_sigma)

    # --- V/H: DIRECT-VALUE fusion, not index-blend-then-reconstruct ---
    #
    # This is the actual fix for "the MW-only view looks like a new color
    # table applied to a satellite image instead of a real MW pass": the
    # OLD approach reduced real MW's actual measured V/H down to an
    # abstract 0-1 "response index," fused THAT index with GOES's index,
    # then reconstructed v37/h37/v89/h89 through the SAME smooth parametric
    # formula regardless of source. Structurally, the output was ALWAYS
    # just the GOES vortex model shape, nudged by a summary number from
    # real data -- never actually informed by real MW's genuine spatial
    # texture/noise, no matter how much nominal "weight" it had.
    #
    # Fixed by building three CANDIDATE V/H value fields (one per source)
    # and fusing the actual Kelvin values directly with the same priority-
    # weighted mechanism:
    #   - "goes": the parametric formula (as before), smoothed -- this is
    #     the clean fallback wherever real data doesn't reach.
    #   - "mw": real_swath's ACTUAL v37/h37/v89/h89, regridded but NOT
    #     smoothed -- preserving real MW's genuine texture is the entire
    #     point of this change.
    #   - "radar": radar doesn't measure V/H directly, so this is still a
    #     model estimate (same formula, driven by radar_response instead
    #     of goes_response) -- but radar's own spatial detail (individual
    #     cells, not just a smooth envelope) still comes through via
    #     radar_response's texture before the formula is applied.
    #
    # IMPORTANT FIX: goes and radar are merged into ONE combined "backbone"
    # response BEFORE the parametric V/H formula is applied, rather than
    # being two separate candidates that compete directly against MW in
    # the final blend. A real run caught why the separate-candidates
    # version was wrong: wherever radar covered a pixel, its priority
    # weight (0.6) directly outweighed MW's real texture (0.3) in the
    # final blend, making MW's actual contribution collapse from 75%
    # (goes+mw only, no radar) down to 30% (goes+mw+radar) purely because
    # radar happened to also reach that pixel -- radar has no real texture
    # of its own to justify outweighing MW's genuine measured detail, it's
    # still just a reshaped parametric guess. This showed up as a visible
    # "bubble" -- a smoother, differently-textured patch with a crisp
    # circular edge exactly at radar's coverage radius, sitting inside an
    # otherwise MW-textured field. Fixed by merging radar into the
    # backbone first, so MW's weight relative to the backbone stays
    # constant (0.3 vs 0.7) regardless of whether radar also happens to
    # cover a given pixel -- radar still fully shapes the backbone's
    # structure/intensity wherever it has data, it just doesn't separately
    # out-compete MW's real texture on top of that.
    # NOTE: the 37 and 89 GHz backbone responses are built from the same
    # sources with the same weights, so they are identical by
    # construction. Computed once and shared rather than twice.
    #
    # Worth being explicit that this is a modelling simplification, not
    # an oversight: all frequency dependence in the backbone lives in the
    # CALIBRATION constants (emission/depression magnitudes and V/H
    # baselines), none of it in the spatial response. Real 37 and 89 GHz
    # do differ spatially -- 89 GHz responds to smaller ice particles and
    # resolves finer structure, 37 GHz weights liquid emission more --
    # so a genuinely frequency-dependent response would be a real
    # improvement. Until then, computing the same array twice was pure
    # duplicated work.
    backbone_response, _ = _weighted_fuse(
        {"goes": goes_response, "radar": radar_response},
        {"goes": weights.get("goes", 0.1), "radar": weights.get("radar", 0.6)},
    )
    # Shape the shared response per frequency (mw_surface). 89 GHz sees
    # ICE above the freezing level -- scattering is threshold-like, so the
    # response is concentrated on the deepest convection. 37 GHz sees
    # LIQUID below it -- a deep, diffuse layer extending well into
    # stratiform rain -- so it is broadened and smoothed.
    #
    # Until 0.114 these were literally the same array: the 0.112 PSF gave
    # the two frequencies different RESOLUTION, but their underlying
    # hydrometeor response was identical, leaving every real difference to
    # the calibration constants.
    #
    # An earlier attempt at this in-lined a gamma-plus-envelope shaper
    # here that referenced four CALIBRATION keys which were never added
    # (response_gamma_37, envelope_scale_37, ...). It would have raised
    # KeyError on the first generate. Replaced with the tested module
    # rather than patched, so there is one implementation with coverage
    # behind it instead of two.
    try:
        import mw_surface
        backbone_response_37 = mw_surface.frequency_response(backbone_response, 37, lat, lon)
        backbone_response_89 = mw_surface.frequency_response(backbone_response, 89, lat, lon)
    except Exception:
        backbone_response_37 = backbone_response_89 = backbone_response

    # --- Surface-aware backgrounds ------------------------------------
    # Blend the ocean and land background constants by land fraction
    # before the emission/scattering terms are applied. Smoothed with the
    # same sigma as the fields themselves so a coastline does not appear
    # as a one-pixel step, which would read as an artifact rather than as
    # a shoreline.
    try:
        import surface_type
        land_frac = surface_type.land_fraction(lat, lon, progress_callback=progress_callback)
        # Report WHICH backend answered. surface_type.backend_name() has
        # existed since 0.97 and was never called, so a run gave no way to
        # tell an active land mask from the all-ocean fallback -- and the
        # fallback's warning fires once per process, easy to lose in a
        # long log. Over land that difference is the whole point: ocean
        # emissivity there mimics heavy precipitation.
        if progress_callback:
            _bk = surface_type.backend_name()
            if _bk == "none":
                progress_callback("Land mask: NONE (all-ocean assumed) -- "
                                  "install global-land-mask if land is in view.")
            elif np.any(land_frac > 0.01):
                progress_callback(f"Land mask [{_bk}]: land covers "
                                  f"{100*float(np.mean(land_frac > 0.5)):.1f}% of the grid.")
        if np.any(land_frac > 0):
            land_frac = gaussian_filter(land_frac.astype(np.float64), sigma=smoothing_sigma)
        land_frac = np.clip(land_frac, 0.0, 1.0)
        _surface_elevation = surface_type.surface_elevation_m(lat, lon, land_frac)
    except Exception:
        land_frac = np.zeros_like(ir_tb, dtype=float)
        _surface_elevation = np.zeros_like(ir_tb, dtype=float)

    def _bg(name):
        """Ocean/land blended background for a CALIBRATION key."""
        ocean = cal[name]
        land = cal.get(f"{name}_land", ocean)
        return ocean * (1.0 - land_frac) + land * land_frac

    bg_v_37_f, bg_h_37_f = _bg("bg_v_37"), _bg("bg_h_37")
    bg_v_89_f, bg_h_89_f = _bg("bg_v_89"), _bg("bg_h_89")

    # --- Wind roughening (0.114) --------------------------------------
    # A TC is a wind field, and wind raises ocean emissivity sharply at
    # H-pol. Real GMI 37H in Lowell's outer region measured 182-193 K
    # against a calm-ocean 150-155 K; that difference is the storm's own
    # wind. With uniform backgrounds it had been absorbed into the
    # constant, which made it right at one radius and wrong at every
    # other. Now the background rises toward the core as it should.
    try:
        import mw_surface

        # --- Atmospheric emission anomaly (0.132) ---------------------
        # The background constants carry a MEAN atmospheric contribution
        # (~25 K at 37H over tropical ocean) folded into a fixed number,
        # so it could not vary with the moisture field. That is a
        # systematic, spatially-varying error sitting in a constant, and
        # it matches the symptom exactly: bias is 76% of the error budget
        # and did not move between 298 and 799 training examples.
        #
        # Driven by band 9 (mid-level WV), which is always fetched. Li et
        # al.'s saliency makes low-level WV a dominant predictor; band 10
        # or the 13-15 split window would be better proxies still, but
        # both are optional here and band 9 is guaranteed.
        _atm = lambda f, p: mw_surface.atmospheric_anomaly_k(wv_tb, f, p, land_frac)
        bg_v_37_f = bg_v_37_f + _atm(37, "v")
        bg_h_37_f = bg_h_37_f + _atm(37, "h")
        bg_v_89_f = bg_v_89_f + _atm(89, "v")
        bg_h_89_f = bg_h_89_f + _atm(89, "h")

        _r_km = _distance_km_from(lat, lon, storm_fix.lat, storm_fix.lon)
        _wind = mw_surface.surface_wind_ms(_r_km, storm_fix.vmax_kt, rmw_km)
        bg_v_37_f = bg_v_37_f + mw_surface.roughening_k(_wind, 37, "v", land_frac)
        bg_h_37_f = bg_h_37_f + mw_surface.roughening_k(_wind, 37, "h", land_frac)
        bg_v_89_f = bg_v_89_f + mw_surface.roughening_k(_wind, 89, "v", land_frac)
        bg_h_89_f = bg_h_89_f + mw_surface.roughening_k(_wind, 89, "h", land_frac)
        if progress_callback:
            progress_callback(
                f"Wind roughening: {_wind.max():.0f} m/s peak -> 37H background "
                f"{bg_h_37_f.min():.0f}-{bg_h_37_f.max():.0f} K")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Wind roughening skipped ({type(e).__name__}: {e}).")

    # --- Optical-depth saturation (0.133) -----------------------------
    # Emission and scattering saturate as the layer becomes optically
    # thick; a linear ramp calibrated to be right at full response is too
    # weak everywhere below it, which is a one-signed error over the bulk
    # of every frame. Normalized so the endpoints -- and therefore the
    # existing constants -- are unchanged.
    import mw_surface
    _sat37 = mw_surface.saturate(backbone_response_37)
    _sat89 = mw_surface.saturate(backbone_response_89)

    v37_backbone = bg_v_37_f + cal["emission_boost_v37"] * _sat37 * (1 - scat_pot_vh_37) - cal["max_depression_v37"] * _sat37 * scat_pot_vh_37
    h37_backbone = bg_h_37_f + cal["emission_boost_h37"] * _sat37 * (1 - scat_pot_vh_37) - cal["max_depression_h37"] * _sat37 * scat_pot_vh_37
    # 89 GHz now splits emission from scattering, exactly as 37 GHz does.
    # Ice depression is additionally scaled by latitude: the freezing
    # level drops poleward, so the same cloud top implies less ice aloft
    # at 35 degrees than in the deep tropics.
    _ice = mw_surface.ice_depth_factor(lat)
    v89_backbone = (bg_v_89_f
                    + cal["emission_boost_v89"] * _sat89 * (1 - scat_pot_vh_89)
                    - cal["max_depression_v89"] * _sat89 * scat_pot_vh_89 * _ice)
    h89_backbone = (bg_h_89_f
                    + cal["emission_boost_h89"] * _sat89 * (1 - scat_pot_vh_89)
                    - cal["max_depression_h89"] * _sat89 * scat_pot_vh_89 * _ice)

    # Backbone always fully covers the grid (goes_response has no NaN, and
    # _weighted_fuse guarantees full coverage wherever "goes" is valid) --
    # safe to smooth normally, no NaN-boundary concerns like the old
    # radar-only candidate had.
    v37_backbone = gaussian_filter(v37_backbone, sigma=smoothing_sigma)
    h37_backbone = gaussian_filter(h37_backbone, sigma=smoothing_sigma)
    v89_backbone = gaussian_filter(v89_backbone, sigma=smoothing_sigma)
    h89_backbone = gaussian_filter(h89_backbone, sigma=smoothing_sigma)

    # --- ML correction, ALWAYS applied (per direct guidance -- not a
    # toggle), to the BACKBONE specifically, before the real-MW fusion
    # blend below. This is deliberate: the existing fusion already
    # weights backbone-vs-real-MW by confidence (fresh/high-confidence
    # real MW dominates; aging or absent real MW lets the backbone carry
    # more of the final result) -- correcting the backbone itself means
    # that SAME confidence weighting automatically makes the ML
    # correction's influence on the FINAL output inversely related to
    # real-MW confidence too, with no new blending logic needed here.
    # Degrades completely gracefully (see ml_inference.py) to a no-op
    # when no trained checkpoint exists yet -- the current real state --
    # so this is safe to leave unconditionally wired in.
    # Regrid the supplementary IR bands onto the working grid once, and
    # keep them: the ML step needs them now, and the training exporter
    # needs exactly these arrays later. Regridding twice would risk the
    # two drifting apart.
    _extra_ir_grids = {}
    for _b, _img in (extra_ir or {}).items():
        if _img is None:
            continue
        try:
            _extra_ir_grids[_b] = _regrid_to(lat, lon, _img)
        except Exception as _e:
            # Do NOT swallow this. A band that fails to regrid silently
            # becomes a neutral plane, so the model quietly loses an input
            # it was trained with and nothing says so -- exactly the kind
            # of degradation that shows up later as unexplained accuracy
            # loss rather than as an error.
            if progress_callback:
                progress_callback(f"  extra IR band {_b} failed to regrid "
                                  f"({type(_e).__name__}) -- channel will be neutral.")

    ml_correction_applied = False
    ml_stats: dict = {}
    # Per-channel record of exactly what the ML step added, kept so the
    # training exporter can subtract it back out (see the backbone_*
    # diagnostics below). Zero when no correction was applied.
    ml_delta = {"v37": 0.0, "h37": 0.0, "v89": 0.0, "h89": 0.0}
    try:
        import ml_inference

        _pre_ml = (v37_backbone.copy(), h37_backbone.copy(), v89_backbone.copy(), h89_backbone.copy())
        v37_backbone, h37_backbone, v89_backbone, h89_backbone = ml_inference.apply_ml_correction(
            ir_tb, wv_tb, swir_tb,
            v37_backbone, h37_backbone, v89_backbone, h89_backbone,
            lat, lon,
            storm_lat=storm_fix.lat, storm_lon=storm_fix.lon,
            storm_vmax_kt=storm_fix.vmax_kt,
            storm_rmw_nm=storm_fix.rmw_nm,
            # Same RMW the parametric backbone above was actually built
            # with, converted back to nm, so the model isn't told
            # "average storm" while sitting on top of a backbone shaped
            # by a Willoughby estimate.
            fallback_rmw_nm=rmw_km / 1.852,
            # Extra ABI bands (ml_constants.EXTRA_IR_BANDS) plus surface
            # conditioning. All optional: absent inputs become neutral
            # planes, so the channel count never changes.
            extra_ir=_extra_ir_grids,
            land_fraction=land_frac,
            elevation_m=_surface_elevation,
            flash_density=flash_density,
            strength=ml_strength,
            stats_out=ml_stats,
            progress_callback=progress_callback,
        )
        ml_delta = {
            "v37": v37_backbone - _pre_ml[0],
            "h37": h37_backbone - _pre_ml[1],
            "v89": v89_backbone - _pre_ml[2],
            "h89": h89_backbone - _pre_ml[3],
        }
        ml_correction_applied = True  # a checkpoint was found and applied without error --
        # doesn't distinguish "genuinely corrected" from "corrected but the
        # patch fell entirely outside the grid," both count as "attempted
        # successfully" here; ml_inference's own progress_callback messages
        # carry the more specific detail when something is available to log to.
    except Exception:
        pass  # import failure or anything else -- same graceful no-op as ml_inference's own internal fallback

    # --- Inject REAL GOES-derived spatial texture into the backbone ---
    #
    # Without this, the backbone above is a smooth parametric radial
    # model EVERYWHERE -- fine wherever real MW data dominates the final
    # blend, but visually jarring at the boundary where MW coverage ends
    # and the backbone alone takes over: real MW is naturally grainy,
    # the backbone was perfectly smooth, so even a well-feathered WEIGHT
    # transition still reads as "two different things touching," because
    # the two regions have fundamentally different visual/statistical
    # character, not just a soft edge between them.
    #
    # Uses _multispectral_texture_field (all 4 GOES bands: IR, WV, SWIR,
    # and VIS when available -- "not just band 13" per the original ask),
    # injected across the WHOLE domain, not just where MW happens to cover.
    #
    # Amplitude: when real MW data is available, MEASURE its actual local
    # texture amplitude directly (per-channel) and use THAT instead of a
    # fixed guessed Kelvin constant -- a three-way real-data comparison
    # (GOES-only vs. GOES+realMW single frame vs. GOES+realMW loop) showed
    # the fixed-amplitude version had genuinely fixed the "flat region"
    # bug (GOES-only: consistent grain everywhere, confirmed) but left a
    # subtler residual seam specifically where real MW was blended in --
    # not flat-vs-textured anymore, just a texture-STYLE mismatch between
    # a fixed guess and this particular pass's actual character. See
    # _measure_real_texture_amplitude's docstring. Falls back to the fixed
    # defaults when no real data is available (that path is unaffected --
    # already verified working via the GOES-only comparison).
    texture_field = _multispectral_texture_field(ir_tb, wv_tb, swir_tb, vis)

    if real_swath is not None:
        measured_v37 = _measure_real_texture_amplitude(mw_v37_clean)
        measured_h37 = _measure_real_texture_amplitude(mw_h37_clean)
        measured_v89 = _measure_real_texture_amplitude(mw_v89_clean)
        measured_h89 = _measure_real_texture_amplitude(mw_h89_clean)
    else:
        measured_v37 = measured_h37 = measured_v89 = measured_h89 = None

    tex_amp_v37 = measured_v37 if measured_v37 is not None else TEXTURE_INJECTION_V37_K
    tex_amp_h37 = measured_h37 if measured_h37 is not None else TEXTURE_INJECTION_H37_K
    tex_amp_v89 = measured_v89 if measured_v89 is not None else TEXTURE_INJECTION_V89_K
    tex_amp_h89 = measured_h89 if measured_h89 is not None else TEXTURE_INJECTION_H89_K

    # --- Baseline COLOR calibration from real data (separate from texture) ---
    #
    # A direct real-storm comparison showed a genuine mismatch even after
    # the texture-amplitude fix above: GOES-only regions at 37 GHz
    # rendered a noticeably more saturated, vivid green than the muted
    # tone seen where real MW was actually blended in -- a baseline
    # Kelvin LEVEL mismatch, not a texture/variance one. Fixed the same
    # way as texture: measure real data's typical background value
    # directly (median, robust to the storm core being a small pixel
    # fraction) and shift the backbone to match, instead of relying
    # purely on the fixed CALIBRATION baseline constants.
    #
    # Clamped (unlike the texture amplitude, which has no clamp) because
    # an UNCLAMPED baseline shift is exactly the mechanism that caused a
    # real saturation bug a few rounds back (a large one-shot correction
    # pushed V/H past the color table's usable range) -- this is
    # deliberately more generous than calibration_state's V/H clamp
    # (+-5K, tuned to be very conservative after that incident) since
    # this is a direct per-pass measurement rather than a compounding
    # EMA correction, but still bounded rather than trusting an arbitrary
    # measured shift completely.
    BASELINE_SHIFT_CLAMP_K = 20.0

    def _clamped_baseline_shift(measured, assumed_baseline):
        if measured is None:
            return 0.0
        return float(np.clip(measured - assumed_baseline, -BASELINE_SHIFT_CLAMP_K, BASELINE_SHIFT_CLAMP_K))

    # Default to zero so the GOES-only path is defined. These are read
    # unconditionally further down when recording the ML-free backbone,
    # and were previously assigned only inside this branch.
    baseline_shift_v37 = baseline_shift_h37 = 0.0
    baseline_shift_v89 = baseline_shift_h89 = 0.0

    # --- M-PERC-style EWRC diagnosis (0.138) --------------------------
    #
    # REAL MICROWAVE ONLY, deliberately. Kossin et al. (2023) are explicit
    # that IR cannot see through the cirrus canopy that covers a TC, which
    # is why M-PERC uses microwave at all -- a secondary eyewall is a
    # convective ring and in IR it is usually hidden. Running this on the
    # synthetic field would be circular: the backbone derives its response
    # from IR, so any ring it shows is one IR already implied.
    # --- WVIR ERC staging (0.139) -------------------------------------
    #
    # GOES-ONLY and runs on every frame, which is the point. The
    # microwave EWRC score needs a real pass and those are hours apart;
    # Sanabia et al. mapped an entire ERC from WVIR radial profiles at
    # geostationary cadence. Microwave anchors, WVIR tracks progression
    # between overpasses.
    _wvir_result = None
    try:
        import wvir as _wvir
        _hist = _wvir_history_for(storm_fix.storm_id) if storm_fix.storm_id else None
        _wvir_result = _wvir.analyze(wv_tb, ir_tb, lat, lon,
                                     storm_fix.lat, storm_fix.lon,
                                     history=_hist, vmax_kt=storm_fix.vmax_kt)
        if storm_fix.storm_id:
            _wvir_remember(storm_fix.storm_id, _wvir_result)
        if progress_callback and _wvir_result.get("stage") is not None:
            for _line in _wvir.format_wvir(_wvir_result).splitlines():
                progress_callback("  " + _line)
    except Exception as _e:
        if progress_callback:
            progress_callback(f"  WVIR staging unavailable ({type(_e).__name__}).")

    _ewrc_result = None
    if real_swath is not None:
        try:
            import ewrc as _ewrc
            _tb89 = getattr(real_swath, "h89", None)
            if _tb89 is None:
                _tb89 = getattr(real_swath, "v89", None)
            if _tb89 is not None:
                _e = _ewrc.ewrc_confidence(
                    _tb89, real_swath.lat, real_swath.lon,
                    storm_fix.lat, storm_fix.lon,
                    vmax_kt=storm_fix.vmax_kt, rmw_km=rmw_km)
                _ewrc_result = _e
                if progress_callback:
                    for _line in _ewrc.format_ewrc(_e).splitlines():
                        progress_callback("  " + _line)
        except Exception as _e:
            if progress_callback:
                progress_callback(f"  EWRC diagnosis unavailable "
                                  f"({type(_e).__name__}).")

    if real_swath is not None:
        baseline_shift_v37 = _clamped_baseline_shift(_measure_real_baseline_value(mw_v37_clean), cal["bg_v_37"])
        baseline_shift_h37 = _clamped_baseline_shift(_measure_real_baseline_value(mw_h37_clean), cal["bg_h_37"])
        baseline_shift_v89 = _clamped_baseline_shift(_measure_real_baseline_value(mw_v89_clean), cal["bg_v_89"])
        baseline_shift_h89 = _clamped_baseline_shift(_measure_real_baseline_value(mw_h89_clean), cal["bg_h_89"])
        v37_backbone = v37_backbone + baseline_shift_v37
        h37_backbone = h37_backbone + baseline_shift_h37
        v89_backbone = v89_backbone + baseline_shift_v89
        h89_backbone = h89_backbone + baseline_shift_h89

    v37_backbone = v37_backbone + tex_amp_v37 * texture_field
    h37_backbone = h37_backbone + tex_amp_h37 * texture_field
    v89_backbone = v89_backbone + tex_amp_v89 * texture_field
    h89_backbone = h89_backbone + tex_amp_h89 * texture_field

    # --- Baseline sensor-noise-like texture FLOOR ---
    #
    # A REAL bug, confirmed via direct testing after a genuine screenshot
    # still showed the exact seam the texture injection above was meant to
    # fix: _multispectral_texture_field normalizes by the WHOLE DOMAIN's
    # 95th percentile -- correct for keeping real storm-scale structure
    # dominant where it genuinely exists, but it means a calm, genuinely
    # smooth clear-sky region sitting in the same domain as an intense
    # storm collapses to near-zero texture (confirmed directly: a
    # realistic combined test scene showed the clear-sky region's
    # normalized texture ~80x smaller than the storm region's). Real GOES
    # IR/WV over calm open ocean CAN be genuinely almost perfectly smooth
    # -- but real MW imagery has INHERENT footprint-to-footprint sensor
    # noise regardless of whether the underlying meteorology is "textured"
    # or not. Fixed with a small, independent noise floor (spatially
    # correlated at a rough MW-footprint scale, not per-pixel white noise)
    # that doesn't depend on GOES's own local relative texture magnitude,
    # so no region -- however meteorologically calm -- ever renders
    # perfectly flat next to a genuinely textured one.
    #
    # Amplitude also uses the measured real values above when available
    # (scaled down, since this floor is meant to be a SMALL baseline under
    # the main texture injection, not a second full-strength copy of it).
    # Seeded deterministically, NOT from a fresh unseeded RNG.
    #
    # This drew `np.random.default_rng()` with no seed, so every call
    # produced an independent noise field. In a single still that is
    # harmless. In a frame loop it is not: measured on a real 5-frame
    # Lowell GIF, a calm far-field ocean corner changed 17-18 RGB levels
    # between consecutive 10-minute frames while the STORM CORE changed
    # only 6-7. The most meteorologically stable part of the scene was
    # the most visually volatile -- the background boiled while the storm
    # sat nearly still, which is the opposite of how a real MW loop
    # behaves and reads as a rendering artifact.
    #
    # Seeding from the storm ID and grid shape makes the field fixed in
    # grid space across every frame of a loop (so the floor stays put and
    # only genuine meteorology animates) while still differing between
    # storms and domains, so it never looks like one baked-in template.
    # zlib.crc32 rather than hash(), because Python salts string hashing
    # per process -- hash() would have reintroduced the same flicker
    # between separate runs.
    import zlib
    noise_seed = zlib.crc32(
        f"{storm_fix.storm_id}|{ir_tb.shape[0]}x{ir_tb.shape[1]}".encode()
    )
    footprint_noise = _normalize_texture(
        gaussian_filter(np.random.default_rng(noise_seed).standard_normal(ir_tb.shape), sigma=1.3)
    )
    FLOOR_FRACTION = 0.4  # the noise floor is a modest fraction of the main injection amplitude
    floor_amp_v37 = tex_amp_v37 * FLOOR_FRACTION if measured_v37 is not None else 2.0
    floor_amp_h37 = tex_amp_h37 * FLOOR_FRACTION if measured_h37 is not None else 2.5
    floor_amp_v89 = tex_amp_v89 * FLOOR_FRACTION if measured_v89 is not None else 1.8
    floor_amp_h89 = tex_amp_h89 * FLOOR_FRACTION if measured_h89 is not None else 2.2
    v37_backbone = v37_backbone + floor_amp_v37 * footprint_noise
    h37_backbone = h37_backbone + floor_amp_h37 * footprint_noise
    v89_backbone = v89_backbone + floor_amp_v89 * footprint_noise
    h89_backbone = h89_backbone + floor_amp_h89 * footprint_noise
    # --- Sensor antenna pattern (0.112) -------------------------------
    # Applied to the BACKBONE ONLY, and before texture injection.
    #
    # Only the backbone is artificially sharp: it is rendered at the GOES
    # grid spacing with no antenna pattern, so its 37 GHz field carries
    # detail no 37 GHz radiometer could resolve. Real MW fused in later
    # already carries its own sensor's PSF, and blurring the FUSED field
    # would convolve genuine observations a second time.
    #
    # AFTER texture and the noise floor, which is the correction to a
    # first attempt that applied it before them. Real radiometer noise is
    # per-FOOTPRINT: adjacent grid cells inside one footprint see the same
    # sample, so noise generated at grid scale and left unblurred is
    # finer than the instrument can produce. Applying the PSF before
    # texture measured a 37/89 sharpness ratio of 1.00 -- the grid-scale
    # texture, identical across frequencies, simply overwrote the blur.
    # Convolving the whole assembled backbone reproduces both the
    # footprint smoothing and correlated footprint-scale noise.
    #
    # This is also what finally gives 37 and 89 GHz different SPATIAL
    # responses. Until now every frequency difference lived in the
    # calibration constants and the two fields had identical structure,
    # which no real sensor pair does.
    try:
        import mw_psf
        psf_sensor = (real_swath.sensor if real_swath is not None
                      and getattr(real_swath, "sensor", None) in mw_psf.SENSOR_IFOV_KM
                      else mw_psf.DEFAULT_SENSOR)
        v37_backbone = mw_psf.apply_sensor_psf(v37_backbone, lat, lon, 37, psf_sensor)
        h37_backbone = mw_psf.apply_sensor_psf(h37_backbone, lat, lon, 37, psf_sensor)
        v89_backbone = mw_psf.apply_sensor_psf(v89_backbone, lat, lon, 89, psf_sensor)
        h89_backbone = mw_psf.apply_sensor_psf(h89_backbone, lat, lon, 89, psf_sensor)
        if progress_callback:
            progress_callback(mw_psf.describe(lat, lon, psf_sensor))
    except Exception as e:
        if progress_callback:
            progress_callback(f"Sensor PSF skipped ({type(e).__name__}: {e}).")


    v37_mw = h37_mw = v89_mw = h89_mw = None
    mw_confidence_37 = mw_confidence_89 = None
    if real_swath is not None:
        lat37_src, lon37_src = real_swath.grid_for(37)
        v37_mw = _regrid_external_with_mask(lat37_src, lon37_src, mw_v37_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        h37_mw = _regrid_external_with_mask(lat37_src, lon37_src, mw_h37_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        lat89_src, lon89_src = real_swath.grid_for(89)
        v89_mw = _regrid_external_with_mask(lat89_src, lon89_src, mw_v89_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        h89_mw = _regrid_external_with_mask(lat89_src, lon89_src, mw_h89_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        # Light NaN-aware smoothing -- see MW_VALUE_SMOOTHING_SIGMA's
        # comment above for why (guards against a resampling moire
        # artifact confirmed in testing, at the cost of only a small
        # fraction of real MW's genuine texture).
        v37_mw = _nan_aware_light_smooth(v37_mw, MW_VALUE_SMOOTHING_SIGMA)
        h37_mw = _nan_aware_light_smooth(h37_mw, MW_VALUE_SMOOTHING_SIGMA)
        v89_mw = _nan_aware_light_smooth(v89_mw, MW_VALUE_SMOOTHING_SIGMA)
        h89_mw = _nan_aware_light_smooth(h89_mw, MW_VALUE_SMOOTHING_SIGMA)

        # Feather MW's coverage edge -- see _edge_feather_taper's docstring.
        # Full confidence out to 70% of the fusion radius, tapering to
        # zero at the radius itself (210km/300km with the default
        # MW_FUSION_MAX_DISTANCE_KM), separately per frequency since 37
        # and 89 GHz can have different native swath footprints.
        mw_confidence_37 = _edge_feather_taper(
            lat37_src, lon37_src, lat, lon,
            taper_start_km=MW_FUSION_MAX_DISTANCE_KM * 0.7, taper_end_km=MW_FUSION_MAX_DISTANCE_KM,
        )
        mw_confidence_89 = _edge_feather_taper(
            lat89_src, lon89_src, lat, lon,
            taper_start_km=MW_FUSION_MAX_DISTANCE_KM * 0.7, taper_end_km=MW_FUSION_MAX_DISTANCE_KM,
        )

    # --- SECOND real MW pass: the "after" pass, for MIMIC-TC-style
    # crossfading (see generate_synthetic_mw's docstring for
    # real_swath_after/mw_confidence_after). Deliberately SCOPED DOWN
    # relative to the "before" pass above: this only feeds the direct-
    # value V/H blend, NOT the response-index/backbone-shaping
    # computation (idx37_real etc. above) -- the after-pass's role is
    # specifically to make the handoff to the next real pass smooth
    # during a short crossfade window, not to reshape the overall storm
    # intensity/structure model, which stays driven by GOES/radar/the
    # before-pass exactly as before. Physical-bounds QC still applies
    # (same MW_PHYSICAL_BOUNDS check as the before-pass) -- an upcoming
    # pass is just as capable of having a corrupted swath edge.
    v37_mw_after = h37_mw_after = v89_mw_after = h89_mw_after = None
    mw_after_confidence_37 = mw_after_confidence_89 = None
    if real_swath_after is not None:
        after_v37_clean = _apply_physical_bounds(real_swath_after.v37, MW_PHYSICAL_BOUNDS["v37"])
        after_h37_clean = _apply_physical_bounds(real_swath_after.h37, MW_PHYSICAL_BOUNDS["h37"])
        after_v89_clean = _apply_physical_bounds(real_swath_after.v89, MW_PHYSICAL_BOUNDS["v89"])
        after_h89_clean = _apply_physical_bounds(real_swath_after.h89, MW_PHYSICAL_BOUNDS["h89"])
        # Same despeckle pass as the before-pass -- see _despeckle_mw_field's docstring.
        after_v37_clean = _despeckle_mw_field(after_v37_clean)
        after_h37_clean = _despeckle_mw_field(after_h37_clean)
        after_v89_clean = _despeckle_mw_field(after_v89_clean)
        after_h89_clean = _despeckle_mw_field(after_h89_clean)

        lat37_after_src, lon37_after_src = real_swath_after.grid_for(37)
        v37_mw_after = _regrid_external_with_mask(lat37_after_src, lon37_after_src, after_v37_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        h37_mw_after = _regrid_external_with_mask(lat37_after_src, lon37_after_src, after_h37_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        lat89_after_src, lon89_after_src = real_swath_after.grid_for(89)
        v89_mw_after = _regrid_external_with_mask(lat89_after_src, lon89_after_src, after_v89_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")
        h89_mw_after = _regrid_external_with_mask(lat89_after_src, lon89_after_src, after_h89_clean, lat, lon, MW_FUSION_MAX_DISTANCE_KM, method="linear")

        v37_mw_after = _nan_aware_light_smooth(v37_mw_after, MW_VALUE_SMOOTHING_SIGMA)
        h37_mw_after = _nan_aware_light_smooth(h37_mw_after, MW_VALUE_SMOOTHING_SIGMA)
        v89_mw_after = _nan_aware_light_smooth(v89_mw_after, MW_VALUE_SMOOTHING_SIGMA)
        h89_mw_after = _nan_aware_light_smooth(h89_mw_after, MW_VALUE_SMOOTHING_SIGMA)

        mw_after_confidence_37 = _edge_feather_taper(
            lat37_after_src, lon37_after_src, lat, lon,
            taper_start_km=MW_FUSION_MAX_DISTANCE_KM * 0.7, taper_end_km=MW_FUSION_MAX_DISTANCE_KM,
        )
        mw_after_confidence_89 = _edge_feather_taper(
            lat89_after_src, lon89_after_src, lat, lon,
            taper_start_km=MW_FUSION_MAX_DISTANCE_KM * 0.7, taper_end_km=MW_FUSION_MAX_DISTANCE_KM,
        )

    # Backbone weight = goes + radar combined -- radar's priority is fully
    # expressed in SHAPING the backbone (above), not as a separate
    # competitor against MW's real texture here.
    backbone_weight = weights.get("goes", 0.1) + (weights.get("radar", 0.6) if radar_response is not None else 0.0)
    mw_weight = weights.get("mw", 0.3) * mw_age_confidence
    # MW's weight is feathered per-pixel near its coverage edge (see
    # _edge_feather_taper) -- as confidence falls, backbone naturally
    # picks up the remainder via _weighted_fuse's renormalization, giving
    # a graceful transition instead of a hard seam at the swath boundary.
    mw_weight_37 = mw_weight * mw_confidence_37 if mw_confidence_37 is not None else mw_weight
    mw_weight_89 = mw_weight * mw_confidence_89 if mw_confidence_89 is not None else mw_weight

    # "After" pass weight: base mw priority weight, scaled by BOTH the
    # crossfade-in confidence (how close target_time is to the after-
    # pass's own time -- see _crossfade_confidence_toward_after) and its
    # own edge-feather taper, mirroring the before-pass's weight
    # construction exactly. mw_confidence_after defaults to 0.0 (no
    # after-pass given), so this is a pure no-op addition when the
    # feature isn't used.
    mw_after_weight = weights.get("mw", 0.3) * mw_confidence_after
    mw_after_weight_37 = mw_after_weight * mw_after_confidence_37 if mw_after_confidence_37 is not None else mw_after_weight
    mw_after_weight_89 = mw_after_weight * mw_after_confidence_89 if mw_after_confidence_89 is not None else mw_after_weight

    value_blend_weights_37 = {"backbone": backbone_weight, "mw": mw_weight_37, "mw_after": mw_after_weight_37}
    value_blend_weights_89 = {"backbone": backbone_weight, "mw": mw_weight_89, "mw_after": mw_after_weight_89}

    v37, _cov_v37 = _weighted_fuse({"backbone": v37_backbone, "mw": v37_mw, "mw_after": v37_mw_after}, value_blend_weights_37)
    h37, _cov_h37 = _weighted_fuse({"backbone": h37_backbone, "mw": h37_mw, "mw_after": h37_mw_after}, value_blend_weights_37)
    v89, _cov_v89 = _weighted_fuse({"backbone": v89_backbone, "mw": v89_mw, "mw_after": v89_mw_after}, value_blend_weights_89)
    h89, _cov_h89 = _weighted_fuse({"backbone": h89_backbone, "mw": h89_mw, "mw_after": h89_mw_after}, value_blend_weights_89)

    # Apply the persisted auto-calibration offset (calibration_state.py) --
    # a running EMA bias correction learned from real MW comparisons on
    # previous runs (see that module's docstring for why an additive
    # offset, not a direct refit of CALIBRATION).
    #
    # SCALAR (tb37/tb89) gets the FULL persisted offset -- this drives
    # mw_compare.py's quantitative comparisons and should track real data
    # as closely as possible. V/H get a SEPARATELY, MUCH MORE TIGHTLY
    # clamped version of the same offset -- a real run showed that even a
    # modest +5.1K persisted nudge alone was enough to push V89 into
    # visibly saturated cyan, because NRL's 89 GHz blue-channel range is
    # only 20K wide ([270,290]). No baseline placement inside a window
    # that narrow survives an uncapped shift derived from a full
    # quantitative correction. This mirrors the same scalar-vs-V/H
    # decoupling in mw_compare.apply_bias_calibration, and for the same
    # reason: keep the composite colors in NRL's designed usable range
    # even when the underlying Tb correction is large.
    import calibration_state

    VH_OFFSET_CLAMP_K = 5.0

    def _clamped_offset_for_vh(freq: int) -> float:
        state = calibration_state.load_state()
        raw_offset = state.get(f"offset_{freq}", 0.0)
        return max(-VH_OFFSET_CLAMP_K, min(VH_OFFSET_CLAMP_K, raw_offset))

    tb37 = calibration_state.apply_offset(tb37, 37)
    tb89 = calibration_state.apply_offset(tb89, 89)

    vh_offset_37 = _clamped_offset_for_vh(37)
    vh_offset_89 = _clamped_offset_for_vh(89)
    v37 = v37 - vh_offset_37
    h37 = h37 - vh_offset_37
    v89 = v89 - vh_offset_89
    h89 = h89 - vh_offset_89

    # --- Exact ML-free counterpart of the fields above ---------------
    #
    # _weighted_fuse is a per-pixel linear combination whose weights
    # depend only on which sources have coverage, never on the source
    # VALUES. The ML correction enters through the backbone alone. So
    # re-running the same fuse with the backbone replaced by the recorded
    # ML delta -- and the other sources replaced by zeros carrying their
    # original NaN pattern, so the identical weight renormalization
    # happens -- yields exactly the ML step's contribution to the fused
    # output. Subtracting it recovers what this frame would have looked
    # like with the correction off.
    #
    # This makes an A/B a SINGLE run. Doing it as two runs (strength 1
    # then 0) re-draws the noise floor and advances the calibration EMA
    # between them, so part of any measured difference was run-to-run
    # randomness rather than the correction -- which is exactly the
    # ambiguity that made the first A/B pair hard to read.
    ml_free = {}
    if ml_correction_applied and not np.isscalar(ml_delta["v37"]):
        def _zeros_like_coverage(arr):
            if arr is None:
                return None
            return np.where(np.isfinite(arr), 0.0, np.nan)

        def _ml_contribution(delta, mw_arr, mw_after_arr, blend_weights):
            contrib, _ = _weighted_fuse(
                {"backbone": delta,
                 "mw": _zeros_like_coverage(mw_arr),
                 "mw_after": _zeros_like_coverage(mw_after_arr)},
                blend_weights,
            )
            return contrib

        ml_free = {
            "v37": v37 - _ml_contribution(ml_delta["v37"], v37_mw, v37_mw_after, value_blend_weights_37),
            "h37": h37 - _ml_contribution(ml_delta["h37"], h37_mw, h37_mw_after, value_blend_weights_37),
            "v89": v89 - _ml_contribution(ml_delta["v89"], v89_mw, v89_mw_after, value_blend_weights_89),
            "h89": h89 - _ml_contribution(ml_delta["h89"], h89_mw, h89_mw_after, value_blend_weights_89),
        }

    # --- Independent IR centre check ---------------------------------
    #
    # Purely diagnostic: nothing is repositioned from this. It exists
    # because the centre the whole field is built around is often
    # projected hours past the last best-track entry, and a projected
    # centre looks exactly like an observed one in the output.
    center_check = None
    try:
        import tc_center_fix

        center_check = tc_center_fix.detect_center(
            ir_tb, lat, lon, storm_fix.lat, storm_fix.lon,
        )
        if progress_callback:
            progress_callback(tc_center_fix.describe(
                center_check,
                projected_hours=float(getattr(storm_fix, "extrapolated_hours", 0.0) or 0.0),
            ))
    except Exception as e:
        if progress_callback:
            progress_callback(f"IR centre check skipped ({type(e).__name__}: {e}).")

    # --- Structural scoring -------------------------------------------
    #
    # Descriptive structure statistics (mw_structure_metrics.py) for this
    # frame, and where an ML-free counterpart exists, for that too. This
    # is the only quantitative channel in the pipeline that can actually
    # see what the correction did: the residual in mw_compare is measured
    # on the scalar tb37/tb89 fields, which the correction never touches.
    structure = {}
    try:
        import mw_structure_metrics as msm

        for freq, (fv, fh) in ((37, (v37, h37)), (89, (v89, h89))):
            structure[f"corrected_{freq}"] = msm.structure_metrics(
                fv, fh, freq, lat, lon, storm_fix.lat, storm_fix.lon, rmw_km=rmw_km,
            )
        if ml_free:
            for freq, (fv, fh) in ((37, (ml_free["v37"], ml_free["h37"])),
                                   (89, (ml_free["v89"], ml_free["h89"]))):
                structure[f"ml_free_{freq}"] = msm.structure_metrics(
                    fv, fh, freq, lat, lon, storm_fix.lat, storm_fix.lon, rmw_km=rmw_km,
                )
            if progress_callback:
                for freq in (37, 89):
                    progress_callback(
                        "Structure " + msm.compare_structure(
                            structure[f"ml_free_{freq}"], structure[f"corrected_{freq}"]
                        )
                    )
    except Exception as e:
        # Scoring is diagnostic only -- it must never be able to fail a
        # generate that otherwise succeeded.
        if progress_callback:
            progress_callback(f"Structure metrics skipped ({type(e).__name__}: {e}).")

    # Final QC guard: fail loudly rather than handing matplotlib a NaN.
    assert_finite("freq_37ghz", tb37)
    assert_finite("freq_89ghz", tb89)
    assert_finite("v37_synth", v37)
    assert_finite("h37_synth", h37)
    assert_finite("v89_synth", v89)
    assert_finite("h89_synth", h89)

    return SyntheticMWResult(
        scene_time=band13.scene_time,
        storm_id=storm_fix.storm_id,
        freq_37ghz=tb37,
        freq_89ghz=tb89,
        lat=lat,
        lon=lon,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        diagnostics={
            # M-PERC-style EWRC diagnosis; None unless a real MW pass was
            # supplied. Assigned to a local first because there is no
            # `diagnostics` dict in scope until this call -- writing to it
            # earlier raised NameError inside a try/except that swallowed
            # it and printed "unavailable", which is the silent-failure
            # shape this project keeps finding.
            "ewrc": _ewrc_result,
            "wvir": _wvir_result,
            "scattering_potential": scat_pot,
            "wv_mask": wv_mask,
            "texture": texture,
            "convective_signal": convective_signal,
            "radial_weight": radial_w,
            "rmw_km": rmw_km,
            # Hours the storm CENTRE was projected past the last real
            # best-track entry (0.0 when the fix is real or interpolated).
            # Surfaced because a projected centre looks exactly like an
            # observed one in the output, and everything in the frame --
            # radial profile, eyewall ring, ML patch -- is positioned
            # from it.
            "fix_extrapolated_hours": float(getattr(storm_fix, "extrapolated_hours", 0.0) or 0.0),
            # Independent IR-derived centre estimate (tc_center_fix.py),
            # or None. Diagnostic only -- never fed back into positioning.
            "center_check": center_check,
            # Surface conditioning, recorded so training_data_export can
            # persist exactly what the model was conditioned on.
            "land_fraction": land_frac,
            # Supplementary IR bands already regridded onto this grid, so
            # the exporter stores precisely what the model was fed.
            "extra_ir_regridded": _extra_ir_grids,
            # GLM flash density on this grid (zeros when unavailable), so
            # the exporter stores exactly what conditioned the model.
            "flash_density": flash_density,
            "elevation_m": _surface_elevation,
            # The centre the field was actually built around. Recorded so
            # the GUI can mark it without needing the StormFix passed
            # alongside the result.
            "storm_lat": float(storm_fix.lat),
            "storm_lon": float(storm_fix.lon),
            "roci_km": roci_km,
            "is_daytime": is_daytime,
            "band2_used": band2_used,
            "calibration_state": calibration_state.load_state(),
            "fusion_weights": dict(weights),
            "fusion_coverage_37": coverage_37,
            "fusion_coverage_89": coverage_89,
            "fusion_sources_used": {
                "goes": True,
                "mw": real_swath is not None,
                "radar": radar_response is not None,
            },
            # Regridded real MW V/H (already QC'd and on the same GOES
            # grid as everything else), exposed for two purposes: (1) so
            # a caller can inspect exactly what real data fed the fusion
            # for a given pixel, (2) as the "ground truth" target half of
            # a paired (GOES input, real MW output) training example --
            # see training_data_export.py, which is exactly what this was
            # added for.
            "mw_regridded_v37": v37_mw,
            "mw_regridded_h37": h37_mw,
            "mw_regridded_v89": v89_mw,
            "mw_regridded_h89": h89_mw,
            # Parametric BACKBONE values (GOES-only, before real MW is
            # blended in) -- added specifically for ML residual/
            # correction training: the actual learning target for "teach
            # a model to correct the parametric algorithm's systematic
            # errors" is (real_MW - backbone), not the real MW value
            # alone. Exposed here rather than recomputed elsewhere so the
            # training pipeline always sees exactly what THIS run's
            # algorithm actually produced pre-fusion, not a separately
            # re-run approximation of it.
            #
            # The ML correction's own contribution is SUBTRACTED back out
            # here. This matters and is not cosmetic: the ML step runs on
            # the backbone earlier in this same function, so once a
            # checkpoint exists, whatever the current model predicted was
            # being baked into every newly-mined training example -- both
            # as an input channel and inside the (real_MW - backbone)
            # residual target. Training a v2 model on that data would
            # teach it the residual of an ALREADY-corrected backbone,
            # while inference still applies it to an uncorrected one, so
            # v2 would silently under-correct by exactly v1's
            # contribution, and each retraining round would compound it.
            # Because the correction is purely additive, subtracting the
            # recorded delta recovers the true ML-free backbone exactly.
            # ALSO subtract the per-frame baseline shift (0.128).
            #
            # baseline_shift_* is measured from THIS FRAME'S real MW and
            # added to the backbone, clamped to +-20 K. It is applied only
            # when a real pass exists -- which is every training example,
            # and no GOES-only frame.
            #
            # Leaving it in produced two problems at once. The stored
            # backbone was partially pre-aligned to the target, so the
            # model learned a SMALLER correction than the GOES-only case
            # needs; and GOES-only inference then sees a backbone without
            # the shift, a train/inference distribution mismatch in
            # exactly the mode this tool exists for. The clamp is 20 K
            # against a measured bias of 11.5 K, so this is the same order
            # of magnitude as the error being corrected.
            #
            # Subtracting it stores the pure parametric backbone -- the
            # one a GOES-only frame actually produces. The residual target
            # grows accordingly, which is the point: the model should be
            # learning that systematic offset, not having it handed over.
            "backbone_v37": v37_backbone - ml_delta["v37"] - baseline_shift_v37,
            "backbone_h37": h37_backbone - ml_delta["h37"] - baseline_shift_h37,
            "backbone_v89": v89_backbone - ml_delta["v89"] - baseline_shift_v89,
            "backbone_h89": h89_backbone - ml_delta["h89"] - baseline_shift_h89,
            # Recorded so the effect is auditable rather than implicit.
            "baseline_shift_k": {"v37": baseline_shift_v37, "h37": baseline_shift_h37,
                                 "v89": baseline_shift_v89, "h89": baseline_shift_h89},
            # Diagnostics from the ML step itself (empty dict if no
            # checkpoint was applied) -- magnitude, clamp saturation, and
            # peak excursion in polarization-corrected space.
            "ml_stats": ml_stats,
            # Exact ML-free counterparts of v37/h37/v89/h89 (empty dict
            # when no correction was applied), so the GUI can render a
            # true A/B from one generate instead of two.
            "ml_free_fields": ml_free,
            # Structural scoring for the corrected fields and, where
            # available, the ML-free ones.
            "structure": structure,
            # Which real sensor (if any) supplied the fusion/training
            # target for this frame -- lets a training pipeline filter to
            # specific sensors (e.g. GMI/AMSR2 only, per direct guidance:
            # higher native resolution than WSFM/SSMIS, and the two
            # sensors whose operational era cleanly overlaps the GOES-R
            # ABI series this project's inputs come from) without having
            # to parse it back out of a free-text source_note string.
            "mw_sensor": real_swath.sensor if real_swath is not None else None,
        },
    )


# Computed at import, once CALIBRATION exists. Any tuning change to a
# physics constant alters this automatically.
VH_PHYSICS_ID = _compute_vh_physics_id()
