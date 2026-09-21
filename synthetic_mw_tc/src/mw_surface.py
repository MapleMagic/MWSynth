"""
Surface wind roughening, and frequency-dependent hydrometeor response.

Two physics gaps closed here, both found reviewing the parametric
algorithm rather than an image.


1. WIND ROUGHENING OF THE OCEAN BACKGROUND
------------------------------------------
A calm ocean is a poor, strongly polarizing emitter at 37 GHz -- about
150 K at H-pol. Wind roughens it: capillary waves and especially foam
raise the emissivity, and H-pol is far more sensitive than V because it
starts much further from unity. Published sensitivities are roughly
0.75 K per m/s at 37H and 0.25 K per m/s at 37V, somewhat less at 89 GHz.

This matters here because a tropical cyclone IS a wind field. Measured
real GMI 37H in Lowell's outer region was 182-193 K against a calm-ocean
150-155 K, and the difference is the storm's own wind. Until now the
backgrounds were UNIFORM constants, so `bg_h_37 = 182` had quietly
absorbed a storm-relative, radially-varying effect into a single number
that is right at one radius and wrong everywhere else.

Modelling it explicitly means the background now rises toward the core as
it should, and the constant returns to being what it claims to be: the
calm-ocean value.

Only applied over water. Land emissivity is already near unity and does
not roughen (see surface_type.py).


2. FREQUENCY-DEPENDENT RESPONSE
-------------------------------
`backbone_response_37` and `backbone_response_89` were literally the same
array. The 0.112 sensor PSF gave the two frequencies different
RESOLUTION, but their underlying hydrometeor response was identical,
which is wrong for a reason that matters:

  * 89 GHz responds to ICE above the freezing level. Ice scattering is
    threshold-like -- it needs substantial frozen mass aloft -- so the
    response should be PEAKED, concentrated on the deepest convection.
  * 37 GHz responds mainly to LIQUID below the freezing level. Liquid
    extends well beyond the deep-ice cores into stratiform rain, so the
    response should be BROADER and flatter, and smoother, because the
    emitting layer is deep rather than a thin scattering shell.

Implemented as a gamma exponent per frequency plus extra smoothing at
37 GHz. That is a shape correction, not a radiative transfer model, but
it moves both fields in the direction the physics requires and it is
testable: after this, 89 GHz response must be more concentrated than
37 GHz on the same scene.
"""
from __future__ import annotations

import numpy as np

# --- Wind roughening -------------------------------------------------
# dTb / dU in K per (m/s), over ocean, per frequency and polarization.
WIND_SENSITIVITY = {
    37: {"v": 0.25, "h": 0.75},
    89: {"v": 0.20, "h": 0.50},
}

# Foam coverage saturates at hurricane wind speeds, so the linear
# relation cannot be extended indefinitely -- without a cap a 70 m/s
# eyewall would push H-pol emissivity past unity, which is unphysical.
MAX_ROUGHENING_K = {37: {"v": 18.0, "h": 50.0}, 89: {"v": 14.0, "h": 35.0}}

# Modified-Rankine decay exponent outside the RMW. 0.5 is the usual
# choice for TC tangential wind and is what the radial model elsewhere in
# this project is implicitly shaped around.
RANKINE_ALPHA = 0.5

# Ambient wind assumed far from the storm, m/s. Not zero: the tropical
# ocean is never calm, and a background of exactly the calm-ocean value
# would make the far field colder than any real observation.
AMBIENT_WIND_MS = 7.0

# --- Frequency-dependent response ------------------------------------
# Gamma applied to the shared convective response. >1 concentrates
# (89 GHz ice, threshold-like), <1 broadens (37 GHz liquid, diffuse).
RESPONSE_GAMMA = {37: 0.70, 89: 1.35}

# Extra smoothing at 37 GHz, in km: the liquid layer is deep and
# horizontally diffuse compared with the thin ice shell 89 GHz sees.
LIQUID_SMOOTH_KM = 12.0


def surface_wind_ms(r_km, vmax_kt: float, rmw_km: float) -> np.ndarray:
    """Modified-Rankine surface wind profile, in m/s.

    Linear inside the RMW, decaying as r^-alpha outside, blended into an
    ambient background far out. Crude next to a real wind model, but the
    quantity it feeds -- an emissivity increment -- is itself an
    empirical linear fit, so more sophistication here would be false
    precision.
    """
    r = np.asarray(r_km, dtype=np.float64)
    vmax_ms = float(vmax_kt) * 0.514444
    rmw = max(float(rmw_km), 1e-3)
    inner = vmax_ms * (r / rmw)
    outer = vmax_ms * (rmw / np.maximum(r, 1e-3)) ** RANKINE_ALPHA
    wind = np.where(r <= rmw, inner, outer)
    # Never below ambient; the storm adds to the trade-wind background
    # rather than replacing it.
    return np.maximum(wind, AMBIENT_WIND_MS)


def roughening_k(wind_ms, freq: int, pol: str, land_fraction=None) -> np.ndarray:
    """Brightness-temperature increment from wind roughening, in K.

    Referenced to AMBIENT_WIND_MS, so the far field reproduces whatever
    the calm-ocean constant is set to and only the storm-relative excess
    is added. Suppressed over land, where emissivity is already near
    unity and wind does not roughen the surface.
    """
    u = np.asarray(wind_ms, dtype=np.float64)
    slope = WIND_SENSITIVITY.get(freq, WIND_SENSITIVITY[37])[pol]
    cap = MAX_ROUGHENING_K.get(freq, MAX_ROUGHENING_K[37])[pol]
    inc = np.clip(slope * (u - AMBIENT_WIND_MS), 0.0, cap)
    if land_fraction is not None:
        inc = inc * (1.0 - np.clip(np.asarray(land_fraction, dtype=np.float64), 0.0, 1.0))
    return inc


def frequency_response(response, freq: int, lat=None, lon=None) -> np.ndarray:
    """Shape the shared convective response for one frequency.

    89 GHz is concentrated (ice scattering is threshold-like); 37 GHz is
    broadened and smoothed (liquid emission comes from a deep, diffuse
    layer that extends into stratiform rain).
    """
    a = np.clip(np.asarray(response, dtype=np.float64), 0.0, 1.0)
    out = a ** RESPONSE_GAMMA.get(freq, 1.0)

    if freq == 37 and lat is not None and lon is not None:
        try:
            from scipy.ndimage import gaussian_filter
            import mw_psf
            km_row, km_col = mw_psf._grid_spacing_km(lat, lon)
            sigma = (LIQUID_SMOOTH_KM / max(km_row, 1e-6),
                     LIQUID_SMOOTH_KM / max(km_col, 1e-6))
            if max(sigma) > 0.5:
                out = gaussian_filter(out, sigma=sigma, mode="nearest")
        except Exception:
            pass
    return np.clip(out, 0.0, 1.0)


def concentration(field) -> float:
    """Fraction of the field's total that sits in its top 10% of pixels.

    A scalar for 'how peaked is this', used to verify that the 89 GHz
    response really is more concentrated than 37 GHz rather than just
    differently scaled.
    """
    a = np.asarray(field, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0 or a.sum() <= 0:
        return 0.0
    k = max(1, int(0.1 * a.size))
    return float(np.sort(a)[-k:].sum() / a.sum())


# --- Atmospheric emission (0.132) ------------------------------------
#
# THE MISSING TERM. Flagged in 0.114 and not built until now.
#
# `bg_h_37 = 182 K` is documented as "ambient tropical background
# INCLUDING the mean atmospheric contribution". Clear calm ocean at 37H is
# nearer 150-155 K; the remaining 25-30 K is water vapour and cloud liquid
# emitting, plus its reflection off a poorly-emitting sea surface. That is
# real physics folded into a constant -- so it cannot vary with the
# moisture field, which varies a great deal.
#
# The measured consequence is exactly what a missing term looks like:
# bias is 76% of the error budget (11.5 K of 18.2 K) and did not move at
# all between 298 and 799 training examples. More data cannot learn
# something the input never varies with.
#
# Li et al. (2026) point straight at the fix. Their saliency analysis
# makes IR channel 10 (low-level water vapour) a dominant predictor, and
# finds channels 13/15 contribute "baseline constraints on cloud optical
# thickness and total precipitable water" through differential water
# vapour absorption. The information is in the channels this project
# already fetches.
#
# Parameterized as an ANOMALY about the tropical mean, not an absolute
# addition -- the constants already carry the mean, and adding the whole
# thing again would double-count. Same discipline as the wind term, which
# is referenced to an ambient wind rather than to calm.

# Mid-level WV (band 9) brightness temperature spanning dry to moist.
# WARMER band 9 means radiation escaping from lower, warmer levels, i.e.
# a DRIER mid-troposphere -- so the relationship is inverted.
WV_DRY_K, WV_MOIST_K = 250.0, 225.0

# Peak-to-peak atmospheric swing, K, across that range. H-pol moves more
# than V: over ocean the surface reflects downwelling atmospheric
# radiation, and H-pol reflects far more of it.
#
# FIRST ESTIMATES. They are the right sign and order of magnitude, and
# they are exactly what calibrate_constants should refine once there is a
# dataset generated with this term active.
ATMOS_SWING_K = {37: {"v": 10.0, "h": 22.0}, 89: {"v": 8.0, "h": 16.0}}


def atmospheric_anomaly_k(wv_tb, freq: int, pol: str, land_fraction=None):
    """Moisture-driven departure from the mean atmospheric contribution.

    Returns K to add to the background. Zero at the midpoint of the
    dry/moist range by construction, so a typical tropical column
    reproduces the existing constants exactly and only the ANOMALY moves.

    Suppressed over land, where the background is already near-blackbody
    and there is no poorly-emitting surface for downwelling radiation to
    reflect off.
    """
    wv = np.asarray(wv_tb, dtype=np.float64)
    # 0 at the dry end, 1 at the moist end (note the inversion).
    frac = np.clip((WV_DRY_K - wv) / (WV_DRY_K - WV_MOIST_K), 0.0, 1.0)
    swing = ATMOS_SWING_K.get(freq, ATMOS_SWING_K[37])[pol]
    anomaly = (frac - 0.5) * swing
    if land_fraction is not None:
        anomaly = anomaly * (1.0 - np.clip(np.asarray(land_fraction, dtype=np.float64),
                                           0.0, 1.0))
    return anomaly


# --- Optical-depth saturation (0.133) --------------------------------
#
# The backbone maps response to brightness temperature LINEARLY:
#     Tb = bg + E*resp - D*resp
# Real radiative transfer does not. Emission and scattering both saturate
# as the layer becomes optically thick, so Tb approaches a limit
# asymptotically rather than tracking optical depth in a straight line.
#
# The difference is not small, and it is concentrated where most pixels
# live:
#
#     response   linear   saturating   ratio
#       0.10      0.10       0.27       2.7
#       0.25      0.25       0.56       2.2
#       0.50      0.50       0.82       1.6
#       1.00      1.00       1.00       1.0
#
# A linear ramp calibrated to be right at full response is therefore too
# WEAK everywhere below it -- a systematic, one-signed error over the bulk
# of every frame, which is what an unexplained bias that data cannot fix
# looks like.
#
# Normalized so sat(0)=0 and sat(1)=1 EXACTLY. That matters for two
# reasons: the existing calibration constants keep their meaning at the
# endpoints, and calibrate_constants still works unchanged, because it
# solves for the PRODUCTS resp*(1-scat) and resp*scat rather than for
# resp itself -- substituting sat(resp) leaves that algebra untouched.
#
# SATURATION_K is a free parameter standing in for the unknown mapping
# from this project's normalized "response" to actual optical depth.
#
# REDUCED 3.0 -> 0.75 after checking it against the one metric with a real
# reference. The 89 GHz core area was ALREADY 23% larger than the observed
# Lowell value on a linear ramp (11,480 against ~9,355 km2), and stronger
# saturation inflates it further:
#
#     linear    11,480 km2     +0%
#     k=0.75    13,538 km2    +18%
#     k=1.50    15,054 km2    +31%
#     k=3.00    17,545 km2    +53%
#
# The physical argument for a saturating curve is sound and unchanged. The
# problem is that every calibration constant was tuned against the LINEAR
# form, so changing the shape without refitting over-strengthens the whole
# field -- and it does so in the direction of a discrepancy already known
# to exist.
#
# 0.75 keeps the correct shape while limiting the magnitude change until
# calibrate_constants can fit the constants to it. Raise it once the fit
# has run and the core area can be checked against real data rather than
# against a synthetic scene.
SATURATION_K = 0.75


def saturate(response):
    """Optical-depth saturation curve. sat(0)=0, sat(1)=1."""
    r = np.clip(np.asarray(response, dtype=np.float64), 0.0, 1.0)
    k = SATURATION_K
    if k <= 0:
        return r
    return (1.0 - np.exp(-k * r)) / (1.0 - np.exp(-k))


# --- Freezing level (0.133) ------------------------------------------
#
# Ice scattering at 89 GHz depends on how much frozen mass sits above the
# freezing level, and the freezing level drops poleward -- roughly 4.8 km
# in the deep tropics against 3.5 km near 35 degrees. The same cloud-top
# temperature therefore implies LESS ice aloft at higher latitude, so a
# model with no latitude dependence over-depresses poleward storms and
# under-depresses tropical ones.
#
# A modest effect next to the others here, but it is systematic and
# free: latitude is already known for every frame.
FREEZING_LEVEL_KM = {0.0: 4.9, 15.0: 4.7, 25.0: 4.3, 35.0: 3.6, 45.0: 2.9}


def ice_depth_factor(lat) -> np.ndarray:
    """Scaling on ice-scattering depression, 1.0 in the deep tropics."""
    a = np.abs(np.asarray(lat, dtype=np.float64))
    lats = np.array(sorted(FREEZING_LEVEL_KM))
    vals = np.array([FREEZING_LEVEL_KM[k] for k in lats])
    fl = np.interp(a, lats, vals)
    return np.clip(fl / FREEZING_LEVEL_KM[0.0], 0.5, 1.0)


# --- Outward tilt of eyewall convection (0.145) ----------------------
#
# Best-track RMW is a SURFACE WIND radius. What this project synthesises
# is a cloud-top and hydrometeor signature, and the two are not at the
# same radius: eyewall convection tilts outward with height.
#
# Sanabia et al. (2015) quantify it from SFMR and IR along aircraft
# transects through Typhoon Sinlaku: correlations between surface wind
# and IR brightness temperature are most negative when the winds lag the
# IR by 10 km, with the IR located radially OUTWARD. They note this
# agrees with Sanabia et al. (2014), which found an outward tilt of
# eyewall convective clouds in IR profiles.
#
# This resolves an open question from 0.119. That version measured
# eyewall/RMW drifting from 0.77 on a broad storm to 1.17 on a compact
# one, ruled out the sensor PSF, grid resolution, the radial weight and
# eye suppression, and recorded the drift as unexplained. A FIXED outward
# offset produces exactly that shape -- 10 km is 45% of a 22 km RMW and
# only 9% of a 111 km one -- so the trend was physical all along, and the
# constant it was hunting did not need changing.
EYEWALL_TILT_KM = 10.0


def tilted_ring_radius_km(rmw_km: float) -> float:
    """Radius at which to centre the convective ring.

    Capped relative to RMW so a very compact storm does not get a ring
    displaced most of its own radius outward: the 10 km figure comes from
    a single typhoon's transects, and extrapolating it to a 15 km RMW
    would be pushing one case further than it can carry.
    """
    rmw = max(float(rmw_km), 1.0)
    return rmw + min(EYEWALL_TILT_KM, 0.35 * rmw)
