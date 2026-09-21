"""
M-PERC-style eyewall replacement cycle (ERC) diagnosis from real 89 GHz
passive microwave imagery.

WHAT THIS IS, AND IS NOT. Kossin et al. (2023, Wea. Forecasting 38,
1405) build M-PERC from ARCHER ring-score radial profiles reduced by PCA
over 1787 profiles from 47 Atlantic TCs, feeding an 18-predictor logistic
regression whose coefficients are not published in usable form. This
module does NOT reproduce that model and must not be described as
M-PERC. It implements the mechanism M-PERC rests on -- a ring-score
profile computed outward from TC center and interrogated for a SECONDARY
MAXIMUM -- and reports a calibrated confidence rather than a probability.

Their construction, which this follows:

  - ARCHER "ring score" measures how well brightness-temperature
    gradients fit the shape of a circle at a given radius, scoring
    convective features by circular symmetry
  - scores are computed every 6 km outward from TC center to 200 km
  - the resulting radial profile is searched for secondary maxima
  - following Kossin and Sitkowski (2009), an outer ring counts only if
    it forms at least 75% of a complete circle, clearly separated from
    the primary eyewall

WHY 89 GHz AND NOT IR. Kossin et al. are explicit about this: infrared
sensors do not see through upper-level cirrus, and TCs carry a thick
cirrus canopy that obscures the convective structure beneath. Microwave
is essentially transparent to cirrus. A secondary eyewall is a
convective ring, and in IR it is usually hidden under the canopy -- so
this runs on REAL microwave only and is disabled otherwise. Running it
on the synthetic field would be circular: the backbone builds its
response from IR, so any ring it shows is one IR already implied.

Sanabia et al. (2015, Mon. Wea. Rev. 143, 3406) track the same cycle in
WV-minus-IR radial profiles and find inner-eyewall decay detectable
EARLIER in WVIR than in IR alone. That is a genuine geostationary-only
signal and a natural extension, but it diagnoses a different stage of
the cycle and is deliberately not folded into this score.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Kossin et al. sample every 6 km out to 200 km.
RADIAL_STEP_KM = 6.0
MAX_RADIUS_KM = 200.0

# Azimuthal bins for the circular-symmetry measure. 36 gives 10-degree
# resolution, fine enough to resolve the 75% closure criterion without
# making each bin so small that a single noisy pixel dominates it.
AZIMUTH_BINS = 36

# Brightness-temperature range from scene background to deep 89 GHz
# scattering, used to normalize convective strength. Fixed rather than
# scene-derived -- see ring_score_profile.
DEEP_CONVECTION_SPAN_K = 100.0

# KS09's criterion: an outer ring must form at least 75% of a complete
# circle to count as secondary eyewall formation.
MIN_RING_CLOSURE = 0.75

# A secondary maximum must sit outside the primary eyewall with a real
# moat between them, not be a shoulder on the same peak.
MIN_SEPARATION_KM = 18.0
MIN_MOAT_DEPTH = 0.12


def ring_score_profile(tb89, lat, lon, center_lat: float, center_lon: float,
                       max_radius_km: float = MAX_RADIUS_KM):
    """Radial profile of ring score from a real 89 GHz field.

    Returns (radii_km, score, closure). `score` is the circular symmetry
    of the convective signature at each radius in 0-1; `closure` is the
    fraction of azimuth where convection is present, which is what the
    75% criterion is applied to.
    """
    tb = np.asarray(tb89, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    coslat = np.cos(np.radians(center_lat))
    dy = (lat - center_lat) * 111.32
    dx = (lon - center_lon) * 111.32 * coslat
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    radii = np.arange(RADIAL_STEP_KM, max_radius_km + RADIAL_STEP_KM,
                      RADIAL_STEP_KM)

    # Convection at 89 GHz is a DEPRESSION, so invert into a 0-1 measure
    # of scattering strength. The reference is the scene's own warm
    # background rather than a fixed constant, which keeps this working
    # across sensors with different calibration and across basins.
    finite = np.isfinite(tb)
    if finite.sum() < 100:
        return radii, np.zeros_like(radii), np.zeros_like(radii)
    # Warm reference from the scene (robust), cold reference FIXED.
    #
    # Both ends were originally percentiles, and that fails exactly where
    # it matters. A real eyewall covers well under 1% of a 200 km-radius
    # scene, so the 2nd percentile lands in the BACKGROUND, span collapses
    # to ~1 K, and every pixel with any depression saturates to 1.0. The
    # profile then reads flat-topped from 12 to 48 km on a textbook single
    # ring at 30 km -- no peak at all, so no primary eyewall and no
    # secondary maximum.
    #
    # 100 K below the scene background is the physical dynamic range of
    # 89 GHz deep convection over ocean (~280 K background against 180 K
    # in strong scattering, per the observed Norbert and Lowell frames),
    # so this is anchored rather than tuned.
    warm = float(np.nanpercentile(tb[finite], 90))
    span = DEEP_CONVECTION_SPAN_K
    strength = np.clip((warm - tb) / span, 0.0, 1.0)

    # Pixel scale, for adaptive azimuth binning below.
    try:
        px_km = float(np.nanmedian(np.abs(np.diff(lat, axis=0)))) * 111.32
    except Exception:
        px_km = 4.0
    px_km = max(px_km, 0.5)

    score = np.zeros_like(radii)
    closure = np.zeros_like(radii)
    for i, rad in enumerate(radii):
        sel = (r >= rad - RADIAL_STEP_KM / 2) & (r < rad + RADIAL_STEP_KM / 2) & finite
        if sel.sum() < 12:
            continue

        # ADAPTIVE azimuth binning. A fixed 36 bins leaves most of them
        # EMPTY at small radius -- a 28 km ring has a 176 km circumference,
        # about 50 pixels at 3.5 km resolution, so a third of the bins get
        # nothing and the per-bin scatter explodes. Uniformity then reads
        # near zero and a textbook inner eyewall scores lower than a
        # partial outer arc, which is the opposite of the truth.
        #
        # Scaling bins with circumference keeps roughly one pixel per bin
        # at worst. The 75% closure criterion is a FRACTION, so it is
        # unaffected by how many bins that fraction is measured over.
        nb = int(np.clip(2 * np.pi * rad / max(3.0 * px_km, 1.0), 8, AZIMUTH_BINS))
        b_all = ((theta + np.pi) / (2 * np.pi) * nb).astype(int)
        b_all = np.clip(b_all, 0, nb - 1)

        s = strength[sel]
        b = b_all[sel]
        per = np.full(nb, np.nan)
        for k in range(nb):
            m = b == k
            if m.any():
                per[k] = float(np.mean(s[m]))
        ok = np.isfinite(per)
        if ok.sum() < nb * 0.5:
            continue
        vals = per[ok]
        mean = float(np.mean(vals))
        # Ring score: strong AND uniform around the circle. A single
        # intense cell scores low because it is not circularly symmetric,
        # which is exactly the discrimination ARCHER's ring score makes.
        uniformity = 1.0 - float(np.std(vals)) / max(mean, 1e-6)
        score[i] = max(0.0, mean * np.clip(uniformity, 0.0, 1.0))
        # Closure: fraction of azimuth carrying meaningful convection.
        closure[i] = float(np.mean(vals > max(0.25, 0.5 * mean)))

    return radii, score, closure


def _smooth(a, width: int = 3):
    if a.size < width:
        return a
    k = np.ones(width) / width
    return np.convolve(a, k, mode="same")


def ewrc_confidence(tb89, lat, lon, center_lat: float, center_lon: float,
                    vmax_kt: Optional[float] = None,
                    rmw_km: Optional[float] = None) -> dict:
    """Confidence that a secondary eyewall is present, 0-1.

    Reports the microwave-based score AND an intensity-only baseline,
    following Kossin et al.'s practice of displaying both the full model
    and a reduced Vmax-only model side by side: the comparison tells a
    forecaster whether the convective presentation is more or less
    indicative than intensity alone would suggest.
    """
    radii, score, closure = ring_score_profile(tb89, lat, lon,
                                               center_lat, center_lon)
    out = {
        "radii_km": radii.tolist(),
        "ring_score": score.tolist(),
        "closure": closure.tolist(),
        "primary_radius_km": None,
        "secondary_radius_km": None,
        "moat_depth": None,
        "secondary_closure": None,
        "confidence": 0.0,
        "intensity_baseline": _intensity_baseline(vmax_kt),
        "note": "",
    }
    if not np.any(score > 0):
        out["note"] = "no usable ring structure in the 89 GHz field"
        return out

    sm = _smooth(score)
    primary_i = int(np.argmax(sm))
    out["primary_radius_km"] = float(radii[primary_i])

    # Search OUTSIDE the primary for a secondary maximum, requiring a
    # genuine moat between them rather than a shoulder on one peak.
    best = None
    for i in range(primary_i + 2, len(sm) - 1):
        if radii[i] - radii[primary_i] < MIN_SEPARATION_KM:
            continue
        if not (sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]):
            continue
        moat = float(np.min(sm[primary_i:i]))
        depth = min(sm[primary_i], sm[i]) - moat
        if depth < MIN_MOAT_DEPTH:
            continue
        if best is None or sm[i] > sm[best[0]]:
            best = (i, depth, moat)

    if best is None:
        out["note"] = "single convective ring; no separated secondary maximum"
        return out

    i, depth, _moat = best
    out["secondary_radius_km"] = float(radii[i])
    out["moat_depth"] = float(depth)
    out["secondary_closure"] = float(closure[i])

    # KS09's 75% criterion is a gate, not a weight: an arc that does not
    # close most of the way round is not a secondary eyewall.
    if closure[i] < MIN_RING_CLOSURE:
        out["note"] = (f"outer ring closes only {closure[i]*100:.0f}% of the "
                       f"circle; KS09 require {MIN_RING_CLOSURE*100:.0f}%")
        out["confidence"] = float(np.clip(0.35 * closure[i] / MIN_RING_CLOSURE,
                                          0.0, 0.35))
        return out

    strength = float(np.clip(sm[i] / max(sm[primary_i], 1e-6), 0.0, 1.0))
    conf = 0.45 * strength + 0.35 * float(np.clip(depth / 0.35, 0, 1)) \
        + 0.20 * float(np.clip((closure[i] - MIN_RING_CLOSURE) / 0.25, 0, 1))

    # Kossin et al. note ERCs are well correlated with intensity: more
    # intense TCs are more likely to undergo one. Used as a mild
    # modifier, never as the signal -- the microwave presentation is the
    # evidence and intensity is prior expectation.
    conf *= 0.85 + 0.30 * out["intensity_baseline"]
    out["confidence"] = float(np.clip(conf, 0.0, 1.0))
    out["note"] = "separated secondary ring meeting the closure criterion"
    return out


def _intensity_baseline(vmax_kt: Optional[float]) -> float:
    """Vmax-only expectation, standing in for M-PERC's reduced model.

    Kossin et al. run a second logistic regression on the three
    intensity predictors alone and display it beside the full model, so a
    forecaster can see whether the cloud presentation is saying more than
    intensity already did. This is a monotone stand-in for that baseline,
    not a fitted probability: ERCs concentrate in strong TCs and are rare
    below hurricane strength.
    """
    if vmax_kt is None:
        return 0.3
    return float(np.clip((float(vmax_kt) - 75.0) / 60.0, 0.0, 1.0))


def format_ewrc(result: dict) -> str:
    """One-line summary for the log, or a short block when a ring is found."""
    if result.get("secondary_radius_km") is None:
        return (f"EWRC: no secondary eyewall detected "
                f"({result.get('note', '')})")
    lines = [
        f"EWRC confidence {result['confidence']:.2f}  "
        f"(intensity-only baseline {result['intensity_baseline']:.2f})",
        f"  primary eyewall  {result['primary_radius_km']:.0f} km",
        f"  secondary ring   {result['secondary_radius_km']:.0f} km, "
        f"closes {result['secondary_closure']*100:.0f}% of the circle",
        f"  moat depth       {result['moat_depth']:.2f}",
    ]
    return "\n".join(lines)
