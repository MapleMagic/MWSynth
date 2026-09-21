"""
Independent storm-centre estimation from GOES IR, used to CHECK the
best-track centre the synthetic field is built around.

WHY THIS EXISTS: best track lands every 6 h, so a live frame is routinely
built around a centre projected several hours forward (see
besttrack.interpolate_fix). A projected centre is indistinguishable from
an observed one in the output -- everything downstream is positioned from
it, and if it is wrong the whole field is confidently misplaced beside
the storm it describes. That happened on Lowell (EP12) at 11:04Z with a
frozen 06Z centre and roughly 80 km of unaccounted motion.

It matters more since 0.91 narrowed the eyewall ring. The old ring was
wide enough (36 km for a 19 km RMW) that a centre error of a few tens of
km still landed some of the ring on real convection. A correctly narrow
ring has no such tolerance: miss the eyewall and the radial response
collapses, taking the 37 GHz emission signature with it. Accidental
error-tolerance disappeared along with the bug that provided it, so the
error now needs measuring directly.

DELIBERATELY NOT a replacement for best track. This does not feed the
generation path and nothing is repositioned from it. It produces a
number and a marker so a misplaced centre is visible instead of silent.
Automated centre-fixing from imagery is a real discipline (ARCHER and
similar) and a Gaussian-blur-and-find-the-warm-hole heuristic is not a
substitute for one.

METHOD, and why it is shaped this way: earlier ad-hoc attempts at this
kept locking onto gaps in the cirrus canopy tens of km from the storm --
a warm hole surrounded by cold cloud looks locally identical to an eye.
Three constraints fix that, and all three are necessary:
  1. Search only within search_radius_km of the prior centre. An eye is
     never 200 km from the best-track position; a canopy gap easily is.
  2. Require the warm region to be fully ENCLOSED by the cold CDO
     (binary_fill_holes), not merely warm and nearby.
  3. Score candidates on thermal contrast, size plausibility AND
     distance from the prior, rather than taking the largest or warmest.

When no eye is present -- a sheared or pre-eye storm, which is most of
them -- there is no honest centroid to report, so this returns a
low-confidence CDO centroid clearly labelled as such, or None.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

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

# Cloud colder than this counts as the central dense overcast.
CDO_TB_K = 235.0

# A candidate eye must be at least this much warmer than the CDO around
# it. Below this it is texture in the canopy, not a warm core.
MIN_EYE_CONTRAST_K = 8.0

# Plausible eye areas. Real eyes run from a few km across (pinhole) to
# ~100 km (large annular); the wide bounds are deliberate, since the
# distance and contrast terms do most of the discriminating.
MIN_EYE_AREA_KM2 = 30.0
MAX_EYE_AREA_KM2 = 20000.0

# How far from the prior centre to look. Generous enough to catch a
# genuinely stale projected centre, tight enough to exclude the canopy
# gaps that broke earlier attempts.
DEFAULT_SEARCH_RADIUS_KM = 200.0


def _distance_km(lat, lon, clat, clon):
    dlat = lat - clat
    dlon = _wrap_lon_delta(lon - clon) * np.cos(np.radians(clat))
    return np.sqrt(dlat ** 2 + dlon ** 2) * 111.32


def _pixel_area_km2(lat, lon):
    if lat.shape[0] < 2 or lat.shape[1] < 2:
        return 1.0
    dlat = abs(float(np.nanmedian(np.diff(lat, axis=0)))) * 111.32
    dlon = abs(float(np.nanmedian(np.diff(lon, axis=1)))) * 111.32 * np.cos(
        np.radians(float(np.nanmean(lat)))
    )
    return max(dlat * dlon, 1e-6)


def detect_center(
    ir_tb: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    prior_lat: float,
    prior_lon: float,
    search_radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
) -> Optional[dict]:
    """Estimate the storm centre from IR, near a prior centre.

    Returns None when there isn't enough cold cloud to say anything, or a
    dict with:
        lat, lon        estimated centre
        method          "eye" | "cdo_centroid"
        confidence      0-1, heuristic
        offset_km       distance from (prior_lat, prior_lon)
        offset_bearing  degrees clockwise from north, prior -> estimate
        eye_area_km2    None unless method == "eye"
        eye_contrast_k  None unless method == "eye"
    """
    if ir_tb is None or ir_tb.size == 0:
        return None

    px_km2 = _pixel_area_km2(lat, lon)
    r_prior = _distance_km(lat, lon, prior_lat, prior_lon)
    in_range = r_prior <= search_radius_km

    cold = (ir_tb < CDO_TB_K) & np.isfinite(ir_tb)
    if not np.any(cold & in_range):
        return None

    # Work with the CDO component that actually overlaps the search area,
    # not merely the largest cold blob in the scene -- a big rainband
    # complex elsewhere in the sector should not define the centre.
    labels, n = ndimage.label(cold)
    if n == 0:
        return None
    overlap = [(int(np.sum((labels == i) & in_range)), i) for i in range(1, n + 1)]
    overlap.sort(reverse=True)
    if overlap[0][0] == 0:
        return None
    cdo = labels == overlap[0][1]

    filled = ndimage.binary_fill_holes(cdo)
    holes = filled & ~cdo

    best = None
    if np.any(holes):
        hlabels, hn = ndimage.label(holes)
        for i in range(1, hn + 1):
            hole = hlabels == i
            # Reject holes that touch off-grid/invalid data. A GOES crop
            # is a parallelogram inside a rectangular array, so the white
            # corners are enclosed by cloud from the array's point of
            # view and register as enormous, infinitely-warm "eyes". The
            # contrast term then evaluates to NaN, and NaN comparisons
            # are False, which let such a candidate slip through the
            # score test and win by default.
            if not np.all(np.isfinite(ir_tb[hole])):
                continue
            area = float(np.sum(hole)) * px_km2
            if not (MIN_EYE_AREA_KM2 <= area <= MAX_EYE_AREA_KM2):
                continue
            cy, cx = ndimage.center_of_mass(hole)
            hlat = float(lat[int(round(cy)), int(round(cx))])
            hlon = float(lon[int(round(cy)), int(round(cx))])
            dist = float(_distance_km(np.array(hlat), np.array(hlon), prior_lat, prior_lon))
            if dist > search_radius_km:
                continue

            # Contrast against the CDO immediately surrounding this hole.
            ring = ndimage.binary_dilation(hole, iterations=3) & cdo
            if not np.any(ring):
                continue
            contrast = float(np.nanmean(ir_tb[hole]) - np.nanmean(ir_tb[ring]))
            if contrast < MIN_EYE_CONTRAST_K:
                continue

            # Compactness: a real eye is roughly round. A long thin gap
            # between rainbands is not, and this is what most often
            # survives the other tests.
            ys, xs = np.nonzero(hole)
            extent = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
            equiv_d = 2.0 * np.sqrt(np.sum(hole) / np.pi)
            compact = float(np.clip(equiv_d / max(extent, 1), 0, 1))

            score = (
                min(contrast / 20.0, 1.5)
                + compact
                + 1.5 * (1.0 - min(dist / search_radius_km, 1.0))
            )
            if not np.isfinite(score):
                continue
            cand = {
                "lat": hlat, "lon": hlon, "method": "eye",
                "eye_area_km2": area, "eye_contrast_k": contrast,
                "offset_km": dist, "_score": score, "_compact": compact,
            }
            if best is None or score > best["_score"]:
                best = cand

    if best is None:
        # No eye. Fall back to the centroid of the coldest core of the
        # CDO, which is a weak estimate -- an asymmetric or sheared storm
        # has its coldest cloud downshear of the circulation centre, not
        # on it. Reported at low confidence precisely so it is not read
        # as a real fix.
        core = cdo & in_range & (ir_tb < np.nanpercentile(ir_tb[cdo & in_range], 20))
        if not np.any(core):
            return None
        cy, cx = ndimage.center_of_mass(core)
        clat = float(lat[int(round(cy)), int(round(cx))])
        clon = float(lon[int(round(cy)), int(round(cx))])
        dist = float(_distance_km(np.array(clat), np.array(clon), prior_lat, prior_lon))
        return {
            "lat": clat, "lon": clon, "method": "cdo_centroid",
            "confidence": 0.25, "offset_km": dist,
            "offset_bearing": _bearing(prior_lat, prior_lon, clat, clon),
            "eye_area_km2": None, "eye_contrast_k": None,
        }

    conf = float(np.clip(
        0.35 * min(best["eye_contrast_k"] / 20.0, 1.0)
        + 0.35 * best["_compact"]
        + 0.30 * (1.0 - min(best["offset_km"] / search_radius_km, 1.0)),
        0.0, 1.0,
    ))
    return {
        "lat": best["lat"], "lon": best["lon"], "method": "eye",
        "confidence": conf, "offset_km": best["offset_km"],
        "offset_bearing": _bearing(prior_lat, prior_lon, best["lat"], best["lon"]),
        "eye_area_km2": best["eye_area_km2"],
        "eye_contrast_k": best["eye_contrast_k"],
    }


def _bearing(lat0, lon0, lat1, lon1) -> float:
    """Compass bearing (degrees clockwise from north) from point 0 to 1."""
    dlat = lat1 - lat0
    dlon = (lon1 - lon0) * np.cos(np.radians(lat0))
    return float((np.degrees(np.arctan2(dlon, dlat)) + 360.0) % 360.0)


def describe(result: Optional[dict], projected_hours: float = 0.0) -> str:
    """One-line summary for the log / figure title."""
    if result is None:
        return "IR centre check: no CDO found near the best-track position."
    if result["method"] == "cdo_centroid":
        return (f"IR centre check: no eye resolved; coldest-cloud centroid is "
                f"{result['offset_km']:.0f} km from the fix (low confidence -- "
                f"cold cloud sits downshear of the centre, so this is not a fix)")
    note = ""
    if result["offset_km"] > 40.0 and projected_hours > 0.5:
        note = (f"  [the fix is projected {projected_hours:.1f}h; an offset this large "
                f"means the field is built around the wrong point]")
    return (f"IR centre check: eye found {result['offset_km']:.0f} km from the fix "
            f"at bearing {result['offset_bearing']:.0f} deg "
            f"(contrast {result['eye_contrast_k']:.0f}K, confidence {result['confidence']:.2f}){note}")
