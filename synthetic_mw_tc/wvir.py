"""
WV-minus-IR (WVIR) deep-convection profiling and eyewall replacement
cycle stage tracking, after Sanabia et al. (2015, Mon. Wea. Rev. 143,
3406).

WHY THIS COMPLEMENTS THE MICROWAVE EWRC SCORE. `ewrc.py` diagnoses a
secondary eyewall from a real 89 GHz pass, which is the right instrument
for the job -- but passes are hours apart and, as Kossin et al. put it,
suffer temporal and spatial data gaps that can be large, particularly in
the tropics. Geostationary scans arrive every 5-15 minutes. Sanabia et
al. mapped an entire ERC in Typhoon Sinlaku (2008) from WVIR radial
profiles alone and found the progression "effectively mapped", with one
result that matters operationally: decay of the inner eyewall was
detected EARLIER in WVIR than in IR.

So this fills the gap between overpasses. The microwave score anchors;
WVIR tracks progression.

THE PHYSICS. The broad weighting functions of IR and WV each make it
hard to isolate deep convective cores from surrounding cloud. Their
DIFFERENCE does not: WV minus IR exploits the temperature inversion at
the tropopause, and positive values indicate convection that has
PENETRATED the tropopause (Fritz and Laszlo 1993; Olander and Velden
2009). Cirrus canopy and anvil, which sit below the tropopause, do not
produce a positive difference.

That is the same discrimination this project's cirrus gate (0.99)
approximates with a water-vapour mask, arrived at from a different
direction and with a physical basis rather than a tuned blend.

THE SIX STAGES, as Sanabia et al. define them for Sinlaku:

  1  single eyewall        one convective ring, broad radial extent
                           (15-150 km), WVIR maxima 3.0-3.5 K
  2  outer erosion         deep convection erodes at OUTER radii; radial
                           extent roughly halves (150 -> 75 km)
  3  concentric eyewalls   two maxima within 200 km (~30 and ~120 km),
                           and BOTH weaker, 1.5-2.0 K
  4  decaying inner        inner maximum fades; seen earlier here than
                           in IR
  5  inner eyewall gone    eye free of deep convection, remaining ring
                           near 100 km
  6  contraction           the surviving outer ring contracts inward

The authors are explicit that these stages and their transitions are
subjective, so this reports a stage WITH a confidence and the evidence
behind it, never a bare label.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

RADIAL_STEP_KM = 6.0
MAX_RADIUS_KM = 200.0

# Sanabia et al. report single-eyewall and transition maxima of 3.0-3.5 K
# against 1.5-2.0 K once concentric eyewalls form. This threshold marks
# "deep convection present"; the amplitude itself carries stage
# information and is preserved rather than thresholded away.
DEEP_CONVECTION_K = 1.0

# Two maxima must be genuinely separated to count as concentric rings
# rather than one broad ring, matching the ~30 and ~120 km structure.
MIN_SEPARATION_KM = 40.0
MIN_MOAT_FRACTION = 0.55

STAGE_NAMES = {
    1: "single eyewall",
    2: "outer erosion",
    3: "concentric eyewalls",
    4: "decaying inner eyewall",
    5: "inner eyewall gone",
    6: "outer eyewall contracting",
}


def wvir_profile(wv_tb, ir_tb, lat, lon, center_lat: float, center_lon: float,
                 max_radius_km: float = MAX_RADIUS_KM):
    """Azimuthal-mean POSITIVE WVIR difference against radius.

    Positive only, as in Sanabia et al.: a negative difference means the
    cloud top is below the tropopause and carries no deep-convection
    information, so averaging it in would dilute the signal with the
    anvil this measure exists to exclude.
    """
    wv = np.asarray(wv_tb, dtype=np.float64)
    ir = np.asarray(ir_tb, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    dy = (lat - center_lat) * 111.32
    dx = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
    r = np.sqrt(dx * dx + dy * dy)

    diff = wv - ir
    positive = np.where(np.isfinite(diff) & (diff > 0), diff, 0.0)
    valid = np.isfinite(diff)

    radii = np.arange(RADIAL_STEP_KM, max_radius_km + RADIAL_STEP_KM,
                      RADIAL_STEP_KM)
    prof = np.zeros_like(radii)
    for i, rad in enumerate(radii):
        sel = (r >= rad - RADIAL_STEP_KM / 2) & (r < rad + RADIAL_STEP_KM / 2) & valid
        if sel.sum() >= 8:
            prof[i] = float(np.mean(positive[sel]))
    return radii, prof


def _smooth(a, width: int = 3):
    """3-point MEDIAN, not a mean.

    A mean crushes narrow rings. Sanabia et al.'s concentric stage has an
    inner ring near 30 km; at 6 km radial sampling a 12 km-wide ring is
    about two samples, and averaging over three drops its amplitude below
    the deep-convection threshold entirely. The inner ring then vanishes
    from the profile and a genuine stage 4 (decaying inner eyewall) reads
    as stage 5 (inner gone) -- collapsing exactly the distinction this
    module exists to make, and the one the paper reports WVIR detects
    earlier than IR.

    A median suppresses single-bin noise just as well and preserves peak
    height.
    """
    if a.size < width:
        return a
    out = a.copy()
    half = width // 2
    for i in range(half, a.size - half):
        out[i] = float(np.median(a[i - half:i + half + 1]))
    return out


def _find_rings(radii, prof):
    """Locate convective maxima and whether they are genuinely separated.

    Peaks are found on the RAW profile and the moat is measured on the
    smoothed one. Detecting on the smoothed profile loses narrow rings:
    a decaying inner eyewall near 30 km can occupy a single 6 km bin, and
    suppressing single-bin features is precisely what a median filter is
    for. The inner ring then disappears and a genuine stage 4 reads as
    stage 5 -- collapsing the distinction the paper reports WVIR sees
    before IR does.

    Smoothing still governs the moat, where single-bin noise would
    otherwise manufacture separation between two halves of one ring.
    """
    sm = _smooth(prof)
    if not np.any(prof > DEEP_CONVECTION_K):
        return [], sm

    peaks = []
    for i in range(1, len(prof) - 1):
        if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] \
                and prof[i] > DEEP_CONVECTION_K:
            peaks.append(i)
    if not peaks:
        return [], sm

    # Keep the strongest, then any other peak far enough out with a real
    # gap between -- a shoulder on one broad ring is not a second ring.
    peaks.sort(key=lambda i: -prof[i])
    kept = [peaks[0]]
    for i in peaks[1:]:
        j = kept[0]
        if abs(radii[i] - radii[j]) < MIN_SEPARATION_KM:
            continue
        lo, hi = (min(i, j), max(i, j))
        moat = float(np.min(sm[lo:hi + 1]))
        if moat < MIN_MOAT_FRACTION * min(prof[i], prof[j]):
            kept.append(i)
    kept.sort()
    return kept, sm


def outer_extent_km(radii, prof) -> float:
    """Outermost radius still carrying deep convection.

    Stage 2 in Sanabia et al. is defined by this roughly halving, from
    about 150 km to about 75 km.
    """
    idx = np.flatnonzero(prof > DEEP_CONVECTION_K)
    return float(radii[idx[-1]]) if idx.size else 0.0


def analyze(wv_tb, ir_tb, lat, lon, center_lat: float, center_lon: float,
            history: Optional[list] = None, vmax_kt: Optional[float] = None) -> dict:
    """Classify the ERC stage from a WVIR profile plus recent history.

    `history` is a list of previous `analyze` results, oldest first.
    Several stages are defined by CHANGE rather than by a snapshot --
    erosion, decay and contraction all require a before -- so without
    history only the stages distinguishable from one frame are returned,
    and the rest are reported as undetermined rather than guessed.
    """
    radii, prof = wvir_profile(wv_tb, ir_tb, lat, lon, center_lat, center_lon)
    rings, sm = _find_rings(radii, prof)
    extent = outer_extent_km(radii, prof)
    peak = float(np.max(prof)) if prof.size else 0.0

    out = {
        "radii_km": radii.tolist(),
        "wvir_profile": prof.tolist(),
        "ring_radii_km": [float(radii[i]) for i in rings],
        "ring_peaks_k": [float(prof[i]) for i in rings],
        "outer_extent_km": extent,
        "peak_wvir_k": peak,
        "stage": None,
        "stage_name": "undetermined",
        "confidence": 0.0,
        "evidence": [],
    }

    if peak <= DEEP_CONVECTION_K:
        out["stage_name"] = "no tropopause-penetrating convection"
        out["evidence"].append(f"peak positive WVIR {peak:.1f} K is below the "
                               f"{DEEP_CONVECTION_K:.1f} K deep-convection threshold")
        return out

    prev = history[-1] if history else None
    d_extent = (extent - prev["outer_extent_km"]) if prev else None
    prev_rings = prev["ring_radii_km"] if prev else []

    # --- two separated rings -> stage 3 or 4 -------------------------
    if len(rings) >= 2:
        inner_k, outer_k = prof[rings[0]], prof[rings[-1]]
        out["evidence"].append(
            f"two separated convective rings at {radii[rings[0]]:.0f} and "
            f"{radii[rings[-1]]:.0f} km")
        # Sanabia et al.: once concentric, BOTH maxima weaken to 1.5-2.0 K
        # against 3.0-3.5 K for a single eyewall. The amplitude drop is
        # itself a stage signal, not incidental.
        if peak < 2.5:
            out["evidence"].append(
                f"peak WVIR {peak:.1f} K, consistent with the 1.5-2.0 K "
                f"reported once concentric eyewalls form")
        inner_fading = False
        if prev and prev.get("ring_peaks_k") and len(prev["ring_peaks_k"]) >= 2:
            inner_fading = inner_k < 0.8 * prev["ring_peaks_k"][0]
        if inner_fading:
            out.update(stage=4, stage_name=STAGE_NAMES[4], confidence=0.75)
            out["evidence"].append("inner ring weakening relative to the "
                                   "previous frame (detected earlier in WVIR "
                                   "than in IR, per Sanabia et al.)")
        else:
            out.update(stage=3, stage_name=STAGE_NAMES[3],
                       confidence=0.70 if peak < 2.5 else 0.55)
        return out

    # --- one ring ----------------------------------------------------
    ring_r = float(radii[rings[0]]) if rings else extent
    out["evidence"].append(f"single convective ring near {ring_r:.0f} km")

    # Stage 5: the surviving ring sits far out and the eye is clear --
    # Sanabia et al. put the inner edge near 100 km at this point.
    inner_clear = bool(np.all(prof[:max(1, int(40 / RADIAL_STEP_KM))]
                              <= DEEP_CONVECTION_K))
    if ring_r >= 70.0 and inner_clear:
        if prev and prev.get("ring_radii_km") and d_extent is not None:
            prev_r = prev["ring_radii_km"][-1] if prev["ring_radii_km"] else None
            if prev_r is not None and ring_r < prev_r - 8.0:
                out.update(stage=6, stage_name=STAGE_NAMES[6], confidence=0.70)
                out["evidence"].append(
                    f"outer ring contracted from {prev_r:.0f} to {ring_r:.0f} km")
                return out
        out.update(stage=5, stage_name=STAGE_NAMES[5], confidence=0.65)
        out["evidence"].append("inner 40 km free of deep convection")
        return out

    # Stage 2 needs a before: the outer extent roughly halving.
    if d_extent is not None and prev["outer_extent_km"] > 0:
        shrink = 1.0 - extent / prev["outer_extent_km"]
        if shrink > 0.25:
            out.update(stage=2, stage_name=STAGE_NAMES[2],
                       confidence=float(np.clip(0.4 + shrink, 0, 0.85)))
            out["evidence"].append(
                f"outer extent fell {prev['outer_extent_km']:.0f} -> "
                f"{extent:.0f} km ({shrink*100:.0f}%)")
            return out

    out.update(stage=1, stage_name=STAGE_NAMES[1],
               confidence=0.60 if peak >= 2.5 else 0.45)
    if peak >= 2.5:
        out["evidence"].append(
            f"peak WVIR {peak:.1f} K, in the 3.0-3.5 K range reported for a "
            f"single eyewall")
    if history is None:
        out["evidence"].append("no history supplied; stages defined by change "
                               "(erosion, decay, contraction) cannot be ruled out")
    return out


def format_wvir(result: dict) -> str:
    """Short block for the log."""
    if result.get("stage") is None:
        return f"WVIR: {result.get('stage_name', 'undetermined')}"
    lines = [f"WVIR ERC stage {result['stage']} ({result['stage_name']}), "
             f"confidence {result['confidence']:.2f}",
             f"  peak positive WVIR {result['peak_wvir_k']:.1f} K, "
             f"deep convection out to {result['outer_extent_km']:.0f} km"]
    for e in result.get("evidence", []):
        lines.append(f"  - {e}")
    return "\n".join(lines)
