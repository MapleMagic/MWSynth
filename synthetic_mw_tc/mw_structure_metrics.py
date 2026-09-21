"""
Structural scoring for synthetic MW fields.

WHY THIS EXISTS: every quantitative number this project had for judging a
frame was a bias/RMSE statistic, and none of them can see the thing that
actually matters about a correction.

  - The residual reported by mw_compare.compare_frequency() is measured
    on the SCALAR tb37/tb89 fields, which the ML correction never touches
    (it returns V/H only). That number is structurally incapable of
    responding to the correction at all.
  - Mean/peak |delta| in Kelvin measures magnitude, not direction. It
    cannot distinguish "sharpened the eyewall into the right radius" from
    "erased the core entirely" -- both are just a few K per pixel.

The two real A/B cases that motivated this made the point concretely. On
an intense storm the correction opened an eyewall annulus at ~33 km
against a reported RMW of 33 km, which is right. On a weak-but-organised
storm it warmed the core enough to push the 37 GHz composite out the top
of its colour window, deleting a signature that genuinely existed. Bias
statistics rated those two frames as essentially identical.

WHAT IS MEASURED, and why these three:
  - eyewall_radius_km: radius of peak "redness" in the radial profile,
    compared against the storm's RMW. A physically correct 37 GHz
    emission signature peaks near the RMW; a smeared or displaced one
    does not. This is the single most diagnostic number here.
  - core_fill_fraction: how much of the core disc is saturated rather
    than forming a ring. A real organised storm shows an eye; a filled
    disc means the structure was lost (or never resolved).
  - peak_red_pct_k / pct_window_frac: peak excursion in the space the
    composite is actually rendered in. The 37 GHz red window is only 20 K
    wide, so a modest per-channel change can traverse it entirely -- this
    is what predicts whether a correction is VISIBLE, which per-channel
    Kelvin does not.

All metrics are computed on the colour composite's red-channel PCT
combination (mw_composites.COLOR_RED_THETA), not the standalone PCT
product, because the composite is what gets looked at.

DELIBERATELY NOT a skill score against truth. These are descriptive
structural statistics for comparing two renderings of the SAME frame
(corrected vs uncorrected, or two checkpoints). Comparing them against
real MW would need a co-located real pass and is a separate problem.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mw_composites import COLOR_RED_THETA, COLOR_COMPOSITE_SPEC

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

# A pixel counts as "core" when its redness exceeds this fraction of the
# colour window. 0.5 is deliberately mid-window rather than near-
# saturation: at 89 GHz the window is 90 K wide and a near-1.0 threshold
# would only ever fire on fully clipped pixels.
CORE_REDNESS_THRESHOLD = 0.5

# Radial profile resolution. 4 km bins out to 250 km covers the eyewall
# and inner rainbands of essentially any TC while staying coarse enough
# that a single noisy pixel cannot define the peak.
RADIAL_BIN_KM = 4.0
RADIAL_MAX_KM = 250.0

# Clipping statistics are measured only within this radius of the centre.
# Over the full grid they are dominated by far-field pixels sitting at
# redness 0, which makes them read the same for every frame.
CLIP_STATS_RADIUS_KM = 150.0

# Below this core area there is no structure to describe, so the eyewall
# radius is located on a noise floor rather than on a feature. Roughly a
# 12 km-radius disc -- smaller than any real eyewall signature that would
# be resolvable here anyway.
MIN_MEANINGFUL_CORE_KM2 = 450.0


def color_red_pct(tb_v: np.ndarray, tb_h: np.ndarray, freq: int) -> np.ndarray:
    """The PCT-like combination used by the colour composite's red
    channel. NOT the same coefficients as the standalone PCT product --
    see mw_composites' module docstring."""
    theta = COLOR_RED_THETA[freq]
    return (1.0 + theta) * tb_v - theta * tb_h


def redness(tb_v: np.ndarray, tb_h: np.ndarray, freq: int) -> np.ndarray:
    """Red-channel intensity, 0-1, matching the composite's inverted
    normalization: 1 at the cold/scattering end of the window, 0 at the
    warm end. Values outside the window clip, exactly as the rendered
    image does -- that clipping is not an artifact to avoid here, it IS
    the behaviour being measured."""
    lo, hi = COLOR_COMPOSITE_SPEC[freq]["red_range"]
    pct = color_red_pct(tb_v, tb_h, freq)
    with np.errstate(invalid="ignore"):
        norm = (hi - pct) / (hi - lo)  # inverted
    return np.clip(norm, 0.0, 1.0)


def _distance_km(lat: np.ndarray, lon: np.ndarray, storm_lat: float, storm_lon: float) -> np.ndarray:
    """Great-circle-ish distance from the storm centre, in km. Uses the
    local cosine-latitude approximation rather than full haversine: over
    the few hundred km that matter here the difference is well under the
    grid resolution, and it stays vectorized and cheap."""
    dlat = lat - storm_lat
    dlon = _wrap_lon_delta(lon - storm_lon) * np.cos(np.radians(storm_lat))
    return np.sqrt(dlat ** 2 + dlon ** 2) * 111.32


def radial_pct_profile(
    tb_v: np.ndarray,
    tb_h: np.ndarray,
    freq: int,
    lat: np.ndarray,
    lon: np.ndarray,
    storm_lat: float,
    storm_lon: float,
    bin_km: float = RADIAL_BIN_KM,
    max_km: float = RADIAL_MAX_KM,
) -> tuple:
    """Azimuthally-averaged UNCLIPPED red-channel PCT vs radius.

    This exists because locating the eyewall on the clipped redness field
    does not work, and failed on the first real frame it saw. Redness is
    clipped to [0,1] because that is what the rendered image does -- but
    an intense core saturates the window over a wide annulus, so the
    profile develops a flat plateau of exactly 1.0 and argmax returns the
    FIRST bin of that plateau. The number it produces is then "where
    clipping begins", which depends on the colour-table boundary rather
    than on the storm, and it moves when the whole field shifts even if
    the structure is identical. On a monotonically-decreasing profile it
    degenerates further and returns the innermost bin.

    PCT is unbounded, so its minimum (coldest = most scattering/emission
    signature) is well-defined whether or not the composite can display
    it. Returns (bin_centres_km, mean_pct, n_pixels_per_bin).
    """
    r = _distance_km(lat, lon, storm_lat, storm_lon)

    pct = color_red_pct(tb_v, tb_h, freq)

    edges = np.arange(0.0, max_km + bin_km, bin_km)
    centres = 0.5 * (edges[:-1] + edges[1:])

    valid = np.isfinite(pct) & np.isfinite(r)
    idx = np.digitize(r[valid], edges) - 1
    vals = pct[valid]

    n_bins = len(centres)
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    in_range = (idx >= 0) & (idx < n_bins)
    np.add.at(sums, idx[in_range], vals[in_range])
    np.add.at(counts, idx[in_range], 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return centres, means, counts


def radial_redness_profile(
    tb_v: np.ndarray,
    tb_h: np.ndarray,
    freq: int,
    lat: np.ndarray,
    lon: np.ndarray,
    storm_lat: float,
    storm_lon: float,
    bin_km: float = RADIAL_BIN_KM,
    max_km: float = RADIAL_MAX_KM,
) -> tuple:
    """Azimuthally-averaged redness as a function of radius.

    Returns (bin_centres_km, mean_redness, n_pixels_per_bin). Bins with
    no pixels come back as NaN rather than 0 -- a radius that falls
    entirely outside the grid is unknown, not dark, and collapsing that
    distinction would let an off-grid annulus masquerade as an eye.
    """
    r = _distance_km(lat, lon, storm_lat, storm_lon)
    red = redness(tb_v, tb_h, freq)

    edges = np.arange(0.0, max_km + bin_km, bin_km)
    centres = 0.5 * (edges[:-1] + edges[1:])

    valid = np.isfinite(red) & np.isfinite(r)
    idx = np.digitize(r[valid], edges) - 1
    vals = red[valid]

    n_bins = len(centres)
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    in_range = (idx >= 0) & (idx < n_bins)
    np.add.at(sums, idx[in_range], vals[in_range])
    np.add.at(counts, idx[in_range], 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return centres, means, counts


def structure_metrics(
    tb_v: np.ndarray,
    tb_h: np.ndarray,
    freq: int,
    lat: np.ndarray,
    lon: np.ndarray,
    storm_lat: float,
    storm_lon: float,
    rmw_km: Optional[float] = None,
) -> dict:
    """Descriptive structural statistics for one frequency of one frame.

    Keys returned:
      eyewall_radius_km   radius of peak azimuthal-mean redness
      rmw_ratio           eyewall_radius_km / rmw_km (None if no RMW).
                          ~1.0 means the emission/scattering peak sits at
                          the radius of maximum wind, which is where a
                          real one belongs.
      peak_redness        azimuthal-mean redness at that radius, 0-1
      centre_redness      azimuthal-mean redness in the innermost bin.
                          Much lower than peak_redness = an open eye;
                          comparable = a filled disc.
      eye_contrast        1 - centre/peak. >0 means a ring, ~0 a disc.
      core_fill_fraction  fraction of pixels inside the eyewall radius
                          exceeding CORE_REDNESS_THRESHOLD
      core_area_km2       total area exceeding that threshold
      peak_red_pct_k      most extreme red-channel PCT value present
      pct_window_frac     how far the red PCT range present spans, as a
                          fraction of the rendered colour window. >1
                          means the field exceeds what the composite can
                          display, so structure is being clipped away.
      clipped_fraction    fraction of pixels sitting at 0 or 1 redness,
                          i.e. outside the window entirely
    """
    centres, means, counts = radial_redness_profile(
        tb_v, tb_h, freq, lat, lon, storm_lat, storm_lon
    )
    pct_centres, pct_means, _pct_counts = radial_pct_profile(
        tb_v, tb_h, freq, lat, lon, storm_lat, storm_lon
    )

    out = {
        "freq": freq,
        "eyewall_radius_km": None,
        "rmw_ratio": None,
        "peak_redness": None,
        "centre_redness": None,
        "eye_contrast": None,
        "core_fill_fraction": None,
        "core_area_km2": None,
        "peak_red_pct_k": None,
        "pct_window_frac": None,
        "clipped_fraction": None,
        "clipped_high_fraction": None,
        "clipped_low_fraction": None,
        "saturated_annulus_km": None,
        "eyewall_from": None,
        "ring_primary_km": None,
        "ring_secondary_km": None,
        "ring_moat_km": None,
        "concentric_eyewall": False,
        "pol_diff_far_k": None,
        "pol_diff_core_k": None,
        "pol_collapse_k": None,
    }

    finite = np.isfinite(means)
    if not np.any(finite):
        return out

    # Eyewall radius from the UNCLIPPED PCT minimum. Falls back to the
    # redness maximum only if PCT is entirely unavailable.
    finite_pct_prof = np.isfinite(pct_means)
    if np.any(finite_pct_prof):
        peak_i = int(np.nanargmin(pct_means))
        out["eyewall_radius_km"] = float(pct_centres[peak_i])
        out["eyewall_from"] = "pct_min"
    else:
        peak_i = int(np.nanargmax(means))
        out["eyewall_radius_km"] = float(centres[peak_i])
        out["eyewall_from"] = "redness_max_fallback"
    eyewall_r = out["eyewall_radius_km"]
    out["peak_redness"] = float(np.nanmax(means))

    # Width of the annulus where the composite is saturated. When this is
    # large the eyewall is visually indistinguishable from its
    # surroundings in the rendered image no matter where the true peak
    # is, so the radius above is real but not something a viewer can see.
    sat_bins = int(np.sum(means[finite] >= 1.0 - 1e-9))
    out["saturated_annulus_km"] = float(sat_bins * RADIAL_BIN_KM)

    first_valid = int(np.argmax(finite))
    out["centre_redness"] = float(means[first_valid])
    if out["peak_redness"] > 1e-6:
        out["eye_contrast"] = float(1.0 - out["centre_redness"] / out["peak_redness"])

    if rmw_km and np.isfinite(rmw_km) and rmw_km > 0:
        out["rmw_ratio"] = eyewall_r / float(rmw_km)

    red = redness(tb_v, tb_h, freq)
    r = _distance_km(lat, lon, storm_lat, storm_lon)
    core_mask = red >= CORE_REDNESS_THRESHOLD
    inside = r <= max(eyewall_r, RADIAL_BIN_KM)
    if np.any(inside):
        out["core_fill_fraction"] = float(np.mean(core_mask[inside]))

    # Per-pixel area from the grid spacing, so core area is in real km^2
    # rather than a pixel count that means nothing across different crops.
    if lat.shape[0] > 1 and lat.shape[1] > 1:
        dlat_km = abs(float(np.nanmedian(np.diff(lat, axis=0)))) * 111.32
        dlon_km = abs(float(np.nanmedian(np.diff(lon, axis=1)))) * 111.32 * np.cos(np.radians(storm_lat))
        out["core_area_km2"] = float(np.sum(core_mask) * dlat_km * dlon_km)



    pct = color_red_pct(tb_v, tb_h, freq)
    lo, hi = COLOR_COMPOSITE_SPEC[freq]["red_range"]
    window = hi - lo
    finite_pct = pct[np.isfinite(pct)]
    if finite_pct.size:
        span = float(np.nanmax(finite_pct) - np.nanmin(finite_pct))
        out["pct_window_frac"] = span / window
        # Report the excursion furthest from the window, which is the one
        # that actually costs visible structure.
        out["peak_red_pct_k"] = float(
            max(np.nanmax(finite_pct) - hi, lo - np.nanmin(finite_pct), 0.0)
        )
    finite_red = red[np.isfinite(red)]
    if finite_red.size:
        # Measured INSIDE the storm region only. Computed over the whole
        # grid this number is useless: the far field sits at redness 0
        # (clipped low) almost everywhere, so it reads ~0.95 for every
        # frame regardless of what happened to the core, and swamps the
        # signal it was supposed to carry.
        near = np.isfinite(red) & (r <= CLIP_STATS_RADIUS_KM)
        if np.any(near):
            near_red = red[near]
            out["clipped_fraction"] = float(np.mean((near_red <= 1e-9) | (near_red >= 1 - 1e-9)))
            out["clipped_high_fraction"] = float(np.mean(near_red >= 1 - 1e-9))
            out["clipped_low_fraction"] = float(np.mean(near_red <= 1e-9))

    # Polarization difference (V - H), and how far it collapses in the
    # core. This is the physically meaningful descriptor at 37 GHz, and
    # the redness metrics above are not: a real 37 GHz eyewall is an
    # optically thick, nearly UNPOLARIZED emitting layer, so V-H falls
    # from ~55-70 K over ocean toward zero, and the NRL composite renders
    # that as a bright WHITE core rather than a red one. Checked against a
    # real GMI pass, a synthetic field can score "core area 0 km2" and be
    # judged empty when the real product is barely red either -- while
    # being wrong by over 100 K in a way redness cannot see.
    #
    # At 89 GHz this is still informative (scattering depresses H more
    # than V, so V-H tends to WIDEN) but there it is the scattering
    # depression, not polarization, that carries the signal.
    # --- Ring-score profile (ARCHER / M-PERC style) -------------------
    # Detects a SECOND eyewall, which eyewall_radius_km structurally
    # cannot: it takes the PCT minimum and returns one radius, so every
    # structural comparison in this project has been blind to concentric
    # structure -- the one transition the ERC literature is about.
    try:
        _radii, _scores = ring_score_profile(tb_v, tb_h, freq, lat, lon,
                                             center_lat, center_lon)
        _rings = find_eyewall_rings(_radii, _scores, rmw_km=rmw_km)
        out["ring_primary_km"] = _rings["primary_km"]
        out["ring_secondary_km"] = _rings["secondary_km"]
        out["ring_moat_km"] = _rings["moat_km"]
        out["concentric_eyewall"] = _rings["concentric"]
        out["ring_score_peak"] = _rings["primary_score"]
    except Exception:
        pass    # diagnostic only; must never fail a generate

    pol = tb_v - tb_h
    far_m = np.isfinite(pol) & (r > 2.0 * max(eyewall_r, RADIAL_BIN_KM)) & (r <= RADIAL_MAX_KM)
    core_r = np.isfinite(pol) & (r <= max(eyewall_r, RADIAL_BIN_KM))
    if np.any(far_m):
        out["pol_diff_far_k"] = float(np.nanmedian(pol[far_m]))
    if np.any(core_r):
        out["pol_diff_core_k"] = float(np.nanmedian(pol[core_r]))
    if out["pol_diff_far_k"] is not None and out["pol_diff_core_k"] is not None:
        out["pol_collapse_k"] = out["pol_diff_far_k"] - out["pol_diff_core_k"]

    return out


def compare_structure(before: dict, after: dict) -> str:
    """One-line human summary of what a correction did structurally,
    for the progress log. `before` is the uncorrected (ML-free) metrics,
    `after` the corrected ones."""
    def _f(d, k, fmt="{:.1f}"):
        v = d.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

    # An eyewall radius is only meaningful if there is actually a ring.
    # On a flat filled disc the PCT minimum is arbitrary within the disc,
    # so the radius is noise and reporting it invites reading a trend
    # into it. eye_contrast is the guard: near zero means no ring.
    # An eyewall radius is only meaningful if there is actually a ring,
    # AND if there is a core to have a ring in. On a flat filled disc the
    # PCT minimum is arbitrary within the disc; on a field with no core
    # at all the minimum is a noise floor. Both were being reported as
    # confident radii -- a real frame reported `eyewall/RMW 1.19` on a
    # field whose core area was exactly 0 km2, which is a number computed
    # from nothing. eye_contrast guards the first case, core area the
    # second.
    def _has_core(d):
        area = d.get("core_area_km2")
        return isinstance(area, float) and area >= MIN_MEANINGFUL_CORE_KM2

    ring_before = (isinstance(before.get("eye_contrast"), float) and before["eye_contrast"] > 0.05
                   and _has_core(before))
    ring_after = (isinstance(after.get("eye_contrast"), float) and after["eye_contrast"] > 0.05
                  and _has_core(after))

    if ring_before or ring_after:
        bits = [
            f"eyewall {_f(before,'eyewall_radius_km')}->{_f(after,'eyewall_radius_km')} km",
            f"RMW ratio {_f(before,'rmw_ratio','{:.2f}')}->{_f(after,'rmw_ratio','{:.2f}')}",
        ]
    elif not (_has_core(before) or _has_core(after)):
        bits = ["no core present (radius not meaningful)"]
    else:
        bits = ["no resolved eyewall (filled core, radius not meaningful)"]
    bits += [
        f"eye contrast {_f(before,'eye_contrast','{:.2f}')}->{_f(after,'eye_contrast','{:.2f}')}",
        f"core area {_f(before,'core_area_km2','{:.0f}')}->{_f(after,'core_area_km2','{:.0f}')} km2",
    ]
    if after.get("concentric_eyewall"):
        bits.append(f"CONCENTRIC: rings {after['ring_primary_km']:.0f}/"
                    f"{after['ring_secondary_km']:.0f} km, moat "
                    f"{after['ring_moat_km']:.0f} km")
    pc = after.get("pol_collapse_k")
    pb = before.get("pol_collapse_k")
    if isinstance(pc, float) and isinstance(pb, float):
        bits.append(f"V-H collapse {pb:.0f}->{pc:.0f} K")
    sat = after.get("saturated_annulus_km")
    if isinstance(sat, float) and sat >= 20.0:
        bits.append(f"saturated over {sat:.0f} km of radius")
    # Warning logic has to distinguish two things that look identical in
    # core-area alone. Opening an eye LEGITIMATELY shrinks the core area
    # (a ring covers less than the disc it replaced) -- that is the
    # correction working. Suppressing a real signature also shrinks core
    # area. The discriminator is peak redness: opening an eye leaves the
    # eyewall just as red as the disc was, while pushing the field out of
    # the colour window takes the peak down with it. Requiring BOTH to
    # collapse is what separates them; keying on area alone flagged a
    # known-good disc->ring improvement as a failure.
    warn = ""
    b_area, a_area = before.get("core_area_km2"), after.get("core_area_km2")
    b_peak, a_peak = before.get("peak_redness"), after.get("peak_redness")
    area_collapsed = (
        isinstance(b_area, float) and isinstance(a_area, float)
        and b_area > 500.0 and a_area < 0.35 * b_area
    )
    peak_collapsed = (
        isinstance(b_peak, float) and isinstance(a_peak, float)
        and b_peak > 0.4 and a_peak < 0.6 * b_peak
    )
    if area_collapsed and peak_collapsed:
        warn = (f"  [WARNING: core area -{100*(1-a_area/b_area):.0f}% AND peak redness "
                f"{b_peak:.2f}->{a_peak:.2f}. The signature is being pushed outside the colour "
                f"window, not resolved into structure]")
    elif peak_collapsed:
        warn = (f"  [WARNING: peak redness collapsed {b_peak:.2f}->{a_peak:.2f} -- the signature "
                f"is being pushed outside the colour window]")
    else:
        # Positive note when the eyewall moved TOWARD the RMW, which is
        # the outcome this whole metric exists to detect.
        b_ratio, a_ratio = before.get("rmw_ratio"), after.get("rmw_ratio")
        if isinstance(b_ratio, float) and isinstance(a_ratio, float):
            if abs(a_ratio - 1.0) < abs(b_ratio - 1.0) - 0.15:
                warn = "  [eyewall moved toward the RMW]"
    return f"{before.get('freq', '?')} GHz: " + ", ".join(bits) + warn


# --- Ring-score radial profile (ARCHER / M-PERC style) ----------------

RING_SCORE_STEP_KM = 6.0      # Kossin et al. (2023) evaluate every 6 km
RING_SCORE_MAX_KM = 200.0     # and out to 200 km
RING_AZIMUTH_BINS = 36        # 10-degree sectors


def ring_score_profile(tb_v, tb_h, freq, lat, lon, center_lat, center_lon,
                       step_km: float = RING_SCORE_STEP_KM,
                       max_km: float = RING_SCORE_MAX_KM):
    """Radial profile of how ring-like the convective signature is.

    Modelled on the ARCHER ring score used by M-PERC (Kossin et al. 2023,
    WAF-D-22-0178): a score computed every 6 km along a radial out to
    200 km, whose profile is then "interrogated for secondary maxima" --
    which is how a secondary eyewall is detected at all.

    WHY THIS AND NOT WHAT WAS HERE. `eyewall_radius_km` takes the PCT
    minimum and returns ONE radius. A concentric eyewall has two rings;
    the existing metric reports one of them and gives no signal that a
    second exists. Every structural comparison this project has made has
    therefore been blind to the single structural transition the ERC
    literature is about.

    The score at radius r combines two things a real eyewall has and a
    random cold blob does not:
      * STRENGTH  -- mean redness in the annulus, relative to the frame
      * SYMMETRY  -- how uniformly that redness is spread in azimuth,
                     since ARCHER scores gradients "by how circularly
                     symmetric they are"

    A one-sided convective arc scores low even when intense, which is
    correct: Norbert's 55-kt arc is not an eyewall.

    Returns (radii_km, scores) with scores in 0-1.
    """
    r = _distance_km(lat, lon, center_lat, center_lon)
    red = redness(tb_v, tb_h, freq)
    theta = np.degrees(np.arctan2(
        np.asarray(lat, dtype=np.float64) - center_lat,
        _wrap_lon_delta(np.asarray(lon, dtype=np.float64) - center_lon)
        * np.cos(np.radians(center_lat)))) % 360.0

    radii = np.arange(step_km, max_km + step_km, step_km)
    scores = np.full(radii.shape, np.nan)
    bin_edges = np.linspace(0.0, 360.0, RING_AZIMUTH_BINS + 1)

    for i, rad in enumerate(radii):
        ann = (r >= rad - step_km / 2) & (r < rad + step_km / 2) & np.isfinite(red)
        if ann.sum() < RING_AZIMUTH_BINS:
            continue
        a_red, a_theta = red[ann], theta[ann]
        which = np.clip(np.digitize(a_theta, bin_edges) - 1, 0, RING_AZIMUTH_BINS - 1)
        sector = np.full(RING_AZIMUTH_BINS, np.nan)
        for b in range(RING_AZIMUTH_BINS):
            sel = a_red[which == b]
            if sel.size:
                sector[b] = float(np.mean(sel))
        filled = np.isfinite(sector)
        if filled.sum() < RING_AZIMUTH_BINS * 0.6:
            continue      # too little azimuthal coverage to call it a ring

        strength = float(np.nanmean(sector))
        # Symmetry: 1 when every sector agrees, falling as the signal
        # concentrates into part of the circle. Normalised by the mean so
        # it measures SHAPE, not amplitude.
        spread = float(np.nanstd(sector))
        symmetry = 1.0 / (1.0 + (spread / max(strength, 1e-6)))
        # Fraction of the circle that is convective at all -- Kossin's
        # SEF criterion is a ring forming "at least 75% of a complete
        # circle", so partial arcs must be penalised explicitly rather
        # than averaged away.
        closure = float(np.nanmean(sector[filled] > 0.5 * strength))
        scores[i] = float(np.clip(strength * symmetry * closure, 0.0, 1.0))

    return radii, scores


def find_eyewall_rings(radii, scores, rmw_km=None, min_score: float = 0.05,
                       min_separation_km: float = 24.0):
    """Locate primary and (if present) secondary eyewall rings.

    M-PERC's whole mechanism is reading SECONDARY maxima out of the
    ring-score profile, so this returns both, plus the moat between them.

    min_separation_km keeps two samples of one broad eyewall from being
    reported as concentric rings; 24 km is four profile steps, and
    observed moats are wider than that (Sitkowski et al. 2011).
    """
    radii = np.asarray(radii, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    if ok.sum() < 5:
        return {"primary_km": None, "secondary_km": None, "moat_km": None,
                "concentric": False, "primary_score": None, "secondary_score": None}

    # Local maxima strictly above the floor.
    peaks = []
    for i in range(1, len(s) - 1):
        if not (np.isfinite(s[i-1]) and np.isfinite(s[i]) and np.isfinite(s[i+1])):
            continue
        if s[i] >= s[i-1] and s[i] > s[i+1] and s[i] > min_score:
            peaks.append((float(radii[i]), float(s[i])))
    if not peaks:
        return {"primary_km": None, "secondary_km": None, "moat_km": None,
                "concentric": False, "primary_score": None, "secondary_score": None}

    peaks.sort(key=lambda p: -p[1])
    primary_km, primary_score = peaks[0]

    secondary = None
    for rad, sc in peaks[1:]:
        if abs(rad - primary_km) >= min_separation_km:
            secondary = (rad, sc)
            break

    out = {"primary_km": primary_km, "primary_score": primary_score,
           "secondary_km": None, "secondary_score": None,
           "moat_km": None, "concentric": False}
    if secondary is not None:
        out["secondary_km"], out["secondary_score"] = secondary
        inner, outer = sorted([primary_km, secondary[0]])
        out["moat_km"] = outer - inner
        # A secondary eyewall is OUTSIDE the primary by definition.
        out["concentric"] = bool(secondary[0] > primary_km)
    return out
