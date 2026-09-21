"""
Parallax correction for geostationary cloud-top displacement.

THE PROBLEM. GOES views a tropical cyclone from the equator at a slant.
A cloud top 12-16 km above the surface is therefore SEEN at a position
displaced away from the subsatellite point by roughly h * tan(theta),
where theta is the satellite zenith angle. The microwave target does not
share that displacement -- PMW senses the column, and a low-Earth-orbit
sensor views it from a completely different geometry.

So every training pair has the IR structure offset from the MW structure
by a systematic, spatially-varying amount, and the correction model has
been quietly asked to learn a coordinate transform on top of the physics.
That is a poor use of a few hundred examples and it shows up as bias and
blur rather than as anything obviously wrong.

Measured for real cases in this project (h = 12-16 km):

    Lowell EP12 / GOES-18     20.8 deg    4.6 - 6.1 km
    Karina EP11 / GOES-18     20.6 deg    4.5 - 6.0 km
    Edouard AL05 / GOES-16    34.1 deg    8.1 - 10.8 km
    Atlantic far east / G16   39.8 deg   10.0 - 13.3 km

Against Lowell's 19 km RMW a 6 km displacement is about a third of the
radius of maximum wind, and for a compact storm at the edge of the disk
it approaches the RMW itself. This is not a rounding error.

It also biases the IR centre check in tc_center_fix, though less than the
figures above suggest: the eye is a WARM, low feature, so its apparent
displacement is much smaller than that of the cold tops surrounding it.
That asymmetry is itself a reason to correct per-pixel by height rather
than shifting the whole scene by one number.

WHAT THIS DOES NOT MODEL. Cloud-top height is estimated from brightness
temperature against a fixed tropical lapse rate. Real height retrieval
uses the CO2-slicing or split-window channels, accounts for tropopause
variation, and handles semi-transparent cirrus (where the radiating level
is above the physical cloud base but below its top). A 2 km height error
becomes roughly a 1 km position error at 25 degrees zenith, which is well
inside a grid cell -- so the crude estimate is adequate for the geometry
even though it would not be adequate for cloud physics.
"""
from __future__ import annotations

import numpy as np

# Tropical mean sea-level temperature and lapse rate used to turn a
# brightness temperature into a height. Deliberately simple: the
# geometry is far less sensitive to height error than to being ignored.
SURFACE_TEMP_K = 300.0
LAPSE_RATE_K_PER_KM = 6.5

# Cap on retrieved height. Above the tropopause the lapse rate inverts,
# so colder no longer means higher -- an overshooting top at 190 K would
# otherwise be placed at an absurd altitude and displaced accordingly.
MAX_CLOUD_TOP_KM = 17.0

# Below this height the displacement is under a kilometre for any
# realistic view angle, so correcting is not worth resampling the field.
MIN_CORRECT_KM = 2.0


def cloud_top_height_km(ir_tb) -> np.ndarray:
    """Crude cloud-top height from IR brightness temperature."""
    tb = np.asarray(ir_tb, dtype=np.float64)
    h = (SURFACE_TEMP_K - tb) / LAPSE_RATE_K_PER_KM
    return np.clip(h, 0.0, MAX_CLOUD_TOP_KM)


def view_angle_deg(sat_lon: float, lat, lon) -> np.ndarray:
    """Satellite ZENITH angle at each grid point, in degrees.

    NOT the same quantity as goes_fetch._view_angle_deg(), which returns
    the Earth-CENTRAL angle between the point and the subsatellite point
    (verified: 20.82 deg for Lowell, against 24.40 here). The central
    angle is a fine proxy for a coverage cutoff and goes_fetch's
    thresholds are calibrated to it, so that function is deliberately
    left alone -- but parallax needs the angle from the local vertical,
    which is always the larger of the two, and using the central angle
    would under-correct by 15-20%.
    """
    latr = np.radians(np.asarray(lat, dtype=np.float64))
    dlonr = np.radians(np.asarray(lon, dtype=np.float64) - sat_lon)
    # Angle subtended at Earth centre between the point and the
    # subsatellite point.
    cos_gamma = np.clip(np.cos(latr) * np.cos(dlonr), -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    Re, Rs = 6371.0, 42164.0
    # Zenith angle at the surface point for a satellite at radius Rs.
    return np.degrees(np.arctan2(Rs * np.sin(gamma), Rs * np.cos(gamma) - Re))


def parallax_offsets(lat, lon, height_km, sat_lon: float) -> tuple:
    """Apparent-minus-true displacement, in degrees (dlat, dlon).

    A cloud top is SEEN displaced away from the subsatellite point, so
    these are the offsets to subtract to recover the true position.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h = np.asarray(height_km, dtype=np.float64)

    theta = np.radians(view_angle_deg(sat_lon, lat, lon))
    dist_km = h * np.tan(theta)

    # Direction: away from the subsatellite point, along the great circle.
    # A local flat-Earth bearing is accurate at these distances (the
    # displacement is tens of km at most).
    dlat_dir = lat - 0.0                      # subsatellite latitude is 0
    dlon_dir = (lon - sat_lon + 180.0) % 360.0 - 180.0
    coslat = np.cos(np.radians(lat))
    norm = np.hypot(dlat_dir, dlon_dir * coslat)
    norm = np.where(norm < 1e-9, 1.0, norm)

    dlat = dist_km / 111.32 * (dlat_dir / norm)
    dlon = dist_km / 111.32 * (dlon_dir * coslat / norm) / np.maximum(coslat, 1e-6)
    return dlat, dlon


def correct_field(field, lat, lon, ir_tb, sat_lon: float, order: int = 1):
    """Move an IR-derived field from apparent to true ground position.

    Resamples so the value seen at the APPARENT location is placed at the
    TRUE one. Per-pixel by height, not a single scene shift, because a
    warm eye and the cold tops around it are displaced by very different
    amounts -- which is exactly the structure that matters here.
    """
    from scipy.ndimage import map_coordinates

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    arr = np.asarray(field, dtype=np.float64)
    if lat.shape[0] < 2 or lat.shape[1] < 2:
        return field

    h = cloud_top_height_km(ir_tb)
    if np.nanmax(h) < MIN_CORRECT_KM:
        return field

    dlat, dlon = parallax_offsets(lat, lon, h, sat_lon)

    # Degrees per pixel along each axis, from the grid itself.
    dlat_dr = np.gradient(lat, axis=0)
    dlon_dc = np.gradient(lon, axis=1)
    dlat_dr = np.where(np.abs(dlat_dr) < 1e-12, 1e-12, dlat_dr)
    dlon_dc = np.where(np.abs(dlon_dc) < 1e-12, 1e-12, dlon_dc)

    rows, cols = np.indices(arr.shape, dtype=np.float64)
    # Sample the input at the apparent position of each true pixel.
    src_r = rows + dlat / dlat_dr
    src_c = cols + dlon / dlon_dc

    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    out = map_coordinates(filled, [src_r, src_c], order=order,
                          mode="nearest", prefilter=False)
    if not finite.all():
        w = map_coordinates(finite.astype(np.float64), [src_r, src_c],
                            order=order, mode="nearest", prefilter=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(w > 1e-6, out / w, np.nan)
    return out


def describe(lat, lon, ir_tb, sat_lon: float) -> str:
    """One-line summary, so a correction of this size is never silent."""
    h = cloud_top_height_km(ir_tb)
    theta = view_angle_deg(sat_lon, lat, lon)
    dist = h * np.tan(np.radians(theta))
    return (f"Parallax: view {np.nanmean(theta):.0f} deg, cloud top up to "
            f"{np.nanmax(h):.1f} km -> shift up to {np.nanmax(dist):.1f} km "
            f"(mean {np.nanmean(dist):.1f} km)")
