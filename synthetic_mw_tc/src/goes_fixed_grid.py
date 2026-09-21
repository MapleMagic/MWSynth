"""
GOES-R ABI fixed-grid geolocation, and storm-centred crops from
full-disk / CONUS files.

WHY THIS EXISTS. Mining only ever looked at RadM -- the two steerable
~1000 x 1000 km mesoscale sectors. Operators point those at whatever is
operationally interesting, so most storm-times have both aimed somewhere
else. Measured over a 2018-2025 run: **83% of all attempts (4,023 of
4,867) were lost to "no covering sector"**, against 799 saved. Full disk
covers the whole hemisphere every 10 minutes and CONUS every 5, so
reaching them is worth roughly 3-5x the dataset -- more than every other
outstanding improvement combined.

The objection is size: a full-disk band 13 is 5424 x 5424 at 2 km, tens of
MB compressed, against ~10 MB for a mesoscale file. But a storm-centred
crop is a tiny fraction of that, and `s3_range_reader` already reads HDF5
in place over ranged GETs. Its ranged path is retained specifically for
objects above 64 MB -- which is exactly this case. 0.121 found that
ranging bought nothing for small TC-PRIMED files; here it is the whole
point.

THE PROJECTION. ABI products are on a fixed grid in SCAN ANGLE, not
lat/lon: a geostationary perspective projection where each pixel is a
fixed (x, y) angle pair as seen from the satellite. The forward transform
(geodetic lat/lon -> scan angles) and its inverse are given in the
GOES-R Product Definition and Users' Guide, Volume 5, section 4.2.8, and
are implemented here directly rather than via an external projection
library, which would be a heavy dependency for about forty lines of
trigonometry.

Both directions are needed: forward to locate the storm in the array, and
inverse to attach real lat/lon to the pixels of the crop that comes back.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# GRS80 ellipsoid and GOES orbit, per the PUG. These are exact constants
# of the product definition, not tunable parameters.
R_EQ_M = 6378137.0                 # semi-major axis
R_POL_M = 6356752.31414            # semi-minor axis
H_M = 42164160.0                   # satellite distance from Earth centre
ECC_SQ = 1.0 - (R_POL_M ** 2) / (R_EQ_M ** 2)
_RATIO_SQ = (R_EQ_M ** 2) / (R_POL_M ** 2)


def lat_lon_to_scan_angles(lat, lon, sat_lon: float):
    """Geodetic lat/lon -> ABI fixed-grid scan angles (x, y) in radians.

    Returns (x, y, visible). `visible` is False where the point is beyond
    the limb -- the geometry still produces numbers there, and they are
    meaningless, so the mask must be honoured rather than assumed.
    """
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lon_0 = np.radians(float(sat_lon))

    # Geocentric latitude, and radius at that latitude.
    lat_c = np.arctan((R_POL_M ** 2 / R_EQ_M ** 2) * np.tan(lat))
    r_c = R_POL_M / np.sqrt(1.0 - ECC_SQ * np.cos(lat_c) ** 2)

    # Satellite-to-point vector in the geostationary frame.
    s_x = H_M - r_c * np.cos(lat_c) * np.cos(lon - lon_0)
    s_y = -r_c * np.cos(lat_c) * np.sin(lon - lon_0)
    s_z = r_c * np.sin(lat_c)

    # Beyond-the-limb test from the PUG. Points failing this are on the
    # far side of the Earth and project to plausible-looking but wrong
    # angles.
    visible = H_M * (H_M - s_x) > (s_y ** 2 + _RATIO_SQ * s_z ** 2)

    y = np.arctan(s_z / s_x)
    x = np.arcsin(-s_y / np.sqrt(s_x ** 2 + s_y ** 2 + s_z ** 2))
    return x, y, visible


def scan_angles_to_lat_lon(x, y, sat_lon: float):
    """ABI scan angles -> geodetic lat/lon. Inverse of the above.

    Returns (lat, lon, valid); `valid` is False where the ray misses the
    Earth, which happens for the corners of any rectangular crop near the
    limb.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon_0 = np.radians(float(sat_lon))

    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)

    a = sin_x ** 2 + cos_x ** 2 * (cos_y ** 2 + _RATIO_SQ * sin_y ** 2)
    b = -2.0 * H_M * cos_x * cos_y
    c = H_M ** 2 - R_EQ_M ** 2

    disc = b ** 2 - 4.0 * a * c
    valid = disc >= 0.0
    # Clamp only to keep the sqrt finite; `valid` carries the truth.
    disc = np.where(valid, disc, 0.0)

    r_s = (-b - np.sqrt(disc)) / (2.0 * a)
    s_x = r_s * cos_x * cos_y
    s_y = -r_s * sin_x
    s_z = r_s * cos_x * sin_y

    denom = np.sqrt((H_M - s_x) ** 2 + s_y ** 2)
    lat = np.arctan(_RATIO_SQ * s_z / np.where(denom == 0, 1e-12, denom))
    lon = lon_0 - np.arctan(s_y / np.where((H_M - s_x) == 0, 1e-12, H_M - s_x))
    return np.degrees(lat), np.degrees(lon), valid


def crop_bounds(x_coords, y_coords, center_lat: float, center_lon: float,
                sat_lon: float, half_width_km: float = 512.0
                ) -> Optional[Tuple[int, int, int, int]]:
    """Index bounds (y0, y1, x0, x1) of a storm-centred crop.

    `x_coords` / `y_coords` are the file's 1-D scan-angle axes. Returns
    None if the storm is beyond the limb or falls outside the product's
    coverage -- CONUS in particular covers only part of the disk, so this
    is a normal outcome and not an error.
    """
    x_c, y_c, visible = lat_lon_to_scan_angles(center_lat, center_lon, sat_lon)
    if not bool(np.all(visible)):
        return None

    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)

    # Angular half-width. At the sub-satellite point one radian of scan
    # angle subtends H_M metres; away from nadir the same angle covers
    # more ground, so this OVER-covers toward the limb, which is the safe
    # direction -- a slightly larger crop costs a few hundred KB, a
    # slightly small one loses the storm.
    half_angle = (half_width_km * 1000.0) / H_M

    ix = np.searchsorted(x_coords, [float(x_c) - half_angle, float(x_c) + half_angle])
    # y axis runs north-to-south, i.e. DECREASING, so searchsorted needs
    # the reversed array. Getting this backwards yields an empty slice
    # rather than an error, which is why it is asserted in the tests.
    y_desc = y_coords[0] > y_coords[-1]
    if y_desc:
        rev = y_coords[::-1]
        iy_rev = np.searchsorted(rev, [float(y_c) - half_angle, float(y_c) + half_angle])
        iy = [len(y_coords) - iy_rev[1], len(y_coords) - iy_rev[0]]
    else:
        iy = np.searchsorted(y_coords, [float(y_c) - half_angle, float(y_c) + half_angle])

    x0, x1 = int(np.clip(ix[0], 0, len(x_coords))), int(np.clip(ix[1], 0, len(x_coords)))
    y0, y1 = int(np.clip(iy[0], 0, len(y_coords))), int(np.clip(iy[1], 0, len(y_coords)))
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def crop_lat_lon(x_coords, y_coords, bounds, sat_lon: float):
    """Build 2-D lat/lon arrays for a crop, from the scan-angle axes.

    Invalid (off-Earth) pixels come back as NaN rather than as silently
    wrong coordinates -- near the limb a rectangular crop genuinely has
    corners that miss the Earth.
    """
    y0, y1, x0, x1 = bounds
    xs = np.asarray(x_coords, dtype=np.float64)[x0:x1]
    ys = np.asarray(y_coords, dtype=np.float64)[y0:y1]
    xx, yy = np.meshgrid(xs, ys)
    lat, lon, valid = scan_angles_to_lat_lon(xx, yy, sat_lon)
    lat = np.where(valid, lat, np.nan)
    lon = np.where(valid, lon, np.nan)
    return lat, lon


def estimate_crop_fraction(bounds, x_len: int, y_len: int) -> float:
    """Share of the full array a crop covers -- what ranged reads save."""
    y0, y1, x0, x1 = bounds
    return ((y1 - y0) * (x1 - x0)) / float(max(x_len * y_len, 1))


# --- Reading a crop out of a full-disk / CONUS file -------------------

def read_cropped_radiance(h5, center_lat: float, center_lon: float,
                          sat_lon: float, half_width_km: float = 512.0):
    """Read a storm-centred crop of `Rad` from an OPEN ABI L1b file.

    Takes an already-open h5py-like handle rather than a path, for the
    same reason `read_overpass_as_swath` does: it lets the identical logic
    run against a file streamed from S3 and against an offline fake, with
    no duplicated parsing to drift apart.

    Returns (rad, dqf, lat, lon, meta) or None if the storm is not covered.
    `rad` is descaled to physical radiance; `dqf` may be None if the file
    carries no quality flags.

    Only the covering chunks are read. Over `s3_range_reader` that means a
    512 km crop transfers roughly 0.6% of a 59 MB full-disk array instead
    of all of it -- which is the entire reason full disk is affordable.
    """
    x_coords = _read_coord(h5, "x")
    y_coords = _read_coord(h5, "y")
    if x_coords is None or y_coords is None:
        return None

    bounds = crop_bounds(x_coords, y_coords, center_lat, center_lon,
                         sat_lon, half_width_km=half_width_km)
    if bounds is None:
        return None
    y0, y1, x0, x1 = bounds

    rad_ds = h5["Rad"]
    # THE slice that matters: HDF5 reads only the chunks this touches.
    raw = np.asarray(rad_ds[y0:y1, x0:x1], dtype=np.float64)

    scale = _attr(rad_ds, "scale_factor", 1.0)
    offset = _attr(rad_ds, "add_offset", 0.0)
    fill = _attr(rad_ds, "_FillValue", None)

    rad = raw * scale + offset
    if fill is not None:
        # Compare against the RAW value: the fill sentinel is a stored
        # integer, and descaling it first would make the comparison miss.
        rad = np.where(raw == fill, np.nan, rad)

    dqf = None
    if "DQF" in _keys(h5):
        dqf = np.asarray(h5["DQF"][y0:y1, x0:x1])

    lat, lon = crop_lat_lon(x_coords, y_coords, bounds, sat_lon)

    meta = {
        "bounds": bounds,
        "crop_fraction": estimate_crop_fraction(bounds, len(x_coords), len(y_coords)),
        "planck": {k: _scalar(h5, k) for k in
                   ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")},
        "kappa0": _scalar(h5, "kappa0"),
    }
    return rad, dqf, lat, lon, meta


def _keys(h5):
    try:
        return set(h5.keys())
    except Exception:
        return set()


def _read_coord(h5, name):
    """Read and descale a 1-D coordinate axis.

    ABI stores x/y as scaled shorts. Forgetting the scale_factor here
    produces coordinates off by four orders of magnitude, which yields an
    empty crop rather than an error -- so this is done once, centrally.
    """
    if name not in _keys(h5):
        return None
    ds = h5[name]
    raw = np.asarray(ds[:], dtype=np.float64)
    return raw * _attr(ds, "scale_factor", 1.0) + _attr(ds, "add_offset", 0.0)


def _attr(ds, name, default):
    try:
        v = ds.attrs[name]
    except Exception:
        return default
    try:
        return float(np.asarray(v).reshape(-1)[0])
    except Exception:
        return default


def _scalar(h5, name):
    if name not in _keys(h5):
        return None
    try:
        return float(np.asarray(h5[name][()]).reshape(-1)[0])
    except Exception:
        return None


def radiance_to_brightness_temperature(rad, planck: dict):
    """ABI L1b radiance -> brightness temperature (PUG sec 5.1.2).

    Mirrors the conversion goes_fetch applies to mesoscale files. Kept
    here so the cropped path does not depend on an xarray Dataset, which
    it never has.
    """
    fk1, fk2 = planck.get("planck_fk1"), planck.get("planck_fk2")
    bc1, bc2 = planck.get("planck_bc1"), planck.get("planck_bc2")
    if None in (fk1, fk2, bc1, bc2):
        return None
    rad = np.asarray(rad, dtype=np.float64)
    # Radiance must be strictly positive for the log to be finite; space
    # looks and fill values are non-positive and become NaN rather than
    # inf leaking downstream.
    bad = ~np.isfinite(rad) | (rad <= 0)
    safe = np.where(bad, 1.0, rad)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb = (fk2 / np.log((fk1 / safe) + 1.0) - bc1) / bc2
    return np.where(bad, np.nan, tb)
