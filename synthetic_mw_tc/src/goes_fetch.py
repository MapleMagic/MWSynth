"""
GOES-18 / GOES-19 mesoscale (RadM) Level-1b radiance ingestion.

Buckets are public, unsigned S3 access (no AWS credentials required):
    https://noaa-goes18.s3.amazonaws.com/index.html
    https://noaa-goes19.s3.amazonaws.com/index.html

Key layout:
    ABI-L1b-RadM/<YYYY>/<DDD>/<HH>/
        OR_ABI-L1b-RadM1-M6C13_G18_sYYYYDDDHHMMSSS_eYYYYDDDHHMMSSS_cYYYYDDDHHMMSSS.nc
        OR_ABI-L1b-RadM2-M6C13_G18_...

    RadM1 / RadM2 = the two independently-steerable mesoscale sectors.
    M6 = scan mode 6 (current operational mode, 60s per mesoscale frame).
    C## = channel/band number (01-16).

We only need bands 2, 7, 9, 13 per the project spec:
    Band 2  (0.64  um, visible)              -> reflectance factor
    Band 7  (3.9   um, shortwave IR)          -> brightness temperature (K)
    Band 9  (6.9   um, mid-level water vapor) -> brightness temperature (K)
    Band 13 (10.3  um, clean IR window)       -> brightness temperature (K)
"""
from __future__ import annotations

import io
from dataclasses import dataclass
import math
import threading
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError:  # pragma: no cover
    boto3 = None

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

from qc_utils import sanitize_field


_LISTING_CACHE: dict = {}
_LISTING_CACHE_LOCK = threading.Lock()
_LISTING_CACHE_MAX = 48  # ~2 days of hour prefixes

BUCKETS = {
    "GOES-16": "noaa-goes16",
    "GOES-17": "noaa-goes17",
    "GOES-18": "noaa-goes18",
    "GOES-19": "noaa-goes19",
}

# Operational windows and sub-satellite longitudes, confirmed against
# NOAA/NESDIS and the AWS Open Data registry:
#   GOES-16  GOES-East 75.2W, 2017 -> 2025-04-07 (then standby)
#   GOES-19  GOES-East 75.2W, 2025-04-07 -> present
#   GOES-17  GOES-West 137W,  2018 -> 2023-01-10
#   GOES-18  GOES-West 137W,  2023-01-10 -> present
#
# This matters concretely: a real run produced training examples ONLY
# for 2025 storms, because the code defaulted to GOES-19 and the
# noaa-goes19 bucket does not exist before 2025. Every 2023/2024
# Atlantic storm silently returned "no imagery" -- not because the
# storm was invisible, but because we were asking the wrong satellite.
SATELLITE_ERAS = [
    # (name, subsatellite_lon, start, end)
    ("GOES-16", -75.2, datetime(2017, 7, 10, tzinfo=timezone.utc), datetime(2025, 4, 7, tzinfo=timezone.utc)),
    ("GOES-19", -75.2, datetime(2025, 4, 7, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc)),
    ("GOES-17", -137.0, datetime(2019, 2, 12, tzinfo=timezone.utc), datetime(2023, 1, 10, tzinfo=timezone.utc)),
    ("GOES-18", -137.0, datetime(2023, 1, 10, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc)),
]


def select_satellite(lat: float, lon: float, when: datetime,
                     max_angle_deg: float = None) -> Optional[str]:
    """Pick the GOES satellite that was actually operational at `when`
    AND can see (lat, lon), preferring whichever has the smaller viewing
    angle. Returns None if no GOES satellite covers that place and time
    (e.g. the Indian Ocean, or a date before GOES-16).

    Without this, the caller has to hardcode a satellite name, which is
    wrong in two different ways at once: it breaks for any date outside
    that satellite's operational era, and it ignores that a storm may be
    far better viewed from the other orbital slot.
    """
    when = _as_utc(when)
    if max_angle_deg is None:
        max_angle_deg = MAX_VIEW_ANGLE_DEG

    best = None
    best_angle = None
    for name, sub_lon, start, end in SATELLITE_ERAS:
        # `when` is coerced to aware UTC on entry (see _as_utc); the era
        # bounds are aware too. This comparison is where a naive value
        # used to raise TypeError, failing every mined overpass.
        if not (start <= when < end):
            continue
        angle = _view_angle_deg(sub_lon, lat, lon)
        if angle <= max_angle_deg and (best_angle is None or angle < best_angle):
            best, best_angle = name, angle
    return best


def _view_angle_deg(sub_lon: float, lat: float, lon: float) -> float:
    lat_r = math.radians(lat)
    dlon_r = math.radians(((lon - sub_lon + 180.0) % 360.0) - 180.0)
    cos_gamma = max(-1.0, min(1.0, math.cos(lat_r) * math.cos(dlon_r)))
    return math.degrees(math.acos(cos_gamma))

# Sub-satellite longitude for each GOES position. Used by
# satellite_can_see() to reject storms that are physically outside the
# satellite's disk BEFORE spending any S3 calls on them.
SUBSATELLITE_LON = {
    "GOES-16": -75.2,    # GOES-East (2017 - Apr 2025)
    "GOES-17": -137.0,   # GOES-West (2019 - Jan 2023)
    "GOES-18": -137.0,   # GOES-West (Jan 2023 - present)
    "GOES-19": -75.2,    # GOES-East (Apr 2025 - present)
}

# Max angular distance from the sub-satellite point to still count as
# usable. A geostationary satellite geometrically sees ~81.3 deg, but
# viewing geometry degrades badly toward the limb, so this is
# deliberately tighter -- a storm at 78 deg from nadir is smeared across
# the limb and not something to train on.
MAX_VIEW_ANGLE_DEG = 65.0


def satellite_can_see(satellite: str, lat: float, lon: float,
                      max_angle_deg: float = MAX_VIEW_ANGLE_DEG) -> bool:
    """True if (lat, lon) is within the usable portion of `satellite`'s
    disk. Pure geometry -- no network calls at all.

    This exists because list_available_files() lists an entire hour
    prefix of the RadM bucket (~2000 objects: every band, both meso
    sectors, every minute) and does so ONCE PER BAND. For a storm the
    satellite physically cannot see, that entire cost is wasted. A real
    run spent ~2.5 hours on 1821 WP/IO/SH attempts that could never have
    succeeded. Calling this first turns each of those into a few
    microseconds of arithmetic.
    """
    if satellite not in SUBSATELLITE_LON:
        return True  # unknown satellite -- don't block, let the fetch decide
    return _view_angle_deg(SUBSATELLITE_LON[satellite], lat, lon) <= max_angle_deg

# Bands relevant to this project and what physical quantity they represent.
BAND_INFO = {
    2: {"name": "vis_064um", "kind": "reflectance"},
    7: {"name": "swir_39um", "kind": "brightness_temp"},
    9: {"name": "wv_69um", "kind": "brightness_temp"},
    13: {"name": "ir_103um", "kind": "brightness_temp"},
}


def _get_s3_client():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed. `pip install boto3`")
    # Unsigned config: these buckets are public and do NOT require AWS credentials.
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


@dataclass
class GoesFileRef:
    key: str
    satellite: str
    band: int
    sector: str          # "M1" or "M2"
    scan_start: datetime


def list_available_files(
    satellite: str,
    band: int,
    target_time: datetime,
    sector: Optional[str] = None,
    window_minutes: int = 30,
    product: str = "RadM",
) -> list[GoesFileRef]:
    """List ABI L1b files for a band within +/- window_minutes.

    product: "RadM" (mesoscale, the historical default), "RadC" (CONUS) or
        "RadF" (full disk). RadM alone loses 83% of storm-times to "no
        covering sector" -- the two mesoscale boxes are steerable and are
        usually pointed elsewhere -- so RadF is the fallback that makes
        most of the archive reachable. See goes_fixed_grid.

    sector: "M1", "M2", or None to include both. Ignored for RadC/RadF,
        which have no sub-sectors.
    """
    if satellite not in BUCKETS:
        raise ValueError(f"satellite must be one of {list(BUCKETS)}")
    if band not in BAND_INFO:
        raise ValueError(f"band {band} not in supported set {list(BAND_INFO)}")

    s3 = _get_s3_client()
    bucket = BUCKETS[satellite]

    results: list[GoesFileRef] = []
    # Mesoscale scans happen every ~1 minute, so we need to check every hour
    # prefix that falls inside the time window (usually just 1, sometimes 2).
    start = target_time - timedelta(minutes=window_minutes)
    end = target_time + timedelta(minutes=window_minutes)

    hour_cursor = start.replace(minute=0, second=0, microsecond=0)
    seen_prefixes = set()
    while hour_cursor <= end:
        prefix = f"ABI-L1b-{product}/{hour_cursor:%Y}/{hour_cursor:%j}/{hour_cursor:%H}/"
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            # Cache the raw key list per (bucket, prefix). One hour prefix
            # holds ~2000 objects (every band, both meso sectors, every
            # minute) and this function is called once PER BAND -- so
            # fetching bands 13/9/7 for one scene previously re-listed the
            # exact same ~2000 objects three times over the network. The
            # cache makes that one listing reused three times.
            cache_key = (bucket, prefix)
            with _LISTING_CACHE_LOCK:
                keys = _LISTING_CACHE.get(cache_key)
            if keys is None:
                keys = []
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        keys.append(obj["Key"])
                with _LISTING_CACHE_LOCK:
                    if len(_LISTING_CACHE) >= _LISTING_CACHE_MAX:
                        _LISTING_CACHE.clear()  # crude bound; these are big lists
                    _LISTING_CACHE[cache_key] = keys
            for key in keys:
                ref = _parse_key(key, satellite)
                if ref is None:
                    continue
                if ref.band != band:
                    continue
                if sector is not None and ref.sector != sector:
                    continue
                if start <= ref.scan_start <= end:
                    results.append(ref)
        hour_cursor += timedelta(hours=1)

    results.sort(key=lambda r: r.scan_start)
    return results


def _parse_key(key: str, satellite: str) -> Optional[GoesFileRef]:
    """Parse e.g. .../OR_ABI-L1b-RadM1-M6C13_G18_s20242451230123_e...nc

    Also handles RadC / RadF, which carry no sector digit -- the sector
    field comes back None for those, which is correct rather than a
    missing value: CONUS and full disk have no sub-sectors to choose.
    """
    fname = key.rsplit("/", 1)[-1]
    if not fname.startswith("OR_ABI-L1b-Rad"):
        return None

    # Filename layouts:
    #   OR_ABI-L1b-RadM1-M6C13_G18_s2024245...  (mesoscale, sector digit)
    #   OR_ABI-L1b-RadC-M6C13_G18_s2024245...   (CONUS, no digit)
    #   OR_ABI-L1b-RadF-M6C13_G18_s2024245...   (full disk, no digit)
    import re

    m = re.search(r"Rad([MCF])(\d?)-M\dC(\d{2})_G\d+_s(\d{13})", fname)
    if not m:
        return None
    kind, sector_num, band_str, sdate = m.groups()
    # CONUS and full disk have no sub-sectors; None is the correct value
    # there, not a missing one.
    sector_num = sector_num or None
    band = int(band_str)
    scan_start = _parse_goes_time(sdate)
    return GoesFileRef(
        key=key,
        satellite=satellite,
        band=band,
        # "M1"/"M2" for mesoscale; "C"/"F" for CONUS and full disk,
        # which have no sub-sector to name.
        sector=(f"M{sector_num}" if kind == "M" and sector_num else kind),
        scan_start=scan_start,
    )


def _parse_goes_time(s: str) -> datetime:
    """GOES filename timestamps are sYYYYDDDHHMMSSS (last digit = tenths of sec)."""
    year = int(s[0:4])
    doy = int(s[4:7])
    hour = int(s[7:9])
    minute = int(s[9:11])
    second = int(s[11:13])
    # tz-aware UTC. GOES filenames are UTC by definition, and every other
    # timestamp in this project is aware -- a naive value here would only
    # blow up later, at whichever comparison happened to come first.
    return (datetime(year, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second))


def find_nearest_file(
    satellite: str, band: int, target_time: datetime, sector: Optional[str] = None
) -> Optional[GoesFileRef]:
    # Coerce here as well as in get_band_image: this is called directly
    # from several places, and scan_start (parsed from the filename) is
    # tz-aware, so a naive target_time would raise on the subtraction
    # below rather than anywhere obvious.
    target_time = _as_utc(target_time)
    candidates = list_available_files(satellite, band, target_time, sector=sector)
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs((r.scan_start - target_time).total_seconds()))


def _default_cache_dir() -> str:
    """Platform-correct scratch directory.

    This was hardcoded to "/tmp/goes_cache", which on Windows resolves to
    a path the process cannot create, producing
    `PermissionError: [WinError 5] Access is denied` -- observed once in a
    4,867-attempt run. Rare only because it needs the download path rather
    than the cached path; it would fail every time on a clean machine.
    """
    import tempfile
    return os.path.join(tempfile.gettempdir(), "goes_cache")


def download_and_open(ref: GoesFileRef, local_dir: Optional[str] = None):
    """Download a RadM NetCDF file and return an opened xarray.Dataset."""
    local_dir = local_dir or _default_cache_dir()
    if xr is None:
        raise RuntimeError("xarray is not installed. `pip install xarray netCDF4`")
    import os

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, ref.key.rsplit("/", 1)[-1])
    if not os.path.exists(local_path):
        s3 = _get_s3_client()
        bucket = BUCKETS[ref.satellite]
        # Download to a thread-unique temp path, then atomically rename.
        # boto3's download_file writes straight to the destination, so
        # with concurrent workers two threads can write the same path at
        # once, and a third can open a half-written file and get a
        # corrupt/truncated dataset. os.replace() is atomic, so a reader
        # either sees no file or a complete one -- never a partial. Same
        # lesson as the PPS NRT truncated-cache bug earlier in this
        # project.
        tmp_path = f"{local_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            s3.download_file(bucket, ref.key, tmp_path)
            os.replace(tmp_path, local_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    return xr.open_dataset(local_path)


def radiance_to_physical(ds, band: int) -> tuple[np.ndarray, float]:
    """Convert raw ABI radiance to reflectance (band 2) or brightness temp
    (others), using the calibration coefficients embedded in the file per
    the ABI L1b spec.

    QC step: GOES scenes always contain some QC-flagged / off-limb / fill
    pixels. We mask those using the file's DQF (data quality flag) array
    (0 = good, nonzero = bad) plus a finite-value check on the converted
    physical field, then nearest-fill so the returned array is always
    finite. Returns (clean_array, bad_fraction) so callers/GUI can warn if
    a scene was mostly bad data rather than silently trusting a heavily
    inpainted field.
    """
    rad = ds["Rad"].values.astype(np.float64)
    dqf = ds["DQF"].values if "DQF" in ds.variables else None

    if BAND_INFO[band]["kind"] == "reflectance":
        kappa0 = float(ds["kappa0"].values)
        refl = rad * kappa0
        refl = np.clip(refl, 0, 1.3)
        return sanitize_field(refl, dqf=dqf)

    # Brightness temperature conversion (ABI L1b product guide, sec 5.1.2):
    # T = (fk2 / (ln((fk1/Rad) + 1)) - bc1) / bc2
    # Radiance must be strictly positive for the log term to be finite;
    # non-positive radiance (space look, fill values) is marked invalid
    # up front rather than allowed to produce NaN/inf that then leaks
    # downstream.
    fk1 = float(ds["planck_fk1"].values)
    fk2 = float(ds["planck_fk2"].values)
    bc1 = float(ds["planck_bc1"].values)
    bc2 = float(ds["planck_bc2"].values)

    extra_invalid = rad <= 0
    safe_rad = np.where(extra_invalid, 1.0, rad)  # placeholder, gets overwritten by QC fill anyway

    with np.errstate(invalid="ignore", divide="ignore"):
        tb = (fk2 / np.log((fk1 / safe_rad) + 1.0) - bc1) / bc2

    return sanitize_field(tb, dqf=dqf, extra_invalid=extra_invalid)


def _as_utc(t):
    """Coerce a datetime to tz-aware UTC.

    Defensive boundary, added after a naive/aware mismatch here cost a
    full mining run: TC PRIMED scene times became tz-aware in 0.105 while
    this module's satellite-operational windows and parsed filenames
    stayed naive, so every comparison raised TypeError. 4,867 overpasses
    attempted, 0 saved. Assuming naive means UTC is safe -- everything
    feeding this module is UTC -- and is far better than raising on an
    input that is merely under-specified.
    """
    if t is not None and getattr(t, "tzinfo", None) is None:
        return t.replace(tzinfo=timezone.utc)
    return t


def get_band_image(
    satellite: str,
    band: int,
    target_time: datetime,
    sector: Optional[str] = None,
    local_dir: Optional[str] = None,
):
    """High-level convenience: find nearest file, download, convert to physical
    units, and return a data_types.BandImage. Returns None if no file found."""
    from data_types import BandImage  # local import avoids circularity at module load

    target_time = _as_utc(target_time)
    ref = find_nearest_file(satellite, band, target_time, sector=sector)
    if ref is None:
        return None

    ds = download_and_open(ref, local_dir=local_dir)

    if band == 2:
        # Band 2 (0.64um visible) is natively 0.5km resolution -- 4x finer
        # per dimension than bands 7/9/13 (2km), i.e. up to 16x more
        # pixels for the same physical area. That extra detail is wasted
        # here: synthetic_algorithm.py regrids band2 down onto band13's
        # coarser grid anyway (_regrid_to) before using it at all, so the
        # fine resolution never survives to matter. Downsampling here --
        # BEFORE the expensive per-pixel fixed-grid lat/lon computation
        # and QC masking below, not after -- cuts band2's processing cost
        # substantially. This is the most likely real explanation for
        # band2 fetches taking noticeably longer than the other 3 bands
        # (genuinely far more data to move/process, not a hang).
        #
        # NOTE: written against xarray's standard coarsen() API but could
        # not be tested against a real GOES file OR even a real xarray
        # install in this environment (no network access here to install
        # xarray at all, let alone fetch a real file). Wrapped in a
        # try/except specifically so this optimization can't become a NEW
        # way for band2 fetching to fail outright -- if coarsen() doesn't
        # work as expected against a real file's actual structure, this
        # falls back to the original full-resolution behavior (slower,
        # but was already confirmed working) rather than raising.
        try:
            ds = ds.coarsen(y=4, x=4, boundary="trim").mean()
        except Exception as _e:
            # Was a bare `pass`. An optimization that silently no-ops is
            # indistinguishable from one that works, which is how the
            # uint16 packing sat dead for several versions. Warn instead:
            # the fallback is still correct, just slower, and now visible.
            import warnings
            warnings.warn(f"band2 coarsen failed ({type(_e).__name__}); using "
                          f"full 0.5km resolution, which is slower.",
                          RuntimeWarning)

    values, bad_fraction = radiance_to_physical(ds, band)
    if bad_fraction > 0.5:
        import warnings

        warnings.warn(
            f"GOES {satellite} band {band} scene at {ref.scan_start} is "
            f"{bad_fraction:.0%} QC-flagged/invalid pixels before nearest-fill "
            "-- treat this frame with caution."
        )

    # Fixed-grid x/y -> lat/lon requires the projection info in the file.
    lat, lon = fixed_grid_to_latlon(ds)
    # Off-limb pixels can produce a small number of NaNs in the projection
    # math (sqrt of a slightly negative discriminant at the earth limb).
    # Mesoscale sectors are normally well inside the disk, but sanitize
    # defensively so a stray edge pixel can't crash rendering downstream.
    lat, _ = sanitize_field(lat)
    lon, _ = sanitize_field(lon)

    return BandImage(
        band=band,
        satellite=satellite,
        scene_time=ref.scan_start,
        values=values,
        lat=lat,
        lon=lon,
        units="reflectance" if BAND_INFO[band]["kind"] == "reflectance" else "K",
        mesoscale_sector=ref.sector,
        source_key=ref.key,
    )


def fixed_grid_to_latlon(ds):
    """Convert ABI fixed-grid x/y (radians) to lat/lon using the geostationary
    projection parameters stored in goes_imager_projection."""
    proj_info = ds["goes_imager_projection"]
    """Convert ABI fixed-grid x/y (scan angles, in radians) to lat/lon using
    the geostationary projection parameters stored in goes_imager_projection.

    IMPORTANT: x and y are already the scan angles in radians as stored in
    the file -- they must NOT be scaled by perspective_point_height before
    use. (An earlier version of this function did that, which fed huge
    arguments into sin/cos, wrapped chaotically, and produced nonsense
    lat/lon ranges like -300 to 0 for longitude. Fixed here.)

    This follows the standard NOAA/NESDIS ABI navigation algorithm
    (fixed grid -> geodetic), verified against the known nadir case:
    x=y=0 must map to lat=0, lon=longitude_of_projection_origin.
    """
    proj_info = ds["goes_imager_projection"]
    lon_origin_deg = float(proj_info.attrs["longitude_of_projection_origin"])
    lambda0 = np.radians(lon_origin_deg)
    H = float(proj_info.attrs["perspective_point_height"]) + float(
        proj_info.attrs["semi_major_axis"]
    )
    r_eq = float(proj_info.attrs["semi_major_axis"])
    r_pol = float(proj_info.attrs["semi_minor_axis"])

    x = ds["x"].values  # radians -- scan angle, use as-is
    y = ds["y"].values  # radians -- scan angle, use as-is
    x2d, y2d = np.meshgrid(x, y)

    a = np.sin(x2d) ** 2 + np.cos(x2d) ** 2 * (
        np.cos(y2d) ** 2 + (r_eq**2 / r_pol**2) * np.sin(y2d) ** 2
    )
    b = -2 * H * np.cos(x2d) * np.cos(y2d)
    c = H**2 - r_eq**2

    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        sx = rs * np.cos(x2d) * np.cos(y2d)
        sy = -rs * np.sin(x2d)
        sz = rs * np.cos(x2d) * np.sin(y2d)

        lat = np.degrees(
            np.arctan((r_eq**2 / r_pol**2) * (sz / np.sqrt((H - sx) ** 2 + sy**2)))
        )
        lon = np.degrees(lambda0 - np.arctan(sy / (H - sx)))

    return lat, lon


def fetch_extra_ir_bands(satellite, target_time, sector=None, bands=None,
                         limit=None, progress_callback=None,
                         center_lat=None, center_lon=None) -> dict:
    """Fetch the supplementary ABI IR bands used as extra model inputs.

    Returns {band_number: BandImage}, omitting any band that failed. A
    missing band is NOT an error: ml_inference fills the corresponding
    channel with a neutral plane, so a partial fetch degrades the model's
    information rather than breaking its input shape. That property is
    what makes it safe to fetch these best-effort on a latency budget.

    Bands are requested in ml_constants.EXTRA_IR_BANDS order, which is
    sorted by the per-channel saliency Li et al. (2026) measured, so
    trimming with `limit` drops the least informative channels first.
    """
    from ml_constants import EXTRA_IR_BANDS, EXTRA_IR_FETCH_LIMIT

    if bands is None:
        bands = EXTRA_IR_BANDS
    if limit is None:
        limit = EXTRA_IR_FETCH_LIMIT
    bands = tuple(bands)[:max(0, int(limit))]
    if not bands:
        return {}

    # IN PARALLEL. Each band is an independent list+read, and on a fast
    # link the cost is round trips rather than bytes -- three serial
    # fetches is three times the latency for no reason. Bounded at 4 so
    # this cannot compete with the mining pool's own workers for
    # connections.
    from concurrent.futures import ThreadPoolExecutor

    def _one(b):
        try:
            # Same sector fallback as the primary bands. Without this the
            # extra channels can come from mesoscale while band 13 came
            # from full disk, putting them on a different grid than the
            # frame they are meant to condition.
            if center_lat is not None and center_lon is not None:
                img = get_band_image_any_sector(satellite, b, target_time,
                                                center_lat, center_lon, sector=sector)
            else:
                img = get_band_image(satellite, b, target_time, sector=sector)
            return b, img
        except Exception as e:
            if progress_callback:
                progress_callback(f"Extra IR band {b} unavailable ({type(e).__name__}) -- "
                                  f"that channel will be neutral for this frame.")
            return b, None

    out = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(bands)))) as pool:
        for b, img in pool.map(_one, bands):
            if img is not None:
                out[b] = img
    if progress_callback and out:
        progress_callback(f"Extra IR bands fetched: {sorted(out)} "
                          f"(of {len(bands)} requested).")
    return out


def get_band_image_any_sector(satellite: str, band: int, target_time,
                              center_lat: float, center_lon: float,
                              sector: Optional[str] = None,
                              half_width_km: float = 512.0,
                              progress_callback=None):
    """Fetch a band image covering (center_lat, center_lon), falling back
    from mesoscale to full disk.

    This is the fix for the single largest loss in mining: RadM alone
    discarded 83% of storm-times (4,023 of 4,867) as "no covering sector",
    because the two mesoscale boxes are steerable and usually pointed
    elsewhere. Full disk covers the whole hemisphere every 10 minutes.

    Order is deliberate. Mesoscale is tried FIRST because it is 1-minute
    cadence, already-tested, and a small whole-file read; full disk is the
    fallback, read as a storm-centred crop through ranged GETs
    (~0.6% of a 59 MB array). Preferring full disk would be slower and
    lower cadence for the majority of cases that mesoscale already covers.

    Returns a BandImage, or None.
    """
    from data_types import BandImage
    import goes_fixed_grid as gfg

    target_time = _as_utc(target_time)

    # 1. Mesoscale, as before.
    try:
        img = get_band_image(satellite, band, target_time, sector=sector)
        if img is not None and _image_covers(img, center_lat, center_lon):
            return img
    except Exception as e:
        if progress_callback:
            progress_callback(f"  RadM fetch failed ({type(e).__name__}); trying full disk.")

    # 2. Full disk, cropped.
    #
    # NOT for band 2. It is 0.5 km, so full disk is 21696 x 21696 -- 941 MB
    # against 59 MB for a 2 km band -- and synthetic_algorithm regrids it
    # straight down onto band 13's grid, so every one of those extra pixels
    # is discarded before it is used. Band 2 is the least valuable band and
    # by far the most expensive to reach this way. Returning None here
    # means the frame is built without the visible channel, which is
    # already the normal case at night.
    if band == 2:
        if progress_callback:
            progress_callback("  band 2 skipped on full disk (0.5 km, 941 MB, "
                              "and downsampled away immediately).")
        return None

    sat_lon = SUBSATELLITE_LON.get(satellite)
    if sat_lon is None:
        return None
    try:
        refs = list_available_files(satellite, band, target_time,
                                    window_minutes=10, product="RadF")
        if not refs:
            return None
        ref = min(refs, key=lambda r: abs((r.scan_start - target_time).total_seconds()))

        from s3_range_reader import open_s3_hdf5
        # prefer_ranged: this reads well under 1% of the object, so the
        # whole-object shortcut must not apply however small the file is.
        h5, reader = open_s3_hdf5(_get_s3_client(), BUCKETS[satellite], ref.key,
                                  prefer_ranged=True)
        try:
            out = gfg.read_cropped_radiance(h5, center_lat, center_lon, sat_lon,
                                            half_width_km=half_width_km)
        finally:
            h5.close()
        if out is None:
            return None
        rad, dqf, lat, lon, meta = out

        if BAND_INFO[band]["kind"] == "reflectance":
            k0 = meta.get("kappa0")
            values = np.clip(rad * k0, 0, 1.3) if k0 else None
        else:
            values = gfg.radiance_to_brightness_temperature(rad, meta["planck"])
        if values is None:
            return None
        if dqf is not None:
            values = np.where(np.asarray(dqf) == 0, values, np.nan)

        img = BandImage(band=band, satellite=satellite, scene_time=ref.scan_start,
                        values=values, lat=lat, lon=lon, units="K",
                        mesoscale_sector="F")

        # Logging AFTER the result exists, and in its own try. This line
        # previously sat before the return and called format_bytes(),
        # which lives in tcprimed_ingest and is not imported here -- so
        # every full-disk crop was read successfully and then discarded by
        # a NameError in the progress message. Diagnostics must never be
        # able to fail the operation they describe.
        if progress_callback:
            try:
                st = reader.stats()
                # Says "per band" because only band 13 is given a
                # progress_callback by the mining path -- bands 9, 7 and 2
                # take the same route silently, so a reader seeing one
                # line per frame was under-counting the real transfer by
                # about 4x.
                progress_callback(
                    f"  full-disk crop band {band}: "
                    f"{meta['crop_fraction']*100:.2f}% of the array, "
                    f"{st['bytes_fetched']/1e6:.1f} MB fetched "
                    f"({st['requests']} request(s), per band; "
                    f"3-4 bands per frame)")
            except Exception:
                pass
        return img
    except Exception as e:
        if progress_callback:
            progress_callback(f"  full-disk fallback failed ({type(e).__name__}: {e}).")
        return None


def _image_covers(img, lat: float, lon: float) -> bool:
    """Does this image actually contain the point? goes_fetch matches on
    TIME ONLY, so without this a storm gets paired with whichever sector
    happened to be scanning, anywhere on Earth."""
    try:
        la, lo = np.asarray(img.lat), np.asarray(img.lon)
        return bool(np.nanmin(la) <= lat <= np.nanmax(la)
                    and np.nanmin(lo) <= lon <= np.nanmax(lo))
    except Exception:
        return False


# --- GLM lightning ----------------------------------------------------

def get_glm_flashes(satellite: str, start_time, end_time, progress_callback=None):
    """Flash locations from GOES GLM between two times.

    Returns (lat, lon, energy) arrays, empty if nothing is found.

    WHY GLM. Both Li et al. papers, and this project's own measurements,
    hit the same wall: cloud-top IR under-determines the sub-cloud
    hydrometeor column. Lightning requires graupel and supercooled water
    colliding in strong updrafts, which is close to a direct observation
    of what 89 GHz scattering measures -- and it is information IR does
    NOT carry, since a cirrus canopy and an active core can share a
    cloud-top temperature and differ completely in flash rate.

    Measured on the Norbert frame: the IR cold shield is 16% of the scene
    while 89 GHz deep scattering is 9.2%, and the COLDEST IR tops are only
    0.9%. The answer sits between two thresholds an order of magnitude
    apart in area, which is exactly the discrimination lightning speaks to.

    GLM L2 LCFA granules cover 20 s each, so a few minutes either side is
    tens of small files. They are read whole (each is well under a MB) via
    the same streaming reader, so nothing lands on disk.
    """
    import numpy as np

    start_time, end_time = _as_utc(start_time), _as_utc(end_time)
    bucket = BUCKETS.get(satellite)
    if bucket is None:
        return np.array([]), np.array([]), np.array([])

    client = _get_s3_client()
    keys = []
    cursor = start_time.replace(minute=0, second=0, microsecond=0)
    while cursor <= end_time:
        prefix = f"GLM-L2-LCFA/{cursor:%Y}/{cursor:%j}/{cursor:%H}/"
        # Reuse the same listing cache as the ABI path. An hour prefix
        # holds ~180 granules and consecutive overpasses of one storm
        # often fall in the same hour, so this is frequently a free hit.
        cache_key = (bucket, prefix)
        with _LISTING_CACHE_LOCK:
            hour_keys = _LISTING_CACHE.get(cache_key)
        if hour_keys is None:
            hour_keys = []
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    hour_keys.append(obj["Key"])
            with _LISTING_CACHE_LOCK:
                _LISTING_CACHE[cache_key] = hour_keys
        for k in hour_keys:
            t = _parse_glm_key_time(k)
            if t is not None and start_time <= t <= end_time:
                keys.append(k)
        cursor += timedelta(hours=1)

    if not keys:
        return np.array([]), np.array([]), np.array([])

    from concurrent.futures import ThreadPoolExecutor
    from s3_range_reader import open_s3_hdf5

    def _read_one(key):
        """One GLM granule. Whole-object: a 20 s file is tiny, and ranging
        it would be the mistake 0.121 measured on TC PRIMED files."""
        try:
            h5, _reader = open_s3_hdf5(client, bucket, key)
            try:
                names = set(h5.keys())
                if not {"flash_lat", "flash_lon"} <= names:
                    return None
                la = _descale(h5["flash_lat"])
                lo = _descale(h5["flash_lon"])
                en = (_descale(h5["flash_energy"]) if "flash_energy" in names
                      else np.ones_like(la))
            finally:
                h5.close()
            return la, lo, en
        except Exception:
            return None    # one bad granule must not lose the window

    # IN PARALLEL. A +/-5 min window is ~30 granules, and read serially that
    # is ~30 round trips of pure latency per frame -- on a 250 Mbps link the
    # bottleneck here is round trips, not bytes, since each granule is well
    # under a megabyte. Threads are the right tool: this is entirely
    # network-bound, so the GIL costs nothing.
    lats, lons, energies = [], [], []
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(keys)))) as pool:
        for got in pool.map(_read_one, keys):
            if got is None:
                continue
            lats.append(got[0])
            lons.append(got[1])
            energies.append(got[2])

    if not lats:
        return np.array([]), np.array([]), np.array([])
    out = (np.concatenate(lats), np.concatenate(lons), np.concatenate(energies))
    if progress_callback:
        progress_callback(f"  GLM: {out[0].size} flash(es) from {len(keys)} granule(s)")
    return out


def _parse_glm_key_time(key: str):
    """Start time from a GLM LCFA key, e.g.
    OR_GLM-L2-LCFA_G18_s20242451230000_e..._c....nc"""
    import re
    m = re.search(r"_s(\d{13})", key.rsplit("/", 1)[-1])
    return _parse_goes_time(m.group(1)) if m else None


def _descale(ds):
    """Read a GLM variable, applying scale_factor/add_offset.

    GLM stores flash_lat/lon as scaled int16. Skipping the scaling yields
    coordinates off by orders of magnitude, which places every flash
    outside the grid and silently returns an empty density field rather
    than raising -- the same shape of bug as the ABI coordinate scaling.
    """
    import numpy as np
    raw = np.asarray(ds[:], dtype=np.float64)
    def attr(name, default):
        try:
            return float(np.asarray(ds.attrs[name]).reshape(-1)[0])
        except Exception:
            return default
    return raw * attr("scale_factor", 1.0) + attr("add_offset", 0.0)


def fetch_bands_parallel(satellite: str, target_time, center_lat: float,
                         center_lon: float, bands, sector=None,
                         progress_callback=None) -> dict:
    """Fetch several bands for one frame concurrently.

    The base bands (13, 9, 7 and optionally 2) were fetched one after
    another, and each may be a full-disk crop -- a LIST plus several
    ranged GETs against a ~30 MB object. At an observed ~36 s per saved
    example against a ~4 s CPU floor, the mining loop is bound by serial
    round trips, not by computation.

    The bands are independent, so this is the same fix already applied to
    the supplementary IR bands and the GLM granules. Bounded at 4 so a
    single frame cannot monopolise connections the mining pool's own
    workers need.
    """
    from concurrent.futures import ThreadPoolExecutor

    bands = list(bands)

    def _one(b):
        try:
            return b, get_band_image_any_sector(
                satellite, b, target_time, center_lat, center_lon,
                sector=sector, progress_callback=progress_callback if b == 13 else None)
        except Exception:
            return b, None

    out = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(bands)))) as pool:
        for b, img in pool.map(_one, bands):
            out[b] = img
    return out


def fetch_frame_bands(satellite: str, target_time, center_lat: float,
                      center_lon: float, base_bands=(13, 9, 7),
                      extra_limit: Optional[int] = None,
                      sector=None, progress_callback=None):
    """Fetch the base AND supplementary bands for one frame in ONE pool.

    These were two pools of three, run back to back: `fetch_bands_parallel`
    for 13/9/7 and then `fetch_extra_ir_bands` for the rest. Both are
    latency-bound, and the second cannot start until the first finishes,
    so a frame paid two round-trip waits where one would do.

    Returns (base_dict, extra_dict). One pool of six, bounded so the outer
    mining pool still governs total concurrency.
    """
    from concurrent.futures import ThreadPoolExecutor
    from ml_constants import EXTRA_IR_BANDS, EXTRA_IR_FETCH_LIMIT

    if extra_limit is None:
        extra_limit = EXTRA_IR_FETCH_LIMIT
    extra_bands = [b for b in EXTRA_IR_BANDS[:extra_limit]
                   if b not in base_bands]
    wanted = list(base_bands) + extra_bands

    def _one(b):
        try:
            return b, get_band_image_any_sector(
                satellite, b, target_time, center_lat, center_lon,
                sector=sector,
                progress_callback=progress_callback if b == 13 else None)
        except Exception:
            return b, None

    got = {}
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(wanted)))) as pool:
        for b, img in pool.map(_one, wanted):
            got[b] = img

    base = {b: got.get(b) for b in base_bands}
    extra = {b: got[b] for b in extra_bands if got.get(b) is not None}
    return base, extra
