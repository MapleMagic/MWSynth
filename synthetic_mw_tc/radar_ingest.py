"""
NEXRAD Level 2 radar ingestion, used as an additional, CONUS/territory-
limited calibration cross-check for the synthetic algorithm -- NOT a
replacement for the real-MW calibration pipeline (mw_compare.py), since
radar reflectivity and microwave brightness temperature are physically
different quantities with no direct Tb-equivalent conversion. This is
meant to answer "does the synthetic algorithm's implied convective
structure roughly match what radar actually sees," not to bias-correct
Tb values the way mw_compare.py does.

DELIBERATE SCOPE LIMIT (per spec): only pulls from a radar if the storm's
best-track center is within max_range_mi (default 200) of a NEXRAD site,
and only the single nearest in-range site -- never scans/pulls from every
station. See nexrad_stations.find_nearest_station_in_range().

DATA SOURCE: s3://unidata-nexrad-level2/<Year>/<Month>/<Day>/<STATION>/
<STATION><YYYYMMDD>_<HHMMSS>_V06 -- confirmed directly against two
authoritative sources (Unidata's own bucket-migration announcement, and
the awslabs/open-data-docs README), not assumed:
  https://www.unidata.ucar.edu/blogs/news/entry/important-changes-to-noaa-nexrad
  https://github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-nexrad/README.md
Files after 2016-06-02 have no .gz extension; older files are gzip-
compressed. Bucket is public/anonymous (same unsigned-boto3 pattern as
goes_fetch.py).

READING THE DATA: uses Py-ART (`arm_pyart` on PyPI), the standard NEXRAD
Level 2 reader. IMPORTANT CAVEAT: Py-ART is not installed in the
environment this was built in (no network access to install/test it
live), so the reflectivity-extraction code below is written against
Py-ART's documented, long-stable API (read_nexrad_archive, get_slice,
gate_latitude/gate_longitude) but has NOT been run against a real file.
Treat this the same way GMI-NRT was treated earlier in this project:
written correctly against the spec, first live run is the real test.
"""
from __future__ import annotations

import timeutil

import gzip
import os
import re
import shutil
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

import nexrad_stations
from data_types import StormFix

CACHE_DIR = os.path.expanduser("~/.synthetic_mw_tc/radar")
BUCKET = "unidata-nexrad-level2"


def _get_s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def find_radar_for_storm(storm_fix: StormFix, max_range_mi: float = 200.0):
    """Wrapper around nexrad_stations.find_nearest_station_in_range using
    a StormFix's lat/lon. Returns (station, distance_mi) or (None, None)."""
    return nexrad_stations.find_nearest_station_in_range(
        storm_fix.lat, storm_fix.lon, max_range_mi=max_range_mi
    )


def _list_scans_for_day(s3, station_icao: str, day: datetime) -> list[str]:
    prefix = f"{day:%Y}/{day:%m}/{day:%d}/{station_icao}/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


_SCAN_TIME_RE = re.compile(r"([A-Z0-9]{4})(\d{8})_(\d{6})")


def _parse_scan_time(key: str) -> Optional[datetime]:
    fname = key.rsplit("/", 1)[-1]
    m = _SCAN_TIME_RE.search(fname)
    if not m:
        return None
    _icao, datestr, timestr = m.groups()
    try:
        return timeutil.as_utc(datetime.strptime(datestr + timestr, "%Y%m%d%H%M%S"))
    except ValueError:
        return None


def find_nearest_scan(
    station_icao: str, target_time: datetime, lookback_hours: float = 3.0
) -> Optional[tuple[str, datetime]]:
    """Find the S3 key of the volume scan closest to target_time (within
    the past lookback_hours), for the given station. Returns (key,
    scan_time) or None if nothing found."""
    s3 = _get_s3_client()
    start = target_time - timedelta(hours=lookback_hours)

    days = set()
    cursor = start
    while cursor <= target_time:
        days.add(cursor.date())
        cursor += timedelta(hours=1)
    days.add(target_time.date())

    candidates = []
    for day in sorted(days):
        day_dt = timeutil.as_utc(datetime(day.year, day.month, day.day))
        keys = _list_scans_for_day(s3, station_icao, day_dt)
        for key in keys:
            scan_time = _parse_scan_time(key)
            if scan_time is None:
                continue
            if start <= scan_time <= target_time:
                candidates.append((scan_time, key))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)  # most recent first
    scan_time, key = candidates[0]
    return key, scan_time


def download_scan(key: str, local_dir: str = CACHE_DIR) -> str:
    """Download a volume scan file (handling the pre-2016-06-02 .gz case)
    and return the local path to the (decompressed, if needed) file."""
    os.makedirs(local_dir, exist_ok=True)
    fname = key.rsplit("/", 1)[-1]
    local_path = os.path.join(local_dir, fname)

    if not os.path.exists(local_path):
        s3 = _get_s3_client()
        s3.download_file(BUCKET, key, local_path)

    if local_path.endswith(".gz"):
        decompressed_path = local_path[: -len(".gz")]
        if not os.path.exists(decompressed_path):
            with gzip.open(local_path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_path

    return local_path


def get_echo_top_gates_near_storm(
    local_path: str,
    storm_lat: float,
    storm_lon: float,
    radius_km: float = 150.0,
    dbz_threshold: float = 18.0,
    bin_deg: float = 0.02,
):
    """Estimate echo-top height (km, above the radar's reference altitude
    -- i.e. gate_altitude, roughly MSL) for spotting overshooting tops and
    vigorous convective towers (VHTs), which base reflectivity alone
    doesn't capture -- a storm can show strong low-level reflectivity
    without deep vertical extent, or a comparatively modest-looking
    reflectivity core that's actually punching to unusually high altitude.

    Unlike get_reflectivity_gates_near_storm (which only reads the lowest
    sweep, since base reflectivity is inherently a single-tilt product),
    this reads ALL elevation sweeps in the volume scan. Different sweeps
    hit very different lat/lon at the same range/azimuth (beam height
    increases with both range and elevation angle), so gates from
    different sweeps don't share coordinates -- a raw nearest-neighbor
    regrid of the unaggregated point cloud would effectively pick an
    arbitrary sweep's altitude near each query point, not the genuine
    "top of the column" at that location. This bins gates meeting
    dbz_threshold into a coarse lat/lon grid (bin_deg, ~2km default) and
    takes the MAX altitude per bin -- the actual echo-top definition.

    Returns (bin_lat_centers, bin_lon_centers, max_alt_km) flat arrays,
    one entry per non-empty bin (possibly empty arrays if nothing in
    range meets the threshold).

    NOTE: same Py-ART API caveat as the reflectivity functions (written
    against the documented API, not run against a real file in this
    environment -- see module docstring).
    """
    try:
        import pyart
    except ImportError as e:
        raise RuntimeError(
            "The 'arm_pyart' package isn't installed (pip install arm_pyart). "
            "This is the standard NEXRAD Level 2 reader -- required for radar "
            "ingestion specifically, not for anything else in this project."
        ) from e

    radar = pyart.io.read_nexrad_archive(local_path)

    if "reflectivity" not in radar.fields:
        raise RuntimeError(
            f"No 'reflectivity' field found in {local_path}. Fields present: "
            f"{list(radar.fields.keys())}."
        )

    # Deliberately NOT sliced to one sweep -- need every elevation tilt in
    # the volume to find the true column maximum.
    refl = np.ma.filled(radar.fields["reflectivity"]["data"], np.nan)
    lats = radar.gate_latitude["data"]
    lons = radar.gate_longitude["data"]
    alts_km = radar.gate_altitude["data"] / 1000.0  # meters -> km

    dist_km = _haversine_km_grid(lats, lons, storm_lat, storm_lon)
    mask = (dist_km <= radius_km) & np.isfinite(refl) & (refl >= dbz_threshold)

    if not mask.any():
        return np.array([]), np.array([]), np.array([])

    valid_lats = lats[mask]
    valid_lons = lons[mask]
    valid_alts = alts_km[mask]

    # Bin into a coarse lat/lon grid and take the max altitude per bin --
    # a numpy-only "groupby-max" via sort + reduceat (no pandas dependency).
    lat_bins = np.round(valid_lats / bin_deg) * bin_deg
    lon_bins = np.round(valid_lons / bin_deg) * bin_deg
    keys = lat_bins.astype(np.float64) * 1e4 + lon_bins.astype(np.float64)

    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_alts = valid_alts[order]
    sorted_lat_bins = lat_bins[order]
    sorted_lon_bins = lon_bins[order]

    _, first_idx = np.unique(sorted_keys, return_index=True)
    max_alts = np.maximum.reduceat(sorted_alts, first_idx)
    result_lats = sorted_lat_bins[first_idx]
    result_lons = sorted_lon_bins[first_idx]

    return result_lats, result_lons, max_alts


def _despeckle_gates(lats: np.ndarray, lons: np.ndarray, values: np.ndarray,
                      min_neighbors: int = 3, neighbor_radius_km: float = 3.0,
                      despeckle_threshold: float = 35.0):
    """Remove isolated single-gate 'spikes' with no spatial support from
    nearby gates -- a standard radar QC technique for filtering ground
    clutter, anomalous propagation, or biological scatterers (birds,
    insects), which characteristically show up as isolated intense
    single-bin echoes with no areal extent, unlike real precipitation
    (which has spatial coherence -- nearby gates should also show
    elevated values). Added after a real run showed a thin, isolated
    linear "spike" artifact in fused output at long range from the
    radar -- exactly the shape you'd get from linear-interpolating a
    single anomalous point against its sparse, otherwise-unremarkable
    neighbors. Only checks gates ABOVE despeckle_threshold (weaker
    echoes aren't tested -- an isolated weak reading is far less likely
    to visually distort the fused output, and legitimately sparse weak
    echo is common and not worth discarding).

    Genuine, spatially-coherent intense convection is NOT removed by
    this -- it correctly has plenty of nearby gates also showing
    elevated reflectivity, so it passes the neighbor-count check.
    """
    if len(lats) == 0:
        return lats, lons, values

    from scipy.spatial import cKDTree

    lat0 = lats.mean()
    points_km = np.column_stack([lats * 111.0, lons * 111.0 * np.cos(np.radians(lat0))])
    tree = cKDTree(points_km)

    keep = np.ones(len(lats), dtype=bool)
    candidates = np.where(values >= despeckle_threshold)[0]
    for i in candidates:
        neighbor_idx = tree.query_ball_point(points_km[i], neighbor_radius_km)
        n_neighbors = len(neighbor_idx) - 1  # exclude the point itself
        if n_neighbors < min_neighbors:
            keep[i] = False

    return lats[keep], lons[keep], values[keep]


def get_reflectivity_gates_near_storm(
    local_path: str,
    storm_lat: float,
    storm_lon: float,
    radius_km: float = 150.0,
):
    """Read a NEXRAD Level 2 volume scan with Py-ART, extract the lowest-
    elevation-tilt reflectivity field, and return the raw (irregular)
    gate lat/lon/dBZ arrays within radius_km of the storm center as flat
    1D arrays -- suitable for regridding (nearest-neighbor) onto an
    analysis grid, e.g. synthetic_algorithm.py's multi-source fusion.
    Returns (lat, lon, dbz) flat arrays, possibly empty if no gates are
    in range or all are masked/invalid.

    NOTE: same Py-ART API caveat as summarize_reflectivity_near_storm
    (written against the documented API, not run against a real file in
    this environment -- see module docstring).
    """
    try:
        import pyart
    except ImportError as e:
        raise RuntimeError(
            "The 'arm_pyart' package isn't installed (pip install arm_pyart). "
            "This is the standard NEXRAD Level 2 reader -- required for radar "
            "ingestion specifically, not for anything else in this project."
        ) from e

    radar = pyart.io.read_nexrad_archive(local_path)
    sweep_slice = radar.get_slice(0)  # lowest elevation tilt

    if "reflectivity" not in radar.fields:
        raise RuntimeError(
            f"No 'reflectivity' field found in {local_path}. Fields present: "
            f"{list(radar.fields.keys())}."
        )

    refl = np.ma.filled(radar.fields["reflectivity"]["data"][sweep_slice], np.nan)
    lats = radar.gate_latitude["data"][sweep_slice]
    lons = radar.gate_longitude["data"][sweep_slice]

    dist_km = _haversine_km_grid(lats, lons, storm_lat, storm_lon)
    mask = (dist_km <= radius_km) & np.isfinite(refl)

    kept_lats, kept_lons, kept_refl = _despeckle_gates(lats[mask], lons[mask], refl[mask])
    return kept_lats, kept_lons, kept_refl


def summarize_reflectivity_near_storm(
    local_path: str,
    storm_lat: float,
    storm_lon: float,
    radius_km: float = 150.0,
    dbz_thresholds: tuple = (20.0, 30.0, 40.0, 50.0),
) -> dict:
    """Read a NEXRAD Level 2 volume scan with Py-ART, extract the lowest-
    elevation-tilt reflectivity field, crop to radius_km around the storm
    center, and compute simple summary stats -- max reflectivity and the
    fraction of in-range gates exceeding each dBZ threshold. This is a
    coarse, qualitative cross-check (does radar show intense convection
    where the synthetic algorithm's response field says it should).

    NOTE: written against Py-ART's documented API but not run against a
    real file in this environment (no Py-ART install, no network) -- see
    module docstring. If this errors, the traceback + the specific
    Py-ART call that failed is the fastest way to identify what needs
    adjusting.
    """
    lats, lons, in_range_refl = get_reflectivity_gates_near_storm(local_path, storm_lat, storm_lon, radius_km)

    if in_range_refl.size == 0:
        return {
            "n_gates": 0,
            "max_dbz": None,
            "coverage_fraction": {t: None for t in dbz_thresholds},
        }

    coverage = {
        t: float(np.mean(in_range_refl >= t)) for t in dbz_thresholds
    }

    return {
        "n_gates": int(in_range_refl.size),
        "max_dbz": float(np.max(in_range_refl)),
        "mean_dbz": float(np.mean(in_range_refl)),
        "coverage_fraction": coverage,
    }


def _haversine_km_grid(lats: np.ndarray, lons: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    R = 6371.0
    lat1 = np.radians(center_lat)
    lon1 = np.radians(center_lon)
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _find_and_download_scan_for_storm(storm_fix: StormFix, target_time: datetime, max_range_mi: float = 200.0):
    """Shared lookup used by both auto_radar_check (summary-only) and
    fetch_radar_for_fusion (gridded data): find the nearest in-range
    station, find its closest scan, download it. Returns
    (local_path, station, scan_time, dist_mi) or None if no station is in
    range or no scan is found."""
    station, dist_mi = find_radar_for_storm(storm_fix, max_range_mi=max_range_mi)
    if station is None:
        return None

    found = find_nearest_scan(station.icao, target_time)
    if found is None:
        return None
    key, scan_time = found

    local_path = download_scan(key)
    return local_path, station, scan_time, dist_mi


def auto_radar_check(storm_fix: StormFix, target_time: datetime, max_range_mi: float = 200.0) -> Optional[dict]:
    """High-level convenience for the Generate tab's automatic flow:
    find the nearest in-range station, fetch the closest scan, summarize
    reflectivity near the storm center. Returns None (not an exception)
    if no station is in range or no scan is found -- callers should treat
    that as "radar check unavailable for this storm/time," not an error.
    Any deeper failure (Py-ART missing, file read error) DOES raise, since
    that's an actionable problem rather than an expected "no data" case.
    """
    found = _find_and_download_scan_for_storm(storm_fix, target_time, max_range_mi)
    if found is None:
        return None
    local_path, station, scan_time, dist_mi = found

    summary = summarize_reflectivity_near_storm(local_path, storm_fix.lat, storm_fix.lon)
    summary["station"] = station.icao
    summary["station_name"] = station.name
    summary["distance_mi"] = dist_mi
    summary["scan_time"] = scan_time
    return summary


def fetch_radar_for_fusion(
    storm_fix: StormFix,
    target_time: datetime,
    max_range_mi: float = 200.0,
    radius_km: float = 150.0,
    include_echo_tops: bool = True,
    echo_top_dbz_threshold: float = 18.0,
) -> Optional[dict]:
    """High-level convenience for synthetic_algorithm.py's multi-source
    fusion: find the nearest in-range station, fetch the closest scan,
    return the raw gridded (lat, lon, dbz) reflectivity arrays AND
    (unless include_echo_tops=False) gridded echo-top height arrays,
    ready to hand to generate_synthetic_mw's radar_*/echo_top_* parameters,
    plus the same metadata auto_radar_check returns for diagnostics/display.

    Returns None if no station is in range or no scan is found -- treat
    that as "no radar available for this storm/time," not an error. If
    reflectivity succeeds but echo-top extraction fails for some reason,
    that failure is logged into the returned dict rather than aborting
    the whole radar fetch (reflectivity alone is still useful).
    """
    found = _find_and_download_scan_for_storm(storm_fix, target_time, max_range_mi)
    if found is None:
        return None
    local_path, station, scan_time, dist_mi = found

    lats, lons, dbz = get_reflectivity_gates_near_storm(local_path, storm_fix.lat, storm_fix.lon, radius_km)
    if lats.size == 0:
        return None

    result = {
        "lat": lats,
        "lon": lons,
        "dbz": dbz,
        "station": station.icao,
        "station_name": station.name,
        "station_lat": station.lat,
        "station_lon": station.lon,
        "distance_mi": dist_mi,
        "scan_time": scan_time,
        "max_dbz": float(np.max(dbz)) if dbz.size else None,
        "echo_top_lat": None,
        "echo_top_lon": None,
        "echo_top_km": None,
        "max_echo_top_km": None,
    }

    if include_echo_tops:
        try:
            et_lat, et_lon, et_km = get_echo_top_gates_near_storm(
                local_path, storm_fix.lat, storm_fix.lon, radius_km, dbz_threshold=echo_top_dbz_threshold
            )
            if et_lat.size > 0:
                result["echo_top_lat"] = et_lat
                result["echo_top_lon"] = et_lon
                result["echo_top_km"] = et_km
                result["max_echo_top_km"] = float(np.max(et_km))
        except Exception as e:
            result["echo_top_error"] = str(e)

    return result
