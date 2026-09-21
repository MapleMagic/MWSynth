"""
TLE-based overpass prediction for the PPS NRT sensors, using real orbital
elements from Celestrak and SGP4 propagation via `skyfield` (which wraps
the lower-level `sgp4` library and handles the TEME-to-geodetic subpoint
math correctly, rather than re-deriving that transform by hand).

This buys two things, both requested directly:
  1. A genuine "did this satellite even pass near the storm in this
     window" answer, distinct from "we searched PPS and found nothing" --
     lets you tell a real miss apart from a data/access problem.
  2. A way to avoid expensive/error-prone PPS directory listings on the
     SSMIS/WSFM feeds, which are large flat folders with NO date
     subdirectories, spanning the whole ~2-week NRT retention window (this
     is almost certainly why a wildcard listing there returned a
     permissions-flavored error -- scanning that much unstructured data
     server-side is expensive, unlike GMI's /1CR/ which behaves better).
     Only query PPS when a predicted overpass says there's actually
     something to find, and narrow the query to the specific predicted
     hour(s) instead of an entire day/multi-week folder when we do.

NORAD catalog IDs (as provided):
  GMI (GPM Core Observatory): 39574
  AMSR2 (GCOM-W1):            38337
  WSF-M (MWI):                59483
  SSMIS / DMSP F16:           28054
  SSMIS / DMSP F17:           29522
  SSMIS / DMSP F18:           36032

Swath half-widths below are approximate -- used only to decide "close
enough to plausibly be in the swath," not for precision geolocation.
GMI's and AMSR2's are well-documented published values. SSMIS's is a
typical published figure. WSF-M/MWI's isn't publicly confirmed (same
"too new to have a solid public reference" situation as its channel
layout in mw_ingest.py) so it's assumed similar to AMSR2/SSMIS
(conically-scanning imagers of a similar class) -- flagged as an
assumption, not a verified number.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

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

TLE_CACHE_DIR = os.path.expanduser("~/.synthetic_mw_tc/tle_cache")
TLE_MAX_AGE_HOURS = 24  # refresh at least once a day; TLE accuracy degrades with age

NORAD_IDS = {
    "GMI": 39574,
    "AMSR2": 38337,
    "WSFM": 59483,
    "SSMIS-F16": 28054,
    "SSMIS-F17": 29522,
    "SSMIS-F18": 36032,
    # AMSR3 (GOSAT-GW/IBUKI-GW platform) deliberately has NO entry here.
    # No confirmed NORAD ID was available to add without risking silently
    # tracking the wrong satellite (worse than the current honest gap).
    # predict_overpasses("AMSR3", ...) raises KeyError as a result, which
    # mw_ingest.py's fetch_amsr3_swath_nrt catches and treats as "TLE
    # narrowing unavailable, fall back to a broader search" -- confirmed
    # to be why AMSR3 has been working in practice, not because its TLE
    # prediction succeeds. Add a real entry here once GOSAT-GW's NORAD ID
    # is confirmed, to give AMSR3 the benefit of actual overpass
    # narrowing instead of relying on this fallback indefinitely.
}

# Approximate swath half-widths in km -- see module docstring.
SWATH_HALF_WIDTH_KM = {
    "GMI": 440,        # ~885 km full swath, well documented
    "AMSR2": 725,       # ~1450 km full swath
    "SSMIS-F16": 850,    # ~1700 km full swath (SSMIS family)
    "SSMIS-F17": 850,
    "SSMIS-F18": 850,
    "WSFM": 700,           # NOT publicly confirmed -- assumed similar to AMSR2/SSMIS
}


class NoPredictedOverpass(Exception):
    """Raised (and meant to be caught) when TLE propagation predicts no
    pass of the requested satellite near the target location in the
    requested window -- a likely genuine miss, not an error."""


def _fetch_tle(norad_id: int) -> tuple:
    """Fetch (and cache locally, refreshed daily) the two TLE lines for a
    NORAD catalog ID from Celestrak's current GP data service."""
    import requests

    os.makedirs(TLE_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(TLE_CACHE_DIR, f"{norad_id}.tle")

    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < TLE_MAX_AGE_HOURS:
            with open(cache_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) >= 2:
                return lines[-2], lines[-1]

    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or "No GP data found" in text:
        raise RuntimeError(f"Celestrak returned no TLE data for NORAD ID {norad_id}.")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected TLE response for NORAD ID {norad_id}: {text!r}")
    line1, line2 = lines[-2], lines[-1]

    with open(cache_path, "w") as f:
        f.write(f"{line1}\n{line2}\n")

    return line1, line2


def _subpoints(line1: str, line2: str, name: str, times: list) -> tuple:
    """Propagate a TLE across a list of UTC datetimes and return
    (lat_array_deg, lon_array_deg) of the sub-satellite point at each
    time, using skyfield's WGS84 subpoint calculation (not a hand-rolled
    ECI/TEME -> geodetic transform, to avoid getting that subtly wrong)."""
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale(builtin=True)  # no network needed for the timescale itself
    sat = EarthSatellite(line1, line2, name, ts)

    t = ts.utc(
        [dt.year for dt in times], [dt.month for dt in times], [dt.day for dt in times],
        [dt.hour for dt in times], [dt.minute for dt in times],
        [dt.second + dt.microsecond / 1e6 for dt in times],
    )
    geocentric = sat.at(t)
    subpoint = wgs84.subpoint(geocentric)
    return np.asarray(subpoint.latitude.degrees), np.asarray(subpoint.longitude.degrees)


def predict_overpasses(
    sensor_key: str,
    center_lat: float,
    center_lon: float,
    start_time: datetime,
    end_time: datetime,
    step_seconds: int = 20,
) -> list:
    """Returns a list of UTC datetimes within [start_time, end_time] where
    sensor_key's satellite ground track is predicted to pass within that
    sensor's approximate swath half-width of (center_lat, center_lon).
    An empty list means no predicted overpass in this window.

    sensor_key must be a key in NORAD_IDS (e.g. "GMI", "AMSR2", "WSFM",
    or one of "SSMIS-F16"/"SSMIS-F17"/"SSMIS-F18" -- see
    predict_ssmis_satellite() for checking all three at once).
    """
    norad_id = NORAD_IDS[sensor_key]
    line1, line2 = _fetch_tle(norad_id)

    n_steps = max(2, int((end_time - start_time).total_seconds() // step_seconds) + 1)
    times = [start_time + timedelta(seconds=i * step_seconds) for i in range(n_steps)]

    sat_lats, sat_lons = _subpoints(line1, line2, sensor_key, times)

    half_width_km = SWATH_HALF_WIDTH_KM.get(sensor_key, 700)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    dlat = (sat_lats - center_lat) * km_per_deg_lat
    dlon = _wrap_lon_delta(sat_lons - center_lon) * km_per_deg_lon
    dist_km = np.hypot(dlat, dlon)

    return [times[i] for i in range(len(times)) if dist_km[i] <= half_width_km]


def predict_ssmis_satellite(
    center_lat: float,
    center_lon: float,
    start_time: datetime,
    end_time: datetime,
    step_seconds: int = 20,
) -> tuple:
    """SSMIS flies on 3 active DMSP satellites (F16/F17/F18) -- check all
    three and return (satellite_key, hit_times) for whichever has the
    most recent predicted overpass, or (None, []) if none do. A
    per-satellite TLE fetch failure (e.g. one satellite temporarily
    absent from Celestrak) doesn't abort the others."""
    best_key, best_hits = None, []
    for key in ("SSMIS-F16", "SSMIS-F17", "SSMIS-F18"):
        try:
            hits = predict_overpasses(key, center_lat, center_lon, start_time, end_time, step_seconds)
        except Exception:
            continue
        if hits and (not best_hits or hits[-1] > best_hits[-1]):
            best_key, best_hits = key, hits
    return best_key, best_hits


def hour_strings(hits: list) -> list:
    """Collapse a list of hit datetimes into unique 'YYYYMMDDHH' strings,
    for narrowing a PPS directory query to just the predicted hour(s)
    instead of a whole day/folder."""
    seen = []
    for dt in hits:
        s = dt.strftime("%Y%m%d%H")
        if s not in seen:
            seen.append(s)
    return seen
