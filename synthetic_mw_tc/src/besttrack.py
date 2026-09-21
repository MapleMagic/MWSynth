"""
NHC ATCF best-track ingestion.

Files live at:
    https://ftp.nhc.noaa.gov/atcf/btk/b<BB><NN><YYYY>.dat     (best track, "b-deck")

    BB = basin: al (Atlantic), ep (East Pacific), cp (Central Pacific), wp, etc.
    NN = 2-digit storm number for the season
    YYYY = 4-digit year

Format is comma-delimited ATCF "b-deck" records, NOT fixed-width. Reference:
    https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt

Relevant fields (0-indexed after split on comma + strip):
    0  BASIN
    1  CY (storm number)
    2  YYYYMMDDHH (valid time)
    3  TECHNUM/MIN
    4  TECH ("BEST" for best track)
    5  TAU (forecast hour, 0 for best track fixes)
    6  LatN/S  e.g. "251N" -> 25.1 N
    7  LonE/W  e.g. "800W" -> -80.0
    8  VMAX (knots)
    9  MSLP (mb)
    10 storm type (TD, TS, HU, EX, etc.)
    11 RAD (wind radii threshold this row describes: 34/50/64, or 0)
    12 WINDCODE
    13-16 RAD1-RAD4 (quadrant wind radii, not used here)
    17 POUTER (pressure of the outermost closed isobar, mb)
    18 ROUTER (radius of the outermost closed isobar, nm) -- "ROCI"
    19 RMW (radius of max wind, nm)

    A single valid_time can have multiple rows (one per RAD threshold:
    34/50/64 kt), and ROCI/RMW are typically only populated on a subset of
    those rows (often just the 34kt row) with the rest left blank/zero. We
    keep the largest non-zero value seen across all rows for a given
    valid_time rather than whichever row happens to be parsed last.
"""
from __future__ import annotations

import csv
import dataclasses
import gzip
import io
import os
from datetime import datetime, timedelta
from typing import Optional

import requests

from data_types import StormFix
import timeutil

BASE_URL = "https://ftp.nhc.noaa.gov/atcf/btk"
ARCHIVE_BASE_URL = "https://ftp.nhc.noaa.gov/atcf/archive"

# NHC's btk server (per direct confirmation: "NHC analyzes current and past
# tropical cyclone locations and intensities for its area of responsibility")
# only covers the basins NHC/CPHC are actually responsible for. Everything
# else (Western Pacific, North Indian Ocean, Southern Hemisphere) is JTWC's
# area of responsibility, and JTWC's own site is a webpage-based portal
# rather than a simple directory listing -- not something to scrape
# reliably. IBTrACS (NOAA/NCEI's own unified, multi-agency archive,
# confirmed at https://www.ncei.noaa.gov/data/international-best-track-
# archive-for-climate-stewardship-ibtracs/v04r01/access/csv/) already
# combines NHC's and JTWC's data into one consistent format and is used as
# the fallback for any basin NHC's server doesn't cover.
NHC_BASINS = ("AL", "EP", "CP")

IBTRACS_ALL_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
IBTRACS_CACHE_PATH = os.path.expanduser("~/.synthetic_mw_tc/ibtracs_cache/ibtracs.ALL.csv")
IBTRACS_CACHE_MAX_AGE_DAYS = 7  # historical rows don't change; recent-storm
# rows do get revised, but a week-old cache is a reasonable balance against
# re-downloading a large multi-basin file on every single storm lookup.


def build_url(basin: str, storm_num: int, year: int) -> str:
    basin = basin.lower()
    return f"{BASE_URL}/b{basin}{storm_num:02d}{year}.dat"


def build_archive_url(basin: str, storm_num: int, year: int) -> str:
    """NHC's /atcf/btk/ directory holds ONLY the current season -- confirmed
    directly against the live listing, which contained nothing but 2026
    storms. Historical best tracks live under /atcf/archive/<year>/ and are
    gzipped. A real run showed every 2023/2024 AL/EP/CP storm 404ing from
    the btk path, which is why that whole phase "took mere seconds" and
    produced 90 no_best_track errors -- silently losing exactly the basins
    that GOES can actually see."""
    basin = basin.lower()
    return f"{ARCHIVE_BASE_URL}/{year}/b{basin}{storm_num:02d}{year}.dat.gz"


def fetch_recent_btk_storms(hours: float = 24.0, timeout: int = 30) -> list:
    """List ATCF best-track .dat files on NHC's public btk server that
    have been modified within the last `hours` hours, for populating a
    storm-selection dropdown instead of typing basin/storm#/year into
    three separate boxes.

    REVISED after a real run against the live server (with the correct
    ?C=M;O=D sort-by-modified-date URL) returned zero storms -- meaning
    the original approach (regex-parsing a displayed date column out of
    the directory listing's HTML, assuming a specific "DD-Mon-YYYY
    HH:MM" format) didn't match NHC's actual page. Rather than guess at
    a different date format and risk the exact same kind of silent
    failure again, this now decouples the two concerns:

    1. Get the list of candidate filenames from the directory listing
       (a much simpler, more robust regex -- just extracting href="..."
       values, not also trying to parse a date in the same pattern).
    2. For each candidate file, get its actual modification time from
       its own HTTP "Last-Modified" response header (via a HEAD request)
       -- a standardized, RFC 7231-defined format, completely independent
       of however NHC's directory-listing HTML happens to display dates
       for humans to read. This is far more robust than scraping a
       display column, at the cost of one extra request per candidate
       file -- kept fast by issuing all the HEAD requests concurrently.

    Returns a list of dicts, most-recently-modified first:
        {"filename":.., "basin":.., "storm_num":.., "year":..,
         "last_modified": datetime (UTC), "display_name": str}

    HONEST CAVEAT, still: the filename-list regex is simpler and more
    likely to be robust than the original date-column approach, but
    still assumes NHC's page contains plain href="....dat" links
    somewhere -- not re-verified against the live server directly (no
    network access here). If NHC's page structure is different enough
    that even this fails, this returns an empty list (a visible "nothing
    found," not a crash), same as before.
    """
    import re
    from datetime import timedelta, timezone
    from email.utils import parsedate_to_datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        resp = requests.get(f"{BASE_URL}/", timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return []

    # Just extract filenames from href="..." attributes -- decoupled
    # from also trying to parse a date in the same pass, which is what
    # broke against the real server.
    href_pattern = re.compile(r'href="([^"]+)"', re.IGNORECASE)
    name_pattern = re.compile(r"b([a-zA-Z]{2})(\d{2})(\d{4})\.dat$", re.IGNORECASE)

    candidate_files = set()
    for href in href_pattern.findall(resp.text):
        basename = href.rsplit("/", 1)[-1]
        if name_pattern.match(basename):
            candidate_files.add(basename)

    if not candidate_files:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    def _check_one_file(filename):
        try:
            head_resp = requests.head(f"{BASE_URL}/{filename}", timeout=timeout)
            head_resp.raise_for_status()
        except Exception:
            return None
        last_mod_str = head_resp.headers.get("Last-Modified")
        if not last_mod_str:
            return None
        try:
            mod_time = parsedate_to_datetime(last_mod_str)
        except (TypeError, ValueError):
            return None
        if mod_time.tzinfo is None:
            mod_time = mod_time.replace(tzinfo=timezone.utc)
        if mod_time < cutoff:
            return None
        return filename, mod_time

    results = []
    # HEAD requests are lightweight (no body transferred) and independent
    # of each other -- run them concurrently rather than one at a time,
    # since there could be a couple hundred candidate files in the
    # directory (most seasons' worth, not just active storms).
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_check_one_file, f) for f in candidate_files]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            filename, mod_time = result
            m = name_pattern.match(filename)
            basin, storm_num_str, year_str = m.groups()
            results.append({
                "filename": filename,
                "basin": basin.upper(),
                "storm_num": int(storm_num_str),
                "year": int(year_str),
                "last_modified": mod_time,
                "display_name": f"{basin.upper()}{storm_num_str} ({year_str})",
            })

    results.sort(key=lambda r: r["last_modified"], reverse=True)
    return results


def _parse_latlon(lat_raw: str, lon_raw: str) -> tuple[float, float]:
    lat_hemi = lat_raw[-1]
    lat_val = float(lat_raw[:-1]) / 10.0
    if lat_hemi == "S":
        lat_val = -lat_val

    lon_hemi = lon_raw[-1]
    lon_val = float(lon_raw[:-1]) / 10.0
    if lon_hemi == "W":
        lon_val = -lon_val

    return lat_val, lon_val


def fetch_best_track(basin: str, storm_num: int, year: int, timeout: int = 30) -> list[StormFix]:
    """Download and parse a full best-track file into a list of StormFix,
    one per unique valid_time (TAU==0, TECH=='BEST' records only).

    NHC's own btk server only covers AL/EP/CP (its actual area of
    responsibility) -- any other basin (WP, IO, SH, or anything else)
    falls back to IBTrACS instead, which already has this data via JTWC.
    """
    if basin.upper() in NHC_BASINS:
        # Try the current-season btk path first, then fall back to the
        # gzipped archive for past seasons. Order matters: btk holds only
        # the CURRENT season, so for any historical storm the first
        # request 404s and the archive is what actually has the data.
        try:
            resp = requests.get(build_url(basin, storm_num, year), timeout=timeout)
            resp.raise_for_status()
            return parse_best_track_text(resp.text, basin=basin, storm_num=storm_num)
        except requests.exceptions.HTTPError:
            resp = requests.get(build_archive_url(basin, storm_num, year), timeout=timeout)
            resp.raise_for_status()
            text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
            return parse_best_track_text(text, basin=basin, storm_num=storm_num)

    return fetch_best_track_ibtracs(basin, storm_num, year, timeout=timeout)


def _ensure_ibtracs_cache(timeout: int = 120) -> str:
    """Download the IBTrACS ALL-basin CSV if not already cached locally,
    or if the cache is older than IBTRACS_CACHE_MAX_AGE_DAYS. Returns the
    local path. This is a genuinely large, multi-basin, 1840s-to-present
    file -- cached specifically so a run processing many WP/IO/SH storms
    (like a bulk TC-PRIMED mining run) downloads it once, not once per
    storm."""
    os.makedirs(os.path.dirname(IBTRACS_CACHE_PATH), exist_ok=True)

    needs_download = True
    if os.path.exists(IBTRACS_CACHE_PATH):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(IBTRACS_CACHE_PATH))
        if age < timedelta(days=IBTRACS_CACHE_MAX_AGE_DAYS):
            needs_download = False

    if needs_download:
        resp = requests.get(IBTRACS_ALL_URL, timeout=timeout)
        resp.raise_for_status()
        tmp_path = IBTRACS_CACHE_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        os.replace(tmp_path, IBTRACS_CACHE_PATH)  # atomic -- avoids a
        # truncated/partial cache file if the download is interrupted
        # partway through, same lesson as an earlier truncated-file bug
        # in this project's live NRT download path.

    return IBTRACS_CACHE_PATH


_IBTRACS_INDEX: dict = {}
_IBTRACS_INDEX_PATH = None


def _get_ibtracs_index(cache_path: str) -> dict:
    """Build (once) and return an in-memory index of the IBTrACS CSV,
    keyed by USA_ATCF_ID.

    This exists because the previous implementation streamed the ENTIRE
    file through csv.DictReader on every single storm lookup. IBTrACS
    ALL covers every basin from 1840 to present -- roughly 700k rows of
    ~180 columns -- and DictReader allocates a dict per row. Looking up
    100 storms therefore meant 100 full passes and tens of millions of
    throwaway dicts. A real run spent hours on WP/IO/SH storms doing
    exactly this, only to discard each storm immediately afterwards as
    being outside GOES coverage.

    Built with csv.reader and fixed column indices (much cheaper than
    DictReader) and stores only the eight fields actually used, parsed
    to their final types, so per-storm lookup becomes a dict hit.
    """
    global _IBTRACS_INDEX, _IBTRACS_INDEX_PATH
    if _IBTRACS_INDEX_PATH == cache_path and _IBTRACS_INDEX:
        return _IBTRACS_INDEX

    def _f(v):
        v = v.strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    index: dict = {}
    with open(cache_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return {}
        col = {name: i for i, name in enumerate(header)}
        try:
            i_id = col["USA_ATCF_ID"]; i_time = col["ISO_TIME"]
            i_lat = col["USA_LAT"]; i_lon = col["USA_LON"]
            i_wind = col["USA_WIND"]; i_pres = col["USA_PRES"]
            i_rmw = col["USA_RMW"]; i_roci = col["USA_ROCI"]
            i_status = col["USA_STATUS"]
        except KeyError as e:
            raise RuntimeError(f"IBTrACS CSV missing expected column {e}") from e

        next(reader, None)  # units row

        n_cols = len(header)
        for row in reader:
            if len(row) < n_cols:
                continue
            atcf = row[i_id].strip()
            if not atcf:
                continue
            lat, lon = _f(row[i_lat]), _f(row[i_lon])
            if lat is None or lon is None:
                continue
            try:
                vt = timeutil.as_utc(datetime.strptime(row[i_time].strip(), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            index.setdefault(atcf, []).append(
                (vt, lat, lon, _f(row[i_wind]), _f(row[i_pres]),
                 _f(row[i_rmw]), _f(row[i_roci]), row[i_status].strip())
            )

    _IBTRACS_INDEX = index
    _IBTRACS_INDEX_PATH = cache_path
    return index


def fetch_best_track_ibtracs(basin: str, storm_num: int, year: int, timeout: int = 120) -> list[StormFix]:
    """Fetch best-track data for a basin NHC's server doesn't cover
    (WP/IO/SH), via IBTrACS's own USA_ATCF_ID column -- which directly
    matches the same BASIN+NN+YYYY convention as NHC's b-decks and this
    project's own StormFix.storm_id (e.g. "SH062023"), confirmed
    directly against a real IBTrACS data sample. Filtering by
    USA_ATCF_ID specifically sidesteps IBTrACS's own basin subdivision
    (it splits the Southern Hemisphere into SI/South Indian and
    SP/South Pacific separately, unlike JTWC/TC-PRIMED's single combined
    "SH" -- matching on the ATCF ID avoids needing to know or guess
    which of those two a given SH storm actually falls under).

    USA_* columns (not WMO_*) used throughout for wind/pressure/RMW/ROCI
    -- WMO_WIND can be a 10-minute-average convention depending on which
    agency is "currently responsible" for a basin (e.g. JMA for WP, IMD
    for NIO), whereas USA_WIND is consistently JTWC's 1-minute
    convention, matching what NHC's b-decks (and the rest of this
    project) already assume.
    """
    cache_path = _ensure_ibtracs_cache(timeout=timeout)
    target_atcf_id = f"{basin.upper()}{storm_num:02d}{year}"

    fixes: dict[datetime, StormFix] = {}
    storm_id = target_atcf_id

    # Serve from the in-memory index if it has been built. See
    # _build_ibtracs_index() for why this matters so much.
    index = _get_ibtracs_index(cache_path)
    for row in index.get(target_atcf_id, ()):
        valid_time, lat, lon, wind, pres, rmw, roci, status = row
        fixes[valid_time] = StormFix(
            storm_id=storm_id, valid_time=valid_time, lat=lat, lon=lon,
            vmax_kt=wind if wind is not None else float("nan"),
            mslp_mb=pres, rmw_nm=rmw, roci_nm=roci,
            storm_type=status, basin=basin.upper(),
        )
    return sorted(fixes.values(), key=lambda fx: fx.valid_time)


def _legacy_ibtracs_scan(cache_path, target_atcf_id, storm_id, basin):
    """Original row-by-row scan, kept only as a reference for what the
    index replaces. Not called."""
    fixes: dict = {}
    with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        # Row immediately after the header is a units row (e.g. "N/A",
        # "kts", "mb", ...), not a data row -- confirmed directly against
        # real IBTrACS CSV structure. Skip it explicitly rather than
        # trying to parse it as data and silently failing/skipping it
        # via a caught exception.
        first_row = next(reader, None)

        for row in reader:
            if row.get("USA_ATCF_ID", "").strip() != target_atcf_id:
                continue

            iso_time_str = row.get("ISO_TIME", "").strip()
            if not iso_time_str:
                continue
            try:
                valid_time = timeutil.as_utc(datetime.strptime(iso_time_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue

            def _f(key: str) -> Optional[float]:
                val = row.get(key, "").strip()
                if not val:
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            lat = _f("USA_LAT")
            lon = _f("USA_LON")
            if lat is None or lon is None:
                continue  # no position -- nothing usable for this row

            fixes[valid_time] = StormFix(
                storm_id=storm_id,
                valid_time=valid_time,
                lat=lat,
                lon=lon,
                vmax_kt=_f("USA_WIND") or float("nan"),
                mslp_mb=_f("USA_PRES"),
                rmw_nm=_f("USA_RMW"),
                roci_nm=_f("USA_ROCI"),
                storm_type=row.get("USA_STATUS", "").strip(),
                basin=basin.upper(),
            )

    return sorted(fixes.values(), key=lambda fx: fx.valid_time)


def parse_best_track_text(text: str, basin: str, storm_num: int) -> list[StormFix]:
    fixes: dict[datetime, StormFix] = {}
    storm_id = f"{basin.upper()}{storm_num:02d}{{year}}"  # filled in once we see a timestamp

    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 11:
            continue

        tech = fields[4]
        tau = fields[5]
        if tech != "BEST" or tau != "0":
            continue

        valid_time = timeutil.as_utc(datetime.strptime(fields[2], "%Y%m%d%H"))
        lat, lon = _parse_latlon(fields[6], fields[7])
        vmax = float(fields[8]) if fields[8] else float("nan")
        mslp = float(fields[9]) if fields[9] else None
        storm_type = fields[10]

        def _safe_float(idx: int) -> Optional[float]:
            if idx < len(fields) and fields[idx] not in ("", "0"):
                try:
                    return float(fields[idx])
                except ValueError:
                    return None
            return None

        roci = _safe_float(18)
        rmw = _safe_float(19)

        sid = f"{basin.upper()}{storm_num:02d}{valid_time.year}"

        existing = fixes.get(valid_time)
        if existing is not None:
            # Multiple rows (34/50/64kt radii) can share a valid_time; keep
            # the largest non-zero ROCI/RMW seen across them rather than
            # letting a later blank row erase an earlier real value.
            if existing.roci_nm is not None:
                roci = max(roci, existing.roci_nm) if roci is not None else existing.roci_nm
            if existing.rmw_nm is not None:
                rmw = max(rmw, existing.rmw_nm) if rmw is not None else existing.rmw_nm

        fixes[valid_time] = StormFix(
            storm_id=sid,
            valid_time=valid_time,
            lat=lat,
            lon=lon,
            vmax_kt=vmax,
            mslp_mb=mslp,
            rmw_nm=rmw,
            roci_nm=roci,
            storm_type=storm_type,
            basin=basin.upper(),
        )

    return sorted(fixes.values(), key=lambda f: f.valid_time)


def nearest_fix(fixes: list[StormFix], target_time: datetime) -> Optional[StormFix]:
    if not fixes:
        return None
    return min(fixes, key=lambda f: abs((f.valid_time - target_time).total_seconds()))


def _wrap180(lon: float) -> float:
    """Normalize a longitude to [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def _lon_delta(lon_from: float, lon_to: float) -> float:
    """Shortest signed longitude difference, dateline-safe.

    A storm crossing 180 has consecutive best-track fixes at, say, +179.5
    and -179.5 -- a 1 degree westward move. Subtracting them naively gives
    -359 degrees, and a linear interpolation between the two then sweeps
    BACKWARDS across the entire globe: at the midpoint it placed a west
    Pacific typhoon at longitude 0, in the Atlantic. Everything positioned
    from the fix followed it there.

    This never fired in the Atlantic or east Pacific, which is why it
    survived; it would fire on the first west Pacific storm to cross the
    dateline. Taking the shortest path around the circle is the only
    correct reading -- no tropical cyclone moves 359 degrees in six hours.
    """
    return _wrap180(lon_to - lon_from)


# Motion for extrapolation is estimated over this many hours of prior
# fixes rather than from the last pair alone. Best-track positions are
# quantized to 0.1 degree, so a single 6h pair has ~0.1 deg of rounding
# noise in it; averaging over a longer arm damps that without smearing
# genuine curvature too badly.
MOTION_BASELINE_HOURS = 12.0

# Beyond this gap, fall back to the old freeze-at-last-fix behaviour. A
# linear projection over half a day cannot represent recurvature or a
# trough interaction, and a confidently wrong centre 12 h out is worse
# than an obviously stale one. A gap this large also means the best-track
# feed is broken, which is its own problem.
MAX_EXTRAPOLATION_HOURS = 12.0

# Implied translation speeds above this are treated as a data error (a
# mis-parsed or duplicated fix) rather than a real motion vector.
MAX_PLAUSIBLE_SPEED_KT = 50.0


def _motion_per_hour(fixes: list, anchor, backward: bool = False):
    """Estimate (dlat/hr, dlon/hr) from the fixes adjacent to `anchor`.
    Returns (0.0, 0.0) when motion can't be estimated or looks unphysical."""
    if backward:
        window = [f for f in fixes
                  if 0 < (f.valid_time - anchor.valid_time).total_seconds() <= MOTION_BASELINE_HOURS * 3600]
        other = min(window, key=lambda f: f.valid_time) if window else None
    else:
        window = [f for f in fixes
                  if 0 < (anchor.valid_time - f.valid_time).total_seconds() <= MOTION_BASELINE_HOURS * 3600]
        other = max(window, key=lambda f: f.valid_time) if window else None
    if other is None:
        return 0.0, 0.0

    dt_hr = abs((anchor.valid_time - other.valid_time).total_seconds()) / 3600.0
    if dt_hr <= 0:
        return 0.0, 0.0

    dlat = (anchor.lat - other.lat) / dt_hr
    dlon = _lon_delta(other.lon, anchor.lon) / dt_hr
    if backward:
        dlat, dlon = -dlat, -dlon

    # Sanity-check the implied speed. 1 deg lat = 60 nm; longitude is
    # scaled by cos(lat) so a fast-moving high-latitude storm isn't
    # falsely rejected.
    import math
    speed_kt = math.hypot(dlat, dlon * math.cos(math.radians(anchor.lat))) * 60.0
    if not math.isfinite(speed_kt) or speed_kt > MAX_PLAUSIBLE_SPEED_KT:
        return 0.0, 0.0
    return dlat, dlon


def _extrapolated(anchor, fixes: list, target_time: datetime, backward: bool = False):
    """Project `anchor`'s POSITION to target_time along its recent motion.

    Position only. Intensity, RMW and ROCI are carried forward unchanged
    (persistence), deliberately: position extrapolates well over a few
    hours because storm motion is smooth, whereas linearly extrapolating
    a rapidly intensifying storm's vmax produces values that never occur,
    and RMW/ROCI are noisy enough between advisories that a projected
    trend is mostly noise. Those fields degrade gracefully when stale;
    position does not.
    """
    gap_hr = abs((target_time - anchor.valid_time).total_seconds()) / 3600.0
    if gap_hr > MAX_EXTRAPOLATION_HOURS:
        return anchor

    dlat_hr, dlon_hr = _motion_per_hour(fixes, anchor, backward=backward)
    if dlat_hr == 0.0 and dlon_hr == 0.0:
        return anchor

    return dataclasses.replace(
        anchor,
        valid_time=target_time,
        lat=anchor.lat + dlat_hr * gap_hr,
        lon=_wrap180(anchor.lon + dlon_hr * gap_hr),
        extrapolated_hours=gap_hr,
    )


def interpolate_fix(fixes: list[StormFix], target_time: datetime) -> Optional[StormFix]:
    """Linear interpolation between the two bracketing best-track fixes.
    Best track is only every 6h, so for anything algorithm-critical (storm
    center for the synthetic MW radial model) this matters more than
    nearest-neighbor snapping.

    OUTSIDE the best-track range the position is EXTRAPOLATED along the
    storm's recent motion rather than frozen at the endpoint fix. This
    used to return `before[-1]` unchanged, which is a real problem for
    live use: best track lands every 6 h, so a frame generated at 11:04Z
    was being built around the 06:00Z centre with five hours of motion
    ignored. For a storm moving 10-12 kt that is roughly a degree of
    longitude, and it puts the entire synthetic field -- eyewall ring,
    radial profile, ML patch -- beside the eye that is plainly visible in
    the GOES imagery underneath it. Observed on Lowell (EP12) at 11:04Z,
    where the real eye sat clearly west of the synthetic core.

    Because every consumer routes through here, this also fixes
    mw_ingest.morph_swath_to_time, which advects a real pass by the
    displacement between two calls to this function: with the endpoint
    frozen, a pass observed after the last best-track entry computed a
    zero displacement and was left unshifted. The morphing logic was
    correct; the positions it was given were not.
    """
    if not fixes:
        return None
    # Coerce BOTH sides. Best-track times are aware at the source now, but
    # a caller may still pass naive, and this comparison is the single
    # point every mined overpass passes through -- it failed 613 times in
    # a row when the two conventions met here.
    target_time = timeutil.as_utc(target_time)
    fixes = [dataclasses.replace(f, valid_time=timeutil.as_utc(f.valid_time))
             for f in fixes]
    before = [f for f in fixes if f.valid_time <= target_time]
    after = [f for f in fixes if f.valid_time >= target_time]
    if not before:
        return _extrapolated(after[0], fixes, target_time, backward=True)
    if not after:
        return _extrapolated(before[-1], fixes, target_time, backward=False)

    f0 = max(before, key=lambda f: f.valid_time)
    f1 = min(after, key=lambda f: f.valid_time)
    if f0.valid_time == f1.valid_time:
        return f0

    total = (f1.valid_time - f0.valid_time).total_seconds()
    frac = (target_time - f0.valid_time).total_seconds() / total

    def lerp(a, b):
        if a is None or b is None:
            return a
        return a + (b - a) * frac

    def lerp_lon(a, b):
        # Dateline-safe: interpolate along the SHORTEST arc (see
        # _lon_delta) rather than through the naive difference.
        if a is None or b is None:
            return a
        return _wrap180(a + _lon_delta(a, b) * frac)

    return StormFix(
        storm_id=f0.storm_id,
        valid_time=target_time,
        lat=lerp(f0.lat, f1.lat),
        lon=lerp_lon(f0.lon, f1.lon),
        vmax_kt=lerp(f0.vmax_kt, f1.vmax_kt),
        mslp_mb=lerp(f0.mslp_mb, f1.mslp_mb),
        rmw_nm=lerp(f0.rmw_nm, f1.rmw_nm),
        roci_nm=lerp(f0.roci_nm, f1.roci_nm),
        storm_type=f0.storm_type,
        basin=f0.basin,
    )
