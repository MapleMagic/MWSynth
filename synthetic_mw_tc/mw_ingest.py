"""
Real passive-microwave ingestion.

PRIMARY, actively-maintained sensors (per direct guidance: AMSR2-NRT
retired transmission after Aug 31 2026, SSMIS shuts down entirely in
September 2026 -- both deliberately deprioritized as a result):
  - GMI-NRT, WSFM-NRT, AMSR3-NRT via PPS's jsimpsonhttps near-real-time
    server (see PPS_NRT_BASE below), with TLE-based overpass narrowing,
    dynamic channel discovery (_discover_wsfm_channels -- generic
    despite the name), and MIMIC-TC-style advection/morphing.
  - GMI archive (NASA Earthdata / GES DISC, via `earthaccess`) as a
    fallback when none of the three NRT sensors have anything usable --
    higher latency, but far more reliable, and not limited to NRT's
    ~7-day retention window.

SECONDARY / lower-priority paths still present in this file:
  - AMSR2 (JAXA G-Portal, via `gportal`) and SSMIS (NOAA/NCEI archive,
    ~1-month latency) -- both implemented and functional, but not the
    focus of ongoing work given their sunset above.
  - SSMIS-NRT (PPS, same server as the primary trio) -- implemented but
    not part of the primary sensor rotation the GUI searches by default.

This docstring is deliberately kept current (previous versions describing
SSMIS as "not solved" and GMI-NRT as "not implemented" were stale for a
long time before being corrected here) -- if you're reading this while
debugging a real failure, the accurate status of each sensor's actual
fetch function is in that function's own docstring, not assumed from here.
"""
from __future__ import annotations

import os
import tempfile
import re
import glob
import dataclasses
from datetime import datetime, timedelta, timezone
import timeutil
from typing import Optional

import numpy as np

import besttrack
from data_types import MWSwath
from qc_utils import sanitize_field

# Platform temp dir, not a hardcoded POSIX path. The literal
# "/tmp/mw_cache" fails on Windows with PermissionError [WinError 5] --
# the same bug found in goes_fetch, present here too and never hit only
# because this path runs less often.
CACHE_DIR = os.path.join(tempfile.gettempdir(), "mw_cache")
# NRT downloads specifically go under the persistent app-config directory
# instead of /tmp -- requested directly, since /tmp's permission/lifetime
# behavior varies by OS and isn't a great place for files you actually
# want to keep around and re-inspect.
NRT_CACHE_DIR = os.path.expanduser("~/.synthetic_mw_tc/NRT")


def clear_nrt_cache() -> tuple:
    """Delete all downloaded PPS NRT files (GMI/SSMIS/WSFM/AMSR3, under
    NRT_CACHE_DIR) and clear the in-process directory-listing cache.
    Added since these are full swath files (SSMIS/AMSR2/AMSR3 in
    particular can span well over an hour of orbit, ~15-25MB+ each) that
    accumulate over time with normal use -- this is a manual cleanup
    tool, not automatic, since there's no way to know from here whether
    the user still wants a given cached file around.

    Returns (files_deleted, bytes_freed). Safe to call even if the cache
    directory doesn't exist yet (returns (0, 0)).
    """
    if not os.path.isdir(NRT_CACHE_DIR):
        return 0, 0

    files_deleted = 0
    bytes_freed = 0
    for root, _dirs, files in os.walk(NRT_CACHE_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                bytes_freed += os.path.getsize(fpath)
                os.remove(fpath)
                files_deleted += 1
            except OSError:
                continue  # best-effort -- a locked/permission-denied file shouldn't abort the whole cleanup

    _PPS_LISTING_CACHE.clear()
    return files_deleted, bytes_freed


# ---------------------------------------------------------------------------
# GMI NRT (PPS "jsimpson" near-real-time server) -- UNVERIFIED, see docstring
# ---------------------------------------------------------------------------
# This is a genuinely different system from the GES DISC archive used by
# fetch_gmi_swath(): NASA's Precipitation Processing System (PPS) hosts a
# separate near-real-time feed (typically available within a few hours of
# observation, vs. the multi-day latency seen from the standard archive).
#
# Access requires a SEPARATE registration step at
# https://registration.pps.eosdis.nasa.gov/registration/ -- specifically
# opting in to NRT access (having an Earthdata/GES DISC login is not
# sufficient by itself). PPS documentation describes using your registered
# email as both username and password for basic-auth-style access to the
# jsimpsonhttps server.
#
# I could not verify this implementation against a live server (no network
# in this sandbox), so treat it as a documented-but-untested best effort:
# it queries PPS's custom "/text/" listing endpoint (which supports
# wildcard patterns server-side, per NASA's own retrieval documentation),
# parses the returned file list for the observation time embedded in each
# filename, and downloads the closest match. If the auth mechanism, listing
# response format, or download path differs from what's implemented here,
# it should fail with a clear HTTP error rather than silently returning
# wrong data -- but this is the one ingestion path in this file that
# genuinely needs a live test before you trust it.
PPS_NRT_BASE = "https://jsimpsonhttps.pps.eosdis.nasa.gov"

# In-process cache for full directory listings (subdir -> (fetched_at, files)).
# These directories are confirmed flat and "massive" (no date subfolders,
# full ~2-week NRT retention, multiple satellites for SSMIS) -- refetching
# per lookback tier (6/9/12/24/48h) within one search would be wasteful.
_PPS_LISTING_CACHE: dict = {}
_PPS_LISTING_CACHE_TTL_SECONDS = 300


def _pps_text_wildcard_listing(pattern_suffix: str, username: str, password: str) -> list:
    """Query PPS's documented plain-text + wildcard endpoint directly --
    e.g. GET /text/1C/SSMIS/*F17*20260822-S17*, which returns just the
    matching full paths as plain text (one per line), with the SERVER
    doing the filtering rather than the client fetching and parsing an
    entire (possibly huge) directory listing first.

    This is exactly the documented, PPS-recommended approach for script-
    based access (per PPS's own "Accessing the PPS Near Real Time Data
    using HTTPS" guide, Cohoon & Kelley, 11 June 2020) -- confirmed with
    a real worked example querying /text/1C/*/*<date>*<hour>* and getting
    back real file paths, including under /1C/SSMIS/ specifically.

    An EARLIER version of this project tried a "/text/" endpoint and
    found it "confirmed broken: repeated 404s" -- that conclusion is
    being revisited here rather than trusted at face value, now that
    official documentation shows a concrete working example against the
    exact directory (SSMIS) that's been failing. The most likely
    explanation is a URL construction difference (wrong wildcard syntax
    or pattern shape), not that the endpoint itself doesn't exist.

    Per that documentation, if nothing matches the server returns a 404
    (not an empty 200 body) -- handled here as "no files found," not an
    error worth raising.

    Returns a list of full paths (e.g. "/1C/SSMIS/1C.F17.SSMIS...RT-H5"),
    or an empty list if nothing matched or the request itself failed for
    any reason (caller should fall back to the HTML listing approach in
    that case, not treat this as fatal).
    """
    import requests

    url = f"{PPS_NRT_BASE}/text/{pattern_suffix}"
    try:
        resp = requests.get(url, auth=(username, password), timeout=90)
    except requests.exceptions.RequestException:
        return []
    if resp.status_code == 404:
        return []
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        return []
    return resp.text.split()


def _pps_html_listing(subdir: str, username: str, password: str) -> list:
    """Fetch the REAL Apache-style directory listing (the same one you get
    browsing e.g. https://jsimpsonhttps.pps.eosdis.nasa.gov/1CR/?C=M;O=D in
    a browser) and parse out filenames directly from the HTML.

    This replaces an earlier approach that used a documented-but-apparently-
    no-longer-working "/text/" wildcard-query endpoint (confirmed broken:
    repeated 404s across multiple subdirectories and query patterns, not a
    one-off) -- since revisited for SSMIS specifically via
    _pps_text_wildcard_listing, see that function's docstring. Since these
    directories are flat (NO date subfolders -- confirmed directly, don't
    reintroduce that assumption) and can be large, results are cached
    in-process for a few minutes so repeated searches within one run
    (including find_swath_that_hit_storm's retry loop, which can make
    several searches per sensor) don't refetch the same huge listing
    over and over.
    """
    import re as _re
    import time as _time
    import requests

    now = _time.time()
    cached = _PPS_LISTING_CACHE.get(subdir)
    if cached and (now - cached[0]) < _PPS_LISTING_CACHE_TTL_SECONDS:
        return cached[1]

    url = f"{PPS_NRT_BASE}/{subdir}/"
    resp = requests.get(url, auth=(username, password), timeout=90)
    resp.raise_for_status()

    # Standard Apache autoindex rows: <a href="FILENAME">FILENAME</a>.
    # Exclude sort-header links (href="?C=...") and parent-directory links.
    hrefs = _re.findall(r'href="([^"]+)"', resp.text)
    files = [h for h in hrefs if not h.startswith("?") and not h.startswith("/") and h != "../"]

    _PPS_LISTING_CACHE[subdir] = (now, files)
    return files


def _open_gmi_swath1_group(local_path: str):
    """Open the GMI 1C-R swath-1 group ('S1', containing Tc/Latitude/
    Longitude with the 10-89 GHz channels) from a downloaded file.

    Tries the expected group name first (matches the GES DISC archive
    convention); if that fails -- e.g. the NRT ".RT-NC" repackaging uses
    different group naming -- falls back to inspecting the file directly
    via netCDF4 and raises a clear, specific error listing what's actually
    in there, rather than xarray's more opaque KeyError.
    """
    import xarray as xr

    try:
        return xr.open_dataset(local_path, group="S1")
    except Exception as first_error:
        try:
            import netCDF4

            with netCDF4.Dataset(local_path, "r") as nc:
                groups = list(nc.groups.keys())
                top_level_vars = list(nc.variables.keys())
        except Exception:
            groups = None
            top_level_vars = None

        raise RuntimeError(
            f"Could not open the 'S1' group in {local_path} ({first_error}). "
            f"Groups actually found in the file: {groups}. Top-level variables: "
            f"{top_level_vars}. The NRT ('.RT-NC') file structure may not match "
            "the GES DISC archive convention this code assumes -- open the file "
            "with netCDF4/ncdump to find the right group/variable names and "
            "update _open_gmi_swath1_group() in mw_ingest.py."
        ) from first_error


def _pps_nrt_find_and_download(
    subdir: str,
    target_time: datetime,
    lookback_hours: float,
    username: str,
    password: str,
    local_dir: str,
    hour_filters: Optional[list] = None,
    name_filter: Optional[str] = None,
    use_wildcard_text_query: bool = False,
    search_direction: str = "backward",
    progress_callback=None,
    exclude_filenames: Optional[set] = None,
):
    """Shared PPS NRT logic: list `subdir` (a flat directory -- these are
    confirmed to have NO date subfolders, don't reintroduce that
    assumption), filter to files within the search window, pick the best
    match, download it. Returns (local_path, observation_start_time) or
    (None, None) if nothing found. Used by GMI-NRT, SSMIS-NRT, WSFM-NRT,
    and AMSR3-NRT -- they only differ in which subdir to search and how
    to parse channels out of the downloaded file.

    hour_filters: if given (a list of 'YYYYMMDDHH' strings, typically from
        tle_predict.hour_strings() on a TLE-predicted overpass), only
        filenames containing one of those hour tokens are considered.
        If None, falls back to filtering by whole date(s) covering the
        search window instead (used when TLE prediction itself isn't
        available/failed).
    name_filter: if given (e.g. "F17"), only filenames containing that
        substring are considered -- relevant for SSMIS, which has
        multiple active satellites sharing one flat directory.
    exclude_filenames: if given (a set of bare filenames, not paths),
        candidates matching one of these are skipped when selecting the
        best match -- lets a caller retry with progressively older
        candidates from the SAME listing after downloading and checking
        a more recent one whose actual geographic coverage turned out
        not to include the target location (confirmed as a real,
        concrete failure mode: a file can match by TIME alone while its
        satellite's orbit was nowhere near the target). The underlying
        listing is cached (_PPS_LISTING_CACHE), so retrying like this
        doesn't re-fetch the directory listing over the network each
        time, only re-selects from what's already been fetched.
    use_wildcard_text_query: if True, TRIES the documented server-side
        wildcard text-listing endpoint first (_pps_text_wildcard_listing,
        one query per hour_filters token, combining name_filter and the
        hour token into the wildcard pattern) -- only relevant when
        hour_filters is available, since a wildcard query needs an actual
        pattern to filter by. Falls back to the existing full-listing
        approach (_pps_html_listing) if the wildcard query raises,
        returns nothing, or hour_filters isn't available.
    search_direction: "backward" (default, and the ONLY behavior this
        function had until now -- preserved exactly for anyone not
        passing this) searches [target_time - lookback_hours,
        target_time] and picks the MOST RECENT match. "forward" instead
        searches [target_time, target_time + lookback_hours] (reusing
        lookback_hours as the forward window's width) and picks the
        EARLIEST match. Added for MIMIC-TC-style crossfade morphing
        between a "before" pass and an "after" pass -- fetching an
        "after" pass for a target_time in ITS past is legitimate for
        historical/archived generation, where a pass that occurred after
        target_time has still already happened and its data already
        exists to fetch, even though a live/real-time run could never
        know about it in advance.
    """
    import requests

    if search_direction == "forward":
        start = target_time
        window_end = target_time + timedelta(hours=lookback_hours)
    else:
        start = target_time - timedelta(hours=lookback_hours)
        window_end = target_time

    if hour_filters is not None:
        tokens = list(hour_filters)
    else:
        dates_to_check = set()
        cursor = start
        while cursor <= window_end:
            dates_to_check.add(cursor.strftime("%Y%m%d"))
            cursor += timedelta(hours=1)
        dates_to_check.add(window_end.strftime("%Y%m%d"))
        tokens = sorted(dates_to_check)

    listing = None
    if use_wildcard_text_query and hour_filters:
        wildcard_results = []
        for tok in tokens:
            date_part, hour_part = tok[:8], tok[8:10]
            name_part = f"*{name_filter}*" if name_filter else "*"
            pattern = f"{subdir}/{name_part}{date_part}-S{hour_part}*"
            wildcard_results.extend(_pps_text_wildcard_listing(pattern, username, password))
        if wildcard_results:
            # Text-listing returns full paths (e.g. "/1C/SSMIS/1C.F17...");
            # strip to bare filenames to match what _pps_html_listing
            # returns, so the rest of this function doesn't need to care
            # which listing strategy actually produced the results.
            listing = [os.path.basename(p) for p in wildcard_results]

    if listing is None:
        try:
            listing = _pps_html_listing(subdir, username, password)
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"PPS NRT server rejected the directory listing request for {subdir} "
                f"({e}). This usually means the account isn't registered for NRT "
                "access yet -- see https://registration.pps.eosdis.nasa.gov/registration/ "
                "and select the NRT access option, separate from a normal Earthdata login."
            ) from e

    def _matches(fname: str) -> bool:
        if name_filter and name_filter not in fname:
            return False
        return any(tok in fname for tok in tokens)

    all_files = [f for f in listing if _matches(f)]

    candidates = []
    parseable_count = 0
    for fpath in all_files:
        if exclude_filenames and os.path.basename(fpath) in exclude_filenames:
            # Already tried this exact file in a previous attempt (see
            # the retry loop in each sensor's fetch_*_swath_nrt function)
            # and its geographic coverage didn't include the target box
            # -- skip it so the next-best candidate by time gets a turn,
            # instead of picking the same already-known-irrelevant file
            # again.
            continue
        m = re.search(r"(\d{8})-S(\d{6})-E(\d{6})", fpath)
        if not m:
            continue
        datestr, sstr, _estr = m.groups()
        try:
            stime = timeutil.as_utc(datetime.strptime(datestr + sstr, "%Y%m%d%H%M%S"))
        except ValueError:
            continue
        parseable_count += 1
        if start <= stime <= window_end:
            candidates.append((stime, fpath))

    if not candidates:
        if progress_callback:
            # Stage-by-stage breakdown -- added directly because a real
            # report showed all three primary sensors returning "nothing
            # found" with zero errors raised anywhere, giving no way to
            # tell WHICH stage actually came up empty: an empty raw
            # listing, a listing that has entries but none matching the
            # name/token filter, entries that match but have no
            # parseable S/E timestamp in their filename, or a parseable
            # timestamp that just falls outside the search window.
            # Mirrors the exact approach that found the previous two real
            # bugs (the truncated file, the engine-ordering regression)
            # -- get a specific enough error that the next failure is
            # immediately diagnosable instead of requiring another round
            # of guessing.
            progress_callback(
                f"MW search debug [{subdir}]: raw listing had {len(listing)} entries, "
                f"{len(all_files)} matched name/token filter "
                f"(tokens={tokens[:5]}{'...' if len(tokens) > 5 else ''}, name_filter={name_filter!r}), "
                f"{parseable_count} had a parseable S/E timestamp, "
                f"0 fell within [{start:%Y-%m-%d %H:%M}, {window_end:%Y-%m-%d %H:%M}]."
            )
        return None, None

    if search_direction == "forward":
        candidates.sort(key=lambda c: c[0])  # EARLIEST first -- soonest after target_time
    else:
        candidates.sort(key=lambda c: c[0], reverse=True)  # most recent first (original behavior)
    stime, best_path = candidates[0]

    # Normalize to a full path under /<subdir>/ regardless of whether the
    # listing returned bare filenames, paths relative to subdir, or
    # already-absolute paths (uncertain which the server does).
    if best_path.startswith("http"):
        download_url = best_path
    elif best_path.startswith("/"):
        download_url = f"{PPS_NRT_BASE}{best_path}"
    elif best_path.startswith(f"{subdir}/"):
        download_url = f"{PPS_NRT_BASE}/{best_path}"
    else:
        download_url = f"{PPS_NRT_BASE}/{subdir}/{best_path}"

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(best_path))

    needs_download = not os.path.exists(local_path)
    if not needs_download:
        # A cached file exists at this path -- but a REAL failure showed
        # this alone doesn't mean it's actually complete: a download
        # interrupted partway (network drop, timeout, the app closing
        # mid-write) can leave a truncated file sitting at the exact
        # final filename, and this check would previously trust it
        # forever, since nothing ever re-verified it. Confirmed directly:
        # a real AMSR3 file stuck at 48 bytes when it should have been
        # much larger, causing the exact same "channels not found"
        # failure on every subsequent run, since it kept reusing the
        # same broken cached file. Verify the cached file's size against
        # the server's real Content-Length before trusting it; force a
        # re-download on any mismatch.
        try:
            head_resp = requests.head(download_url, auth=(username, password), timeout=30)
            remote_size = int(head_resp.headers.get("Content-Length", -1))
            local_size = os.path.getsize(local_path)
            if remote_size >= 0 and local_size != remote_size:
                needs_download = True
        except Exception:
            pass  # can't verify right now -- trust the existing local file rather than break offline use

    if needs_download:
        r = requests.get(download_url, auth=(username, password), timeout=120)
        r.raise_for_status()
        # Write to a temp path and rename (atomic on the same filesystem)
        # rather than writing directly to local_path -- this is the other
        # half of the fix: even with the size-check above catching an
        # ALREADY-truncated file, writing straight to the final filename
        # would let a NEW interruption leave another truncated file right
        # back at that same path. With this, local_path only ever exists
        # in a fully-written state; an interrupted download leaves an
        # orphaned ".part" file instead of a broken "real" one.
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp_path, local_path)

    return local_path, stime


def fetch_gmi_swath_nrt(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 6.0,
    local_dir: str = os.path.join(NRT_CACHE_DIR, "GMI"),
    search_direction: str = "backward",
    progress_callback=None,
) -> Optional[MWSwath]:
    """Search PPS's NRT GMI feed for an overpass in the lookback window,
    download the closest match, and return an MWSwath. Requires PPS NRT
    registration (separate from Earthdata/GES DISC access).

    search_direction: "backward" (default) searches [target_time -
        lookback_hours, target_time]; "forward" searches [target_time,
        target_time + lookback_hours] instead, picking the EARLIEST
        match -- for MIMIC-TC-style crossfade morphing between a
        "before" pass and an "after" pass.

    Tries "1C/GMI" first -- both because it matches the consistent
    "/1C/<SENSOR>/" pattern confirmed against the live server for AMSR2,
    SSMIS, WSFM, and AMSR3, AND for a real scientific reason: 1C and 1CR
    are genuinely different products, not just different paths to the
    same data. 1CR specifically intercalibrates and CO-REGISTERS GMI's
    low-frequency channels (10-89 GHz) with its high-frequency channels
    (166-183 GHz) -- solving a footprint-size mismatch between those two
    groups. This project only ever uses 37 GHz and 89 GHz, both already
    within the SAME low-frequency group 1CR's co-registration was built
    to align against the high-frequency group -- meaning 1CR's specific
    benefit doesn't actually apply to anything this project uses, and
    plain 1C (GMI's standard common-calibrated baseline product) is the
    more directly appropriate choice, not just a fallback-of-convenience.
    1CR is kept as a fallback for resilience (in case 1C genuinely lacks
    data for some date range), not because it's expected to differ in any
    way that matters here.

    Channel extraction uses the SAME dynamic discovery mechanism as
    WSFM-NRT/AMSR3-NRT (_discover_wsfm_channels -- generic despite the
    name, see its docstring), NOT a fixed group="S1"/hardcoded-index
    assumption. This replaces an earlier version that assumed a fixed
    "S1" group and fixed Tc channel indices (5,6,7,8) matching the GES
    DISC archive convention -- confirmed NOT reliable for real NRT data:
    GMI-NRT and SSMIS-NRT (both hardcoded this way) were reported to
    consistently fail to actually retrieve data, while WSFM-NRT and
    AMSR3-NRT (both already using dynamic discovery) worked. Rather than
    guess at what the NRT ".RT-NC" repackaging's actual group/index
    layout is, this scans for it directly the same way that already
    works for the other two sensors.

    Uses tle_predict to narrow the PPS query to the specific hour(s) GMI's
    real orbit actually passed near (center_lat, center_lon), instead of
    scanning the whole window blindly. If TLE prediction confidently finds
    no overpass in this window, returns None immediately without even
    querying PPS. If TLE prediction itself fails (e.g. no network to
    Celestrak), falls back to the original unnarrowed whole-window query.
    """
    hour_filters = None
    try:
        import tle_predict

        if search_direction == "forward":
            tle_start, tle_end = target_time, target_time + timedelta(hours=lookback_hours)
        else:
            tle_start, tle_end = target_time - timedelta(hours=lookback_hours), target_time
        hits = tle_predict.predict_overpasses("GMI", center_lat, center_lon, tle_start, tle_end)
        hour_filters = tle_predict.hour_strings(hits) if hits else None
    except Exception:
        hour_filters = None  # TLE prediction unavailable -- fall back to broad query

    # NOTE: an EMPTY hits list (TLE ran successfully but predicts no
    # overpass) is treated the SAME as TLE being unavailable at all
    # (hour_filters=None -> fall back to a broader, date-based search),
    # not as a confident "definitely no data here, stop searching."
    # A previous version hard-returned None on empty hits -- reported
    # directly that GMI/SSMIS (which have real NORAD IDs, so their TLE
    # geometry math actually runs) consistently failed to retrieve data,
    # while AMSR3 (missing from NORAD_IDS entirely -- a real, separate
    # gap) "worked" purely because its lookup KeyErrors, gets caught by
    # the try/except above, and accidentally falls back to exactly this
    # same broad-search behavior. Since removing this hard-stop can only
    # ever WIDEN a search (never cause a WRONG match the way a matching-
    # tolerance change could), it's safe to apply here even without being
    # able to verify the TLE geometry math directly -- if that math has
    # any bug causing false "no overpass" negatives, this stops trusting
    # it with the same confidence as a real positive.

    # NOTE: previously retried against progressively older candidates
    # when the most recent one missed geographically. Removed per direct
    # feedback and confirmed by a real log: GMI's NRT files are ~5-minute
    # granules, so 5 "different" retry candidates were often just 5
    # slices of the SAME single orbital pass -- if that pass doesn't
    # cross near the target, every granule within it shares that same
    # fate, so the retry loop was burning attempts without any real
    # chance of finding something different, not "trying a few candidates"
    # in any meaningful sense. A single attempt per sensor, cut off
    # cleanly by lookback_hours, is both simpler and no less effective in
    # practice -- the diagnostic message below still explains exactly why
    # a miss happened, without the noise of repeated near-identical misses.
    import xarray as xr  # deferred until we actually have a file to open

    local_path = stime = None
    for candidate_subdir in ("1C/GMI", "1CR"):
        local_path, stime = _pps_nrt_find_and_download(
            candidate_subdir, target_time, lookback_hours, username, password, local_dir,
            hour_filters=hour_filters, search_direction=search_direction,
            progress_callback=progress_callback,
        )
        if local_path is not None:
            break
    if local_path is None:
        return None

    channels = _discover_wsfm_channels(local_path)  # generic despite the name -- see its docstring

    opened = {}

    def _get(label):
        group, idx = channels[label]
        if group not in opened:
            opened[group] = _open_group_robust(local_path, group)
        ds = opened[group]
        return ds["Tc"].values[..., idx], ds["Latitude"].values, ds["Longitude"].values

    v37, lat37, lon37 = _get("37V")
    h37, _, _ = _get("37H")
    v89, lat89, lon89 = _get("89V")
    h89, _, _ = _get("89H")

    lat37, lon37, v37, h37 = _crop_swath_to_box(lat37, lon37, center_lat, center_lon, box_deg, v37, h37)
    lat89, lon89, v89, h89 = _crop_swath_to_box(lat89, lon89, center_lat, center_lon, box_deg, v89, h89)
    if lat37.size == 0 or lat89.size == 0:
        # Real, confirmed failure mode: a file WAS found by its timestamp
        # falling in the search window and WAS successfully downloaded/
        # opened -- but its actual geographic coverage doesn't include
        # the requested box. Simply means no candidate within
        # lookback_hours had a relevant pass; caller (find_swath_that_hit_storm
        # / find_mw_pair_for_crossfade) handles trying other sensors and
        # eventually falling back to the archive.
        if progress_callback:
            progress_callback(
                f"MW search debug [GMI-NRT]: found and opened {os.path.basename(local_path)} "
                f"(observed {stime:%Y-%m-%d %H:%M} UTC) but its actual swath coverage doesn't "
                f"include the requested box (center {center_lat:.2f},{center_lon:.2f}, "
                f"+/-{box_deg}deg) -- the pass's own orbit track didn't pass near this "
                "location, even though its observation TIME fell within the search window."
            )
        return None

    v37, _ = sanitize_field(v37)
    h37, _ = sanitize_field(h37)
    v89, _ = sanitize_field(v89)
    h89, _ = sanitize_field(h89)

    return MWSwath(
        sensor="GMI-NRT",
        scene_time=stime,
        lat=lat89,
        lon=lon89,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        lat37=lat37,
        lon37=lon37,
        lat89=lat89,
        lon89=lon89,
        source_note=f"PPS-NRT/{os.path.basename(local_path)} (channels: {channels})",
    )


# ---------------------------------------------------------------------------
# SSMIS-NRT (PPS "jsimpson" /1C/SSMIS/, e.g. DMSP F17)
# ---------------------------------------------------------------------------
# Channel layout confirmed against GES DISC's own 1C-SSMIS product
# documentation (the same standardized swath layout used across the whole
# GPM-constellation 1C product family): S1=(19V,19H,22V), S2=(37V,37H),
# S3=(150H,183+/-1H,183+/-3H,183+/-7H), S4=(91V,91H). We need S2 for 37 GHz
# and S4 for "89 GHz" (SSMIS's nearest channel is actually 91 GHz -- this
# is the same substitution already handled by PCT_THETA[89], which is the
# coefficient documented for the 85-91 GHz sensor family broadly, not
# specifically 89.0 GHz). S2 and S4 are separate swaths and may have
# different native resolutions/geolocation (same situation as AMSR2's
# 36.5/89 GHz grids) -- handled the same way, via MWSwath's per-frequency
# lat37/lon37 vs lat89/lon89.
SSMIS_S2_CHANNEL_INDEX = {"37V": 0, "37H": 1}
SSMIS_S4_CHANNEL_INDEX = {"91V": 0, "91H": 1}


def fetch_ssmis_swath_nrt(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 6.0,
    local_dir: str = os.path.join(NRT_CACHE_DIR, "SSMIS"),
    # The body already calls progress_callback(...) in its coverage-debug
    # branch, but never took the parameter -- a latent NameError, found by
    # the static undefined-name scan rather than by running it. SSMIS left
    # the sensor rotation in 0.87, so that branch has not been exercised
    # since; it would have raised the moment it was.
    progress_callback=None,
) -> Optional[MWSwath]:
    """Search PPS's NRT SSMIS feed (confirmed live at /1C/SSMIS/) for an
    overpass, download it, and return an MWSwath. Requires PPS NRT
    registration. This is the real near-real-time SSMIS path that wasn't
    available when this project started (only the ~1-month-latency NCEI
    CDR archive existed as an automatable option then).

    /1C/SSMIS/ is a large FLAT folder (no date subdirectories) covering
    the whole ~2-week NRT retention window across 3 active satellites
    (F16/F17/F18) -- an unnarrowed wildcard/full-listing query there is
    what triggered the misleading "permissions" error, since scanning
    that much unstructured data server-side is expensive/rejected. This
    uses tle_predict.predict_ssmis_satellite() to figure out which
    satellite (if any) actually passed near the target and narrows the
    PPS query to just that satellite's specific hour(s) -- avoiding the
    expensive scan entirely rather than working around it after the fact.
    """
    hour_filters = None
    name_filter = None
    start = target_time - timedelta(hours=lookback_hours)
    try:
        import tle_predict

        sat_key, hits = tle_predict.predict_ssmis_satellite(center_lat, center_lon, start, target_time)
        if sat_key is not None:
            hour_filters = tle_predict.hour_strings(hits)
            name_filter = sat_key.split("-")[-1]  # "F16" / "F17" / "F18"
        # else: TLE ran successfully but predicts none of F16/F17/F18
        # passed near here -- do NOT hard-return on this. A previous
        # version did (`if sat_key is None: return None`), matching a
        # reported failure: SSMIS consistently didn't retrieve data,
        # while sensors whose TLE narrowing accidentally never runs (see
        # fetch_amsr3_swath_nrt's docstring) worked by falling back to a
        # broader search instead of trusting a negative prediction with
        # full confidence. Left as hour_filters=None/name_filter=None
        # here; handled below with an hour-sweep fallback SPECIFIC to
        # SSMIS (unlike GMI/WSFM/AMSR3, an unnarrowed query on SSMIS's
        # huge flat multi-satellite directory previously triggered an
        # actual server-side rejection, not just slowness -- see the
        # docstring above -- so this can't just fall back to the plain
        # full-listing approach the way the other three safely do).
    except Exception:
        hour_filters = None
        name_filter = None

    if hour_filters is None:
        # No usable TLE prediction (either it found nothing, or failed
        # outright) -- sweep every hour in the lookback window instead of
        # giving up. Each hour is still its own server-side-filtered
        # wildcard query (not a full unnarrowed listing), so this stays
        # within the same safe pattern that avoids SSMIS's documented
        # rejection risk, just with more (but still individually cheap)
        # queries than an ideal TLE-narrowed 1-2.
        hour_filters = []
        cursor = start
        while cursor <= target_time:
            hour_filters.append(cursor.strftime("%Y%m%d%H"))
            cursor += timedelta(hours=1)
        hour_filters.append(target_time.strftime("%Y%m%d%H"))

    local_path, stime = _pps_nrt_find_and_download(
        "1C/SSMIS", target_time, lookback_hours, username, password, local_dir,
        hour_filters=hour_filters, name_filter=name_filter,
        use_wildcard_text_query=True,
    )
    if local_path is None:
        return None

    import xarray as xr  # deferred until we actually have a file to open

    # Dynamic channel discovery (same mechanism as WSFM-NRT/AMSR3-NRT --
    # generic despite the function's name, see its docstring), NOT the
    # fixed group="S2"/"S4" + hardcoded-index assumption this replaced.
    # That assumption came with its own caveat acknowledging it might not
    # match the NRT file structure -- confirmed unreliable in practice:
    # SSMIS-NRT (hardcoded) was reported to consistently fail, while
    # WSFM-NRT/AMSR3-NRT (already dynamic) worked.
    #
    # target_89_ghz=91.665 (SSMIS's actual real channel frequency) with a
    # tight, SSMIS-SPECIFIC tolerance -- NOT a shared/global widening.
    # An earlier version widened the shared default tolerance instead,
    # which regressed WSFM/AMSR3 (previously working, then broke) by
    # risking them matching some OTHER nearby channel first under the
    # wider window. Scoping the fix to just this call site leaves
    # everyone else's (already-correct) default behavior untouched.
    channels = _discover_wsfm_channels(local_path, target_89_ghz=91.665, tolerance_89=1.0)

    opened = {}

    def _get(label):
        group, idx = channels[label]
        if group not in opened:
            opened[group] = _open_group_robust(local_path, group)
        ds = opened[group]
        return ds["Tc"].values[..., idx], ds["Latitude"].values, ds["Longitude"].values

    v37, lat37, lon37 = _get("37V")
    h37, _, _ = _get("37H")
    v89, lat89, lon89 = _get("89V")
    h89, _, _ = _get("89H")

    lat37, lon37, v37, h37 = _crop_swath_to_box(lat37, lon37, center_lat, center_lon, box_deg, v37, h37)
    lat89, lon89, v89, h89 = _crop_swath_to_box(lat89, lon89, center_lat, center_lon, box_deg, v89, h89)
    if lat37.size == 0 or lat89.size == 0:
        # Real, confirmed silent-failure path: a file WAS found by its
        # timestamp falling in the search window and WAS successfully
        # downloaded/opened -- but its actual geographic coverage doesn't
        # include the requested box. The search is fundamentally TIME-based
        # (most recent file whose observation time is in range), with no
        # verification the satellite's orbit actually passed near the target
        # location -- a satellite covers the whole globe over a day, so
        # "most recent by time" and "most recent that's actually nearby" are
        # NOT the same thing unless TLE-based narrowing genuinely worked.
        if progress_callback:
            progress_callback(
                f"MW search debug [SSMIS-NRT]: found and opened {os.path.basename(local_path)} "
                f"(observed {stime:%Y-%m-%d %H:%M} UTC) but its actual swath coverage doesn't "
                f"include the requested box (center {center_lat:.2f},{center_lon:.2f}, "
                f"+/-{box_deg}deg) -- the pass's own orbit track didn't pass near this "
                "location, even though its observation TIME fell within the search window."
            )
        return None

    v37, _ = sanitize_field(v37)
    h37, _ = sanitize_field(h37)
    v89, _ = sanitize_field(v89)
    h89, _ = sanitize_field(h89)

    return MWSwath(
        sensor="SSMIS-NRT",
        scene_time=stime,
        lat=lat89,
        lon=lon89,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        lat37=lat37,
        lon37=lon37,
        lat89=lat89,
        lon89=lon89,
        source_note=f"{os.path.basename(local_path)} (channels: {channels})",
    )


# ---------------------------------------------------------------------------
# WSFM-NRT (PPS "jsimpson" /1C/WSFM/, WSF-M satellite / MWI instrument)
# ---------------------------------------------------------------------------
# MWI is a genuinely new sensor (launched 2024) with 17 channels including
# several fully-polarimetric ones (V, H, plus 3rd/4th Stokes at 10.85,
# 18.85, 36.75 GHz) -- unlike GMI/SSMIS, there is no publicly documented,
# confirmed channel-index ordering to trust blindly.
#
# CONFIRMED against a real downloaded file: Tc's LongName attribute for
# group "S1" reads "Intercalibrated Tb for channels 1) 10.85 GHz V-Pol and
# 2) 10.85 GHz H-Pol" -- meaning S1 holds ONLY the 10.85 GHz pair, not all
# 17 channels. So MWI's channels are split across multiple swath groups,
# the same pattern GMI/SSMIS already use (just with more groups, given
# more distinct channel types). This scans groups S1 through S8 looking
# for whichever ones' Tc LongName mentions ~36.75 GHz or ~89 GHz, and
# parses the "N) freq GHz Pol" pattern in that string to get the exact
# channel index WITHIN that specific group -- rather than assuming
# everything lives in one group at a hardcoded index. If the needed
# channels aren't found this way, raises a detailed diagnostic dump
# instead of guessing (a wrong index here would silently mislabel data).
WSFM_MAX_GROUPS_TO_SCAN = 8


def _parse_channel_order_from_longname(long_name: str) -> dict:
    """Parse strings like 'channels 1) 10.85 GHz V-Pol and 2) 10.85 GHz
    H-Pol' into {0-based channel index: (freq_ghz, 'V'/'H')}."""
    entries = re.findall(r"(\d+)\)\s*([\d.]+)\s*GHz\s*([VH])-?Pol", long_name, flags=re.IGNORECASE)
    return {int(idx) - 1: (float(freq), pol.upper()) for idx, freq, pol in entries}


def _discover_wsfm_channels(local_path: str, target_89_ghz: float = 89.0, tolerance_89: float = 2.0) -> dict:
    """Scan groups S1..S8 in the downloaded file for Tc variables whose
    LongName mentions a frequency near 36.75 GHz or target_89_ghz, and
    parse out the exact (group, channel_index) for each of
    37V/37H/89V/89H.

    target_89_ghz/tolerance_89 default to (89.0, 2.0) -- the ORIGINAL
    values, deliberately restored here after a real regression: a
    previous version of this function widened the shared tolerance to
    4.0 (to catch SSMIS's real ~91.665 GHz channel) as a single global
    change affecting every sensor that calls this function -- but that
    also risked WSFM/AMSR3 (already confirmed working with the tighter
    2.0 tolerance) newly matching a DIFFERENT, unrelated nearby channel
    first (this function keeps only the first match per label), silently
    replacing a previously-correct channel assignment with a wrong one.
    Reported directly: NRT data stopped working entirely for AMSR3/WSFM
    after that change, having worked before it.

    The fix is to keep everyone's default behavior EXACTLY as it was
    when WSFM/AMSR3 were last confirmed working (this function's
    defaults), and let SSMIS's own call site pass its OWN specific
    target frequency (91.665 GHz) with a narrow tolerance around THAT
    exact value instead -- precise to what SSMIS actually needs, rather
    than a blanket widening that risked every other sensor.

    Returns {"37V": (group_name, idx), "37H": (...), "89V": (...), "89H": (...)}.
    Raises RuntimeError with full diagnostic info if any of the four
    aren't found -- see module comment above for why this doesn't fall
    back to a guessed index.
    """
    import xarray as xr

    found = {}
    groups_seen = {}
    open_errors = {}  # group -> exception message, for diagnosing a total failure

    for i in range(1, WSFM_MAX_GROUPS_TO_SCAN + 1):
        group = f"S{i}"
        try:
            ds = _open_group_robust(local_path, group)
        except Exception as e:
            open_errors[group] = str(e)
            continue
        if "Tc" not in ds.variables:
            ds.close()
            continue

        long_name = ds["Tc"].attrs.get("LongName") or ds["Tc"].attrs.get("long_name") or ""
        groups_seen[group] = long_name
        parsed = _parse_channel_order_from_longname(long_name)

        for idx, (freq, pol) in parsed.items():
            label = None
            if abs(freq - 36.75) < 1.0 or abs(freq - 36.5) < 1.0:
                label = f"37{pol}"
            elif abs(freq - target_89_ghz) < tolerance_89:
                label = f"89{pol}"
            if label and label not in found:
                found[label] = (group, idx)
        ds.close()

    if not groups_seen:
        # The S1..S8 assumption found NOTHING at all (not even a group
        # that opened successfully) -- confirmed directly against a real
        # AMSR3 file, which apparently doesn't follow that naming
        # convention at all (WSFM's own real file DOES follow it, per a
        # direct real-world confirmation that WSFM-NRT worked correctly;
        # this fallback specifically doesn't touch that already-working
        # path). Rather than keep guessing at fixed group names, fall
        # back to discovering the file's ACTUAL real structure directly
        # via h5py (which can enumerate every dataset in the file
        # regardless of what it's named or how deep it's nested), then
        # apply the exact same frequency-matching logic to whatever
        # structure is actually found.
        found, groups_seen = _discover_channels_via_h5py_fallback(local_path, target_89_ghz, tolerance_89)
        if not groups_seen and open_errors:
            # Even the h5py fallback found nothing -- surface exactly
            # WHY each S1..S8 attempt failed (previously silently
            # swallowed), since a real run confirmed the actual data
            # DOES exist in the file (LongName/Tc metadata directly
            # verified via raw string-extraction) even when this scan
            # reports finding nothing -- meaning the group-OPENING step
            # itself is failing, not "these groups genuinely don't
            # exist." Kept as part of groups_seen (not raised directly
            # here) so it surfaces in the eventual RuntimeError below,
            # in the exact same place a caller already looks for
            # diagnostic info.
            groups_seen["<S1..S8 open errors>"] = str(open_errors)

    needed = {"37V", "37H", "89V", "89H"}
    missing = needed - set(found.keys())
    if missing:
        raise RuntimeError(
            f"Could not identify WSF-M/MWI channels {missing} in {local_path}. "
            f"Groups scanned and their Tc LongName: {groups_seen}. Channels found "
            f"so far: {found}. If a needed frequency genuinely isn't in any group's "
            "LongName text, inspect the file directly (netCDF4/ncdump) and hardcode "
            "the (group, index) pair in _discover_wsfm_channels() -- don't guess, "
            "a wrong index here would silently mislabel data as 37/89 GHz."
        )
    return found


def _open_group_robust(local_path: str, group: str):
    """Open a named group within a downloaded NRT file. Tries plain
    xarray auto-detection FIRST -- exactly what GMI-NRT/WSFM-NRT used
    with zero problems before this function existed at all (confirmed
    directly: comparing this file against the last version those two
    sensors were confirmed fully working in shows the ONLY thing that
    changed for their code path was this function's engine ordering) --
    and only falls back to explicitly-named engines ('h5netcdf', then
    'netcdf4') if that plain attempt fails.

    An EARLIER version of this function tried the explicit engines
    FIRST, auto-detection last, reasoning that explicit engines might be
    more robust for a real AMSR3 file that was failing. That ordering
    change applied to every sensor's group-opening, not just AMSR3's --
    including GMI-NRT and WSFM-NRT, which had no problem to fix in the
    first place. Reported directly: NRT still didn't work after that
    change, for sensors that were previously confirmed working. Since
    forcing a specific engine ahead of whatever auto-detection would
    have picked is exactly the kind of change that can silently behave
    differently for files that were already opening fine, this restores
    auto-detection as the first attempt -- the same lesson as an earlier
    regression in this project (a shared tolerance widened for one
    sensor's benefit, which silently affected two others that didn't
    need it): a fix for one confirmed-broken case shouldn't change the
    code path for cases that were already confirmed working.

    Explicit engines remain as a fallback (not removed) for genuinely
    broken cases like AMSR3's, where auto-detection failing outright was
    the actual confirmed symptom. Raises the LAST exception encountered
    if every attempt fails, so a caller sees a real underlying reason
    instead of just "nothing found."
    """
    import xarray as xr

    last_exc = None
    for engine in (None, "h5netcdf", "netcdf4"):
        try:
            if engine is not None:
                return xr.open_dataset(local_path, group=group, engine=engine)
            return xr.open_dataset(local_path, group=group)
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


def _discover_channels_via_h5py_fallback(local_path: str, target_89_ghz: float, tolerance_89: float):
    """General-purpose fallback for _discover_wsfm_channels when the
    S1..S8 group-name assumption finds NOTHING at all -- confirmed
    directly against a real AMSR3 file. Uses h5py's visititems() to walk
    the file's ACTUAL structure (every dataset, at whatever path/depth it
    really sits, under whatever real names the file uses), rather than
    guessing a fixed naming convention -- then applies the same
    LongName-frequency-matching logic used elsewhere in this file.

    IMPORTANT HONEST CAVEAT: this was written reasoning through the
    failure mode (a real error confirmed zero "S1".."S8" groups existed),
    not verified against a real AMSR3 file directly -- no such file was
    available to inspect here. It looks for datasets literally named
    "Tc", "Tb", or "brightness_temperature" (case-insensitive) with a
    LongName/long_name attribute mentioning a frequency, which is the
    standard GPM-constellation 1C convention this project has confirmed
    elsewhere (GMI, SSMIS, WSFM) -- if AMSR3's real file uses some
    entirely different attribute/variable naming this fallback still
    won't find it, and the resulting error message will show exactly
    what h5py DID find, to make manually hardcoding the right (group,
    index) pair (per the RuntimeError's own advice) as easy as possible.
    """
    import h5py

    candidate_names = {"tc", "tb", "brightness_temperature", "brightnesstemperature"}
    found = {}
    groups_seen = {}

    try:
        with h5py.File(local_path, "r") as f:
            def _visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                var_name = name.split("/")[-1]
                if var_name.lower() not in candidate_names:
                    return
                group_path = "/".join(name.split("/")[:-1]) or "/"
                long_name = ""
                for attr_key in ("LongName", "long_name"):
                    val = obj.attrs.get(attr_key)
                    if val is not None:
                        long_name = val.decode() if isinstance(val, bytes) else str(val)
                        break
                groups_seen[group_path] = long_name
                parsed = _parse_channel_order_from_longname(long_name)
                for idx, (freq, pol) in parsed.items():
                    label = None
                    if abs(freq - 36.75) < 1.0 or abs(freq - 36.5) < 1.0:
                        label = f"37{pol}"
                    elif abs(freq - target_89_ghz) < tolerance_89:
                        label = f"89{pol}"
                    if label and label not in found:
                        found[label] = (group_path, idx)

            f.visititems(_visit)
    except Exception as e:
        groups_seen[f"<h5py fallback itself failed: {e}>"] = ""

    return found, groups_seen


def fetch_wsfm_swath_nrt(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 6.0,
    local_dir: str = os.path.join(NRT_CACHE_DIR, "WSFM"),
    search_direction: str = "backward",
    progress_callback=None,
) -> Optional[MWSwath]:
    """Search PPS's NRT WSF-M/MWI feed (confirmed live at /1C/WSFM/) for
    an overpass and download it. Channel identification scans multiple
    swath groups and parses their LongName metadata rather than assuming
    a fixed group/index (see _discover_wsfm_channels' docstring for why).

    search_direction: "backward" (default) or "forward" -- see
        fetch_gmi_swath_nrt's docstring for the full explanation
        (MIMIC-TC-style crossfade morphing between a "before" and
        "after" pass).

    Like SSMIS-NRT, /1C/WSFM/ is a large flat folder -- narrows the PPS
    query via tle_predict the same way, to avoid an expensive full scan."""
    hour_filters = None
    try:
        import tle_predict

        if search_direction == "forward":
            tle_start, tle_end = target_time, target_time + timedelta(hours=lookback_hours)
        else:
            tle_start, tle_end = target_time - timedelta(hours=lookback_hours), target_time
        hits = tle_predict.predict_overpasses("WSFM", center_lat, center_lon, tle_start, tle_end)
        hour_filters = tle_predict.hour_strings(hits) if hits else None
    except Exception:
        hour_filters = None

    # See fetch_gmi_swath_nrt's comment for why empty hits fall back to a
    # broad search (hour_filters=None) instead of hard-returning None.

    # See fetch_gmi_swath_nrt's comment for why this is a single attempt,
    # not a retry loop -- WSFM's files span longer per pass than GMI's
    # 5-minute granules, but the same principle applies: if the one
    # geographically-relevant window that matters here misses, cycling
    # through more files isn't a reliable way to fix that.
    import xarray as xr  # deferred until we actually have a file to open

    local_path, stime = _pps_nrt_find_and_download(
        "1C/WSFM", target_time, lookback_hours, username, password, local_dir,
        hour_filters=hour_filters, search_direction=search_direction,
        progress_callback=progress_callback,
    )
    if local_path is None:
        return None

    channels = _discover_wsfm_channels(local_path)

    # The four channels may live in different groups (different native
    # resolutions), same situation as AMSR2/SSMIS -- open each distinct
    # group once and pull its own Latitude/Longitude alongside its Tc.
    opened = {}

    def _get(label):
        group, idx = channels[label]
        if group not in opened:
            opened[group] = _open_group_robust(local_path, group)
        ds = opened[group]
        return ds["Tc"].values[..., idx], ds["Latitude"].values, ds["Longitude"].values

    v37, lat37, lon37 = _get("37V")
    h37, _, _ = _get("37H")
    v89, lat89, lon89 = _get("89V")
    h89, _, _ = _get("89H")

    lat37, lon37, v37, h37 = _crop_swath_to_box(lat37, lon37, center_lat, center_lon, box_deg, v37, h37)
    lat89, lon89, v89, h89 = _crop_swath_to_box(lat89, lon89, center_lat, center_lon, box_deg, v89, h89)
    if lat37.size == 0 or lat89.size == 0:
        # Real, confirmed failure mode: a file WAS found by its timestamp
        # falling in the search window and WAS successfully downloaded/
        # opened -- but its actual geographic coverage doesn't include
        # the requested box.
        if progress_callback:
            progress_callback(
                f"MW search debug [WSFM-NRT]: found and opened {os.path.basename(local_path)} "
                f"(observed {stime:%Y-%m-%d %H:%M} UTC) but its actual swath coverage doesn't "
                f"include the requested box (center {center_lat:.2f},{center_lon:.2f}, "
                f"+/-{box_deg}deg) -- the pass's own orbit track didn't pass near this "
                "location, even though its observation TIME fell within the search window."
            )
        return None

    v37, _ = sanitize_field(v37)
    h37, _ = sanitize_field(h37)
    v89, _ = sanitize_field(v89)
    h89, _ = sanitize_field(h89)

    return MWSwath(
        sensor="WSFM-NRT",
        scene_time=stime,
        lat=lat89,
        lon=lon89,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        lat37=lat37,
        lon37=lon37,
        lat89=lat89,
        lon89=lon89,
        source_note=f"{os.path.basename(local_path)} (channels: {channels})",
    )


def fetch_amsr3_swath_nrt(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 6.0,
    local_dir: str = os.path.join(NRT_CACHE_DIR, "AMSR3"),
    search_direction: str = "backward",
    progress_callback=None,
) -> Optional[MWSwath]:
    """Search PPS's NRT AMSR3 feed (/1C/AMSR3/, per direct confirmation
    the directory now exists on the live server -- AMSR3 is a newer
    instrument than AMSR2, presumably the GCOM-W follow-on/successor).

    IMPORTANT CAVEAT: unlike GMI/SSMIS/WSFM (all confirmed against real
    downloaded files at some point in this project), AMSR3's actual 1C
    file structure has NOT been verified here at all -- no real AMSR3
    file has been available to inspect. Rather than guess at a fixed
    group/channel-index layout (which would risk silently mislabeling
    data as 37/89 GHz if wrong), this reuses the SAME channel-discovery
    approach built for WSFM (_discover_wsfm_channels -- genuinely
    generic despite its name: it just scans groups for Tc variables
    with LongName metadata mentioning a frequency near 36.5-36.75 GHz or
    89 GHz, which is the standard GPM-constellation 1C product
    convention, not anything WSFM-specific). If AMSR3's real file
    doesn't follow that same group/LongName convention, this will raise
    a clear diagnostic error (listing every group and LongName it did
    find) rather than silently returning wrong data -- see that
    function's docstring for why guessing isn't an acceptable fallback
    here.

    No confirmed NORAD ID for AMSR3's satellite platform either -- TLE
    prediction will simply fail closed (falls back to the broader
    unnarrowed query, the same graceful degradation used whenever TLE
    prediction fails for any sensor) until one is added to tle_predict.py.
    """
    hour_filters = None
    try:
        import tle_predict

        if search_direction == "forward":
            tle_start, tle_end = target_time, target_time + timedelta(hours=lookback_hours)
        else:
            tle_start, tle_end = target_time - timedelta(hours=lookback_hours), target_time
        hits = tle_predict.predict_overpasses("AMSR3", center_lat, center_lon, tle_start, tle_end)
        hour_filters = tle_predict.hour_strings(hits) if hits else None
    except Exception:
        hour_filters = None

    # See fetch_gmi_swath_nrt's comment for why empty hits fall back to a
    # broad search (hour_filters=None) instead of hard-returning None.
    # For AMSR3 specifically, this except branch is what actually always
    # fires right now (NORAD_IDS has no "AMSR3" entry yet -- see
    # tle_predict.py -- so predict_overpasses always raises a KeyError
    # here), which is WHY this sensor already "works": it's already
    # falling back to exactly this broad-search behavior, just via an
    # accidental path rather than a deliberate one. Adding a real NORAD
    # ID for AMSR3's platform (GOSAT-GW) would let it benefit from actual
    # TLE narrowing instead, but no confirmed ID was available to add
    # here without risking silently tracking the wrong satellite.

    # See fetch_gmi_swath_nrt's comment for why this is a single attempt,
    # not a retry loop.
    import xarray as xr  # deferred until we actually have a file to open

    local_path, stime = _pps_nrt_find_and_download(
        "1C/AMSR3", target_time, lookback_hours, username, password, local_dir,
        hour_filters=hour_filters, search_direction=search_direction,
        progress_callback=progress_callback,
    )
    if local_path is None:
        return None

    channels = _discover_wsfm_channels(local_path)  # generic despite the name -- see docstring above

    opened = {}

    def _get(label):
        group, idx = channels[label]
        if group not in opened:
            opened[group] = _open_group_robust(local_path, group)
        ds = opened[group]
        return ds["Tc"].values[..., idx], ds["Latitude"].values, ds["Longitude"].values

    v37, lat37, lon37 = _get("37V")
    h37, _, _ = _get("37H")
    v89, lat89, lon89 = _get("89V")
    h89, _, _ = _get("89H")

    lat37, lon37, v37, h37 = _crop_swath_to_box(lat37, lon37, center_lat, center_lon, box_deg, v37, h37)
    lat89, lon89, v89, h89 = _crop_swath_to_box(lat89, lon89, center_lat, center_lon, box_deg, v89, h89)
    if lat37.size == 0 or lat89.size == 0:
        # Real, confirmed failure mode: a file WAS found by its timestamp
        # falling in the search window and WAS successfully downloaded/
        # opened -- but its actual geographic coverage doesn't include
        # the requested box.
        if progress_callback:
            progress_callback(
                f"MW search debug [AMSR3-NRT]: found and opened {os.path.basename(local_path)} "
                f"(observed {stime:%Y-%m-%d %H:%M} UTC) but its actual swath coverage doesn't "
                f"include the requested box (center {center_lat:.2f},{center_lon:.2f}, "
                f"+/-{box_deg}deg) -- the pass's own orbit track didn't pass near this "
                "location, even though its observation TIME fell within the search window."
            )
        return None

    v37, _ = sanitize_field(v37)
    h37, _ = sanitize_field(h37)
    v89, _ = sanitize_field(v89)
    h89, _ = sanitize_field(h89)

    return MWSwath(
        sensor="AMSR3-NRT",
        scene_time=stime,
        lat=lat89,
        lon=lon89,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        lat37=lat37,
        lon37=lon37,
        lat89=lat89,
        lon89=lon89,
        source_note=f"{os.path.basename(local_path)} (channels: {channels})",
    )


# ---------------------------------------------------------------------------
# GMI (NASA Earthdata via earthaccess)
# ---------------------------------------------------------------------------
GMI_SHORT_NAME_CANDIDATES = ("GPM_1CGPMGMI", "GPM_1CGMI")
# Swath S1 channel order per the 1C common calibrated Tb product spec:
# 10V, 10H, 19V, 19H, 23V, 37V, 37H, 89V, 89H
GMI_S1_CHANNEL_INDEX = {"37V": 5, "37H": 6, "89V": 7, "89H": 8}


def _earthdata_login(username: str, password: str):
    try:
        import earthaccess
    except ImportError as e:
        raise RuntimeError(
            "The 'earthaccess' package isn't installed in this Python "
            "environment. Run: pip install -r requirements.txt "
            "(and make sure you're running main.py from the same "
            "venv/interpreter you installed it into)."
        ) from e

    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password
    auth = earthaccess.login(strategy="environment")
    if not getattr(auth, "authenticated", False):
        raise RuntimeError(
            "Earthdata login failed. Check the username/password saved in the "
            "'MW Data Credentials' tab."
        )
    return auth


def _resolve_gmi_short_names(progress_callback=None) -> list:
    """Dynamically discover the actual CMR-registered short_name(s) for
    the GPM GMI 1C common-calibrated brightness-temperature product via a
    keyword search, rather than only trusting hardcoded guesses -- zero
    results across every hardcoded candidate and every time window is a
    strong signal the short_name itself doesn't match what's actually
    registered, not that there's no coverage. Returns a list (possibly
    empty on failure -- caller still falls back to the static
    GMI_SHORT_NAME_CANDIDATES list either way, so this is additive, not a
    hard dependency)."""
    import earthaccess

    try:
        collections = earthaccess.search_datasets(
            keyword="GPM GMI Common Calibrated Brightness Temperature",
            provider="GES_DISC",
        )
    except Exception as e:
        if progress_callback:
            progress_callback(f"  (GMI collection keyword search failed: {e})")
        return []

    resolved = []
    for c in collections:
        short_name = None
        try:
            short_name = c["umm"]["ShortName"]
        except Exception:
            try:
                summary = c.summary()
                short_name = summary.get("short-name") or summary.get("short_name")
            except Exception:
                short_name = None
        if short_name and "1C" in short_name.upper() and "GMI" in short_name.upper():
            if short_name not in resolved:
                resolved.append(short_name)

    if progress_callback:
        progress_callback(
            f"  Resolved GMI short_name candidate(s) via CMR keyword search: {resolved or '(none found)'}"
        )
    return resolved


def fetch_gmi_swath(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 6.0,
    local_dir: str = os.path.join(CACHE_DIR, "gmi"),
    progress_callback=None,
) -> Optional[MWSwath]:
    """Search the lookback_hours before target_time (NOT after -- this is
    a "most recent available imagery" search, not a nearest-in-time
    search) for a GMI overpass covering (center_lat, center_lon), download
    the most recent match, and return an MWSwath cropped to a box_deg-wide
    box around the center. Returns None if no overpass is found (which,
    per the module docstring, can mean either no coverage OR that the
    overpass hasn't been processed into the GES DISC archive yet)."""
    try:
        import earthaccess
    except ImportError as e:
        raise RuntimeError(
            "The 'earthaccess' package isn't installed in this Python "
            "environment. Run: pip install -r requirements.txt"
        ) from e
    import xarray as xr

    _earthdata_login(username, password)

    start = target_time - timedelta(hours=lookback_hours)
    end = target_time
    bbox = (center_lon - box_deg, center_lat - box_deg, center_lon + box_deg, center_lat + box_deg)
    # Pass explicit ISO-8601 strings, not raw datetime objects -- every
    # working earthaccess example in the docs uses strings for `temporal`,
    # and zero granules across every short_name AND every window over a
    # 12x12 deg tropical box (where GPM revisits multiple times/day) is
    # far more consistent with a silently-mis-serialized temporal filter
    # than a genuine coverage gap.
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    dynamic_names = _resolve_gmi_short_names(progress_callback)
    # Dynamically-resolved names first (more likely correct), then the
    # static guesses as a fallback, de-duplicated while preserving order.
    short_name_candidates = list(dict.fromkeys(dynamic_names + list(GMI_SHORT_NAME_CANDIDATES)))

    results = []
    short_name_used = None
    for short_name in short_name_candidates:
        r = earthaccess.search_data(short_name=short_name, temporal=(start_str, end_str), bounding_box=bbox)
        if progress_callback:
            progress_callback(
                f"  GMI short_name={short_name}: {len(r)} granule(s) for "
                f"{start_str}-{end_str}, bbox={[round(x, 2) for x in bbox]}"
            )
        if r:
            results = r
            short_name_used = short_name
            break
    if not results:
        return None

    # Prefer the granule whose temporal coverage is closest to target_time.
    def _granule_time(g):
        try:
            t = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            return timeutil.as_utc(datetime.fromisoformat(t.replace("Z", "+00:00")))
        except Exception:
            return target_time  # fall back to "no preference" if metadata shape is unexpected

    results.sort(key=lambda g: abs((_granule_time(g) - target_time).total_seconds()))
    # With the one-sided (past-only) window above, "closest to target_time"
    # is equivalent to "most recent" -- there's nothing after target_time
    # to accidentally prefer.
    best = results[0]
    actual_scene_time = _granule_time(best)

    os.makedirs(local_dir, exist_ok=True)
    try:
        files = earthaccess.download([best], local_dir)
    except Exception as e:
        if "EULA" in str(e):
            raise RuntimeError(
                "NASA Earthdata rejected the download because this account "
                "hasn't accepted the GES DISC End User License Agreement yet "
                "(a one-time, account-level step -- not a bug here). Fix: log "
                "into https://urs.earthdata.nasa.gov -> Applications -> "
                "Authorized Apps -> 'APPROVE MORE APPLICATIONS', search for "
                "'NASA GESDISC DATA ARCHIVE', authorize it, and accept the "
                f"EULA it presents. Then retry. Original error: {e}"
            ) from e
        raise
    if not files:
        return None

    ds = xr.open_dataset(files[0], group="S1")
    tc = ds["Tc"].values  # shape (scan, pixel, channel)
    lat = ds["Latitude"].values
    lon = ds["Longitude"].values

    v37 = tc[..., GMI_S1_CHANNEL_INDEX["37V"]]
    h37 = tc[..., GMI_S1_CHANNEL_INDEX["37H"]]
    v89 = tc[..., GMI_S1_CHANNEL_INDEX["89V"]]
    h89 = tc[..., GMI_S1_CHANNEL_INDEX["89H"]]

    lat, lon, v37, h37, v89, h89 = _crop_swath_to_box(
        lat, lon, center_lat, center_lon, box_deg, v37, h37, v89, h89
    )
    if lat.size == 0:
        return None

    v37, _ = sanitize_field(v37)
    h37, _ = sanitize_field(h37)
    v89, _ = sanitize_field(v89)
    h89, _ = sanitize_field(h89)

    return MWSwath(
        sensor="GMI",
        scene_time=actual_scene_time,
        lat=lat,
        lon=lon,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        source_note=f"{short_name_used}/{os.path.basename(files[0])}",
    )


# ---------------------------------------------------------------------------
# AMSR2 (JAXA G-Portal via the `gportal` package)
# ---------------------------------------------------------------------------
def list_amsr2_datasets():
    """Returns the G-Portal dataset tree under GCOM-W/AMSR2, so you can
    find the correct L1B brightness-temperature dataset_id by hand if
    fetch_amsr2_swath's default lookup doesn't match your account's view
    of the catalog (dataset IDs/paths can change)."""
    import gportal

    datasets = gportal.datasets()
    return datasets.get("GCOM-W/AMSR2", datasets)


def fetch_amsr2_swath(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 6.0,
    local_dir: str = os.path.join(CACHE_DIR, "amsr2"),
) -> Optional[MWSwath]:
    """Search the lookback_hours before target_time (NOT after) for an
    AMSR2 overpass, download the most recent match's L1B brightness-
    temperature granule, and return an MWSwath cropped to a box_deg-wide
    box. Returns None if no overpass is found.

    NOTE: the dataset_id lookup below tries a couple of plausible paths in
    the G-Portal catalog tree for "L1B brightness temperature" -- if none
    match, call list_amsr2_datasets() to see the real tree for your
    account and hardcode the right dataset_id.
    """
    try:
        import gportal
    except ImportError as e:
        raise RuntimeError(
            "The 'gportal' package isn't installed in this Python "
            "environment. Run: pip install -r requirements.txt"
        ) from e
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError(
            "The 'h5py' package isn't installed in this Python environment. "
            "Run: pip install -r requirements.txt"
        ) from e

    gportal.username = username
    gportal.password = password

    dataset_id = _find_amsr2_l1b_dataset_id()
    if dataset_id is None:
        tree_summary = _describe_dataset_tree(list_amsr2_datasets())
        raise RuntimeError(
            "Could not automatically locate the AMSR2 L1B brightness-temperature "
            "dataset in the G-Portal catalog. Here's the top of the tree this "
            "code actually saw (call mw_ingest.list_amsr2_datasets() for the "
            "full tree, and set MW_INGEST_AMSR2_DATASET_ID or hardcode the ID "
            f"once you find it):\n{tree_summary}"
        )

    start = target_time - timedelta(hours=lookback_hours)
    end = target_time
    bbox = [center_lon - box_deg, center_lat - box_deg, center_lon + box_deg, center_lat + box_deg]

    res = gportal.search(dataset_ids=[dataset_id], start_time=start.isoformat(), end_time=end.isoformat(), bbox=bbox)
    products = res.products()
    if not products:
        return None

    # Prefer the most recent product within the window (gportal's product
    # objects expose observation start time; sort defensively in case
    # search() doesn't already return them in time order).
    def _product_time(p):
        for attr in ("start_time", "startTime", "observation_start"):
            t = getattr(p, attr, None)
            if t is None:
                continue
            if isinstance(t, datetime):
                return timeutil.as_utc(t)
            if isinstance(t, str):
                try:
                    return timeutil.as_utc(datetime.fromisoformat(t.replace("Z", "+00:00")))
                except ValueError:
                    continue
        return None

    # IMPORTANT: don't just trust gportal.search()'s own start_time/end_time
    # filtering -- enforce the window client-side too. A real run surfaced a
    # product ~25 hours outside the requested 12h lookback window, which
    # this would have silently accepted before. GMI's fetch already does
    # this kind of explicit re-check after parsing filenames; AMSR2's didn't.
    dated_products = [(p, _product_time(p)) for p in products]
    in_window = [(p, t) for p, t in dated_products if t is not None and start <= t <= end]
    if not in_window:
        return None
    in_window.sort(key=lambda pt: pt[1], reverse=True)
    products = [p for p, _t in in_window]

    os.makedirs(local_dir, exist_ok=True)
    gportal.download(products[:1], local_dir=local_dir)

    downloaded = sorted(
        glob.glob(os.path.join(local_dir, "*.h5")), key=os.path.getmtime, reverse=True
    )
    if not downloaded:
        return None
    fpath = downloaded[0]

    # Parse the actual observation time from the filename itself (format
    # confirmed against a real file: GW1AM2_YYYYMMDDHHMI_...) rather than
    # trusting target_time or gportal's product-object attributes, which
    # may not always be present/parseable.
    actual_scene_time = target_time
    _m = re.search(r"_(\d{12})_", os.path.basename(fpath))
    if _m:
        try:
            actual_scene_time = timeutil.as_utc(datetime.strptime(_m.group(1), "%Y%m%d%H%M"))
        except ValueError:
            pass

    with h5py.File(fpath, "r") as f:
        keys = _list_h5_datasets(f)

        # Confirmed against a real GW1AM2_*_L1SGBTBR_*.h5 file: there is no
        # plain "Latitude"/"Longitude" dataset. This is the RESAMPLED L1B
        # product (the trailing "R" in L1SGBTBR), so every channel --
        # 6.9GHz through 89GHz -- is resampled onto a common grid, and the
        # only geolocation provided is for the 89GHz sub-swaths ("89A" and
        # "89B", AMSR2's two interleaved 89GHz feedhorn samples). We use
        # the 89A geolocation for everything and the 89A channel (not 89B)
        # for the 89GHz brightness temperatures, which is the standard
        # choice (89B is the secondary/offset sample).
        #
        # Also: 37 GHz on AMSR2 is actually the 36.5 GHz channel (nearest
        # AMSR2 channel to the nominal "37 GHz" used by GMI/SSMIS/TMI).
        lat_key = _find_key(
            keys, r"latitude.*89a", flags=re.IGNORECASE,
            exact_first=("Latitude of Observation Point for 89A",),
        )
        lon_key = _find_key(
            keys, r"longitude.*89a", flags=re.IGNORECASE,
            exact_first=("Longitude of Observation Point for 89A",),
        )
        v37_key = _find_key(
            keys, r"36\.?5\s*ghz.*,\s*v\)", flags=re.IGNORECASE,
            exact_first=("Brightness Temperature (36.5GHz,V)",),
        )
        h37_key = _find_key(
            keys, r"36\.?5\s*ghz.*,\s*h\)", flags=re.IGNORECASE,
            exact_first=("Brightness Temperature (36.5GHz,H)",),
        )
        v89_key = _find_key(
            keys, r"89\.?0?\s*ghz-a.*,\s*v\)", flags=re.IGNORECASE,
            exact_first=("Brightness Temperature (89.0GHz-A,V)",),
        )
        h89_key = _find_key(
            keys, r"89\.?0?\s*ghz-a.*,\s*h\)", flags=re.IGNORECASE,
            exact_first=("Brightness Temperature (89.0GHz-A,H)",),
        )

        missing = [n for n, k in [("lat (89A)", lat_key), ("lon (89A)", lon_key),
                                    ("36.5V", v37_key), ("36.5H", h37_key),
                                    ("89.0-A V", v89_key), ("89.0-A H", h89_key)] if k is None]
        if missing:
            raise RuntimeError(
                f"Could not find HDF5 variables for: {missing} in {fpath}. "
                f"Available dataset keys: {keys}. The AMSR2 L1B variable naming "
                "may differ from what this code expects -- inspect the file and "
                "update the regex patterns in fetch_amsr2_swath()."
            )

        lat_hi = f[lat_key][()].astype(np.float64)   # 89GHz-native grid (width 486)
        lon_hi = f[lon_key][()].astype(np.float64)
        # AMSR2 L1B brightness temps are typically stored as scaled 16-bit
        # integers; the scale factor is usually a dataset attribute.
        v37 = _read_scaled(f, v37_key)
        h37 = _read_scaled(f, h37_key)
        v89 = _read_scaled(f, v89_key)
        h89 = _read_scaled(f, h89_key)

        # Confirmed against a real file: 36.5GHz is natively sampled at
        # HALF the along-scan resolution of 89GHz (e.g. 243 vs 486 pixels
        # per scan) -- this is a real hardware/footprint-size difference
        # for AMSR2, not a resampling artifact this file failed to apply.
        # So 37GHz and 89GHz legitimately need separate geolocation grids.
        # Since only the 89A/89B geolocation is provided in this product,
        # we build the 36.5GHz grid by pairwise-averaging adjacent 89A
        # geolocation samples along the scan direction (486 -> 243, an
        # exact 2x decimation matching the 36.5GHz sample spacing).
        if v37.shape[-1] == lat_hi.shape[-1] // 2 and lat_hi.shape[-1] % 2 == 0:
            lat_lo = 0.5 * (lat_hi[:, 0::2] + lat_hi[:, 1::2])
            # Longitude averaging is naive (doesn't handle antimeridian
            # wraparound) -- fine for a storm-centered box far from +/-180,
            # which is the only use case here.
            lon_lo = 0.5 * (lon_hi[:, 0::2] + lon_hi[:, 1::2])
        elif v37.shape == lat_hi.shape:
            lat_lo, lon_lo = lat_hi, lon_hi
        else:
            raise RuntimeError(
                f"Unexpected AMSR2 36.5GHz array shape {v37.shape} vs 89GHz-grid "
                f"shape {lat_hi.shape} (neither equal nor exactly half-width). "
                "The along-scan decimation assumed here doesn't match this file "
                "-- inspect the file's scan geometry and adjust fetch_amsr2_swath()."
            )

    lat89, lon89, v89, h89 = _crop_swath_to_box(
        lat_hi, lon_hi, center_lat, center_lon, box_deg, v89, h89
    )
    lat37, lon37, v37, h37 = _crop_swath_to_box(
        lat_lo, lon_lo, center_lat, center_lon, box_deg, v37, h37
    )
    if lat89.size == 0 or lat37.size == 0:
        return None

    v37, _ = sanitize_field(v37)
    h37, _ = sanitize_field(h37)
    v89, _ = sanitize_field(v89)
    h89, _ = sanitize_field(h89)

    return MWSwath(
        sensor="AMSR2",
        scene_time=actual_scene_time,
        lat=lat89,
        lon=lon89,
        v37=v37,
        h37=h37,
        v89=v89,
        h89=h89,
        lat37=lat37,
        lon37=lon37,
        lat89=lat89,
        lon89=lon89,
        source_note=os.path.basename(fpath),
    )


def _walk_dataset_tree(node, path=()):
    """Yield (path_tuple, value) for every leaf reached while recursively
    walking a gportal dataset-tree node. A leaf is anything that isn't a
    dict (typically a list of dataset_id strings, sometimes a bare
    string) -- gportal's tree depth/shape varies by sensor, so this makes
    no assumption about how many levels deep the real ID list sits."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_dataset_tree(v, path + (str(k),))
    else:
        yield path, node


def _describe_dataset_tree(node, max_depth=3, _depth=0) -> str:
    """Human-readable summary of a dataset tree's top few levels, for
    error messages -- not the full tree (which can be huge), just enough
    to see the real category names."""
    if _depth >= max_depth or not isinstance(node, dict):
        return ""
    lines = []
    for k in node:
        lines.append("  " * _depth + f"- {k}")
        lines.append(_describe_dataset_tree(node[k], max_depth, _depth + 1))
    return "\n".join(l for l in lines if l)


def _find_amsr2_l1b_dataset_id() -> Optional[str]:
    """Recursively search the ENTIRE GCOM-W/AMSR2 dataset tree for a path
    that looks like "L1B brightness temperature", rather than assuming a
    fixed tree depth or exact key spelling (the earlier 2-level-only
    version missed real trees that nest differently). Prefers paths that
    also mention brightness/temperature/tb over a bare "L1B" match.

    An explicit MW_INGEST_AMSR2_DATASET_ID environment variable, if set,
    always wins -- once you've found the right ID by hand (via
    list_amsr2_datasets()), this lets you pin it without editing code.
    """
    override = os.environ.get("MW_INGEST_AMSR2_DATASET_ID")
    if override:
        return override

    import gportal

    datasets = gportal.datasets()
    amsr2 = datasets.get("GCOM-W/AMSR2", datasets)

    candidates = []
    for path, ids in _walk_dataset_tree(amsr2):
        path_str = " ".join(path).lower().replace("-", "").replace(" ", "")
        if "l1b" in path_str or "level1b" in path_str:
            candidates.append((path, ids))

    if not candidates:
        return None

    def _specificity(path):
        joined = " ".join(path).lower()
        return 0 if any(t in joined for t in ("bright", "temperature", "tb")) else 1

    candidates.sort(key=lambda pc: _specificity(pc[0]))
    _, ids = candidates[0]
    id_list = ids if isinstance(ids, list) else [ids]
    return id_list[0] if id_list else None


def _read_scaled(f, key: str) -> np.ndarray:
    ds = f[key]
    data = ds[()].astype(np.float64)
    scale = ds.attrs.get("SCALE FACTOR", ds.attrs.get("Scale", 1.0))
    try:
        scale = float(np.asarray(scale).ravel()[0])
    except Exception:
        scale = 1.0
    return data * scale


def _list_h5_datasets(h5obj, prefix: str = "") -> list[str]:
    out = []
    for key in h5obj.keys():
        path = f"{prefix}/{key}" if prefix else key
        item = h5obj[key]
        if hasattr(item, "keys"):
            out.extend(_list_h5_datasets(item, path))
        else:
            out.append(path)
    return out


def _find_key(keys: list[str], pattern: str, flags=0, exact_first: tuple = ()) -> Optional[str]:
    for name in exact_first:
        for k in keys:
            if k.split("/")[-1] == name:
                return k
    rx = re.compile(pattern, flags)
    for k in keys:
        if rx.search(k):
            return k
    return None


# ---------------------------------------------------------------------------
# SSMIS -- NCEI CDR archive (works today, ~1 month latency; see module
# docstring for why NOAA CLASS near-real-time isn't automated here yet)
# ---------------------------------------------------------------------------
def fetch_ssmis_swath(
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    **kwargs,
) -> Optional[MWSwath]:
    """SSMIS via NOAA CLASS near-real-time ordering is NOT implemented
    (see module docstring). This raises rather than silently returning
    stale/wrong data, so the GUI can show you a clear message instead of
    an empty or misleading result.

    If you want SSMIS for backtesting a storm from a month or more ago,
    use the NCEI CDR archive directly (not wired into the GUI yet):
    https://www.ncei.noaa.gov/products/climate-data-records/ssmis-brightness-temperature-rss
    """
    raise NotImplementedError(
        "Near-real-time SSMIS isn't automated yet -- NOAA CLASS requires an "
        "order-and-retrieve workflow without a simple, verified public API. "
        "The NCEI SSMIS Climate Data Record is a solid archive source but "
        "carries ~1 month latency, so it wasn't wired into 'fetch a scene for "
        "right now.' See the mw_ingest.py module docstring for details."
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def diagnose_gmi_availability(hours: int = 48) -> dict:
    """Standalone diagnostic (not called automatically): for each GMI
    short_name candidate, checks (a) ANY granules with no filters at all,
    (b) granules in the past `hours` globally (no bounding box). Comparing
    these tells you whether a short_name is registered/has data at all,
    vs. whether the temporal filter specifically is the problem, vs.
    whether it's a real location/latency gap.

    Usage from a Python shell (after earthaccess is installed and you've
    logged in once):
        import mw_ingest
        mw_ingest.diagnose_gmi_availability()
    """
    import earthaccess

    # datetime.utcnow() is deprecated from Python 3.12 and returns a
    # NAIVE datetime, which is inconsistent with the tz-aware UTC used
    # everywhere else in this project.
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    dynamic = _resolve_gmi_short_names()
    candidates = list(dict.fromkeys(dynamic + list(GMI_SHORT_NAME_CANDIDATES)))

    report = {}
    for short_name in candidates:
        entry = {}
        try:
            r_any = earthaccess.search_data(short_name=short_name, count=5)
            entry["any_granules_no_filters"] = len(r_any)
        except Exception as e:
            entry["any_granules_no_filters"] = f"ERROR: {e}"
        try:
            r_time = earthaccess.search_data(short_name=short_name, temporal=(start_str, end_str), count=5)
            entry[f"past_{hours}h_global"] = len(r_time)
        except Exception as e:
            entry[f"past_{hours}h_global"] = f"ERROR: {e}"
        report[short_name] = entry
    return report


# ---------------------------------------------------------------------------
# Local file ingest -- failsafe for when NRT network access isn't working,
# or for reprocessing a file you've already downloaded/saved.
# ---------------------------------------------------------------------------
def detect_sensor_from_filename(filename: str) -> Optional[str]:
    """Best-effort sensor detection from a local NetCDF filename. Returns
    'GMI', 'SSMIS', 'WSFM', or None if it can't tell -- callers should
    treat None as "ask the user" rather than guessing further."""
    name = os.path.basename(filename).upper()
    if "SSMIS" in name:
        return "SSMIS"
    if "WSFM" in name or "MWI" in name:
        return "WSFM"
    if "GMI" in name:
        return "GMI"
    return None


def _parse_scene_time_from_filename(filename: str, fallback: datetime) -> datetime:
    """Try the PPS-style '...YYYYMMDD-SHHMMSS-EHHMMSS...' pattern first
    (GMI/SSMIS/WSFM NRT and archive filenames), then the AMSR2-style
    '..._YYYYMMDDHHMI_...' pattern, then give up and use fallback."""
    base = os.path.basename(filename)
    m = re.search(r"(\d{8})-S(\d{6})-E(\d{6})", base)
    if m:
        datestr, sstr, _estr = m.groups()
        try:
            return timeutil.as_utc(datetime.strptime(datestr + sstr, "%Y%m%d%H%M%S"))
        except ValueError:
            pass
    m2 = re.search(r"_(\d{12})_", base)
    if m2:
        try:
            return timeutil.as_utc(datetime.strptime(m2.group(1), "%Y%m%d%H%M"))
        except ValueError:
            pass
    return fallback


def load_local_swath(
    file_path: str,
    center_lat: float,
    center_lon: float,
    box_deg: float = 6.0,
    sensor: Optional[str] = None,
) -> MWSwath:
    """Load a real MW swath from an already-downloaded local NetCDF file
    instead of fetching over the network -- a failsafe for when NRT
    ingest isn't working, or for reprocessing files you've saved
    (including from ~/.synthetic_mw_tc/NRT/ itself). Sensor type is
    auto-detected from the filename unless explicitly given. Reuses the
    same channel-identification logic as the network-fetch paths (the
    confirmed GMI S1 indices, SSMIS S2/S4 indices, and WSFM's multi-group
    LongName-parsing discovery) rather than duplicating separate logic
    that could drift out of sync.

    Raises RuntimeError (not a silent guess) if the sensor can't be
    determined or the file doesn't cover the requested location.
    """
    import xarray as xr

    if sensor is None:
        sensor = detect_sensor_from_filename(file_path)
    if sensor is None:
        raise RuntimeError(
            f"Could not determine sensor type from filename "
            f"'{os.path.basename(file_path)}' -- expected it to contain 'GMI', "
            "'SSMIS', or 'WSFM'/'MWI'. Rename the file to include one of those, "
            "or pass sensor= explicitly if calling load_local_swath() directly."
        )

    # Fallback must be tz-aware UTC. datetime.now() returns a NAIVE
    # datetime in LOCAL time: every other timestamp in this project is
    # tz-aware UTC, so comparing them (as best-track matching and pass-age
    # confidence both do) raises "can't compare offset-naive and
    # offset-aware datetimes", and even where it didn't the value would be
    # wrong by the machine's UTC offset. Only reachable when a filename
    # can't be parsed, which is exactly when a local file is least likely
    # to be well-formed.
    scene_time = _parse_scene_time_from_filename(file_path, fallback=datetime.now(timezone.utc))

    if sensor == "GMI":
        ds = _open_gmi_swath1_group(file_path)
        tc = ds["Tc"].values
        lat = ds["Latitude"].values
        lon = ds["Longitude"].values
        v37 = tc[..., GMI_S1_CHANNEL_INDEX["37V"]]
        h37 = tc[..., GMI_S1_CHANNEL_INDEX["37H"]]
        v89 = tc[..., GMI_S1_CHANNEL_INDEX["89V"]]
        h89 = tc[..., GMI_S1_CHANNEL_INDEX["89H"]]
        lat, lon, v37, h37, v89, h89 = _crop_swath_to_box(
            lat, lon, center_lat, center_lon, box_deg, v37, h37, v89, h89
        )
        if lat.size == 0:
            raise RuntimeError(
                f"'{os.path.basename(file_path)}' loaded fine, but none of it falls "
                f"within {box_deg} deg of ({center_lat:.2f}, {center_lon:.2f}) -- "
                "wrong storm/time selected for this file?"
            )
        v37, _ = sanitize_field(v37)
        h37, _ = sanitize_field(h37)
        v89, _ = sanitize_field(v89)
        h89, _ = sanitize_field(h89)
        return MWSwath(
            sensor="GMI (local)", scene_time=scene_time, lat=lat, lon=lon,
            v37=v37, h37=h37, v89=v89, h89=h89,
            source_note=f"local file: {os.path.basename(file_path)}",
        )

    elif sensor == "SSMIS":
        try:
            ds_s2 = xr.open_dataset(file_path, group="S2")
            ds_s4 = xr.open_dataset(file_path, group="S4")
        except Exception as e:
            raise RuntimeError(f"Could not open S2/S4 groups in {file_path} ({e}).") from e

        tc_s2 = ds_s2["Tc"].values
        lat37 = ds_s2["Latitude"].values
        lon37 = ds_s2["Longitude"].values
        v37 = tc_s2[..., SSMIS_S2_CHANNEL_INDEX["37V"]]
        h37 = tc_s2[..., SSMIS_S2_CHANNEL_INDEX["37H"]]

        tc_s4 = ds_s4["Tc"].values
        lat89 = ds_s4["Latitude"].values
        lon89 = ds_s4["Longitude"].values
        v89 = tc_s4[..., SSMIS_S4_CHANNEL_INDEX["91V"]]
        h89 = tc_s4[..., SSMIS_S4_CHANNEL_INDEX["91H"]]

        lat37, lon37, v37, h37 = _crop_swath_to_box(lat37, lon37, center_lat, center_lon, box_deg, v37, h37)
        lat89, lon89, v89, h89 = _crop_swath_to_box(lat89, lon89, center_lat, center_lon, box_deg, v89, h89)
        if lat37.size == 0 or lat89.size == 0:
            raise RuntimeError(
                f"'{os.path.basename(file_path)}' loaded fine, but none of it falls "
                f"within {box_deg} deg of ({center_lat:.2f}, {center_lon:.2f})."
            )
        v37, _ = sanitize_field(v37)
        h37, _ = sanitize_field(h37)
        v89, _ = sanitize_field(v89)
        h89, _ = sanitize_field(h89)
        return MWSwath(
            sensor="SSMIS (local)", scene_time=scene_time, lat=lat89, lon=lon89,
            v37=v37, h37=h37, v89=v89, h89=h89,
            lat37=lat37, lon37=lon37, lat89=lat89, lon89=lon89,
            source_note=f"local file: {os.path.basename(file_path)}",
        )

    elif sensor == "WSFM":
        channels = _discover_wsfm_channels(file_path)
        opened = {}

        def _get(label):
            group, idx = channels[label]
            if group not in opened:
                opened[group] = xr.open_dataset(file_path, group=group)
            ds = opened[group]
            return ds["Tc"].values[..., idx], ds["Latitude"].values, ds["Longitude"].values

        v37, lat37, lon37 = _get("37V")
        h37, _, _ = _get("37H")
        v89, lat89, lon89 = _get("89V")
        h89, _, _ = _get("89H")

        lat37, lon37, v37, h37 = _crop_swath_to_box(lat37, lon37, center_lat, center_lon, box_deg, v37, h37)
        lat89, lon89, v89, h89 = _crop_swath_to_box(lat89, lon89, center_lat, center_lon, box_deg, v89, h89)
        if lat37.size == 0 or lat89.size == 0:
            raise RuntimeError(
                f"'{os.path.basename(file_path)}' loaded fine, but none of it falls "
                f"within {box_deg} deg of ({center_lat:.2f}, {center_lon:.2f})."
            )
        v37, _ = sanitize_field(v37)
        h37, _ = sanitize_field(h37)
        v89, _ = sanitize_field(v89)
        h89, _ = sanitize_field(h89)
        return MWSwath(
            sensor="WSFM (local)", scene_time=scene_time, lat=lat89, lon=lon89,
            v37=v37, h37=h37, v89=v89, h89=h89,
            lat37=lat37, lon37=lon37, lat89=lat89, lon89=lon89,
            source_note=f"local file: {os.path.basename(file_path)} (channels: {channels})",
        )

    else:
        raise RuntimeError(f"Unknown sensor '{sensor}'.")


def diagnose_overpass_prediction(
    sensor: str,
    center_lat: float,
    center_lon: float,
    target_time: datetime,
    lookback_hours: float,
) -> str:
    """Best-effort, TLE-based explanation for why a real-MW search came up
    empty across all lookback tiers: distinguishes "the satellite's ground
    track never got near this location" (genuine coverage miss) from "it
    passed close by, so this is more likely a data-latency/access problem
    than a coverage gap." Never raises -- returns an empty or explanatory
    string either way, meant to be appended to a "not found" message."""
    sensor_keys = {
        "GMI": ["GMI"], "GMI-NRT": ["GMI"],
        "AMSR2": ["AMSR2"],
        "WSFM-NRT": ["WSFM"],
        "SSMIS": ["SSMIS-F16", "SSMIS-F17", "SSMIS-F18"],
        "SSMIS-NRT": ["SSMIS-F16", "SSMIS-F17", "SSMIS-F18"],
    }.get(sensor)
    if not sensor_keys:
        return ""

    try:
        import tle_predict

        start = target_time - timedelta(hours=lookback_hours)
        hit_summaries = []
        for key in sensor_keys:
            hits = tle_predict.predict_overpasses(key, center_lat, center_lon, start, target_time)
            if hits:
                hit_summaries.append(f"{key} near {hits[-1]:%Y-%m-%d %H:%M}Z")

        if hit_summaries:
            return (
                " TLE prediction says " + "; ".join(hit_summaries) + " DID pass near this "
                "location in the searched window -- so this looks more like a data-latency/"
                "access problem than a genuine coverage gap."
            )
        else:
            return (
                f" TLE prediction says no {'/'.join(sensor_keys)} pass came within the "
                "sensor's approximate swath of this location in the searched window -- "
                "this looks like a genuine coverage miss, not a data-access problem."
            )
    except Exception as e:
        return f" (TLE-based overpass check unavailable: {e})"


def fetch_recent_swath(
    sensor: str,
    target_time: datetime,
    center_lat: float,
    center_lon: float,
    username: str,
    password: str,
    box_deg: float = 6.0,
    lookback_hours: float = 12.0,
    progress_callback=None,
    search_direction: str = "backward",
) -> tuple[Optional[MWSwath], float]:
    """Search the past `lookback_hours` (default 12h) for the most recent
    overpass. Returns (swath_or_None, lookback_hours) -- the second value
    is kept in the return shape for backward compatibility with callers
    that log/display it, even though it's now always just the same fixed
    input value rather than "whichever tier succeeded."

    Previously tried progressively wider windows (6h, then 9h, then 12h)
    one at a time, on the theory that a narrower search might turn up a
    fresher/more-recent pass. That theory doesn't actually hold: every
    tier's search already selects the MOST RECENT candidate within
    whatever window it searches, and every candidate found within 6h or
    9h is *also* within 12h -- so a single 12h search finds exactly the
    same result a 6->9->12h tiered search would have, just without up to
    2 wasted intermediate queries per sensor. Simplified directly per
    that reasoning.

    sensor must be "GMI" (GES DISC archive, multi-day latency but
    reliable), "GMI-NRT" (PPS near-real-time feed, confirmed live at
    /1CR/), "SSMIS-NRT" (PPS near-real-time feed, confirmed live at
    /1C/SSMIS/ -- this is the real near-real-time SSMIS path; "SSMIS"
    below is the ~1-month-latency NCEI archive fallback), "WSFM-NRT" (PPS
    near-real-time WSF-M/MWI, channel identification not hardcoded -- see
    fetch_wsfm_swath_nrt's docstring), "AMSR2", or "SSMIS".
    progress_callback(str), if given, is called with a status message
    before the search.

    search_direction: "backward" (default) or "forward" -- passed through
        to the underlying NRT fetch functions (GMI-NRT/WSFM-NRT/AMSR3-NRT
        only; the archive/AMSR2/SSMIS-archive paths don't support this
        and ignore it) for MIMIC-TC-style crossfade morphing between a
        "before" and "after" pass. "forward" means lookback_hours is
        actually searched AHEAD of target_time, not behind it.
    """
    if sensor == "GMI":
        fetch_fn = fetch_gmi_swath
    elif sensor == "GMI-NRT":
        fetch_fn = fetch_gmi_swath_nrt
    elif sensor == "SSMIS-NRT":
        fetch_fn = fetch_ssmis_swath_nrt
    elif sensor == "WSFM-NRT":
        fetch_fn = fetch_wsfm_swath_nrt
    elif sensor == "AMSR3-NRT":
        fetch_fn = fetch_amsr3_swath_nrt
    elif sensor == "AMSR2":
        fetch_fn = fetch_amsr2_swath
    elif sensor == "SSMIS":
        fetch_fn = fetch_ssmis_swath
    else:
        raise ValueError(f"Unknown sensor {sensor}")

    supports_direction = sensor in ("GMI-NRT", "SSMIS-NRT", "WSFM-NRT", "AMSR3-NRT")

    if progress_callback:
        direction_word = "ahead of" if search_direction == "forward" else "past"
        progress_callback(f"Searching {lookback_hours:g}h {direction_word} {sensor} imagery...")
    kwargs = dict(box_deg=box_deg, lookback_hours=lookback_hours)
    if sensor in ("GMI", "GMI-NRT", "WSFM-NRT", "AMSR3-NRT"):
        # NOTE: previously only "GMI" (the archive path) got this -- a
        # real, confirmed gap: GMI-NRT/WSFM-NRT/AMSR3-NRT (the three
        # sensors this project actually depends on day to day) had NO
        # path for ANY diagnostic message to reach the user at all,
        # which is why a real failure report showed nothing between
        # "Searching Xh past SENSOR imagery..." and the final "not
        # found" -- no visibility into what happened in between. Fixed
        # by adding progress_callback support to all three fetch
        # functions and actually passing it through here. SSMIS-NRT
        # intentionally not included -- it doesn't accept
        # progress_callback either, but it's not one of the three
        # sensors this project is currently built around, so lower
        # priority to extend right now.
        kwargs["progress_callback"] = progress_callback
    if supports_direction:
        kwargs["search_direction"] = search_direction
    swath = fetch_fn(target_time, center_lat, center_lon, username, password, **kwargs)
    return swath, lookback_hours


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km -- works on scalars or broadcasting
    arrays (used both for a single center-to-center distance and for
    center-to-swath-grid distances)."""
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_swath_that_hit_storm(
    sensor: str,
    fixes: list,
    target_time: datetime,
    storm_lat: float,
    storm_lon: float,
    username: str,
    password: str,
    core_radius_km: float = 75.0,
    max_total_lookback_hours: float = 24.0,
    progress_callback=None,
    search_direction: str = "backward",
):
    """Like fetch_recent_swath, but keeps searching progressively further
    in time (backward by default) if the nearest pass found doesn't
    actually cover the storm's CORE -- not just somewhere within the
    generic search box. This is the "most recent pass that actually hit
    the TC" piece of a MIMIC-TC-style approach: a technically-in-range
    pass whose swath edge just barely clips the search box near the
    storm, without ever crossing the actual circulation center, isn't
    meaningfully better than no pass at all for showing real eye/core
    structure, and shouldn't be preferred over another pass that
    genuinely observed the storm.

    Core coverage is checked using the swath's OWN scene_time (the storm
    has likely moved between an old pass and target_time) via best-track
    interpolation, not target_time itself.

    search_direction: "backward" (default) searches and retries further
        INTO THE PAST on a miss -- the original behavior. "forward"
        searches and retries further INTO THE FUTURE instead, for
        finding an "after" pass to crossfade toward (legitimate for
        historical/archived generation, where a pass that occurs after
        target_time has already happened and its data already exists).

    Returns (swath_or_None, lookback_hours_used, was_a_hit). If nothing
    hits within max_total_lookback_hours, returns whatever the LAST
    candidate was (even if it missed) so a caller can still fall back to
    "better than nothing" -- was_a_hit tells you which case you're in.
    """
    search_time = target_time
    total_searched_hours = 0.0
    last_swath = None
    last_lookback = 0.0
    previous_scene_time = None
    MAX_ITERATIONS = 50  # hard safety cap, independent of the hours-based
    # termination -- guards against fetch_recent_swath returning the same
    # pass again for a barely-shifted search_time (its own internal
    # lookback search can legitimately land on the same nearest-available
    # candidate), which would otherwise make zero forward progress. In
    # real usage, actual passes for a given sensor/location are spaced by
    # at least an orbital period (~90+ min), so this terminates in a
    # handful of iterations well before the cap; the cap matters mainly
    # for pathological/degenerate cases (confirmed via a worst-case test
    # with a mock returning a distinct-but-still-missing pass every single
    # call -- 30 was too tight for that adversarial case to cover the
    # full max_total_lookback_hours range, 50 gives more headroom).

    for _iteration in range(MAX_ITERATIONS):
        if total_searched_hours >= max_total_lookback_hours:
            break

        swath, lookback_used = fetch_recent_swath(
            sensor, search_time, storm_lat, storm_lon, username, password,
            progress_callback=progress_callback, search_direction=search_direction,
        )
        if swath is None:
            # A REAL, confirmed structural gap: this used to give up
            # immediately here, completely bypassing the retry/push-back
            # logic below (which only ever ran for a "found a swath, but
            # it missed the storm's core" result). But fetch_recent_swath
            # returning None is EXACTLY what a geographic-coverage miss
            # produces too (the sensor's own fetch_*_swath_nrt function
            # finds a time-matching file, downloads it, but its actual
            # swath doesn't cover the target box) -- and per real logs,
            # that's the more common failure mode of the two. Treating
            # "nothing usable found" as an immediate dead end meant the
            # well-tested retry mechanism below never got a chance to run
            # for the case that needed it most.
            #
            # Fixed by treating this the same as a core-miss: push
            # search_time and keep going, bounded by the same
            # max_total_lookback_hours/MAX_ITERATIONS this function
            # already respects. Since we don't have a specific failed
            # candidate's own scene_time here (fetch_recent_swath only
            # returns None, not the time it examined), jump by a full
            # ~90-minute orbital period rather than the 6-minute nudge
            # used for a known core-miss -- a smaller nudge risks landing
            # on another slice of the exact same already-failed pass
            # (confirmed directly: GMI's NRT files are ~5-minute granules,
            # so several nearby search times can all resolve to the same
            # useless swath), while a full period reliably reaches a
            # genuinely different orbital opportunity.
            #
            # NOTE: deliberately NOT logging a message on every one of
            # these pushes (unlike the core-miss case below, which does).
            # A 12h budget at 90-min steps is up to 8 retries per sensor;
            # logging each one produced a wall of largely-redundant
            # output in testing, especially since fetch_recent_swath's
            # own call chain (fetch_*_swath_nrt) already logs a specific
            # "MW search debug [...]" explanation on a geographic miss --
            # this loop doesn't need its own extra line saying the same
            # thing again for the same event. The search + debug messages
            # already in the log are enough to follow what's happening
            # without every single 90-minute step being spelled out too.
            if search_direction == "forward":
                search_time = search_time + timedelta(minutes=90)
            else:
                search_time = search_time - timedelta(minutes=90)
            total_searched_hours = abs((search_time - target_time).total_seconds()) / 3600.0
            continue

        last_swath, last_lookback = swath, lookback_used

        fix_at_pass_time = besttrack.interpolate_fix(fixes, swath.scene_time)
        if fix_at_pass_time is None:
            # Can't validate without a best-track fix at the pass's own
            # time -- trust it rather than discarding good data over a
            # missing validation step.
            return swath, lookback_used, True

        lat_g, lon_g = swath.grid_for(37)
        dist_km = _haversine_km(lat_g, lon_g, fix_at_pass_time.lat, fix_at_pass_time.lon)
        if np.isfinite(dist_km).any() and np.nanmin(dist_km) <= core_radius_km:
            return swath, lookback_used, True

        if progress_callback:
            direction_word = "further ahead" if search_direction == "forward" else "further back"
            progress_callback(
                f"{sensor} pass at {swath.scene_time:%Y-%m-%d %H:%M} UTC missed the storm "
                f"core (nearest approach >{core_radius_km:.0f}km) -- searching {direction_word}..."
            )

        # MISS: push the search window past this pass, in whichever
        # direction we're searching. If the exact same pass came back
        # last time too (no progress from a small nudge), jump more
        # decisively RELATIVE TO THE CURRENT SEARCH_TIME (not the stuck
        # swath's own scene_time -- jumping relative to a value that
        # keeps coming back unchanged would recompute the exact same
        # target every time and never actually escape the stuck state,
        # which a direct test caught).
        stuck = previous_scene_time is not None and swath.scene_time == previous_scene_time
        if search_direction == "forward":
            if stuck:
                search_time = search_time + timedelta(hours=1)
            else:
                search_time = swath.scene_time + timedelta(minutes=6)
        else:
            if stuck:
                search_time = search_time - timedelta(hours=1)
            else:
                search_time = swath.scene_time - timedelta(minutes=6)
        previous_scene_time = swath.scene_time
        total_searched_hours = abs((search_time - target_time).total_seconds()) / 3600.0

    return last_swath, last_lookback, False


def morph_swath_to_time(swath: MWSwath, fixes: list, target_time: datetime) -> MWSwath:
    """MIMIC-TC-style advection: shift a real MW swath's lat/lon by the
    storm's own displacement between the swath's OBSERVATION time and
    target_time, so an aging pass's real pattern gets repositioned to
    align with the storm's CURRENT location. This is what lets a single
    real overpass keep contributing real (if increasingly dated)
    structure across the gap until the next actual pass arrives, rather
    than either misaligning it at its original position or dropping it
    entirely once it's no longer "fresh."

    Only the SPATIAL grid moves -- the V/H data itself is untouched (this
    captures storm TRANSLATION, not genuine structural evolution like
    intensification or eyewall replacement; scene_time is preserved
    unchanged so callers can still tell how old the underlying
    observation actually is, e.g. for confidence weighting).

    Returns a NEW MWSwath (does not mutate the input). If either
    endpoint's best-track fix is unavailable, returns the swath
    unshifted (better than crashing; the caller's downstream distance/
    coverage masks will just treat it as if it were still at its
    original position).
    """
    fix_then = besttrack.interpolate_fix(fixes, swath.scene_time)
    fix_now = besttrack.interpolate_fix(fixes, target_time)
    if fix_then is None or fix_now is None:
        return swath

    dlat = fix_now.lat - fix_then.lat
    # Dateline-safe: a pass observed either side of 180 would otherwise
    # be advected most of the way around the planet.
    dlon = (fix_now.lon - fix_then.lon + 180.0) % 360.0 - 180.0

    updates = {"lat": swath.lat + dlat, "lon": swath.lon + dlon}
    if swath.lat37 is not None:
        updates["lat37"] = swath.lat37 + dlat
        updates["lon37"] = swath.lon37 + dlon
    if swath.lat89 is not None:
        updates["lat89"] = swath.lat89 + dlat
        updates["lon89"] = swath.lon89 + dlon

    return dataclasses.replace(swath, **updates)


# Sensors this project treats as the long-term-viable NRT trio, per
# direct guidance: AMSR2-NRT stops transmitting after Aug 31 2026 (AMSR3
# is its successor and already active), SSMIS shuts down entirely in
# September 2026 -- investing further effort in either would have a
# shelf life of weeks. GMI-NRT, AMSR3-NRT, and WSFM-NRT are the sensors
# worth building the primary pipeline around going forward.
PRIMARY_NRT_SENSORS = ("GMI-NRT", "AMSR3-NRT", "WSFM-NRT")


def compute_checkpoint_frame_indices(n_frames: int) -> list:
    """For multi-frame loops: which frame indices should trigger an
    actual MW search, vs. reusing the nearest searched checkpoints'
    results (still independently MIMIC-TC-morphed to each frame's own
    time -- only the SEARCH is skipped, not the per-frame math).

    Per direct guidance: searching independently for every single frame
    (up to 15 separate multi-sensor searches) is wasteful when a 5-frame
    "distance" between searches is perfectly adequate -- real MW passes
    don't refresh every 10 minutes anyway. Checkpoints are evenly spaced
    across the frame sequence, always including frame 0 and the last
    frame: 5 frames -> 2 checkpoints (first, last), 10 frames -> 3
    (first, ~halfway, last), 15 frames -> 4 (first, ~1/3, ~2/3, last) --
    confirmed directly to match all three examples given.
    """
    if n_frames <= 1:
        return [0]
    n_checkpoints = max(2, int(np.ceil(n_frames / 5)) + 1)
    n_checkpoints = min(n_checkpoints, n_frames)
    raw = np.linspace(0, n_frames - 1, n_checkpoints)
    indices = []
    for x in raw:
        idx = int(round(x))
        if idx not in indices:
            indices.append(idx)
    return indices


def find_mw_pair_for_crossfade(
    fixes: list,
    target_time: datetime,
    storm_lat: float,
    storm_lon: float,
    creds: dict,
    sensors: tuple = PRIMARY_NRT_SENSORS,
    lookback_hours: float = 12.0,
    lookahead_hours: float = 3.0,
    core_radius_km: float = 75.0,
    progress_callback=None,
    search_after: bool = False,
):
    """Gather the best "before" real MW pass across all of `sensors`
    (default: the primary GMI-NRT/AMSR3-NRT/WSFM-NRT trio), searching a
    window of lookback_hours before target_time.

    Per direct guidance: this project is primarily for REAL-TIME use, and
    "searching ahead" of target_time for a future/"after" pass to
    crossfade toward is a wasted network round-trip in that context (a
    real-time run has no future data to find) -- it only ever had value
    for archived/historical generation, and even then NRT data retention
    is only ~7 days, a narrow use case that didn't justify the search
    cost on every single real-time run. search_after=False (the default,
    and the ONLY behavior this project's real-time GUI paths use now)
    skips the forward search entirely -- "after" is always None in the
    returned dict. The underlying forward-search CAPABILITY still exists
    (find_swath_that_hit_storm's search_direction="forward",
    _pps_nrt_find_and_download's mirrored window logic) and can be
    reached by passing search_after=True, in case a future archive-
    specific mode wants it -- nothing about that capability was removed,
    just no longer invoked by default.

    Per-sensor searches (each an independent network round-trip) run
    CONCURRENTLY via a small thread pool rather than one at a time in
    sequence -- a direct speed optimization, since these are independent
    I/O-bound requests to (typically) the same PPS server.

    For each sensor, uses find_swath_that_hit_storm's "keep looking if
    the nearest candidate misses the actual core" logic, not just
    "nearest in time." Across all sensors that found a "before" hit,
    keeps the MOST RECENT one.

    Returns a dict:
        {
            "before": MWSwath or None, "before_sensor": str or None,
            "after": MWSwath or None, "after_sensor": str or None,
        }
    "before" may be None (nothing found across all sensors within the
    window) -- caller should then fall back to the GMI archive
    (fetch_gmi_swath / sensor="GMI"), which has its own (much higher
    latency, multi-day) availability and isn't limited to the NRT ~7-day
    retention window. "after"/"after_sensor" are always None unless
    search_after=True is explicitly passed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    creds_map = {
        "GMI-NRT": creds.get("pps_nrt", {}),
        "AMSR3-NRT": creds.get("pps_nrt", {}),
        "WSFM-NRT": creds.get("pps_nrt", {}),
    }

    def _search_one_sensor_backward(sensor):
        cred = creds_map.get(sensor, {})
        if not cred.get("username"):
            if progress_callback:
                progress_callback(f"MW search: skipping {sensor} (no credentials configured).")
            return None
        try:
            swath, _lb, hit = find_swath_that_hit_storm(
                sensor, fixes, target_time, storm_lat, storm_lon,
                cred.get("username", ""), cred.get("password", ""),
                core_radius_km=core_radius_km, max_total_lookback_hours=lookback_hours,
                progress_callback=progress_callback, search_direction="backward",
            )
            if swath is not None and hit:
                return (swath.scene_time, swath, sensor)
        except Exception as e:
            if progress_callback:
                progress_callback(f"MW search: {sensor} failed ({e}).")
        return None

    def _search_one_sensor_forward(sensor):
        cred = creds_map.get(sensor, {})
        if not cred.get("username"):
            return None
        try:
            swath, _lb, hit = find_swath_that_hit_storm(
                sensor, fixes, target_time, storm_lat, storm_lon,
                cred.get("username", ""), cred.get("password", ""),
                core_radius_km=core_radius_km, max_total_lookback_hours=lookahead_hours,
                progress_callback=progress_callback, search_direction="forward",
            )
            if swath is not None and hit:
                return (swath.scene_time, swath, sensor)
        except Exception as e:
            if progress_callback:
                progress_callback(f"MW search: {sensor} forward search failed ({e}).")
        return None

    before_candidates = []
    with ThreadPoolExecutor(max_workers=max(1, len(sensors))) as executor:
        futures = [executor.submit(_search_one_sensor_backward, s) for s in sensors]
        for future in as_completed(futures):
            found = future.result()
            if found is not None:
                before_candidates.append(found)

    after_candidates = []
    if search_after:
        with ThreadPoolExecutor(max_workers=max(1, len(sensors))) as executor:
            futures = [executor.submit(_search_one_sensor_forward, s) for s in sensors]
            for future in as_completed(futures):
                found = future.result()
                if found is not None:
                    after_candidates.append(found)

    result = {"before": None, "before_sensor": None, "after": None, "after_sensor": None}

    if before_candidates:
        before_candidates.sort(key=lambda c: c[0], reverse=True)  # most recent first
        _stime, swath, sensor = before_candidates[0]
        result["before"] = swath
        result["before_sensor"] = sensor
        if progress_callback:
            progress_callback(f"MW search: best pass is {sensor} @ {swath.scene_time:%Y-%m-%d %H:%M} UTC.")

    if after_candidates:
        after_candidates.sort(key=lambda c: c[0])  # soonest first
        _stime, swath, sensor = after_candidates[0]
        result["after"] = swath
        result["after_sensor"] = sensor
        if progress_callback:
            progress_callback(f"MW search: best 'after' pass is {sensor} @ {swath.scene_time:%Y-%m-%d %H:%M} UTC.")

    return result


def _crop_swath_to_box(lat, lon, center_lat, center_lon, box_deg, *fields):
    """Crop conically-scanning swath arrays (2D, irregular) to a lat/lon
    box by masking rows containing at least one in-box pixel, then masking
    columns similarly -- keeps the arrays 2D and roughly rectangular for
    pcolormesh, rather than flattening to a scattered point cloud."""
    in_box = (
        (lat >= center_lat - box_deg)
        & (lat <= center_lat + box_deg)
        & (lon >= center_lon - box_deg)
        & (lon <= center_lon + box_deg)
    )
    if not in_box.any():
        empty = np.empty((0, 0))
        return (empty, empty) + tuple(empty for _ in fields)

    rows = np.where(in_box.any(axis=1))[0]
    cols = np.where(in_box.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    cropped = [arr[r0:r1, c0:c1] for arr in (lat, lon, *fields)]
    return tuple(cropped)
