"""
TC-PRIMED (NOAA/NESDIS/CIRA "Tropical Cyclone PRecipitation, Infrared,
Microwave, and Environmental Dataset") ingestion, via direct boto3
access to its public, unsigned AWS S3 bucket -- reusing the exact same
unsigned-access pattern already established in goes_fetch.py for the
(also public) GOES buckets, rather than introducing a different style.

WHY THIS MATTERS FOR THIS PROJECT SPECIFICALLY: TC-PRIMED already does
the "does this satellite pass actually cover the storm" validation that
caused a long chain of real bugs in mw_ingest.py's live-NRT search
(geographic-coverage misses returning a time-matching but geographically
irrelevant file). TC-PRIMED's own curation process (documented in
TCPRIMED_v01r01_documentation.pdf, Section 2.2) computes an areal
coverage fraction within 750km of the interpolated storm center
(falling back to a 250km check) and only retains overpasses meeting a
minimum threshold -- meaning every GMI/AMSR2 overpass file TC-PRIMED
provides for a storm has ALREADY been confirmed to have hit that storm,
for free. Historical training-data mining via TC-PRIMED sidesteps the
entire class of bug the live NRT search kept hitting.

CONFIRMED bucket structure (verified directly, not assumed -- via the
official TC PRIMED products page and the AWS Open Data Registry entry,
both independently describing the same layout):

    s3://noaa-nesdis-tcprimed-pds/<version>/<version_type>/<season>/<basin>/<number>/

e.g. s3://noaa-nesdis-tcprimed-pds/v01r01/final/2018/AL/06/, containing
files like:
    TCPRIMED_v01r01-final_AL062018_GMI_GPM_025795_20180912184512.nc
    TCPRIMED_v01r01-final_AL062018_era5_s20180830060000_e20180918120000.nc  (environmental file)

Confirmed public/unsigned: "Since TC PRIMED is a public dataset, you do
not need an AWS credential" (official products page) -- matches this
module's use of the same UNSIGNED boto3 config already used for GOES.

SCOPE: this project only needs GMI and AMSR2 (per direct guidance --
higher native resolution than other GPM-constellation sensors, and
operational eras that cleanly overlap the GOES-R ABI series this
project's own inputs come from). Channel group/name layouts for GMI and
AMSR2 specifically are hardcoded here directly from the documentation's
Table 3 (confirmed, versioned, stable -- unlike the live NRT feeds in
mw_ingest.py, which needed DYNAMIC channel discovery specifically
because their real structure didn't match assumptions). If TC-PRIMED
ever changes these group layouts in a future version, that dynamic-
discovery machinery already exists in mw_ingest.py and could be reused
here too, but isn't needed unless/until a real mismatch is found.

HONEST CAVEAT, consistent with every other network-touching module in
this project: written directly against confirmed, documented bucket
structure and file-format details, but never executed against the live
bucket from this sandbox (no network access here to verify).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None

import timeutil
from data_types import MWSwath

BUCKET = "noaa-nesdis-tcprimed-pds"

# The only instruments this project reads. TC-PRIMED carries the whole
# GPM constellation -- SSMIS, ATMS, MHS, AMSU-B and others -- so an
# UNFILTERED listing returns several times more files than are usable
# here, and read_overpass_as_swath rejects everything else by design.
# Anything counting or streaming overpasses must filter on this or it
# will over-report and waste requests on files it cannot parse.
SUPPORTED_INSTRUMENTS = ("GMI", "AMSR2")
DEFAULT_LOCAL_DIR = os.path.expanduser("~/.synthetic_mw_tc/tcprimed_cache")

# Directly from TCPRIMED_v01r01_documentation.pdf, Table 3 -- kept ONLY
# as a fallback default target frequency for the discovery function
# below, NOT as a hardcoded group assignment anymore. A real run showed
# this hardcoded approach was wrong: requesting group S2 for GMI's
# 36.64/89.0GHz channels actually opened group S1 (confirmed directly
# via the resulting error -- xarray's own "did you mean" suggestion
# named TB_166.0V, an S1-only channel, when TB_36.64V from S2 was
# requested), meaning either the documented group numbering was
# mis-transcribed here, or passing a nested "passive_microwave/S2"
# group string to xarray's group= parameter doesn't reliably open the
# right group in every environment/engine -- a same-shaped
# "OSError: Invalid argument" on Windows for an AMSR2 file
# (group "passive_microwave/S4") is consistent with the latter.
# Rather than keep guessing at hardcoded group numbers, channel
# discovery is now dynamic (see _discover_tcprimed_channels), reading
# via h5py directly rather than constructing nested group-path strings
# for xarray at all -- sidesteps both failure modes at the root instead
# of patching around them.
GMI_TARGET_37_GHZ = 36.64
AMSR2_TARGET_37_GHZ = 36.5
TARGET_89_GHZ = 89.0
_CHANNEL_NAME_PATTERN = re.compile(r"^TB_([AB]?)(\d+\.?\d*)([HV])$")


def _discover_tcprimed_channels(local_path: str, target_37_ghz: float, target_89_ghz: float = TARGET_89_GHZ, tolerance: float = 3.0) -> tuple:
    """Walk the actual file structure via h5py's visititems() (never
    constructing a nested group-path string for xarray to interpret --
    see the module-level note above for why that broke on a real file)
    to find which (group, variable) pair holds each of 37V/37H/89V/89H,
    based on the frequency encoded directly in TC-PRIMED's own variable
    naming (e.g. "TB_36.64V"), not an assumed group number.

    Deliberately matches ONLY simple "TB_[AB]?<freq><pol>" names (the
    regex's end-of-string anchor rejects e.g. "TB_183.31_3.0V", which
    has an extra underscore-separated offset segment that isn't a
    channel frequency by itself) -- avoids misinterpreting a
    high-frequency sounding channel as a 37 or 89GHz window channel.

    The optional "[AB]?" handles AMSR2's 89GHz channels specifically,
    which are split across two interlaced sub-swaths named with an A/B
    prefix (e.g. "TB_A89.0V", "TB_B89.0V") -- confirmed directly in an
    earlier round's documentation reading, then genuinely dropped by
    mistake when this function was rewritten from a hardcoded-group
    approach to dynamic discovery. A real run surfaced the gap: AMSR2's
    89GHz channels went missing entirely (37/18.7/23.8/10.65GHz all
    still found fine, since those have no A/B prefix), which looked at
    first like it might be specific to one basin but is actually just
    "any AMSR2 file with 89GHz data, wherever it happens to be
    encountered first in processing order." A-scan is preferred
    deterministically when both exist (they're interlaced at
    essentially the same locations, so this is a reasonable, simple
    default rather than needing to merge both) -- NOT just "whichever
    h5py's visititems() happens to walk first," which could vary by
    file or platform and silently disagree from one run to the next.

    Returns ({"37V": (group, varname), ...}, groups_seen) -- groups_seen
    is a dict of every group path actually found with TB_* variables in
    it, included so a failure to find all four channels produces a
    genuinely useful error rather than just "not found."
    """
    import h5py

    found = {}
    found_is_b_scan = set()  # labels currently satisfied only by a B-scan match,
    # still eligible to be upgraded to A-scan if one is found later in the walk
    groups_seen: dict = {}

    # Accepts either a path (opens/closes it here) or an ALREADY-OPEN
    # h5py File handle -- read_overpass_as_swath passes its own open
    # handle so a single file is opened once for discovery+reading
    # rather than twice. Kept path-accepting too so this stays
    # independently callable (e.g. for diagnosing a single file by hand).
    if isinstance(local_path, str):
        opened_here = h5py.File(local_path, "r")
        f = opened_here
    else:
        opened_here = None
        f = local_path

    try:
        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            var_name = name.rsplit("/", 1)[-1]
            m = _CHANNEL_NAME_PATTERN.match(var_name)
            if not m:
                return
            scan = m.group(1)  # "" (no prefix, e.g. GMI), "A", or "B"
            freq = float(m.group(2))
            pol = m.group(3)
            group_path = name.rsplit("/", 1)[0] if "/" in name else ""
            groups_seen.setdefault(group_path, []).append(var_name)

            label = None
            if abs(freq - target_37_ghz) < tolerance:
                label = f"37{pol}"
            elif abs(freq - target_89_ghz) < tolerance:
                label = f"89{pol}"
            if not label:
                return

            if label not in found:
                found[label] = (group_path, var_name)
                if scan == "B":
                    found_is_b_scan.add(label)
            elif label in found_is_b_scan and scan == "A":
                # Upgrade a previously-found B-scan match to A-scan, per
                # the documented preference -- visititems' walk order
                # isn't something to rely on for which one gets seen first.
                found[label] = (group_path, var_name)
                found_is_b_scan.discard(label)

        f.visititems(_visit)
    finally:
        if opened_here is not None:
            opened_here.close()

    return found, groups_seen


def _get_s3_client():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed. `pip install boto3`")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_storm_overpass_files(
    basin: str,
    storm_num: int,
    season: int,
    instrument_filter: Optional[str] = None,
    version: str = "v01r01",
    version_type: str = "final",
) -> list:
    """List all TC-PRIMED overpass files (.nc, excluding the single
    per-storm "era5" environmental file) for one storm, optionally
    filtered to a specific instrument (e.g. "GMI" or "AMSR2" -- matches
    the <instrument> segment of the filename directly, case-sensitive,
    matching TC-PRIMED's own naming).

    Returns a list of dicts: {"key": full S3 key, "filename": str,
    "instrument": str, "platform": str, "granule": str, "timestamp":
    datetime}. Empty list if the storm/prefix doesn't exist in the
    bucket (not an error -- most seasons/basins/numbers won't for any
    given query, that's expected).
    """
    basin = basin.upper()
    prefix = f"{version}/{version_type}/{season}/{basin}/{storm_num:02d}/"

    s3 = _get_s3_client()
    results = []
    continuation_token = None
    while True:
        kwargs = dict(Bucket=BUCKET, Prefix=prefix)
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        try:
            resp = s3.list_objects_v2(**kwargs)
        except ClientError:
            return []

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            filename = os.path.basename(key)
            if not filename.endswith(".nc"):
                continue
            parsed = _parse_overpass_filename(filename)
            if parsed is None:
                continue  # environmental ("era5") file, or unrecognized naming -- skip
            if instrument_filter and parsed["instrument"] != instrument_filter:
                continue
            parsed["key"] = key
            parsed["filename"] = filename
            parsed["size_bytes"] = obj.get("Size", 0)  # already present in the LIST response --
            # no extra API call or download needed to know this, useful for
            # storage estimation without actually pulling any file content.
            results.append(parsed)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    results.sort(key=lambda r: r["timestamp"])
    return results


def _parse_overpass_filename(filename: str) -> Optional[dict]:
    """Parse e.g. "TCPRIMED_v01r01-final_AL062018_GMI_GPM_025795_20180912184512.nc"
    into {"instrument": "GMI", "platform": "GPM", "granule": "025795",
    "timestamp": datetime(2018,9,12,18,45,12), "basin": "AL",
    "storm_num": 6, "season": 2018}. Returns None for the per-storm
    "era5" environmental file (different naming, not an overpass file)
    or anything that doesn't match the expected pattern -- deliberately
    conservative, since silently mis-parsing a filename into wrong
    metadata is worse than skipping a file this doesn't recognize.
    """
    stem = filename[:-3] if filename.endswith(".nc") else filename
    parts = stem.split("_")
    # Expected: TCPRIMED, v01r01-final, AL062018, GMI, GPM, 025795, 20180912184512
    if len(parts) != 7 or parts[0] != "TCPRIMED":
        return None
    storm_code, instrument, platform, granule, ts_str = parts[2], parts[3], parts[4], parts[5], parts[6]
    if instrument == "era5":
        return None
    try:
        timestamp = timeutil.as_utc(datetime.strptime(ts_str, "%Y%m%d%H%M%S"))
    except ValueError:
        return None
    # storm_code is e.g. "AL062018": 2-letter basin, 2-digit storm number,
    # 4-digit season -- same ATCF convention this project already uses
    # elsewhere (besttrack.py's build_url).
    if len(storm_code) != 8:
        return None
    basin = storm_code[:2]
    try:
        storm_num = int(storm_code[2:4])
        season = int(storm_code[4:8])
    except ValueError:
        return None
    return {
        "instrument": instrument, "platform": platform, "granule": granule, "timestamp": timestamp,
        "basin": basin, "storm_num": storm_num, "season": season,
    }


def open_overpass_streaming(key: str, size: Optional[int] = None):
    """Open a TC-PRIMED overpass file IN PLACE on S3, without downloading.

    Returns (h5py.File, reader). Close the file when done; reader.stats()
    reports how much of the file actually crossed the network.

    This is the path that makes a large training set possible on a small
    disk. netCDF4 is HDF5, HDF5 is random-access, and the exporter needs
    only a few variables per file -- so ranged GETs pull roughly a quarter
    of each file and write nothing to local storage. The download path
    below is retained for cases where a file will be read repeatedly.
    """
    from s3_range_reader import open_s3_hdf5
    return open_s3_hdf5(_get_s3_client(), BUCKET, key, size=size)


def download_overpass_file(key: str, local_dir: str = DEFAULT_LOCAL_DIR) -> str:
    """Download one overpass file by its full S3 key, if not already
    cached locally. Returns the local path. No integrity re-check
    against a truncated prior download here (unlike mw_ingest.py's PPS
    NRT path) -- S3 downloads via boto3's download_file are already
    atomic/verified by the library itself, a genuinely different
    reliability profile than the plain-HTTP-GET-into-a-file pattern
    that needed that fix elsewhere in this project."""
    s3 = _get_s3_client()
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(key))
    if not os.path.exists(local_path):
        s3.download_file(BUCKET, key, local_path)
    return local_path


def _normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """Convert 0..360 longitude to -180..180.

    CONFIRMED against real files, not assumed: TC-PRIMED stores longitude
    in 0..360 (an AL012023 swath reported lon range [274.29, 298.96] and
    its own storm at 289.54), while every other component of this project
    -- GOES grids, NHC/IBTrACS best-track, the PPS NRT swaths -- uses
    -180..180. Without this conversion a storm at 70W sits at 290 in the
    swath and 0 percent of the regridded MW ever lands on the GOES grid,
    anywhere. That produced training examples whose supervision mask was
    empty in every single patch, which in turn made the training loss
    exactly 0.0000 for 50 straight epochs while the model learned
    nothing.

    Note on the dateline: a swath spanning it becomes discontinuous in
    -180..180 (values jump from +180 to -180). That is the same
    convention every other swath source in this project already uses, so
    this does not introduce a new problem, but West Pacific storms near
    the dateline remain a known rough edge for the regridding generally.
    """
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def read_overpass_as_swath(local_path: str, instrument: str,
                           handle=None, label: Optional[str] = None) -> MWSwath:
    """Read a downloaded TC-PRIMED overpass file's passive_microwave
    group into an MWSwath. Reads directly via h5py throughout (channel
    discovery AND the actual data), rather than xarray's group=
    parameter with a constructed "parent/child" path string -- a real
    run showed that approach opening the wrong group entirely (see
    _discover_tcprimed_channels' docstring), and a same-shaped Windows
    OSError for a different sensor is consistent with the same root
    cause. Reading directly via h5py's own path navigation sidesteps
    both failure modes rather than patching around them.

    instrument: "GMI" or "AMSR2" -- selects which target 37GHz frequency
        to search for (GMI: 36.64, AMSR2: 36.5 -- both documented,
        confirmed values). Any other value raises ValueError rather than
        guessing.
    handle: an already-open h5py.File. When given, `local_path` is
        ignored and the handle is NOT closed here -- the caller owns it.
        This is what lets the identical reading logic run against a file
        streamed from S3 (s3_range_reader) instead of one on disk, with
        no duplicated parsing code to drift apart.
    label: name to use in messages and source_note when reading a handle,
        since there is no path to derive one from.
    """
    # Argument validation BEFORE the h5py import. Cheap checks first is
    # the general rule, but here it also means a bad instrument name
    # reports itself as a bad instrument name rather than as a missing
    # optional dependency -- and stays checkable on a machine that has no
    # h5py at all.
    if instrument == "GMI":
        target_37 = GMI_TARGET_37_GHZ
    elif instrument == "AMSR2":
        target_37 = AMSR2_TARGET_37_GHZ
    else:
        raise ValueError(f"read_overpass_as_swath only supports GMI/AMSR2, got {instrument!r}")

    import h5py

    # Single file open for BOTH discovery and reading -- previously this
    # opened the same file twice (once inside _discover_tcprimed_channels,
    # once here). Modest but real, and it matters more when processing
    # hundreds of cached files in a bulk mining run.
    name = label or (os.path.basename(local_path) if local_path else "<stream>")
    f = h5py.File(local_path, "r") if handle is None else handle
    try:
        channels, groups_seen = _discover_tcprimed_channels(f, target_37_ghz=target_37)
        missing = [label for label in ("37V", "37H", "89V", "89H") if label not in channels]
        if missing:
            raise RuntimeError(
                f"Could not find channel(s) {missing} in {name} "
                f"(instrument={instrument}). Groups with TB_* variables actually found: "
                f"{ {g: sorted(v) for g, v in groups_seen.items()} }"
            )

        def _read(group_path: str, var_name: str):
            full_path = f"{group_path}/{var_name}" if group_path else var_name
            group_obj = f[group_path] if group_path else f
            data = np.asarray(f[full_path][()], dtype=np.float64)
            lat = np.asarray(group_obj["latitude"][()], dtype=np.float64)
            lon = np.asarray(group_obj["longitude"][()], dtype=np.float64)
            return data, lat, _normalize_longitude(lon)

        group37, var37v = channels["37V"]
        _, var37h = channels["37H"]
        group89, var89v = channels["89V"]
        _, var89h = channels["89H"]

        v37, lat37, lon37 = _read(group37, var37v)
        h37, _, _ = _read(group37, var37h)
        v89, lat89, lon89 = _read(group89, var89v)
        h89, _, _ = _read(group89, var89h)

        # overpass_storm_metadata/time is the observation midpoint time --
        # the correct scene_time for this swath (see documentation: "Time of
        # the passive microwave observation of the tropical cyclone: the
        # subset observation midpoint time").
        scene_time_epoch = float(np.asarray(f["overpass_storm_metadata/time"][()]).flat[0])
    finally:
        # Only close what we opened. A streamed handle belongs to the
        # caller, which also owns the underlying S3 reader.
        if handle is None:
            f.close()

    # datetime.utcfromtimestamp() is deprecated from Python 3.12 and
    # returns a NAIVE datetime, while every other timestamp in this
    # project is tz-aware UTC. Comparing the two raises "can't compare
    # offset-naive and offset-aware datetimes" -- the same bug fixed in
    # mw_ingest at 0.93, still present here.
    scene_time = datetime.fromtimestamp(scene_time_epoch, tz=timezone.utc)

    # TC-PRIMED's fill value for brightness temperature is -9999.9 (per
    # documentation) -- mask it to NaN so it's treated as "no data" by
    # everything downstream, not as a genuinely cold, physically absurd
    # brightness temperature.
    def _mask_fill(arr):
        arr = arr.copy()
        arr[arr <= -9990.0] = np.nan
        return arr

    v37, h37, v89, h89 = _mask_fill(v37), _mask_fill(h37), _mask_fill(v89), _mask_fill(h89)

    return MWSwath(
        sensor=instrument,
        scene_time=scene_time,
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
        source_note=f"TC-PRIMED/{name}",
    )


def read_overpass_streaming(key: str, instrument: str, size: Optional[int] = None,
                           progress_callback=None) -> MWSwath:
    """Read one overpass straight off S3, writing nothing to disk.

    Reports the fetched fraction, because that number is the whole point
    and is worth watching on real files: it was tuned against a simulated
    HDF5 access pattern, and if it comes back far from ~25% the block
    size in s3_range_reader wants revisiting.
    """
    f, reader = open_overpass_streaming(key, size=size)
    try:
        swath = read_overpass_as_swath(None, instrument, handle=f,
                                       label=os.path.basename(key))
    finally:
        f.close()
    if progress_callback:
        st = reader.stats()
        progress_callback(
            f"  streamed {os.path.basename(key)}: {format_bytes(st['bytes_fetched'])} of "
            f"{format_bytes(st['size_bytes'])} ({st['fetched_fraction']*100:.0f}%) "
            f"in {st['requests']} request(s)"
        )
    return swath


def fetch_storm_swaths(
    basin: str,
    storm_num: int,
    season: int,
    instrument: str = "GMI",
    local_dir: str = DEFAULT_LOCAL_DIR,
    stream: bool = True,
    progress_callback=None,
) -> list:
    """Convenience wrapper: list, download, and read every overpass of
    the given instrument for one storm, in one call. Returns a list of
    MWSwath objects, sorted chronologically. Individual files that fail
    to download or parse are skipped (logged via progress_callback if
    given), not fatal to the whole batch -- one bad file shouldn't lose
    every other pass for a storm that otherwise has good data.
    """
    files = list_storm_overpass_files(basin, storm_num, season, instrument_filter=instrument)
    swaths = []
    for f in files:
        try:
            # Stream by default: read the overpass in place on S3 rather
            # than landing a copy on disk. Storage, not bandwidth, is what
            # caps the size of the training set on a laptop, and each
            # file is read exactly once during mining -- so there is
            # nothing for a local copy to amortise. stream=False keeps
            # the download path for anything read repeatedly.
            if stream:
                swath = read_overpass_streaming(
                    f["key"], instrument, size=f.get("size_bytes"),
                    progress_callback=progress_callback)
            else:
                local_path = download_overpass_file(f["key"], local_dir=local_dir)
                swath = read_overpass_as_swath(local_path, instrument)
            swaths.append(swath)
            if progress_callback:
                progress_callback(f"TC-PRIMED: loaded {instrument} pass @ {swath.scene_time:%Y-%m-%d %H:%M} UTC")
        except Exception as e:
            if progress_callback:
                progress_callback(f"TC-PRIMED: skipped {f['filename']} ({type(e).__name__}: {e})")
            continue
    return swaths


# Basin codes per the documentation (Section 2.3.1) -- all seven TC-PRIMED
# uses. Walking every basin for every season is the only way to discover
# which storms actually exist without already knowing their numbers in
# advance (storm numbering restarts each season/basin, so there's no way
# to guess valid numbers -- they have to be listed).
ALL_BASINS = ("AL", "SL", "EP", "CP", "WP", "IO", "SH")


def list_available_storms(
    season_start: int,
    season_end: Optional[int] = None,
    basins: tuple = ALL_BASINS,
    version: str = "v01r01",
    version_type: str = "final",
    progress_callback=None,
) -> list:
    """Enumerate every (season, basin, storm_num) combination that
    actually exists in the bucket from season_start through season_end
    (inclusive; defaults to the current year if not given). Uses S3's
    Delimiter='/' listing mode to discover subdirectory names directly
    (the "number" folders under each season/basin) -- this lists
    directory STRUCTURE only, not file contents, so it's cheap even
    across many seasons/basins: no file metadata, no downloads, just
    "what storm numbers exist here."

    Returns a list of dicts: {"season": int, "basin": str, "storm_num": int}.
    A season/basin with nothing in the bucket contributes nothing (not
    an error) -- most basin/season combinations for less-active basins
    will have few or no storms, which is expected.
    """
    if season_end is None:
        season_end = datetime.now().year

    s3 = _get_s3_client()
    storms = []

    for season in range(season_start, season_end + 1):
        for basin in basins:
            prefix = f"{version}/{version_type}/{season}/{basin}/"
            if progress_callback:
                progress_callback(f"TC-PRIMED: checking {prefix}...")
            try:
                resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
            except ClientError:
                continue
            for cp in resp.get("CommonPrefixes", []):
                # cp["Prefix"] looks like "v01r01/final/2018/AL/06/"
                number_str = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                try:
                    storm_num = int(number_str)
                except ValueError:
                    continue
                storms.append({"season": season, "basin": basin, "storm_num": storm_num})

    return storms


def estimate_download_size(
    storms: list,
    instruments: tuple = ("GMI", "AMSR2"),
    version: str = "v01r01",
    version_type: str = "final",
    progress_callback=None,
) -> dict:
    """For a list of storms (as returned by list_available_storms), sum
    up the actual S3 object sizes for the given instruments' overpass
    files -- WITHOUT downloading anything. list_objects_v2 already
    returns each object's size in its response metadata, so this is just
    listing plus arithmetic, not a bulk transfer.

    Returns {"total_bytes": int, "n_files": int, "n_storms_with_data": int,
    "per_storm": {storm_key: bytes, ...}}. A storm contributing 0 files
    (e.g. only had SSMIS/ATMS passes, no GMI/AMSR2) is included in
    per_storm with 0 bytes, not silently dropped, so a caller can see
    exactly which storms had usable data and which didn't.
    """
    s3 = _get_s3_client()
    total_bytes = 0
    n_files = 0
    n_storms_with_data = 0
    per_storm = {}

    for storm in storms:
        season, basin, storm_num = storm["season"], storm["basin"], storm["storm_num"]
        storm_key = f"{basin}{storm_num:02d}{season}"
        prefix = f"{version}/{version_type}/{season}/{basin}/{storm_num:02d}/"
        if progress_callback:
            progress_callback(f"TC-PRIMED: sizing {storm_key}...")

        storm_bytes = 0
        continuation_token = None
        while True:
            kwargs = dict(Bucket=BUCKET, Prefix=prefix)
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            try:
                resp = s3.list_objects_v2(**kwargs)
            except ClientError:
                break

            for obj in resp.get("Contents", []):
                filename = os.path.basename(obj["Key"])
                parsed = _parse_overpass_filename(filename)
                if parsed is None or parsed["instrument"] not in instruments:
                    continue
                storm_bytes += obj["Size"]
                n_files += 1

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

        per_storm[storm_key] = storm_bytes
        total_bytes += storm_bytes
        if storm_bytes > 0:
            n_storms_with_data += 1

    return {
        "total_bytes": total_bytes,
        "n_files": n_files,
        "n_storms_with_data": n_storms_with_data,
        "per_storm": per_storm,
    }


def download_storms(
    storms: list,
    instruments: tuple = ("GMI", "AMSR2"),
    local_dir: str = DEFAULT_LOCAL_DIR,
    progress_callback=None,
) -> dict:
    """Actually download every GMI/AMSR2 overpass file for the given
    storms (same `storms` list shape as estimate_download_size and
    list_available_storms, so the typical flow is: list_available_storms
    -> show the user estimate_download_size's result for confirmation ->
    if confirmed, call this with the SAME storms list). Intended to run
    on a background thread, not the GUI thread directly -- downloading
    a meaningful date range can take a long time. A file that fails to
    download is skipped (logged via progress_callback) rather than
    aborting the whole run -- one bad file shouldn't lose everything
    else that would otherwise succeed.

    Returns the same shape as estimate_download_size, reporting what was
    ACTUALLY downloaded.
    """
    total_bytes = 0
    n_files = 0
    n_storms_with_data = 0
    per_storm = {}

    for storm in storms:
        season, basin, storm_num = storm["season"], storm["basin"], storm["storm_num"]
        storm_key = f"{basin}{storm_num:02d}{season}"
        storm_bytes = 0

        for instrument in instruments:
            try:
                files = list_storm_overpass_files(basin, storm_num, season, instrument_filter=instrument)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"TC-PRIMED: could not list {storm_key} {instrument} ({type(e).__name__}: {e})")
                continue
            for f in files:
                try:
                    download_overpass_file(f["key"], local_dir=local_dir)
                    storm_bytes += f.get("size_bytes", 0)
                    n_files += 1
                    if progress_callback:
                        progress_callback(f"Downloaded {f['filename']} ({format_bytes(f.get('size_bytes', 0))})")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Failed: {f['filename']} ({type(e).__name__}: {e})")

        per_storm[storm_key] = storm_bytes
        total_bytes += storm_bytes
        if storm_bytes > 0:
            n_storms_with_data += 1

    return {
        "total_bytes": total_bytes,
        "n_files": n_files,
        "n_storms_with_data": n_storms_with_data,
        "per_storm": per_storm,
    }


def format_bytes(n: int) -> str:
    """Human-readable size string -- KB/MB/GB, whichever is most
    legible for the magnitude, not a fixed unit regardless of size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

