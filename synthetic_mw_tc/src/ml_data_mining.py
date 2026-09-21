"""
Batch-mine historical (GOES input, GMI/AMSR2 target) training examples
across many past storms, instead of relying only on organic accumulation
through normal day-to-day app use (training_data_export.py's other entry
point) -- which would take far too long to build a useful dataset on its
own.

SCOPE, given real constraints (limited local storage, a single 6GB-VRAM
laptop GPU, this being a first pass at the ML pivot): this is deliberately
NOT "download a decade of everything." It processes one basin/season (or
an explicit list of storms) at a time, at a configurable time step, and
skips (doesn't error on) any storm-time where GOES and/or GMI/AMSR2
coverage isn't available -- most storm-times won't have both, and that's
expected, not a bug. Run it in controlled batches (a season at a time,
say) rather than pointing it at "2012-2026" in one call, both to keep
storage bounded and so a network hiccup partway through doesn't waste
hours of already-mined progress -- each successfully mined example is
written to disk immediately (via training_data_export.export_training_example),
so a partial/interrupted run still keeps everything it mined so far.

HONEST CAVEAT, consistent with every other network-touching module in
this project: none of this has been run against live GOES/GMI/AMSR2
archives from this sandbox (no network access here) -- it's built
directly on top of already-confirmed-working pieces (goes_fetch.get_band_image,
besttrack.fetch_best_track, mw_ingest.fetch_gmi_swath/fetch_amsr2_swath,
synthetic_algorithm.generate_synthetic_mw, training_data_export.export_training_example),
reusing their existing, real error handling rather than adding new
untested network logic -- but the orchestration loop itself needs a real
run to confirm before you trust it at scale.
"""
from __future__ import annotations

import os
import time

import threading

# Per-phase wall-clock totals across saved examples, so a run REPORTS
# where its time went instead of leaving it to be guessed at. Guessing
# has been wrong here before: the block size was tuned on a simulated
# access pattern and the real figure was 93%, not 25%.
_phase_totals = {}

# Attempts to allow before judging a run to be systematically broken.
# Large enough to cross several storms (so a single bad storm cannot trip
# it) and small enough to cost minutes rather than hours.
ABORT_CHECK_AFTER = 150

# Share of failures that must come from one reason for that judgement.
ABORT_DOMINANCE = 0.95

# How often to print a live progress line.
PROGRESS_EVERY = 50
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

import besttrack
import glm_lightning
import goes_fetch
import mw_ingest
import solar
from synthetic_algorithm import generate_synthetic_mw
import training_data_export as tde


def mine_storm(
    basin: str,
    storm_num: int,
    year: int,
    satellite: str = "GOES-19",
    time_step_hours: float = 3.0,
    gmi_credentials: Optional[dict] = None,
    amsr2_credentials: Optional[dict] = None,
    output_dir: str = tde.DEFAULT_EXPORT_DIR,
    progress_callback=None,
    sleep_between_attempts: float = 0.0,
) -> dict:
    """Mine training examples across one storm's full best-track lifetime.

    For each time step, tries GMI first, then AMSR2 (both allowed
    targets -- see training_data_export.py for why only these two), using
    whichever credentials dict is actually provided (pass only the one(s)
    you have registered access for; the other is silently skipped, not
    an error). Skips (doesn't raise) any step where GOES imagery isn't
    available, where neither MW sensor has a pass near that time/location,
    or where any other exception occurs -- logs it via progress_callback
    and moves to the next time step, so one bad step doesn't abort an
    entire storm's mining run.

    gmi_credentials / amsr2_credentials: {"username":.., "password":..}
        dicts, same shape as this project's credentials.py storage.
        Passing neither means this function will find nothing (both
        sensors skipped) -- not an error, just an empty result.

    sleep_between_attempts: optional politeness delay between network-
        touching steps, in seconds -- 0 by default (no delay), set this
        if you're mining many storms back-to-back and want to avoid
        hammering the archive services.

    Returns a summary dict: {"attempted": int, "saved": int,
    "skipped_reasons": {reason: count}, "saved_paths": [str, ...]}.
    """
    fixes = besttrack.fetch_best_track(basin, storm_num, year)
    if not fixes:
        return {"attempted": 0, "saved": 0, "skipped_reasons": {"no_best_track": 1}, "saved_paths": []}

    start_time = fixes[0].valid_time
    end_time = fixes[-1].valid_time
    step = timedelta(hours=time_step_hours)

    attempted = 0
    saved = 0
    skipped_reasons: dict = {}
    saved_paths = []

    def _skip(reason):
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    t = start_time
    while t <= end_time:
        attempted += 1
        if progress_callback:
            progress_callback(f"[{basin}{storm_num:02d}{year} {t:%Y-%m-%d %H:%M}] mining...")

        try:
            fix = besttrack.interpolate_fix(fixes, t)
            if fix is None:
                _skip("no_interpolated_fix")
                t += step
                continue

            # Fetched CONCURRENTLY. Each of these may be a full-disk crop
            # (a LIST plus several ranged GETs on a ~30 MB object), and at
            # ~36 s per saved example against a ~4 s CPU floor the loop is
            # bound by serial round trips rather than computation.
            # Base AND supplementary bands in ONE pool. These were two
            # pools of three run back to back, so a frame paid two
            # round-trip waits where one would do.
            # Band 2 joins the pool when it is daytime, rather than being
            # a fourth serial round trip after the other three.
            _want = (13, 9, 7, 2) if solar.is_daytime(fix.lat, fix.lon, t) else (13, 9, 7)
            _base, extra_ir = goes_fetch.fetch_frame_bands(
                satellite, t, fix.lat, fix.lon, base_bands=_want,
                progress_callback=progress_callback)
            band13, band9, band7 = _base[13], _base[9], _base[7]
            band2 = _base.get(2)
            if band13 is None or band9 is None or band7 is None:
                _skip("no_goes_imagery")
                t += step
                continue

            real_swath = None
            if gmi_credentials and gmi_credentials.get("username"):
                try:
                    real_swath, _lb, hit = mw_ingest.find_swath_that_hit_storm(
                        "GMI", fixes, t, fix.lat, fix.lon,
                        gmi_credentials["username"], gmi_credentials.get("password", ""),
                    )
                    if not hit:
                        real_swath = None
                except Exception:
                    real_swath = None

            if real_swath is None and amsr2_credentials and amsr2_credentials.get("username"):
                try:
                    real_swath, _lb, hit = mw_ingest.find_swath_that_hit_storm(
                        "AMSR2", fixes, t, fix.lat, fix.lon,
                        amsr2_credentials["username"], amsr2_credentials.get("password", ""),
                    )
                    if not hit:
                        real_swath = None
                except Exception:
                    real_swath = None

            if real_swath is None:
                _skip("no_gmi_or_amsr2_pass")
                t += step
                continue

            if real_swath.scene_time != t:
                real_swath = mw_ingest.morph_swath_to_time(real_swath, fixes, t)

            # Supplementary IR bands and lightning.
            #
            # Mining previously called generate WITHOUT extra_ir, so all
            # seven extra channels were neutral in EVERY training
            # example -- the multi-channel IR work from 0.98/0.99 never
            # reached the data, and Li et al. found that ablation to be
            # their single largest gain. The bands were only ever being
            # fetched on the GUI path.
            flash = glm_lightning.flash_density_grid(
                band13.lat, band13.lon, satellite, t)
            result = generate_synthetic_mw(
                band13, band9, band7, fix, band2=band2,
                real_swath=real_swath, extra_ir=extra_ir, flash_density=flash)
            path = tde.export_training_example(band13, band9, band7, band2, fix, result, output_dir=output_dir)
            if path:
                saved += 1
                saved_paths.append(path)
                if progress_callback:
                    progress_callback(f"  saved: {path}")
            else:
                _skip("export_returned_none")

        except Exception as e:
            _skip(f"exception: {type(e).__name__}: {str(e)[:80]}")
            if progress_callback:
                progress_callback(f"  skipped ({type(e).__name__}: {e})")

        if sleep_between_attempts > 0:
            time.sleep(sleep_between_attempts)
        t += step

    return {"attempted": attempted, "saved": saved, "skipped_reasons": skipped_reasons, "saved_paths": saved_paths}


def mine_storm_via_tcprimed(
    basin: str,
    storm_num: int,
    year: int,
    satellite: str = "GOES-19",
    instruments: tuple = ("GMI", "AMSR2"),
    output_dir: str = tde.DEFAULT_EXPORT_DIR,
    progress_callback=None,
) -> dict:
    """Mine training examples for one storm using TC-PRIMED
    (tcprimed_ingest.py) as the real-MW source, instead of the live NRT/
    archive search path mine_storm() above uses.

    WHY THIS IS THE PREFERRED PATH FOR HISTORICAL MINING: TC-PRIMED's
    own curation already validates that each overpass file it provides
    genuinely covers the storm (an areal coverage fraction check within
    750km, falling back to 250km -- see tcprimed_ingest.py's module
    docstring), which is exactly the class of problem (a file matching
    by TIME but not actually covering the storm geographically) that
    caused a long chain of real, confirmed bugs in mw_ingest.py's live
    NRT search. Using TC-PRIMED for historical mining sidesteps that
    whole failure mode instead of needing to work around it.

    Still fetches GOES imagery ourselves (goes_fetch, same as
    mine_storm() above) rather than using TC-PRIMED's own bundled
    infrared data -- TC-PRIMED's IR is a single "clean window" channel
    from a separate curated archive (TC IRAR/HURSAT), not raw GOES ABI,
    so it doesn't include the WV/SWIR channels this project's model
    actually uses as input. Keeping the input side consistent with what
    the operational algorithm sees at inference time matters more than
    convenience here.

    instruments: which TC-PRIMED instruments to pull -- defaults to both
        GMI and AMSR2 (the two this project trains on; see
        training_data_export.py for why not the others). Each overpass
        found is treated as an independent training example, same as
        mine_storm() -- no attempt to deduplicate near-simultaneous
        passes from different instruments, since each is still a
        genuine, independent (GOES input, real MW target) pair.

    Returns the same summary shape as mine_storm().
    """
    import tcprimed_ingest as tp

    fixes = besttrack.fetch_best_track(basin, storm_num, year)
    if not fixes:
        return {"attempted": 0, "saved": 0, "skipped_reasons": {"no_best_track": 1}, "saved_paths": []}

    attempted = 0
    saved = 0
    skipped_reasons: dict = {}
    saved_paths = []

    def _skip(reason):
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    for instrument in instruments:
        try:
            swaths = tp.fetch_storm_swaths(basin, storm_num, year, instrument=instrument, progress_callback=progress_callback)
        except Exception as e:
            _skip(f"tcprimed_fetch_failed: {type(e).__name__}")
            continue

        for real_swath in swaths:
            attempted += 1
            t = real_swath.scene_time
            try:
                fix = besttrack.interpolate_fix(fixes, t)
                if fix is None:
                    _skip("no_interpolated_fix")
                    continue

                # Fetched CONCURRENTLY. Each of these may be a full-disk crop
                # (a LIST plus several ranged GETs on a ~30 MB object), and at
                # ~36 s per saved example against a ~4 s CPU floor the loop is
                # bound by serial round trips rather than computation.
                # Base AND supplementary bands in ONE pool. These were two
                # pools of three run back to back, so a frame paid two
                # round-trip waits where one would do.
                _want = ((13, 9, 7, 2) if solar.is_daytime(fix.lat, fix.lon, t)
                         else (13, 9, 7))
                _base, extra_ir = goes_fetch.fetch_frame_bands(
                    satellite, t, fix.lat, fix.lon, base_bands=_want,
                    progress_callback=progress_callback)
                band13, band9, band7 = _base[13], _base[9], _base[7]
                band2 = _base.get(2)
                if band13 is None or band9 is None or band7 is None:
                    _skip("no_goes_imagery")
                    continue

                # Validate the fetched image actually contains the storm --
                # goes_fetch matches on TIME ONLY (see _goes_covers_storm).
                # Without this, a WP/IO/SH storm gets paired with whatever
                # sector happened to be scanning, anywhere on Earth.
                if not _goes_covers_storm(band13, fix.lat, fix.lon):
                    _skip("goes_does_not_cover_storm")
                    continue


                # Supplementary IR bands and lightning.
                #
                # Mining previously called generate WITHOUT extra_ir, so all
                # seven extra channels were neutral in EVERY training
                # example -- the multi-channel IR work from 0.98/0.99 never
                # reached the data, and Li et al. found that ablation to be
                # their single largest gain. The bands were only ever being
                # fetched on the GUI path.
                _phase["goes"] = time.time() - _tg
                _te = time.time()
                flash = glm_lightning.flash_density_grid(
                    band13.lat, band13.lon, satellite, t)
                _phase["extra_glm"] = time.time() - _te
                _tgen = time.time()
                result = generate_synthetic_mw(
                    band13, band9, band7, fix, band2=band2,
                    real_swath=real_swath, extra_ir=extra_ir, flash_density=flash)
                _phase["generate"] = time.time() - _tgen
                _tx = time.time()
                path = tde.export_training_example(band13, band9, band7, band2, fix, result, output_dir=output_dir)
                if path:
                    saved += 1
                    saved_paths.append(path)
                    if progress_callback:
                        progress_callback(f"  saved: {path}")
                else:
                    _skip("export_returned_none")

            except Exception as e:
                _skip(f"exception: {type(e).__name__}: {str(e)[:80]}")
                if progress_callback:
                    progress_callback(f"  skipped ({type(e).__name__}: {e})")

    return {"attempted": attempted, "saved": saved, "skipped_reasons": skipped_reasons, "saved_paths": saved_paths}


def _goes_covers_storm(band_image, storm_lat: float, storm_lon: float, margin_deg: float = 1.0) -> bool:
    """Check that a fetched GOES image ACTUALLY contains the storm before
    using it.

    This exists because goes_fetch.find_nearest_file selects purely by
    TIME -- it returns whichever mesoscale sector was scanning closest to
    the requested timestamp, with no check of where on Earth that sector
    was pointed. For an Atlantic/EPac storm that's usually fine. For a
    WP/IO/SH storm it is not: GOES physically cannot see the Indian Ocean
    or most of the Western Pacific, so the "nearest in time" file is a
    sector over a completely different part of the planet.

    Without this check, such a file is still returned, still regridded,
    and still exported as a training example -- pairing a real MW target
    over (say) the Bay of Bengal with GOES imagery of the Atlantic. That
    is actively poisoning training data, not merely wasting time. It is
    the same failure mode as the MW geographic-coverage miss found much
    earlier in this project: matching on time without validating
    geography.

    margin_deg requires the storm to sit slightly inside the image edge
    rather than exactly on it, since a storm right at the boundary gives
    almost no usable surrounding structure.
    """
    try:
        lat = band_image.lat
        lon = band_image.lon
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    except Exception:
        return False

    return (
        (lat_min + margin_deg) <= storm_lat <= (lat_max - margin_deg)
        and (lon_min + margin_deg) <= storm_lon <= (lon_max - margin_deg)
    )


_log_lock = threading.Lock()


def mine_local_tcprimed_cache(
    cache_dir: str = None,
    satellite: str = None,   # None = auto-select per scene (see select_satellite)
    output_dir: str = tde.DEFAULT_EXPORT_DIR,
    progress_callback=None,
    max_workers: int = 10,
    max_per_storm: Optional[int] = None,
    exclude_basins: tuple = ("WP", "IO", "SH"),
    stream: bool = False,
    seasons: tuple = (),
    basins: tuple = (),
) -> dict:
    """Process whatever GMI/AMSR2 overpass files are ALREADY sitting in
    the local TC-PRIMED cache (from the GUI's download button, or a
    prior mine_storm_via_tcprimed run) into paired training examples --
    without re-querying S3 at all. This is the missing link between
    "downloaded raw .nc files" and "actual training data": the download
    button only caches files; it doesn't pair them with GOES imagery or
    run them through the parametric algorithm. This function does that
    second step, using only what's already on disk.

    Parses each cached filename directly (tcprimed_ingest._parse_overpass_filename,
    which already extracts basin/storm_num/season/instrument/timestamp
    from the filename alone -- no network call needed to know what a
    locally cached file is), groups by storm so each storm's best-track
    is fetched only once even if it has many cached passes, then for
    each file: reads it, fetches GOES imagery at that observation time,
    runs it through the parametric algorithm, and exports the training
    example (subject to training_data_export.py's own GMI/AMSR2 filter
    and NaN/coverage checks).

    A file that fails at any step (no GOES coverage at that time, a
    corrupt/unreadable cache file, no best-track available) is skipped
    and logged, not fatal to the rest -- this is expected to process
    potentially hundreds of files from a bulk download, and losing a
    few shouldn't lose the run.

    Returns {"attempted": int, "saved": int, "skipped_reasons": {reason: count},
    "saved_paths": [...], "storms_processed": [...]}.
    """
    import tcprimed_ingest as tp

    if cache_dir is None:
        cache_dir = tp.DEFAULT_LOCAL_DIR

    # STREAMING MODE. Instead of walking a local cache, list the overpass
    # files on S3 and read each one in place (s3_range_reader), writing
    # nothing to disk. Everything after the read -- best-track pairing,
    # GOES fetch, the parametric algorithm, export -- is identical, which
    # is the point: the only thing that changes is where the bytes come
    # from, so there is no second pipeline to keep in sync.
    stream_keys: dict = {}
    if stream:
        if not seasons:
            raise ValueError("stream=True requires seasons, e.g. seasons=(2020, 2021)")
        for season in seasons:
            for storm in tp.list_available_storms(season_start=season, season_end=season,
                                                  basins=basins or None):
                if storm["basin"] in exclude_basins:
                    continue
                for f in tp.list_storm_overpass_files(
                    basin=storm["basin"], storm_num=storm["storm_num"],
                    season=storm["season"],
                ):
                    # TC-PRIMED carries the whole GPM constellation;
                    # read_overpass_as_swath handles GMI and AMSR2 only.
                    # Filtering here rather than failing later saves a
                    # ranged GET on every file that could never be used.
                    if f.get("instrument") not in tp.SUPPORTED_INSTRUMENTS:
                        continue
                    stream_keys[os.path.basename(f["key"])] = f
        if progress_callback:
            progress_callback(f"Streaming mode: {len(stream_keys)} overpass file(s) "
                              f"found on S3 across season(s) {list(seasons)}")

    if not stream and not os.path.isdir(cache_dir):
        return {"attempted": 0, "saved": 0, "skipped_reasons": {"cache_dir_not_found": 1}, "saved_paths": [], "storms_processed": []}

    # Parse every cached filename up front, group by (basin, storm_num, season)
    # so best-track is fetched once per storm, not once per file.
    by_storm: dict = {}
    for fname in sorted(stream_keys if stream else os.listdir(cache_dir)):
        if not fname.endswith(".nc"):
            continue
        parsed = tp._parse_overpass_filename(fname)
        if parsed is None:
            continue
        key = (parsed["basin"], parsed["storm_num"], parsed["season"])
        by_storm.setdefault(key, []).append((fname, parsed))

    attempted = 0
    saved = 0
    skipped_reasons: dict = {}
    saved_paths = []
    storms_processed = []
    counters = {"saved": 0}
    _counter_lock = threading.Lock()

    # --- Fail-fast (0.117) --------------------------------------------
    # Two mining runs were lost to a single systematic error: 4,867
    # attempted / 0 saved, then 613 / 0 saved, both grinding to completion
    # while failing identically every time. Nothing stopped them, and the
    # summary only arrived at the end.
    #
    # If the first ABORT_CHECK_AFTER attempts produce nothing and are
    # dominated by ONE reason, the run is not going to recover -- that is
    # a broken pipeline, not an unlucky stretch of storms. Stop and say
    # so, rather than spending hours proving it.
    abort_state = {"stopped": False, "reason": None}

    def _should_abort():
        """True once the evidence says this run cannot succeed."""
        if abort_state["stopped"]:
            return True
        with _counter_lock:
            if counters["saved"] > 0:
                return False            # it works; never abort after a save
            total = sum(skipped_reasons.values())
            if total < ABORT_CHECK_AFTER:
                return False
            top, n = max(skipped_reasons.items(), key=lambda kv: kv[1])
            if n / total < ABORT_DOMINANCE:
                return False            # varied reasons = genuinely unlucky
            # Legitimate skips are expected to dominate early runs, so
            # only ERRORS trigger the abort. "no GOES coverage" for 200
            # storm-times in a row is normal; one exception repeated 200
            # times is not.
            if not top.startswith(("exception:", "no_best_track:",
                                   "tcprimed_fetch_failed:")):
                return False
            abort_state["stopped"] = True
            abort_state["reason"] = f"{top} ({n} of {total} attempts)"
            return True

    def _skip(reason):
        with _counter_lock:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            done = sum(skipped_reasons.values()) + counters["saved"]
        # Live progress. The previous runs printed nothing until the end,
        # so there was no way to tell a broken run from a slow one.
        if done % PROGRESS_EVERY == 0:
            _log(f"  ... {done} attempted, {counters['saved']} saved; "
                 f"most common: {max(skipped_reasons.items(), key=lambda kv: kv[1])[0]}")

    def _log(msg):
        # progress_callback is usually print(); serialise it so concurrent
        # workers don't interleave half-lines into the same output.
        if progress_callback:
            with _log_lock:
                progress_callback(msg)

    for (basin, storm_num, season), file_list in by_storm.items():
        storm_key = f"{basin}{storm_num:02d}{season}"

        # Stop early if this run is systematically failing rather than
        # merely finding unsuitable storms.
        if _should_abort():
            _log("")
            _log("ABORTING: " + abort_state["reason"])
            _log("  Nothing has saved and one failure dominates, so this is a")
            _log("  broken pipeline rather than an unlucky run. Fix the error")
            _log("  above and re-run; no point spending hours confirming it.")
            break

        # Basin filter FIRST, before any best-track fetch. WP/IO/SH are
        # never within any GOES satellite's field of view, so every one
        # of these storms was going to be discarded anyway -- but the
        # discard previously happened only AFTER fetching best-track,
        # which for these basins means an IBTrACS lookup. That was the
        # dominant cost of a real 4-hour run. Filtering here skips the
        # lookup entirely.
        if exclude_basins and basin.upper() in {b.upper() for b in exclude_basins}:
            _skip(f"excluded_basin_{basin.upper()}")
            _log(f"  {storm_key}: basin {basin.upper()} excluded -- skipping "
                 f"{len(file_list)} file(s) without any lookup.")
            continue
        if progress_callback:
            progress_callback(f"=== Processing cached files for {storm_key} ({len(file_list)} file(s)) ===")

        try:
            fixes = besttrack.fetch_best_track(basin, storm_num, season)
        except Exception as e:
            _skip(f"no_best_track: {type(e).__name__}")
            continue
        if not fixes:
            _skip("no_best_track")
            continue

        # Storm-level field-of-view check, BEFORE opening a single file.
        # The per-file check further down runs only after
        # read_overpass_as_swath() has already decompressed and read that
        # file's channel arrays -- for a basin the satellite cannot see at
        # all, that is ~25MB of disk read per file, times every file in
        # the storm, for a result that was never going to be usable. If no
        # best-track position for this storm is within view, skip the
        # whole storm and never touch its files.
        if not any(goes_fetch.select_satellite(fx.lat, fx.lon, fx.valid_time) for fx in fixes):
            _skip("storm_outside_satellite_field_of_view")
            if progress_callback:
                progress_callback(
                    f"  {storm_key}: no GOES satellite covers this track at this date "
                    f"-- skipping all {len(file_list)} file(s) without reading them."
                )
            continue

        storms_processed.append(storm_key)

        # Per-file work is dominated by GOES S3 downloads (network I/O),
        # not CPU, so threads genuinely help here despite the GIL -- both
        # boto3 and the HDF5/numpy reads release it while waiting. Serial
        # downloading leaves most of a fast connection idle; several
        # concurrent transfers actually use it. max_workers is
        # deliberately modest (6) rather than very high: past ~8 the
        # bottleneck shifts to S3 throttling and local CPU for
        # generate_synthetic_mw, and more workers just add contention.
        def _process_one(item):
            fname, parsed = item
            _t0 = time.time()
            _tm = _t0          # defined for the non-streaming path too
            _phase = {}
            try:
                if stream:
                    meta = stream_keys[fname]
                    _tm = time.time()
                    real_swath = tp.read_overpass_streaming(
                        meta["key"], parsed["instrument"],
                        size=meta.get("size_bytes"), progress_callback=_log)
                else:
                    real_swath = tp.read_overpass_as_swath(
                        os.path.join(cache_dir, fname), parsed["instrument"])
            except Exception as e:
                _skip(f"unreadable_cache_file: {type(e).__name__}")
                _log(f"  skipped {fname} (unreadable: {type(e).__name__}: {e})")
                return

            t = real_swath.scene_time
            try:
                fix = besttrack.interpolate_fix(fixes, t)
                if fix is None:
                    _skip("no_interpolated_fix")
                    return

                # Pick the satellite that was actually operational at this
                # scene's time AND can see this position. Hardcoding one
                # name silently produced 2025-only training data, since
                # GOES-19 did not exist as GOES-East before April 2025.
                # `satellite` (if explicitly passed) acts as an override.
                sat = satellite or goes_fetch.select_satellite(fix.lat, fix.lon, t)
                if sat is None:
                    _skip("no_goes_satellite_for_this_time_and_place")
                    return

                # Fetch band 13 ALONE first, then check coverage, and only
                # fetch bands 9 and 7 if the scene actually contains the
                # storm. Previously all three were downloaded before the
                # check ran -- and coverage rejection is by far the most
                # common outcome (1270 of ~1739 attempts in a real run), so
                # two thirds of the download traffic on the dominant code
                # path was being thrown away. goes_fetch matches on TIME
                # ONLY, so this check cannot be skipped, only moved earlier.
                # Mesoscale first, full-disk crop as fallback. RadM alone discarded
                # 83% of storm-times as "no covering sector" -- the two meso
                # boxes are steerable and usually aimed elsewhere.
                _phase["mw_read"] = time.time() - _tm
                _tg = time.time()
                band13 = goes_fetch.get_band_image_any_sector(
                    sat, 13, t, fix.lat, fix.lon, progress_callback=progress_callback)
                if band13 is None:
                    _skip("no_goes_imagery")
                    return

                if not _goes_covers_storm(band13, fix.lat, fix.lon):
                    _skip("goes_does_not_cover_storm")
                    return

                # Everything else for this frame in ONE pool.
                #
                # Band 13 stays alone and first because the coverage check
                # above gates the rest -- that ordering is deliberate and
                # worth keeping. But 9, 7, 2 and the supplementary bands
                # were then fetched ONE AT A TIME, each a separate
                # round-trip and possibly a full-disk crop, where they are
                # entirely independent of each other.
                _rest = (9, 7, 2) if solar.is_daytime(fix.lat, fix.lon, t) else (9, 7)
                _got, extra_ir = goes_fetch.fetch_frame_bands(
                    sat, t, fix.lat, fix.lon, base_bands=_rest)
                band9, band7 = _got.get(9), _got.get(7)
                band2 = _got.get(2)
                if band9 is None or band7 is None:
                    _skip("no_goes_imagery")
                    return

                # Supplementary IR bands and lightning.
                #
                # Mining previously called generate WITHOUT extra_ir, so all
                # seven extra channels were neutral in EVERY training
                # example -- the multi-channel IR work from 0.98/0.99 never
                # reached the data, and Li et al. found that ablation to be
                # their single largest gain. The bands were only ever being
                # fetched on the GUI path.
                _phase["goes"] = time.time() - _tg
                _te = time.time()
                flash = glm_lightning.flash_density_grid(
                    band13.lat, band13.lon, sat, t,
                        progress_callback=progress_callback)
                _phase["extra_glm"] = time.time() - _te
                _tgen = time.time()
                result = generate_synthetic_mw(
                    band13, band9, band7, fix, band2=band2,
                    real_swath=real_swath, extra_ir=extra_ir, flash_density=flash)
                _phase["generate"] = time.time() - _tgen
                _tx = time.time()
                path = tde.export_training_example(band13, band9, band7, band2, fix, result, output_dir=output_dir)
                if path:
                    with _counter_lock:
                        counters["saved"] += 1
                        saved_paths.append(path)
                    _phase["export"] = time.time() - _tx
                    with _counter_lock:
                        for _k, _v in _phase.items():
                            _phase_totals[_k] = _phase_totals.get(_k, 0.0) + _v
                        _phase_totals["_n"] = _phase_totals.get("_n", 0) + 1
                    _log(f"  saved: {os.path.basename(path)}")
                else:
                    _skip("export_returned_none")

            except Exception as e:
                _skip(f"exception: {type(e).__name__}: {str(e)[:80]}")
                _log(f"  skipped {fname} ({type(e).__name__}: {e})")

        # Cap examples per storm, if asked.
        #
        # A storm contributes many overpasses hours apart, and they are
        # highly correlated -- same storm, same basin, similar structure.
        # Fifteen frames of one hurricane add far less than fifteen frames
        # of fifteen different ones, while costing the same time. Spread
        # evenly across the storm's lifetime rather than taking the first
        # N, so intensification and decay both stay represented.
        if max_per_storm and len(file_list) > max_per_storm:
            step = len(file_list) / float(max_per_storm)
            file_list = [file_list[int(i * step)] for i in range(max_per_storm)]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_process_one, file_list))
        with _counter_lock:
            attempted += len(file_list)

    saved = counters["saved"]
    return {
        "attempted": attempted, "saved": saved, "skipped_reasons": skipped_reasons,
        "saved_paths": saved_paths, "storms_processed": storms_processed,
        "aborted": abort_state["stopped"], "abort_reason": abort_state["reason"],
    }


def mine_storm_list(
    storms: list,
    satellite: str = "GOES-19",
    time_step_hours: float = 3.0,
    gmi_credentials: Optional[dict] = None,
    amsr2_credentials: Optional[dict] = None,
    output_dir: str = tde.DEFAULT_EXPORT_DIR,
    progress_callback=None,
) -> dict:
    """Mine a list of storms in sequence: storms = [(basin, storm_num, year), ...].
    Returns a combined summary across all of them, plus a per-storm
    breakdown -- useful for running one whole season (e.g., every
    Atlantic storm in a given year) in a single call, while still being
    able to see which specific storms actually contributed data.
    """
    _phase_totals.clear()
    combined = {"attempted": 0, "saved": 0, "skipped_reasons": {}, "saved_paths": [], "per_storm": {}}
    for basin, storm_num, year in storms:
        storm_key = f"{basin}{storm_num:02d}{year}"
        if progress_callback:
            progress_callback(f"=== Mining {storm_key} ===")
        result = mine_storm(
            basin, storm_num, year, satellite=satellite, time_step_hours=time_step_hours,
            gmi_credentials=gmi_credentials, amsr2_credentials=amsr2_credentials,
            output_dir=output_dir, progress_callback=progress_callback,
        )
        combined["attempted"] += result["attempted"]
        combined["saved"] += result["saved"]
        combined["saved_paths"].extend(result["saved_paths"])
        for reason, count in result["skipped_reasons"].items():
            combined["skipped_reasons"][reason] = combined["skipped_reasons"].get(reason, 0) + count
        combined["per_storm"][storm_key] = {"attempted": result["attempted"], "saved": result["saved"]}

    return combined
