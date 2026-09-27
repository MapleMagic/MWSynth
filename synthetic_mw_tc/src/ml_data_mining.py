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
import tcprimed_env
import goes_fetch
import mw_ingest
import solar
from synthetic_algorithm import generate_synthetic_mw
import training_data_export as tde


# --- Memory-safe worker count (0.159) ----------------------------------
# Measured on a real frame (0.159): one process holds ~1.4 GB regardless
# of workers -- 933 MB of it is global_land_mask's global array, loaded
# once at import and shared by every thread -- and each CONCURRENT overpass
# adds ~0.18 GB in generation plus its band decodes. With the extra IR
# bands actually fetched (0.159) a frame decodes six bands at once, and a
# live 2-worker run peaked at 3.5 GB: (3.5 - 1.4) / 2 = 1.05 GB per worker.
# The first estimate (0.7) predated the extra bands and was too low.
# Through 0.158 the per-storm pool silently capped concurrency at
# --max-per-storm; the global pool removes that cap, and in a 4 GB sandbox
# four truly concurrent workers were OOM-killed. So the worker count is
# now checked against what the machine actually has.
BASE_PROCESS_GB = 1.4
PER_WORKER_GB = 1.05
MEMORY_BUDGET_FRACTION = 0.8


def available_memory_gb():
    """Physical memory currently available, in GB, or None if unknown.
    psutil if installed; otherwise /proc/meminfo (Linux) or
    GlobalMemoryStatusEx (Windows) -- no new dependency."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return ms.ullAvailPhys / 1e9
    except Exception:
        pass
    return None


def process_rss_gb():
    """This process's resident memory in GB, or 0.0 if unknown."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        pass
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1e9
    except Exception:
        pass
    return 0.0


def memory_safe_workers(requested: int, available_gb=None, log=None,
                        own_rss_gb=None, per_worker_gb=None, base_gb=None) -> int:
    """`requested`, capped so BASE + workers x PER_WORKER stays within
    MEMORY_BUDGET_FRACTION of available memory. Never below 1; unchanged
    if memory cannot be measured."""
    avail = available_memory_gb() if available_gb is None else available_gb
    if avail is None:
        return max(1, requested)
    # The base is partly loaded ALREADY (torch, the model) by the time this
    # runs, and "available" no longer includes it -- charging the full base
    # again double-counted it and chose 1 worker where 2 fit.
    per = PER_WORKER_GB if per_worker_gb is None else per_worker_gb
    base = BASE_PROCESS_GB if base_gb is None else base_gb
    own = process_rss_gb() if own_rss_gb is None else own_rss_gb
    still_needed = max(0.0, base - own)
    budget = avail * MEMORY_BUDGET_FRACTION
    fit = int((budget - still_needed) / per)
    safe = max(1, min(requested, fit))
    if log:
        peak = base + safe * per
        if safe < requested:
            log(f"Memory: {avail:.1f} GB available -> {safe} worker(s), not "
                f"{requested} (est. peak ~{peak:.1f} GB; each concurrent "
                f"overpass needs ~{per} GB)")
        else:
            log(f"Memory: {avail:.1f} GB available; {safe} worker(s) est. "
                f"peak ~{peak:.1f} GB")
    return safe


def _env_series(basin, storm_num, year):
    """Cached per process by tcprimed_env; None if unavailable."""
    try:
        return tcprimed_env.load_env_series(basin, storm_num, year)
    except Exception:
        return None


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
            # ml_strength=0.0: the NPZ stores the physics backbone (0.162).
            _env = tcprimed_env.env_at(_env_series(basin, storm_num, year), t)
            result = generate_synthetic_mw(
                band13, band9, band7, fix, band2=band2,
                real_swath=real_swath, extra_ir=extra_ir, flash_density=flash,
                ml_strength=0.0, env=_env)
            path = tde.export_training_example(band13, band9, band7, band2, fix, result,
                                               output_dir=output_dir, env=_env)
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
                # ml_strength=0.0: the NPZ stores the physics backbone (0.162).
                _env = tcprimed_env.env_at(_env_series(basin, storm_num, year), t)
                result = generate_synthetic_mw(
                    band13, band9, band7, fix, band2=band2,
                    real_swath=real_swath, extra_ir=extra_ir, flash_density=flash,
                    ml_strength=0.0, env=_env)
                _phase["generate"] = time.time() - _tgen
                _tx = time.time()
                path = tde.export_training_example(band13, band9, band7, band2, fix, result,
                                                   output_dir=output_dir, env=_env)
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


# --- Process-based mining (0.161) ----------------------------------------
#
# A real 1,237-example run (threads, ~6 in flight) timed every phase ~8x
# slower than the same phase run alone -- mw_read 8.6 s vs 0.8, goes 63 vs
# 6.8, generate 36 vs 5.0 -- INCLUDING generate, which touches no network.
# Throughput came out no better than one worker. Threads in one process
# share Python's GIL and h5py's global lock (every HDF5 call, from any
# thread, is serialised), and every phase is HDF5 or Python-heavy. Separate
# PROCESSES each have their own interpreter and their own HDF5.
#
# Everything a worker needs travels in its job tuple; everything the parent
# needs comes back in the record -- no shared state, so the same function
# runs in a thread (use_processes=False) or a process.
MINING_USE_PROCESSES = True
# Measured per worker PROCESS (0.161, spawn, one overpass at a time): see
# the CHANGELOG. Covers the interpreter + numpy/scipy/h5py/boto3 and one
# frame's peak; the land mask is a shared memory-map, torch is not loaded.
PER_PROCESS_GB = 1.35   # measured 1.33 GB peak per worker process (0.161)
LAST_RUN_STATS: dict = {}


def _mining_process_init():
    """Per worker process. Numeric libraries' own thread pools are held to
    one thread each: N processes x a pool per process oversubscribes the
    cores that the processes themselves are meant to fill."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def peak_rss_gb() -> float:
    """This process's peak resident memory, in GB (0.0 if unknown)."""
    try:
        import resource
        import sys as _sys
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1e9 if _sys.platform == "darwin" else v * 1024 / 1e9
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                    ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1e9
    except Exception:
        pass
    return 0.0


def _extra_band_snapshot():
    eb = goes_fetch.EXTRA_BAND_STATS
    return eb["requested"], dict(eb["missing"])


def _extra_band_delta(before):
    eb = goes_fetch.EXTRA_BAND_STATS
    req0, miss0 = before
    return {"requested": eb["requested"] - req0,
            "missing": {b: n - miss0.get(b, 0) for b, n in eb["missing"].items()
                        if n - miss0.get(b, 0)},
            "last_error": dict(eb["last_error"])}


def _merge_extra_band_stats(delta):
    """Fold a worker PROCESS's extra-band counts into this process's, so
    the end-of-mine summary counts every frame. (Threads share the dict
    already, so this is only called in process mode.)"""
    if not delta:
        return
    eb = goes_fetch.EXTRA_BAND_STATS
    eb["requested"] += delta.get("requested", 0)
    for b, n in delta.get("missing", {}).items():
        eb["missing"][b] = eb["missing"].get(b, 0) + n
    eb["last_error"].update(delta.get("last_error", {}))


def _mine_overpass(job) -> dict:
    """One overpass, start to finish. Returns a record; never raises."""
    import tcprimed_ingest as tp    # only ever imported locally in this module
    (fname, parsed, fixes, env_series, meta, cache_dir, output_dir,
     satellite, stream, in_process) = job
    logs: list = []
    rec = {"fname": fname, "status": "skip", "reason": None, "path": None,
           "phase": {}, "logs": logs, "cpu_s": 0.0, "peak_rss_gb": 0.0,
           "extra_band": None}
    cpu0 = time.process_time()
    eb0 = _extra_band_snapshot() if in_process else None
    ph = rec["phase"]
    try:
        _tm = time.time()
        try:
            if stream:
                real_swath = tp.read_overpass_streaming(
                    meta["key"], parsed["instrument"], size=meta.get("size_bytes"),
                    progress_callback=logs.append)
            else:
                real_swath = tp.read_overpass_as_swath(
                    os.path.join(cache_dir, fname), parsed["instrument"])
        except Exception as e:
            rec["reason"] = f"unreadable_cache_file: {type(e).__name__}"
            logs.append(f"  skipped {fname} (unreadable: {type(e).__name__}: {e})")
            return rec
        t = real_swath.scene_time
        fix = besttrack.interpolate_fix(fixes, t)
        if fix is None:
            rec["reason"] = "no_interpolated_fix"
            return rec
        sat = satellite or goes_fetch.select_satellite(fix.lat, fix.lon, t)
        if sat is None:
            rec["reason"] = "no_goes_satellite_for_this_time_and_place"
            return rec
        ph["mw_read"] = time.time() - _tm
        _tg = time.time()
        band13 = goes_fetch.get_band_image_any_sector(
            sat, 13, t, fix.lat, fix.lon, progress_callback=logs.append)
        if band13 is None:
            rec["reason"] = "no_goes_imagery"
            return rec
        if not _goes_covers_storm(band13, fix.lat, fix.lon):
            rec["reason"] = "goes_does_not_cover_storm"
            return rec
        _rest = (9, 7, 2) if solar.is_daytime(fix.lat, fix.lon, t) else (9, 7)
        # anchor=band13 (0.160): every band from band 13's own scan.
        _got, extra_ir = goes_fetch.fetch_frame_bands(
            sat, t, fix.lat, fix.lon, base_bands=_rest, anchor=band13,
            progress_callback=logs.append)
        band9, band7, band2 = _got.get(9), _got.get(7), _got.get(2)
        if band9 is None or band7 is None:
            rec["reason"] = "no_goes_imagery"
            return rec
        ph["goes"] = time.time() - _tg
        _te = time.time()
        flash = glm_lightning.flash_density_grid(
            band13.lat, band13.lon, sat, t, progress_callback=logs.append)
        ph["extra_glm"] = time.time() - _te
        _tgen = time.time()
        # ml_strength=0.0 (0.161): the NPZ stores the physics backbone with
        # the ML contribution subtracted back out, so running the model here
        # is pure cost. Until 0.160 fixed the checkpoint loader the model
        # never loaded, which hid that; now it would run a full diffusion
        # ensemble on every overpass and throw the answer away.
        # The environment goes INTO generation (0.168): the adopted physics
        # shifts emission and scattering downshear-left using its shear
        # heading. Computed only at export before, the stored backbone would
        # have lacked the asymmetry the replay (which passes it) includes.
        env = tcprimed_env.env_at(env_series, t, fname)
        result = generate_synthetic_mw(
            band13, band9, band7, fix, band2=band2, real_swath=real_swath,
            extra_ir=extra_ir, flash_density=flash, ml_strength=0.0, env=env)
        ph["generate"] = time.time() - _tgen
        _tx = time.time()
        path = tde.export_training_example(band13, band9, band7, band2, fix, result,
                                           output_dir=output_dir, env=env,
                                           source_overpass=fname)
        if path:
            ph["export"] = time.time() - _tx
            rec["status"], rec["path"] = "saved", path
            logs.append(f"  saved: {os.path.basename(path)}")
        else:
            rec["reason"] = "export_returned_none"
        return rec
    except Exception as e:
        rec["reason"] = f"exception: {type(e).__name__}: {str(e)[:80]}"
        logs.append(f"  skipped {fname} ({type(e).__name__}: {e})")
        return rec
    finally:
        rec["cpu_s"] = time.process_time() - cpu0
        if in_process:
            rec["peak_rss_gb"] = peak_rss_gb()
            rec["extra_band"] = _extra_band_delta(eb0)


# --- Resume (0.162) --------------------------------------------------------
# The 0.117 "resumability" lived in export_training_example -- the LAST step
# -- so a resumed mine re-downloaded the overpass, all six GOES bands and
# ~30 GLM granules and regenerated the frame before discovering the file
# existed. The output name needs band 13's scan time, unknown until band 13
# is fetched, so skipping is keyed on what IS known up front: the TC PRIMED
# overpass filename. Sources, in order: the ledger this module appends to;
# `source_overpass` inside NPZs from 0.162 on; and, for older NPZs, same
# storm + sensor with the GOES scan within LEGACY_MATCH_MINUTES of the
# overpass time (one sensor's passes over one storm are hours apart).
# Everything counts only under the CURRENT physics ID: a physics change
# still re-mines for real.
LEDGER_NAME = "mining_ledger.jsonl"
LEGACY_MATCH_MINUTES = 15
# Skip outcomes that re-running cannot change under the same physics.
# Network and exception skips are always retried.
DETERMINISTIC_SKIPS = ("goes_does_not_cover_storm", "no_goes_satellite_for_this_time_and_place",
                       "no_interpolated_fix", "export_returned_none",
                       # A TC PRIMED overpass whose every MW pixel is QC-flagged
                       # or non-finite (17 in a full 2018-2025 mine): the file
                       # will not change, so a resume need not refetch it.
                       "exception: ValueError: Entire field is invalid/missing")


def _overpass_time(fname: str):
    """MW time from a TC PRIMED name (..._YYYYMMDDHHMMSS.nc), or None."""
    try:
        stamp = os.path.basename(fname).rsplit("_", 1)[-1].split(".")[0]
        return datetime.strptime(stamp[:14], "%Y%m%d%H%M%S")
    except Exception:
        return None


class ResumeIndex:
    """Which overpasses are already done, under the current physics."""

    def __init__(self, output_dir: str, physics_id: str):
        import json
        self.physics_id = physics_id
        self.ledger_path = os.path.join(output_dir, LEDGER_NAME)
        self.ledger = {}
        self.by_source = {}
        self.legacy = {}
        self.missing_env = 0
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        self.ledger[r["fname"]] = r
                    except Exception:
                        continue
        for f in tde.list_training_files(output_dir):
            try:
                with np.load(f, allow_pickle=False) as z:
                    pid = str(z["vh_physics_id"]) if "vh_physics_id" in z.files else None
                    src = str(z["source_overpass"]) if "source_overpass" in z.files else ""
                    has_env = "env_shear_deep_ms" in z.files
            except Exception:
                continue                           # unreadable: redo it
            if pid != physics_id:
                continue
            self.missing_env += not has_env
            if src:
                self.by_source[src] = f
                continue
            parts = os.path.basename(f)[:-4].split("_")    # STORM_YYYYMMDDHHMM_SENSOR
            try:
                t = datetime.strptime(parts[1], "%Y%m%d%H%M")
                self.legacy.setdefault((parts[0], parts[2].upper()), []).append(t)
            except Exception:
                continue

    def done(self, fname: str, storm_key: str, instrument: str):
        """(True, why) if this overpass needs no work, else (False, None)."""
        r = self.ledger.get(fname)
        if r and r.get("physics_id") == self.physics_id:
            if r.get("status") == "saved" and r.get("path") and os.path.exists(r["path"]):
                return True, "saved"
            if r.get("status") == "skip" and str(r.get("reason", "")).startswith(DETERMINISTIC_SKIPS):
                return True, "known_skip"
        if fname in self.by_source:
            return True, "saved"
        t = _overpass_time(fname)
        sensor = str(instrument).replace("-", "").upper()
        if t is not None:
            for g in self.legacy.get((storm_key, sensor), ()):
                if abs((g - t).total_seconds()) <= LEGACY_MATCH_MINUTES * 60:
                    return True, "saved"
        return False, None

    def record(self, fh, rec: dict):
        import json
        fh.write(json.dumps({"fname": rec["fname"], "status": rec["status"],
                             "reason": rec["reason"], "path": rec["path"],
                             "physics_id": self.physics_id}) + "\n")
        fh.flush()


def mine_local_tcprimed_cache(
    cache_dir: str = None,
    satellite: str = None,   # None = auto-select per scene (see select_satellite)
    output_dir: str = None,   # None = tde.DEFAULT_EXPORT_DIR, resolved per call
    progress_callback=None,
    max_workers: int = 10,
    max_per_storm: Optional[int] = None,
    exclude_basins: tuple = ("WP", "IO", "SH"),
    stream: bool = False,
    seasons: tuple = (),
    basins: tuple = (),
    use_processes: Optional[bool] = None,   # None = MINING_USE_PROCESSES
    resume: bool = True,
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
    if output_dir is None:
        output_dir = tde.DEFAULT_EXPORT_DIR   # resolved now, so it can be redirected

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
        _storms = [st for season in seasons
                   for st in tp.list_available_storms(season_start=season, season_end=season,
                                                      basins=basins or None)
                   if st["basin"] not in exclude_basins]
        for storm, _files in zip(_storms, tp.list_overpass_files_many(_storms)):
            for f in _files:
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

    # --- 0.159: ONE pool across ALL storms -------------------------------
    #
    # Through 0.158 storms ran ONE AT A TIME, each in its own pool, and the
    # next storm started only after the slowest overpass of the current one
    # finished. With --max-per-storm 6 that capped concurrency at 6 however
    # many workers were asked for, and every storm paid its slowest frame
    # as a barrier. A real 0.157 run on a 250 Mbps link reached AL062018
    # after 74 min -- ~41 storms at ~1.8 min each, on course for ~9 h --
    # after the full-disk fix had already removed 90% of the GOES bytes.
    # Bytes were not the bottleneck; the structure was.
    #
    # Now: prepare every storm first (best track + field-of-view check, in
    # parallel -- each is one small HTTP request), then submit every
    # selected overpass of every storm to ONE pool, so --workers N means N
    # in flight for the whole run.
    #
    # Ordering: season, then basin, then storm number. The old order came
    # from sorted() FILENAMES, which put AL012018, AL012019 ... AL012025
    # before AL022018 -- an accident, and one that made the log useless as
    # a progress measure.
    def _prepare_storm(item):
        (basin, storm_num, season), file_list = item
        storm_key = f"{basin}{storm_num:02d}{season}"
        if _should_abort():
            return None
        # Basin filter FIRST, before any best-track fetch: WP/IO/SH are
        # never in any GOES view, and the lookup was the dominant cost of a
        # real 4-hour run.
        if exclude_basins and basin.upper() in {b.upper() for b in exclude_basins}:
            _skip(f"excluded_basin_{basin.upper()}")
            _log(f"  {storm_key}: basin {basin.upper()} excluded -- skipping "
                 f"{len(file_list)} file(s) without any lookup.")
            return None
        try:
            fixes = besttrack.fetch_best_track(basin, storm_num, season)
        except Exception as e:
            _skip(f"no_best_track: {type(e).__name__}")
            return None
        if not fixes:
            _skip("no_best_track")
            return None
        # Storm-level field-of-view check, BEFORE opening a single file.
        if not any(goes_fetch.select_satellite(fx.lat, fx.lon, fx.valid_time) for fx in fixes):
            _skip("storm_outside_satellite_field_of_view")
            _log(f"  {storm_key}: no GOES satellite covers this track at this date "
                 f"-- skipping all {len(file_list)} file(s) without reading them.")
            return None
        # (The per-storm cap is applied BEFORE prep since 0.162, so resume
        # can drop finished storms without fetching anything for them.)
        # ERA5 environment (0.160): one ranged read per storm (~0.5-5 MB of
        # a ~200 MB file), here in the parallel prep phase so no overpass
        # waits on it. Optional: a storm without it still mines, with NaNs.
        env_series = tcprimed_env.load_env_series(basin, storm_num, season,
                                                  progress_callback=_log)
        return storm_key, fixes, file_list, env_series

    # Per-file work is dominated by GOES S3 downloads (network I/O),
    # not CPU, so threads genuinely help here despite the GIL -- both
    # boto3 and the HDF5/numpy reads release it while waiting. Serial
    # downloading leaves most of a fast connection idle; several
    # concurrent transfers actually use it. max_workers is
    # deliberately modest (6) rather than very high: past ~8 the
    # bottleneck shifts to S3 throttling and local CPU for
    # generate_synthetic_mw, and more workers just add contention.
    procs = MINING_USE_PROCESSES if use_processes is None else bool(use_processes)
    max_workers = memory_safe_workers(max_workers, log=_log,
                                      per_worker_gb=PER_PROCESS_GB if procs else None,
                                      base_gb=0.3 if procs else None)
    ordered = sorted(by_storm.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1]))
    # Storm prep is small HTTP reads, not memory: it gets its own 8 threads
    # rather than the memory-capped worker count (0.161 -- capped to 2 in a
    # small sandbox, prep took 2.4 of a 6.6-minute run).
    # Per-storm cap, then resume -- both BEFORE prep (0.162). The cap
    # spreads picks evenly across the storm's lifetime (not the first N),
    # so intensification and decay both stay represented; it depends only
    # on the file list, so it picks the same files on every run, which is
    # what makes resuming meaningful. Resume then drops finished overpasses,
    # and a storm with none left is never prepared: a resumed run used to
    # fetch every storm's best track and ERA5 file first (135 s to do
    # nothing, on 19 storms).
    # The index is always built so the ledger is always WRITTEN; --fresh
    # (resume=False) only stops it being used to skip. 0.162-0.166 wrote no
    # ledger at all under --fresh, leaving an interrupted fresh run less to
    # resume from.
    idx = None
    _tr = time.time()
    idx = ResumeIndex(output_dir, tde._vh_physics_id())
    _log(f"Resume: index built in {time.time() - _tr:.0f} s "
         f"({len(idx.ledger)} ledger entries, {len(idx.by_source)} tagged + "
         f"{sum(len(v) for v in idx.legacy.values())} older NPZs under current physics)")
    resumed = {"saved": 0, "known_skip": 0}
    todo = []
    for (basin, storm_num, season), file_list in ordered:
        fl = sorted(file_list, key=lambda fp: fp[0])
        if max_per_storm and len(fl) > max_per_storm:
            step = len(fl) / float(max_per_storm)
            fl = [fl[int(i * step)] for i in range(max_per_storm)]
        if idx is not None and resume:
            storm_key = f"{basin}{storm_num:02d}{season}"
            keep = []
            for fname, parsed in fl:
                is_done, why = idx.done(fname, storm_key, parsed.get("instrument", ""))
                if is_done:
                    resumed[why] += 1
                else:
                    keep.append((fname, parsed))
            fl = keep
        if fl:
            todo.append(((basin, storm_num, season), fl))
    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = [p for p in pool.map(_prepare_storm, todo) if p]
    if procs:
        # Build the shared land mask ONCE, here, before any worker exists;
        # otherwise every worker process finds it missing at the same moment
        # and builds it concurrently.
        try:
            import surface_type
            surface_type.ensure_packed_land_mask()
        except Exception:
            pass
    work = []
    for storm_key, fixes, file_list, env_series in prepared:
        storms_processed.append(storm_key)
        for fname, parsed in file_list:
            meta = stream_keys.get(fname) if stream else None
            work.append((fname, parsed, fixes, env_series, meta, cache_dir,
                         output_dir, satellite, bool(stream), procs))
    if resumed["saved"] or resumed["known_skip"]:
        _log(f"Resume: {resumed['saved']} overpass(es) already saved and "
             f"{resumed['known_skip']} known skip(s) -- not redone (--fresh to redo)")
    if idx is not None and idx.missing_env:
        _log(f"  {idx.missing_env} existing NPZ(s) predate the ERA5 environment / extra "
             f"IR -- run backfill_npz.py to add them without re-mining")
    _log(f"Mining {len(work)} overpass(es) from {len(prepared)} storm(s) with "
         f"{max_workers} {'PROCESS' if procs else 'thread'} worker(s)")
    ledger_fh = open(idx.ledger_path, "a", encoding="utf-8") if idx is not None else None

    run_stats = {"cpu_s": 0.0, "peak_rss_gb": 0.0}

    def _consume(rec):
        for m in rec["logs"]:
            _log(m)
        if rec["status"] == "saved":
            with _counter_lock:
                counters["saved"] += 1
                saved_paths.append(rec["path"])
                for _k, _v in rec["phase"].items():
                    _phase_totals[_k] = _phase_totals.get(_k, 0.0) + _v
                _phase_totals["_n"] = _phase_totals.get("_n", 0) + 1
        else:
            _skip(rec["reason"] or "unknown")
        if ledger_fh is not None:
            idx.record(ledger_fh, rec)
        run_stats["cpu_s"] += rec.get("cpu_s", 0.0)
        run_stats["peak_rss_gb"] = max(run_stats["peak_rss_gb"], rec.get("peak_rss_gb", 0.0))
        if procs:
            _merge_extra_band_stats(rec.get("extra_band"))

    _wall0, _cpu0 = time.time(), time.process_time()
    if work and not _should_abort():
        if procs:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx = _mp.get_context("spawn")      # what Windows always uses
            pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx,
                                       initializer=_mining_process_init)
        else:
            from concurrent.futures import as_completed
            pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [pool.submit(_mine_overpass, job) for job in work]
            for fut in as_completed(futures):
                try:
                    _consume(fut.result())
                except Exception as e:           # a worker process died
                    _skip(f"worker_failed: {type(e).__name__}")
                    _log(f"  worker failed ({type(e).__name__}: {e})")
                if _should_abort():
                    for f in futures:
                        f.cancel()
                    break
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    if ledger_fh is not None:
        ledger_fh.close()
    wall = time.time() - _wall0
    cpu = run_stats["cpu_s"] if procs else (time.process_time() - _cpu0)
    if wall > 0 and work:
        _log(f"CPU: {cpu / wall:.1f} core(s) busy on average over {wall/60:.1f} min "
             f"(machine has {os.cpu_count()} logical)"
             + (f"; peak worker memory {run_stats['peak_rss_gb']:.2f} GB" if procs else ""))
        LAST_RUN_STATS.update({"wall_s": wall, "cpu_s": cpu, "cores_busy": cpu / wall,
                               "mode": "processes" if procs else "threads",
                               "workers": max_workers,
                               "peak_worker_rss_gb": run_stats["peak_rss_gb"]})
    if abort_state["stopped"]:
        _log("")
        _log("ABORTING: " + abort_state["reason"])
        _log("  Nothing has saved and one failure dominates, so this is a")
        _log("  broken pipeline rather than an unlucky run. Fix the error")
        _log("  above and re-run; no point spending hours confirming it.")
        attempted = counters["saved"] + sum(skipped_reasons.values())
    else:
        attempted = len(work)      # as before: overpasses actually tried

    saved = counters["saved"]
    return {
        "attempted": attempted, "saved": saved, "skipped_reasons": skipped_reasons,
        "saved_paths": saved_paths, "storms_processed": storms_processed,
        "aborted": abort_state["stopped"], "abort_reason": abort_state["reason"],
        "resumed": resumed,
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
