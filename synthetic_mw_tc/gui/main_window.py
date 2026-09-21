"""
PyQt6 GUI shell for the synthetic TC microwave generator.

Phase 1 scope (per current build plan): generate a single synthetic MW
frame from live GOES + best-track pulls. Frame-time slider / GIF export
are deliberately NOT in this version -- planned for a later phase.
"""
from __future__ import annotations

# Bump this each time a new build is delivered.
#
# NUMBERING SCHEME: a single integer that counts up by one per version --
# 0.86, 0.87, 0.88, 0.89, ... There are no patch components. A fix to the
# previous version is still just the next number, not an "0.88.1"; that
# form appeared once by mistake and is what this note exists to prevent.
#
# Past 0.99 the counter keeps incrementing rather than rolling over to
# 1.0: 0.99 is followed by 0.100, then 0.101, and so on. This looks wrong
# if read as a decimal, and it is deliberate -- the leading 0 means "not
# finished", and MWSynth should not silently promote itself to 1.0 just
# because a counter wrapped. Note that this ordering does NOT sort
# correctly as a string, so anything that sorts versions must compare the
# part after the dot as an integer.
APP_VERSION = "MWSynth 0.154"

import sys
import os
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QDateTimeEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QTextEdit,
    QSplitter,
    QScrollArea,
    QFrame,
    QTabWidget,
    QLabel,
    QFileDialog,
    QCheckBox,
    QSlider,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import besttrack
import goes_fetch
import mw_ingest
import mw_composites
import mw_compare
import radar_ingest
import training_data_export
import ml_inference
from synthetic_algorithm import generate_synthetic_mw
from gui.credentials_tab import CredentialsTab


def fetch_real_mw_for_fusion(creds, fixes, fix, target_time, sensor_order, progress_callback):
    """Module-level (thread-safe -- no shared mutable state, everything
    passed explicitly) version of the real-MW fetch, shared by
    GenerateWorker (single frame) and GenerateLoopWorker (multi-frame,
    threaded).

    Real-time focused, per direct guidance: searches only BACKWARD (the
    most recent real pass, MIMIC-TC-morphed forward to target_time) --
    no forward/"after"-pass search. A real-time run has no future data to
    find, so searching ahead was always a wasted round-trip in that
    context; it only ever had value for archived/historical generation,
    a narrow use case (NRT retention is only ~7 days) that didn't
    justify paying the search cost on every real-time run. The crossfade
    machinery itself (mw_ingest.find_mw_pair_for_crossfade's
    search_after parameter, generate_synthetic_mw's real_swath_after/
    mw_confidence_after) still exists underneath for a possible future
    archive-specific mode -- nothing was removed, this just stopped
    invoking it by default.

    Searches the primary long-term trio (GMI-NRT, AMSR3-NRT, WSFM-NRT;
    see mw_ingest.PRIMARY_NRT_SENSORS -- AMSR2/SSMIS are being retired
    within weeks and were deliberately deprioritized) across the past
    12 hours, querying all three CONCURRENTLY (not one at a time) for
    speed, and falls back to the higher-latency GMI archive only if
    nothing is found there.

    sensor_order is kept as a parameter for backward compatibility with
    existing call sites, but is no longer used -- the actual sensor list
    searched is always mw_ingest.PRIMARY_NRT_SENSORS.

    Returns (swath_or_None, mw_age_confidence, None, 0.0) -- the trailing
    None/0.0 pair is kept in the return shape so callers/generate_synthetic_mw's
    real_swath_after/mw_confidence_after plumbing doesn't need special-
    casing; it's just always empty now. mw_age_confidence is 1.0 for a
    fresh/unmorphed pass, decaying as the pass ages. Never raises, since
    "no real MW available" is an expected, non-blocking outcome (fusion
    just falls back to GOES-only weighting for this frame).
    """
    pair = mw_ingest.find_mw_pair_for_crossfade(
        fixes, target_time, fix.lat, fix.lon, creds,
        sensors=mw_ingest.PRIMARY_NRT_SENSORS,
        lookback_hours=12.0,
        progress_callback=progress_callback,
    )

    before_swath = pair["before"]

    if before_swath is None:
        # Nothing from any of the 3 NRT sensors -- fall back to the GMI
        # archive (higher latency, but far more reliable availability,
        # and not limited to NRT's ~7-day retention window).
        gmi_cred = creds.get("earthdata", {})
        if gmi_cred.get("username"):
            progress_callback("MW search: no NRT pass found from GMI-NRT/AMSR3-NRT/WSFM-NRT -- falling back to the GMI archive...")
            try:
                archive_swath, _lookback, archive_hit = mw_ingest.find_swath_that_hit_storm(
                    "GMI", fixes, target_time, fix.lat, fix.lon,
                    gmi_cred.get("username", ""), gmi_cred.get("password", ""),
                    progress_callback=progress_callback,
                )
            except Exception as e:
                progress_callback(f"MW search: GMI archive fallback failed ({e}).")
                archive_swath = None
            if archive_swath is not None:
                age_hours = (target_time - archive_swath.scene_time).total_seconds() / 3600.0
                if age_hours > 0.1:
                    archive_swath = mw_ingest.morph_swath_to_time(archive_swath, fixes, target_time)
                age_confidence = _morph_age_confidence_for_gui(age_hours)
                progress_callback(f"MW search: using GMI archive pass @ {archive_swath.scene_time:%Y-%m-%d %H:%M} UTC (confidence {age_confidence:.2f}).")
                return archive_swath, age_confidence, None, 0.0
        progress_callback(f"MW search: no real MW pass found anywhere for {target_time:%H:%M} UTC -- proceeding GOES-only for this frame.")
        return None, 1.0, None, 0.0

    age_hours = (target_time - before_swath.scene_time).total_seconds() / 3600.0
    if age_hours > 0.1:
        before_swath = mw_ingest.morph_swath_to_time(before_swath, fixes, target_time)
    mw_age_confidence = _morph_age_confidence_for_gui(age_hours)
    progress_callback(f"MW search: '{pair['before_sensor']}' pass confidence {mw_age_confidence:.2f}.")

    return before_swath, mw_age_confidence, None, 0.0


def _morph_age_confidence_for_gui(age_hours: float) -> float:
    """Thin wrapper so the GUI doesn't need to import synthetic_algorithm
    just for this one small function -- same formula, kept in sync
    manually (small/stable enough that duplication here is lower-risk
    than a cross-module import purely for this)."""
    full_confidence_hours, floor_hours, floor_value = 1.5, 6.0, 0.15
    if age_hours <= full_confidence_hours:
        return 1.0
    if age_hours >= floor_hours:
        return floor_value
    frac = (floor_hours - age_hours) / (floor_hours - full_confidence_hours)
    return floor_value + (1.0 - floor_value) * frac


def _crossfade_confidence_for_gui(minutes_until_after: float) -> float:
    """Thin wrapper, same reasoning as _morph_age_confidence_for_gui --
    kept in sync manually with synthetic_algorithm._crossfade_confidence_toward_after."""
    crossfade_minutes = 60.0
    if minutes_until_after <= 0:
        return 1.0
    if minutes_until_after >= crossfade_minutes:
        return 0.0
    return 1.0 - (minutes_until_after / crossfade_minutes)


def _morph_and_confidence_for_frame(fixes, ft, before_swath_raw, after_swath_raw):
    """Given ALREADY-FOUND (un-morphed) before/after swaths -- typically
    from a checkpoint search shared across several nearby frames in a
    loop, see GenerateLoopWorker.run()'s checkpoint logic -- morph BOTH
    to THIS SPECIFIC frame's target_time and compute their confidences.
    This is the cheap, per-frame part of MIMIC-TC crossfading, factored
    out from the expensive multi-sensor network search so a loop can
    search once at a checkpoint and reuse the result's morph+confidence
    computation independently for every frame between checkpoints,
    instead of re-searching for every single frame.
    """
    before_swath = None
    mw_age_confidence = 1.0
    if before_swath_raw is not None:
        age_hours = (ft - before_swath_raw.scene_time).total_seconds() / 3600.0
        before_swath = before_swath_raw
        if age_hours > 0.1:
            before_swath = mw_ingest.morph_swath_to_time(before_swath_raw, fixes, ft)
        mw_age_confidence = _morph_age_confidence_for_gui(age_hours)

    after_swath = None
    mw_confidence_after = 0.0
    if after_swath_raw is not None:
        minutes_until_after = (after_swath_raw.scene_time - ft).total_seconds() / 60.0
        after_swath = mw_ingest.morph_swath_to_time(after_swath_raw, fixes, ft)
        mw_confidence_after = _crossfade_confidence_for_gui(minutes_until_after)

    return before_swath, mw_age_confidence, after_swath, mw_confidence_after


def fetch_radar_for_fusion_data(fix, target_time, progress_callback):
    """Module-level (thread-safe) version of the radar fetch, shared by
    GenerateWorker and GenerateLoopWorker. If the storm center is within
    200mi of a NEXRAD site, fetch the nearest volume scan's gridded
    reflectivity AND echo-top height for fusion. Never raises -- any
    failure is logged and treated as "no radar available," not an error
    that blocks generation."""
    try:
        station, dist_mi = radar_ingest.find_radar_for_storm(fix)
        if station is None:
            progress_callback(f"Fusion: no NEXRAD site within 200mi of storm center ({target_time:%H:%M} UTC).")
            return None

        progress_callback(f"Fusion: nearest radar {station.icao} ({station.name}), {dist_mi:.0f}mi -- fetching scan for {target_time:%H:%M} UTC...")
        data = radar_ingest.fetch_radar_for_fusion(fix, target_time)
        if data is None:
            progress_callback(f"Fusion: {station.icao} in range but no recent scan found for {target_time:%H:%M} UTC.")
            return None

        et_str = ""
        if data.get("max_echo_top_km") is not None:
            et_str = f", max echo top {data['max_echo_top_km']:.1f} km"
        elif data.get("echo_top_error"):
            et_str = f" (echo tops unavailable: {data['echo_top_error']})"
        progress_callback(
            f"Fusion: {data['station']} @ {data['scan_time']:%H:%M} UTC, "
            f"max {data['max_dbz']:.0f} dBZ near storm center{et_str}."
        )
        return data
    except Exception as e:
        progress_callback(f"Fusion: radar fetch failed (non-blocking): {e}")
        return None


class GenerateWorker(QThread):
    """Runs data fetch + algorithm off the GUI thread. Also handles the
    automatic real-MW ingest + bias calibration step (see run() below)."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    # Tried in this order, stopping at the first sensor that both has
    # credentials configured AND actually returns a swath. GMI-NRT/AMSR2
    # first (recent, fast-ish); GMI archive last (reliable but can lag by
    # days, per mw_ingest.py -- fine as a fallback, not ideal as a first try).
    # Sensor search order. SSMIS-NRT and AMSR2 were removed: SSMIS shut
    # down entirely in September 2026 and AMSR2-NRT retired transmission
    # after Aug 31 2026, so both could only ever fail -- but they sat
    # AHEAD of AMSR3-NRT here, so every generate spent time failing over
    # two dead sensors before reaching a live one. AMSR3-NRT is now tried
    # second, and archive GMI remains last as the deliberate fallback
    # (slower, but no NRT retention window, so it is the only path that
    # works for older cases).
    AUTO_CALIBRATE_SENSOR_ORDER = ("GMI-NRT", "AMSR3-NRT", "WSFM-NRT", "GMI")

    def __init__(self, params: dict, creds: dict, auto_calibrate: bool, radar_check: bool,
                 ml_strength: float = 1.0):
        super().__init__()
        self.params = params
        self.creds = creds
        self.auto_calibrate = auto_calibrate
        self.radar_check = radar_check
        self.ml_strength = ml_strength

    def run(self):
        try:
            self.progress.emit("Fetching best track...")
            fixes = besttrack.fetch_best_track(
                self.params["basin"], self.params["storm_num"], self.params["year"]
            )
            fix = besttrack.interpolate_fix(fixes, self.params["target_time"])
            if fix is None:
                raise RuntimeError("No best-track fix found near requested time.")

            sat = self.params["satellite"]
            sector = self.params["sector"] or None
            target_time = self.params["target_time"]

            self.progress.emit(f"Fetching GOES band 13 ({sat})...")
            band13 = goes_fetch.get_band_image(sat, 13, target_time, sector=sector)

            # Does the sector actually contain the storm?
            #
            # goes_fetch matches on TIME ONLY, and M1/M2 are independently
            # steerable boxes routinely parked over different systems. So
            # a valid, recent file from the wrong sector renders a
            # perfectly convincing frame of somewhere else -- no error,
            # just the wrong storm. The mining path has checked this since
            # 0.98; the GUI never did, and with the sector now chosen
            # explicitly rather than "auto", a wrong pick is easier to
            # make and deserves to be caught.
            if band13 is not None and fix is not None:
                if not goes_fetch._image_covers(band13, fix.lat, fix.lon):
                    self.progress.emit(
                        f"  WARNING: sector {sector} does not contain the storm "
                        f"at {fix.lat:.1f}N {fix.lon:.1f}E. Try the other "
                        f"mesoscale sector.")

            self.progress.emit(f"Fetching GOES band 9 ({sat})...")
            band9 = goes_fetch.get_band_image(sat, 9, target_time, sector=sector)
            self.progress.emit(f"Fetching GOES band 7 ({sat})...")
            band7 = goes_fetch.get_band_image(sat, 7, target_time, sector=sector)

            import solar
            daytime = solar.is_daytime(fix.lat, fix.lon, target_time)
            if daytime:
                self.progress.emit(f"Daytime at storm location -- fetching GOES band 2 ({sat})...")
                band2 = goes_fetch.get_band_image(sat, 2, target_time, sector=sector)
            else:
                self.progress.emit(
                    "Nighttime at storm location -- skipping band 2 fetch "
                    "(visible imagery has no usable signal after dark)."
                )
                band2 = None

            # Supplementary ABI IR bands (ml_constants.EXTRA_IR_BANDS, ordered
            # by the saliency Li et al. (2026) measured). Best-effort: any band
            # that fails is simply absent, and its model channel becomes a
            # neutral plane rather than changing the input shape.
            extra_ir = goes_fetch.fetch_extra_ir_bands(
                sat, target_time, sector=sector, progress_callback=self.progress.emit)

            if band13 is None or band9 is None or band7 is None:
                raise RuntimeError(
                    "Could not find GOES RadM files near the requested time/sector. "
                    "Try a wider window, different sector (M1/M2), or different satellite."
                )

            # Fetch real MW and radar BEFORE generation, not after -- so
            # generate_synthetic_mw can properly FUSE all available
            # sources as weighted equals (radar > MW > GOES priority),
            # instead of generating from GOES alone and then patching a
            # bias correction on top of the result. See
            # synthetic_algorithm.py's fusion docstring for the reasoning.
            real_swath, mw_age_confidence, real_swath_after, mw_confidence_after = (
                self._fetch_real_mw_for_fusion(fixes, fix, target_time) if self.auto_calibrate else (None, 1.0, None, 0.0)
            )
            radar_data = self._fetch_radar_for_fusion(fix, target_time) if self.radar_check else None

            sources_desc = ["GOES"]
            if real_swath is not None:
                sources_desc.append(f"MW-before ({real_swath.sensor}, confidence {mw_age_confidence:.2f})")
            if real_swath_after is not None:
                sources_desc.append(f"MW-after ({real_swath_after.sensor}, confidence {mw_confidence_after:.2f})")
            if radar_data is not None:
                sources_desc.append(f"radar ({radar_data['station']})")
            self.progress.emit(f"Running synthetic MW algorithm, fusing: {', '.join(sources_desc)}...")

            result = generate_synthetic_mw(
                band13, band9, band7, fix, band2=band2,
                extra_ir=extra_ir,
                real_swath=real_swath,
                mw_age_confidence=mw_age_confidence,
                real_swath_after=real_swath_after,
                mw_confidence_after=mw_confidence_after,
                radar_lat=radar_data["lat"] if radar_data else None,
                radar_lon=radar_data["lon"] if radar_data else None,
                radar_dbz=radar_data["dbz"] if radar_data else None,
                radar_site_lat=radar_data.get("station_lat") if radar_data else None,
                radar_site_lon=radar_data.get("station_lon") if radar_data else None,
                echo_top_lat=radar_data.get("echo_top_lat") if radar_data else None,
                echo_top_lon=radar_data.get("echo_top_lon") if radar_data else None,
                echo_top_km=radar_data.get("echo_top_km") if radar_data else None,
                ml_strength=self.ml_strength,
                progress_callback=self.progress.emit,
            )

            # Track the persisted calibration state (measurement-only --
            # feeds calibration_state's EMA for FUTURE runs' baseline, but
            # does NOT apply a further direct correction to this frame's
            # output, since the fusion above already incorporated the real
            # data directly rather than as an after-the-fact patch).
            #
            # Only feed the persisted baseline from reasonably FRESH real
            # data (confidence >= 0.5, i.e. an un-morphed or lightly-aged
            # pass) -- a heavily-morphed old pass's measured "bias" reflects
            # outdated storm conditions (structure/intensity likely
            # changed since that observation), and letting stale-data-
            # driven noise pollute the long-term learned offset would be
            # worse than just not updating it this run. The morphed data
            # is still fully used for THIS frame's direct fusion either
            # way (that's frame-specific, doesn't accumulate).
            if real_swath is not None and mw_age_confidence >= 0.5:
                self._update_persisted_calibration(result, real_swath)
            elif real_swath is not None:
                result.diagnostics["calibration_applied"] = False
                result.diagnostics["calibration_reason"] = (
                    f"Real MW pass used for fusion was too old/low-confidence "
                    f"({mw_age_confidence:.2f}) to trust for the persisted baseline update -- "
                    "still fully used for this frame's direct fusion, just not fed back "
                    "into long-term calibration learning."
                )
            elif self.auto_calibrate:
                import calibration_state
                result.diagnostics["calibration_applied"] = False
                result.diagnostics["calibration_reason"] = (
                    "No real MW pass found (or no credentials configured) for fusion. "
                    f"Persisted offset from past runs still applies -- "
                    f"{calibration_state.get_status_summary()}."
                )
            else:
                result.diagnostics["calibration_applied"] = False
                # Distinguish a DELIBERATE GOES-only frame from a failed
                # search. Both produce a frame with no real MW in it, and
                # reading "no pass found" on a frame where fusion was
                # switched off would send you looking for a data problem
                # that does not exist.
                result.diagnostics["calibration_reason"] = (
                    "Real MW fusion turned OFF -- this is a deliberate GOES-only frame, "
                    "not a failed search. The 37/89 GHz fields are the parametric "
                    "backbone (plus any ML correction) alone."
                )

            if radar_data is not None:
                result.diagnostics["radar_check"] = {
                    "available": True,
                    "station": radar_data["station"],
                    "station_name": radar_data["station_name"],
                    "distance_mi": radar_data["distance_mi"],
                    "scan_time": radar_data["scan_time"],
                    "max_dbz": radar_data["max_dbz"],
                    "max_echo_top_km": radar_data.get("max_echo_top_km"),
                    "fused": True,
                }
            elif self.radar_check:
                result.diagnostics["radar_check"] = {
                    "available": False,
                    "reason": "No NEXRAD site in range, or no recent scan found.",
                }

            self.finished.emit((result, band13, band9, band7, band2, fix))

        except Exception as exc:  # surfaced to the GUI, not swallowed
            self.failed.emit(str(exc))

    def _fetch_real_mw_for_fusion(self, fixes, fix, target_time):
        return fetch_real_mw_for_fusion(
            self.creds, fixes, fix, target_time,
            self.AUTO_CALIBRATE_SENSOR_ORDER,
            self.progress.emit,
        )

    def _fetch_radar_for_fusion(self, fix, target_time):
        return fetch_radar_for_fusion_data(fix, target_time, self.progress.emit)

    def _update_persisted_calibration(self, result, real_swath):
        """Measure the residual bias between the FUSED result and the real
        MW pass, and feed it into calibration_state's persisted EMA offset
        -- for future runs' baseline, not as an additional correction to
        this frame (which already incorporated the real data directly via
        fusion, not a post-hoc patch). Never raises."""
        import calibration_state

        try:
            comparison = mw_compare.compare_both_frequencies(result, real_swath)

            # Surface the FULL comparison, not just the bias. This ran on
            # every fused frame and only its bias_k was consumed -- the
            # RMSE and pixel counts against both real V-pol and real PCT
            # were computed and discarded. That is the one direct
            # quantitative check of synthetic against real this project
            # has, and it was invisible.
            try:
                summary = mw_compare.format_stats_summary(comparison)
                result.diagnostics["real_mw_comparison"] = summary
                if hasattr(self, "progress"):
                    for line in summary.splitlines():
                        self.progress.emit("  " + line)
            except Exception:
                pass

            calib_info = {freq: comparison[freq]["vs_v"] for freq in (37, 89)}
            for freq in (37, 89):
                bias = calib_info[freq]["bias_k"]
                if bias is not None:
                    calibration_state.update_offset(freq, bias)

            result.diagnostics["calibration_applied"] = True
            result.diagnostics["calibration_source"] = (
                f"{real_swath.sensor} ({real_swath.scene_time:%Y-%m-%d %H:%M} UTC) "
                "[fused directly into generation, not a post-hoc bias correction]"
            )
            result.diagnostics["calibration_info"] = calib_info
            result.diagnostics["calibration_state"] = calibration_state.load_state()
            self.progress.emit(
                f"Residual bias vs {real_swath.sensor} after fusion: "
                f"37GHz {calib_info[37]['bias_k']:+.1f}K, 89GHz {calib_info[89]['bias_k']:+.1f}K "
                f"(fed into persisted state: {calibration_state.get_status_summary()})"
            )
        except Exception as e:
            self.progress.emit(f"Calibration tracking failed (non-blocking): {e}")
            result.diagnostics["calibration_applied"] = False
            result.diagnostics["calibration_reason"] = f"Error measuring residual bias: {e}"


class GenerateLoopWorker(QThread):
    """Generates multiple frames at 10-minute intervals (or a multiple of
    10 minutes, via skip_every -- e.g. skip_every=3 covers 3x the time
    span at the same frame count) ending at the selected target time, for
    the frame-slider/GIF-export feature.

    Real MW/radar fusion is now available per-frame (auto_calibrate/
    radar_check flags, same semantics as GenerateWorker) -- accepted as
    slower, per explicit request. To make that trade-off less painful,
    frames are generated CONCURRENTLY via a thread pool whenever fusion
    is enabled (each frame's fetch is largely independent I/O-bound work:
    several separate network round-trips per frame once MW/radar are
    involved, so threading gives a real speedup rather than fighting the
    GIL). Best-track is fetched once up front and shared read-only across
    threads/frames; the persisted calibration_state baseline is
    deliberately NOT updated per-frame here (unlike single-frame
    GenerateWorker) to avoid concurrent-write races on its JSON file --
    multi-frame fusion informs each frame directly, it just doesn't feed
    the long-term learned baseline the way a single-frame run does.
    """

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    # Sensor search order. SSMIS-NRT and AMSR2 were removed: SSMIS shut
    # down entirely in September 2026 and AMSR2-NRT retired transmission
    # after Aug 31 2026, so both could only ever fail -- but they sat
    # AHEAD of AMSR3-NRT here, so every generate spent time failing over
    # two dead sensors before reaching a live one. AMSR3-NRT is now tried
    # second, and archive GMI remains last as the deliberate fallback
    # (slower, but no NRT retention window, so it is the only path that
    # works for older cases).
    AUTO_CALIBRATE_SENSOR_ORDER = ("GMI-NRT", "AMSR3-NRT", "WSFM-NRT", "GMI")
    MAX_CONCURRENT_FRAMES = 5

    def __init__(self, params: dict, n_frames: int, creds: dict = None, auto_calibrate: bool = False, radar_check: bool = False, skip_every: int = 1,
                 ml_strength: float = 1.0):
        super().__init__()
        self.params = params
        self.n_frames = n_frames
        self.creds = creds or {}
        self.auto_calibrate = auto_calibrate
        self.radar_check = radar_check
        self.ml_strength = ml_strength
        self.skip_every = max(1, skip_every)  # 1 = consecutive 10-min steps (original behavior); N = N*10-min steps, same frame count, covers a longer span

    def _generate_one_frame(self, fixes, ft, before_swath_raw=None, after_swath_raw=None, mw_searched=False):
        sat = self.params["satellite"]
        sector = self.params["sector"] or None

        fix = besttrack.interpolate_fix(fixes, ft)
        if fix is None:
            raise RuntimeError(f"No best-track fix found near {ft:%H:%M} UTC.")

        band13 = goes_fetch.get_band_image(sat, 13, ft, sector=sector)
        band9 = goes_fetch.get_band_image(sat, 9, ft, sector=sector)
        band7 = goes_fetch.get_band_image(sat, 7, ft, sector=sector)
        # Supplementary ABI IR bands (ml_constants.EXTRA_IR_BANDS, ordered
        # by the saliency Li et al. (2026) measured). Best-effort: any band
        # that fails is simply absent, and its model channel becomes a
        # neutral plane rather than changing the input shape.
        extra_ir = goes_fetch.fetch_extra_ir_bands(
            sat, ft, sector=sector, progress_callback=None)

        if band13 is None or band9 is None or band7 is None:
            raise RuntimeError(
                f"Could not find GOES RadM files near {ft:%H:%M} UTC. "
                "Multi-frame loops need a GOES scene at every 10-minute "
                "step -- try a different time range or a wider sector search."
            )

        import solar
        band2 = None
        if solar.is_daytime(fix.lat, fix.lon, ft):
            band2 = goes_fetch.get_band_image(sat, 2, ft, sector=sector)

        real_swath = None
        mw_age_confidence = 1.0
        real_swath_after = None
        mw_confidence_after = 0.0
        if self.auto_calibrate:
            if mw_searched:
                # Checkpoint-based flow: the SEARCH already happened
                # (shared across several frames -- see run()'s checkpoint
                # logic), this frame just does its OWN cheap morph +
                # confidence computation against whatever raw swaths were
                # found at the bracketing checkpoints.
                real_swath, mw_age_confidence, real_swath_after, mw_confidence_after = _morph_and_confidence_for_frame(
                    fixes, ft, before_swath_raw, after_swath_raw,
                )
            else:
                # Fallback path (shouldn't normally be reached once the
                # checkpoint flow is wired up in run(), but kept so this
                # method still works standalone if called directly).
                real_swath, mw_age_confidence, real_swath_after, mw_confidence_after = fetch_real_mw_for_fusion(
                    self.creds, fixes, fix, ft, self.AUTO_CALIBRATE_SENSOR_ORDER,
                    self.progress.emit,
                )
        radar_data = None
        if self.radar_check:
            radar_data = fetch_radar_for_fusion_data(fix, ft, self.progress.emit)

        result = generate_synthetic_mw(
            band13, band9, band7, fix, band2=band2,
            extra_ir=extra_ir,
            real_swath=real_swath,
            mw_age_confidence=mw_age_confidence,
            real_swath_after=real_swath_after,
            mw_confidence_after=mw_confidence_after,
            radar_lat=radar_data["lat"] if radar_data else None,
            radar_lon=radar_data["lon"] if radar_data else None,
            radar_dbz=radar_data["dbz"] if radar_data else None,
            radar_site_lat=radar_data.get("station_lat") if radar_data else None,
            radar_site_lon=radar_data.get("station_lon") if radar_data else None,
            echo_top_lat=radar_data.get("echo_top_lat") if radar_data else None,
            echo_top_lon=radar_data.get("echo_top_lon") if radar_data else None,
            echo_top_km=radar_data.get("echo_top_km") if radar_data else None,
            ml_strength=self.ml_strength,
            progress_callback=self.progress.emit,
        )

        # Set calibration_applied/calibration_reason explicitly -- without
        # this, render_result's title falls back to a generic default
        # ("Fusion/calibration off") that directly contradicts the
        # "Fused: GOES + MW(cov ..%)" text right above it whenever real MW
        # WAS actually used for this frame. GenerateWorker (single-frame)
        # sets these via _update_persisted_calibration; loops deliberately
        # skip THAT (avoids concurrent-write races on calibration_state's
        # file across threads), but still need the diagnostics set so the
        # title reports what actually happened for this specific frame.
        if real_swath is not None or real_swath_after is not None:
            result.diagnostics["calibration_applied"] = False
            desc_parts = []
            if real_swath is not None:
                desc_parts.append(f"before-pass {real_swath.sensor} (confidence {mw_age_confidence:.2f})")
            if real_swath_after is not None:
                desc_parts.append(f"after-pass {real_swath_after.sensor} (confidence {mw_confidence_after:.2f})")
            result.diagnostics["calibration_reason"] = (
                f"Real MW ({', '.join(desc_parts)}) fused directly "
                "into this frame; not fed into the persisted baseline (multi-frame loops don't update "
                "long-term calibration, to avoid concurrent-write races across threads)."
            )
        elif self.auto_calibrate:
            result.diagnostics["calibration_applied"] = False
            result.diagnostics["calibration_reason"] = "No real MW pass found for this frame -- GOES-only."
        else:
            result.diagnostics["calibration_applied"] = False
            result.diagnostics["calibration_reason"] = (
                "Real MW fusion turned OFF for this loop -- deliberate GOES-only frames, "
                "not a failed search."
            )

        return (result, band13)

    def run(self):
        try:
            end_time = self.params["target_time"]
            interval = timedelta(minutes=10 * self.skip_every)
            frame_times = [end_time - interval * i for i in range(self.n_frames)][::-1]  # oldest first

            self.progress.emit("Fetching best track...")
            fixes = besttrack.fetch_best_track(
                self.params["basin"], self.params["storm_num"], self.params["year"]
            )

            fusion_on = self.auto_calibrate or self.radar_check
            frames_by_index = [None] * len(frame_times)

            # --- Checkpoint-based MW searching ---
            # Searching independently for every frame (up to 15 separate
            # multi-sensor searches) is wasteful -- real MW passes don't
            # refresh every 10 minutes anyway. Search only at a handful
            # of evenly-spaced checkpoint frames (see
            # mw_ingest.compute_checkpoint_frame_indices: 5 frames -> 2
            # searches, 10 -> 3, 15 -> 4), then every frame (checkpoint or
            # not) reuses the nearest PRECEDING checkpoint's RAW pass,
            # independently MIMIC-TC-morphing it to its OWN specific
            # time -- only the expensive network SEARCH is shared, not
            # the per-frame morph/confidence math, which stays exact.
            #
            # Backward-only (no "after"/forward search): per direct
            # guidance, this project is real-time focused, and searching
            # ahead of any frame's own time only ever made sense for
            # archived/historical generation -- not worth the extra
            # search cost on every loop run. See
            # mw_ingest.find_mw_pair_for_crossfade's search_after
            # parameter if a future archive-specific mode wants it back.
            checkpoint_swaths = {}  # checkpoint_frame_index -> {"before":.., "before_sensor":..}
            if self.auto_calibrate:
                checkpoint_indices = mw_ingest.compute_checkpoint_frame_indices(len(frame_times))
                self.progress.emit(
                    f"Searching for real MW at {len(checkpoint_indices)} checkpoint frame(s) "
                    f"(instead of all {len(frame_times)}) -- reused across nearby frames via MIMIC-TC morphing, "
                    f"all checkpoints searched concurrently."
                )

                def _search_one_checkpoint(idx):
                    ft = frame_times[idx]
                    fix_at_ft = besttrack.interpolate_fix(fixes, ft)
                    if fix_at_ft is None:
                        return idx, {"before": None, "before_sensor": None}
                    pair = mw_ingest.find_mw_pair_for_crossfade(
                        fixes, ft, fix_at_ft.lat, fix_at_ft.lon, self.creds,
                        progress_callback=self.progress.emit,
                    )
                    return idx, pair

                # Checkpoints run CONCURRENTLY, not one after another -- each
                # checkpoint's OWN search is already internally parallel
                # across its 3 sensors, so nesting this means up to
                # len(checkpoint_indices)*3 requests in flight at once,
                # shrinking the whole search phase toward roughly one
                # sensor's slowest single query instead of N sequential
                # rounds of it.
                with ThreadPoolExecutor(max_workers=max(1, len(checkpoint_indices))) as executor:
                    futures = [executor.submit(_search_one_checkpoint, idx) for idx in checkpoint_indices]
                    for future in as_completed(futures):
                        idx, pair = future.result()
                        checkpoint_swaths[idx] = pair

            def _preceding_checkpoint_swath(i):
                """For frame index i, the nearest searched checkpoint AT
                OR BEFORE it supplies the "before" pass -- simple nearest-
                preceding lookup now that there's no "after" pass to
                bracket toward."""
                if not checkpoint_swaths:
                    return None
                indices_sorted = sorted(checkpoint_swaths.keys())
                ck_lo = max(c for c in indices_sorted if c <= i)
                return checkpoint_swaths[ck_lo]["before"]

            if fusion_on:
                self.progress.emit(
                    f"Generating {len(frame_times)} frames WITH MW/radar fusion "
                    f"(up to {self.MAX_CONCURRENT_FRAMES} concurrent)..."
                )
                with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_FRAMES) as executor:
                    future_to_idx = {}
                    for i, ft in enumerate(frame_times):
                        before_raw = _preceding_checkpoint_swath(i) if self.auto_calibrate else None
                        future = executor.submit(
                            self._generate_one_frame, fixes, ft,
                            before_raw, None, self.auto_calibrate,
                        )
                        future_to_idx[future] = i
                    completed = 0
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        frames_by_index[idx] = future.result()  # re-raises here if that frame failed
                        completed += 1
                        self.progress.emit(f"Frame {completed}/{len(frame_times)} complete ({frame_times[idx]:%H:%M} UTC).")
            else:
                for i, ft in enumerate(frame_times):
                    self.progress.emit(f"Frame {i+1}/{len(frame_times)} ({ft:%H:%M} UTC)...")
                    frames_by_index[i] = self._generate_one_frame(fixes, ft)

            self.finished.emit(frames_by_index)
        except Exception as exc:
            self.failed.emit(str(exc))


class GenerateTab(QWidget):
    def __init__(self, credentials_tab: CredentialsTab):
        super().__init__()
        self.credentials_tab = credentials_tab
        self.worker: GenerateWorker | None = None
        self.frames: list = []  # [(SyntheticMWResult, BandImage), ...] -- 1 entry for single-frame, N for a loop

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        controls = self._build_controls()
        splitter.addWidget(controls)

        self.figure = Figure(figsize=(9, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(140)
        # Added to the container's layout, OUTSIDE the scroll area, so the
        # log stays visible while the settings above it scroll. Putting it
        # inside meant scrolling down to read progress output.
        self._controls_container_layout.addWidget(self.status_log, 0)

        self.on_refresh_storms()  # populate the storm dropdown on startup

    def _build_controls(self) -> QWidget:
        """Builds the settings column.

        The controls live inside a QScrollArea rather than directly in the
        splitter. The column had grown past the height of a normal
        (non-maximized) window, which silently CLIPPED the lower group
        boxes -- they were not merely off-screen, there was no way to
        reach them at all without maximizing. Anything added below this
        point now scrolls instead of disappearing.

        Returns the scroll area, not the inner panel, so the caller adds
        the scrollable thing to the splitter. The status log is appended
        to the OUTER container (see _build_controls_container) so it stays
        pinned at the bottom instead of scrolling away with the settings.
        """
        panel = QWidget()
        v = QVBoxLayout(panel)

        live_box = QGroupBox("GOES + best-track settings")
        live_form = QFormLayout(live_box)
        self.satellite_combo = QComboBox()
        self.satellite_combo.addItems(["GOES-18", "GOES-19"])
        self.sector_combo = QComboBox()
        # M1/M2 only. "Auto (either)" passed sector=None, and
        # find_nearest_file matches on TIME ONLY -- so it returned
        # whichever sector happened to scan closest to the requested
        # moment, regardless of where that sector was pointed. The two
        # mesoscale boxes are independently steerable and routinely sit
        # over different storms, so "auto" could quietly hand back a scene
        # of somewhere else entirely.
        self.sector_combo.addItems(["M1", "M2"])
        self.datetime_edit = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self.datetime_edit.setTimeSpec(Qt.TimeSpec.UTC)
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm 'UTC'")
        self.datetime_edit.setCalendarPopup(True)
        datetime_note = QLabel(
            "Enter time in UTC (Zulu). Note: the AWS S3 console's 'Last "
            "modified' column shows your browser's local time zone, NOT "
            "UTC -- for the true scan time, use the timestamp embedded in "
            "the object key name itself (the 's...' field, e.g. "
            "s20242451230123 = day 245 of 2024, 12:30:12 UTC)."
        )
        datetime_note.setWordWrap(True)
        datetime_note.setStyleSheet("color: gray; font-size: 11px;")

        self.storm_combo = QComboBox()
        self.storm_combo.setToolTip(
            "Storms with a best-track (.dat) file modified in the last "
            "24 hours on NHC's public btk server "
            "(https://ftp.nhc.noaa.gov/atcf/btk/) -- replaces typing "
            "basin/storm#/season into three separate boxes. This list is "
            "fetched once when the app starts and whenever you click "
            "Refresh; it isn't re-checked continuously."
        )
        self.refresh_storms_btn = QPushButton("Refresh storm list")
        self.refresh_storms_btn.clicked.connect(self.on_refresh_storms)

        live_form.addRow("Satellite:", self.satellite_combo)
        live_form.addRow("Mesoscale sector:", self.sector_combo)
        live_form.addRow("Target time:", self.datetime_edit)
        live_form.addRow("", datetime_note)
        live_form.addRow("Storm:", self.storm_combo)
        live_form.addRow("", self.refresh_storms_btn)
        v.addWidget(live_box)

        self.mw_fusion_checkbox = QCheckBox("Fuse real MW when available")
        self.mw_fusion_checkbox.setChecked(True)
        self.mw_fusion_checkbox.setToolTip(
            "Searches for a real passive-MW pass that actually observed the "
            "storm's core (GMI-NRT \u2192 AMSR3-NRT \u2192 WSFM-NRT \u2192 GMI archive, "
            "whichever have credentials configured), MIMIC-TC-style morphed to "
            "the frame time if it is aging, and fuses it per-pixel weighted by "
            "coverage and pass age.\n\n"
            "Leave this ON for normal use \u2014 real data is the point of this "
            "tool, and it already falls back to GOES-only by itself when no "
            "pass is found.\n\n"
            "Turn it OFF to force a GOES-only frame even when a pass IS "
            "available. That is a diagnostic mode: it isolates the parametric "
            "backbone and the ML correction, which are otherwise masked by real "
            "data wherever it has coverage. On a 98%-coverage frame the 37 GHz "
            "field is almost entirely real MW, so nothing about the backbone "
            "can be judged from it."
        )
        v.addWidget(self.mw_fusion_checkbox)

        mw_note = QLabel(
            "Real MW is fused automatically when the box above is ticked "
            "(GMI-NRT \u2192 AMSR3-NRT \u2192 WSFM-NRT \u2192 GMI archive, whichever have "
            "credentials configured) -- searches for a pass that actually "
            "observed the storm's core, MIMIC-TC-style morphing if it's aging, "
            "and falls back to GOES-only automatically if nothing is found. "
            "Unticking forces GOES-only even when a pass exists, which is what "
            "you want when testing the backbone or the ML correction in "
            "isolation."
        )
        mw_note.setWordWrap(True)
        mw_note.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(mw_note)


        ml_box = QGroupBox("ML correction")
        ml_form = QFormLayout(ml_box)
        self.ml_strength_spin = QDoubleSpinBox()
        self.ml_strength_spin.setRange(0.0, 2.0)
        self.ml_strength_spin.setDecimals(2)
        self.ml_strength_spin.setSingleStep(0.05)
        self.ml_strength_spin.setValue(ml_inference.DEFAULT_CORRECTION_STRENGTH)
        self.ml_strength_spin.setKeyboardTracking(True)
        self.ml_strength_spin.setToolTip(
            "Scale applied to the trained correction model's output before it "
            "is added to the parametric backbone.\n\n"
            "1.0 = use the model as trained. 0.0 = disable it entirely "
            "(identical to having no checkpoint, without deleting the file), "
            "which makes this the control for A/B-ing the correction against "
            "the raw backbone on the same frame.\n\n"
            "Above 1.0 is allowed but exaggerates a correction the training "
            "set may not support -- useful for seeing WHERE the model is "
            "acting, not for producing an image you'd trust.\n\n"
            "Note the correction is applied to the backbone, so its influence "
            "on the FINAL image also depends on how much weight the backbone "
            "carries: with fresh real MW the backbone is only ~25% of the "
            "result, but with an aged pass and no radar it can exceed 65%, and "
            "with radar fused in it can exceed 90%. The same strength value "
            "therefore does visibly different amounts of work frame to frame."
        )
        ml_form.addRow("Strength:", self.ml_strength_spin)

        self.ml_compare_checkbox = QCheckBox("Show ML-OFF comparison in 4th panel")
        self.ml_compare_checkbox.setChecked(False)
        self.ml_compare_checkbox.setToolTip(
            "Replaces the convective-signal panel with the 89 GHz composite as "
            "it would look with the correction disabled.\n\n"
            "This is reconstructed exactly from the same generate, not a second "
            "run: the fusion is linear in the backbone, so the correction's "
            "contribution can be subtracted back out precisely. That matters "
            "because generating the same frame twice at different strengths "
            "re-draws the noise floor and advances the calibration EMA in "
            "between, so part of the apparent difference is run-to-run "
            "randomness rather than the correction."
        )
        ml_form.addRow(self.ml_compare_checkbox)

        self.ml_novelty_checkbox = QCheckBox("Taper strength on unusual storms")
        # OFF by default. Tapering ML strength on unusual storms
        # suppresses the correction exactly where the backbone is least
        # trustworthy -- rare, extreme or oddly-structured cases, which
        # are the ones worth looking at. It stays available as an
        # opt-in for when a frame looks over-corrected.
        self.ml_novelty_checkbox.setChecked(False)
        self.ml_novelty_checkbox.setToolTip(
            "Reduces the correction's strength when the storm's intensity and "
            "RMW sit far from the training distribution, on the reasoning that "
            "the model is least trustworthy exactly where it is extrapolating.\n\n"
            "Full strength within 1 sigma, tapering to a floor of 0.25 beyond "
            "2.5 sigma. The effective strength is reported in the log and figure "
            "title whenever the taper is actually reducing it, so it is never "
            "silently different from the value in the box above.\n\n"
            "This is a mitigation, not a fix. On the 50 kt Edouard case it would "
            "have applied about 0.77, where preserving the signature needed "
            "roughly 0.17 -- the storm was only ~1.5 sigma out, because what made "
            "it unusual was weak intensity combined WITH an organised core, and a "
            "distance over two scalars cannot see that combination."
        )
        ml_form.addRow(self.ml_novelty_checkbox)

        self.ml_status_label = QLabel("")
        self.ml_status_label.setWordWrap(True)
        self.ml_status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._refresh_ml_status_label()
        ml_form.addRow(self.ml_status_label)
        v.addWidget(ml_box)


        self.calib_status_label = QLabel("")
        self.calib_status_label.setWordWrap(True)
        self.calib_status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._refresh_calib_status_label()
        v.addWidget(self.calib_status_label)

        self.reset_calib_btn = QPushButton("Reset persisted calibration")
        self.reset_calib_btn.setToolTip(
            "Clears the learned bias offset (calibration_state.py), if it's "
            "drifted somewhere clearly wrong. No confirmation dialog -- this "
            "is just a numeric offset, not sensitive data."
        )
        self.reset_calib_btn.clicked.connect(self.on_reset_calibration)
        v.addWidget(self.reset_calib_btn)

        loop_box = QGroupBox("Frame loop")
        loop_form = QFormLayout(loop_box)
        self.frame_count_combo = QComboBox()
        self.frame_count_combo.addItems(["Single frame", "5 frames", "10 frames", "15 frames"])
        self.frame_count_combo.setToolTip(
            "Multiple frames are generated at 10-minute intervals (or a "
            "multiple of 10 minutes, see 'Skip every' below) ending at "
            "the target time above (e.g. 5 frames ending at 19:00 UTC covers "
            "18:20-19:00 at the default 1x step). Real MW fusion happens per-frame (always on) -- "
            "and if the radar checkbox above is checked, radar is searched "
            "per-frame too -- slower, since those are real network searches "
            "for every frame, but frames generate concurrently (up to 4 at "
            "once) to help."
        )
        loop_form.addRow("Frames:", self.frame_count_combo)

        self.skip_every_spin = QSpinBox()
        self.skip_every_spin.setRange(1, 3)  # capped at 3x (30 min) per direct request
        self.skip_every_spin.setValue(1)
        self.skip_every_spin.setSuffix("x (\u00d710 min)")
        self.skip_every_spin.setToolTip(
            "Step size between frames, as a multiple of 10 minutes. 1x (default) "
            "is consecutive 10-minute frames, same as before. Higher values keep "
            "the SAME frame count but space them further apart -- e.g. 15 frames "
            "at 3x covers 30-minute steps (a 7-hour span) instead of 15 frames "
            "at 1x covering just 2h20m. Capped at 3x (30 min/frame)."
        )
        loop_form.addRow("Skip every:", self.skip_every_spin)

        v.addWidget(loop_box)

        self.generate_btn = QPushButton("Generate synthetic MW")
        self.generate_btn.clicked.connect(self.on_generate)
        v.addWidget(self.generate_btn)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        v.addWidget(self.frame_slider)
        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(self.frame_label)

        gif_row = QHBoxLayout()
        self.gif_fps_spin = QSpinBox()
        self.gif_fps_spin.setRange(1, 60)
        self.gif_fps_spin.setValue(4)
        self.gif_fps_spin.setSuffix(" fps")
        self.export_gif_btn = QPushButton("Export GIF...")
        self.export_gif_btn.setEnabled(False)
        self.export_gif_btn.clicked.connect(self.on_export_gif)
        gif_row.addWidget(self.gif_fps_spin)
        gif_row.addWidget(self.export_gif_btn)
        v.addLayout(gif_row)

        self.save_image_btn = QPushButton("Save Image...")
        self.save_image_btn.clicked.connect(self.on_save_image)
        self.save_image_btn.setEnabled(False)  # enabled once something's rendered
        v.addWidget(self.save_image_btn)

        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        # Without this the inner panel keeps its size-hint width and the
        # horizontal scrollbar appears instead of the content fitting.
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Width is constrained on the OUTER scroll area now. Constraining
        # the inner panel instead left no room for the scrollbar, which
        # then overlapped the controls.
        container = QWidget()
        cv = QVBoxLayout(container)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(scroll, 1)
        container.setMaximumWidth(380)
        self._controls_container_layout = cv
        return container

    def _refresh_calib_status_label(self):
        import calibration_state
        self.calib_status_label.setText(f"Persisted calibration: {calibration_state.get_status_summary()}")

    def _refresh_ml_status_label(self):
        """Report whether a trained checkpoint is actually present, so a
        strength value of 1.0 doesn't imply a correction is being applied
        when there's nothing to apply. Previously the only way to find
        this out was to run a generate and read the log."""
        path = ml_inference.DEFAULT_CHECKPOINT_PATH
        if not os.path.exists(path):
            self.ml_status_label.setText(
                "No trained checkpoint found -- strength has no effect until one exists "
                f"at {path}."
            )
            return
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            stamp = mtime.strftime("%Y-%m-%d %H:%M UTC")
            size_mb = os.path.getsize(path) / 1e6
            self.ml_status_label.setText(
                f"Checkpoint: {os.path.basename(path)} ({size_mb:.1f} MB, trained {stamp}). "
                "Set strength to 0 to compare against the uncorrected backbone."
            )
        except OSError:
            self.ml_status_label.setText(f"Checkpoint present at {path}.")

    def on_refresh_storms(self):
        """Populate the storm dropdown from NHC's live btk directory
        listing (best-track files modified in the last 24h). Done
        synchronously (a single quick HTTP GET, called on startup and on
        the Refresh button) rather than via a background worker thread --
        simpler, at the cost of briefly blocking the UI if NHC's server
        is slow to respond; acceptable given how quick this request
        normally is relative to everything else this app already does
        synchronously on click."""
        self.storm_combo.clear()
        try:
            storms = besttrack.fetch_recent_btk_storms(hours=24)
        except Exception as e:
            self.storm_combo.addItem(f"Error fetching storm list: {e}", None)
            return
        if not storms:
            self.storm_combo.addItem("No storms updated in the last 24h", None)
            return
        for s in storms:
            label = f"{s['display_name']} \u2014 updated {s['last_modified']:%Y-%m-%d %H:%M} UTC"
            self.storm_combo.addItem(label, (s["basin"], s["storm_num"], s["year"]))

    def on_reset_calibration(self):
        import calibration_state
        calibration_state.save_state(dict(calibration_state.DEFAULT_STATE))
        self._refresh_calib_status_label()
        self.log("Persisted calibration offset reset to zero.")

    def on_save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "synthetic_mw.png",
            "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
            self.log(f"Saved image to {path}")
        except Exception as e:
            self.log(f"ERROR saving image: {e}")

    def log(self, msg: str):
        self.status_log.append(msg)

    def on_generate(self):
        storm_data = self.storm_combo.currentData()
        if storm_data is None:
            self.log("No storm selected -- click 'Refresh storm list' or wait for it to load, then pick a storm.")
            return
        basin, storm_num, year = storm_data

        self.generate_btn.setEnabled(False)
        self.status_log.clear()
        self.frame_slider.setEnabled(False)
        self.export_gif_btn.setEnabled(False)

        sector = self.sector_combo.currentText()
        qdt = self.datetime_edit.dateTime().toUTC()
        target_time = datetime(
            qdt.date().year(),
            qdt.date().month(),
            qdt.date().day(),
            qdt.time().hour(),
            qdt.time().minute(),
            tzinfo=timezone.utc,
        )   # aware UTC, per timeutil's project convention -- this used to
            # strip back to naive immediately, which is how the GUI joined
            # the mixed-convention problem that cost two mining runs.
        params = {
            "satellite": self.satellite_combo.currentText(),
            "sector": sector,
            "target_time": target_time,
            "basin": basin,
            "storm_num": storm_num,
            "year": year,
        }

        frame_choice = self.frame_count_combo.currentText()
        n_frames = {"Single frame": 1, "5 frames": 5, "10 frames": 10, "15 frames": 15}[frame_choice]

        if n_frames == 1:
            self.log(f"[{APP_VERSION}] Requesting scene near {target_time:%Y-%m-%d %H:%M} UTC")
            creds = self.credentials_tab.get_credentials()
            auto_calibrate = self.mw_fusion_checkbox.isChecked()
            # NEXRAD was retired in 0.112 (arm_pyart commented out of
            # requirements); the checkbox offered a toggle for a path that
            # could no longer run. Removed from the GUI entirely.
            radar_check = False
            ml_strength = float(self.ml_strength_spin.value())
            # Module-level flag read inside apply_ml_correction. Set on the
            # GUI thread before the worker starts, so there is no race.
            ml_inference.NOVELTY_SCALING_ENABLED = self.ml_novelty_checkbox.isChecked()
            self.worker = GenerateWorker(params, creds, auto_calibrate, radar_check, ml_strength)
            self.worker.progress.connect(self.log)
            self.worker.finished.connect(self.on_single_finished)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()
        else:
            skip_every = self.skip_every_spin.value()
            step_minutes = 10 * skip_every
            start_time = target_time - timedelta(minutes=step_minutes * (n_frames - 1))
            self.log(f"[{APP_VERSION}] Requesting {n_frames}-frame loop: {start_time:%H:%M} \u2192 {target_time:%H:%M} UTC ({step_minutes}-min steps)")
            creds = self.credentials_tab.get_credentials()
            auto_calibrate = self.mw_fusion_checkbox.isChecked()
            # NEXRAD was retired in 0.112 (arm_pyart commented out of
            # requirements); the checkbox offered a toggle for a path that
            # could no longer run. Removed from the GUI entirely.
            radar_check = False
            ml_strength = float(self.ml_strength_spin.value())
            # Module-level flag read inside apply_ml_correction. Set on the
            # GUI thread before the worker starts, so there is no race.
            ml_inference.NOVELTY_SCALING_ENABLED = self.ml_novelty_checkbox.isChecked()
            self.log("MW fusion enabled for this loop (always on) -- slower than a hypothetical GOES-only loop, generating frames concurrently to help.")
            self.worker = GenerateLoopWorker(params, n_frames, creds, auto_calibrate, radar_check, skip_every, ml_strength)
            self.worker.progress.connect(self.log)
            self.worker.finished.connect(self.on_loop_finished)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()

    def on_failed(self, msg: str):
        self.log(f"ERROR: {msg}")
        self.generate_btn.setEnabled(True)

    def on_single_finished(self, payload):
        result, band13, band9, band7, band2, fix = payload
        self.log("Done.")
        self.generate_btn.setEnabled(True)
        self.frames = [(result, band13)]
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.export_gif_btn.setEnabled(False)
        self.frame_label.setText("")
        self.render_result(result, band13)

        # Paired GOES+real-MW export removed from the GUI: a 0.65-era
        # holdover from before ml_data_mining existed. Training data
        # now comes from the mining pipeline, which is reproducible,
        # vintage-tagged and resumable. A one-off checkbox export was
        # none of those.
        if False:
            try:
                path = training_data_export.export_training_example(
                    band13, band9, band7, band2, fix, result
                )
                if path:
                    self.log(f"Saved training example: {path}")
            except Exception as e:
                self.log(f"Training data export failed (non-blocking): {e}")

    def on_loop_finished(self, frames):
        self.log(f"Done -- {len(frames)} frames generated.")
        self.generate_btn.setEnabled(True)
        self.frames = frames
        self.frame_slider.setMaximum(len(frames) - 1)
        self.frame_slider.setEnabled(True)
        self.export_gif_btn.setEnabled(len(frames) >= 2)
        # setValue triggers on_frame_slider_changed via valueChanged *only*
        # if the value actually changes -- call it directly too so the
        # most recent frame renders even if the slider was already there.
        self.frame_slider.setValue(len(frames) - 1)
        self.on_frame_slider_changed(len(frames) - 1)

    def on_frame_slider_changed(self, value: int):
        if not self.frames or value >= len(self.frames):
            return
        result, band13 = self.frames[value]
        self.frame_label.setText(
            f"Frame {value + 1}/{len(self.frames)} \u2014 {result.scene_time:%Y-%m-%d %H:%M} UTC"
        )
        self.render_result(result, band13)

    def on_export_gif(self):
        if len(self.frames) < 2:
            self.log("ERROR: need at least 2 frames to export a GIF (generate a multi-frame loop first).")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export GIF", "synthetic_mw_loop.gif", "GIF Image (*.gif);;All Files (*)"
        )
        if not path:
            return
        try:
            from PIL import Image
            import tempfile
            import os as _os

            fps = self.gif_fps_spin.value()
            duration_ms = max(1, round(1000 / fps))

            self.log(f"Rendering {len(self.frames)} frames for GIF export...")
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_paths = []
                for i, (result, band13) in enumerate(self.frames):
                    self.render_result(result, band13)
                    frame_path = _os.path.join(tmpdir, f"frame_{i:03d}.png")
                    self.figure.savefig(frame_path, dpi=100, bbox_inches="tight")
                    frame_paths.append(frame_path)

                images = [Image.open(p).convert("RGB") for p in frame_paths]
                images[0].save(
                    path, save_all=True, append_images=images[1:],
                    duration=duration_ms, loop=0,
                )

            self.log(f"Exported {len(self.frames)}-frame GIF @ {fps}fps to {path}")
            # Restore the display to whatever the slider currently points at.
            self.on_frame_slider_changed(self.frame_slider.value())
        except Exception as e:
            self.log(f"ERROR exporting GIF: {e}")

    def render_result(self, result, band13):
        self.figure.clear()
        axes = self.figure.subplots(2, 2)

        ax = axes[0, 0]
        im = ax.pcolormesh(band13.lon, band13.lat, band13.values, cmap="Greys", shading="auto")
        ax.set_title("Input: GOES IR (band 13)")
        self.figure.colorbar(im, ax=ax, label="K", shrink=0.8)

        # Mark the centre the synthetic field was built around, and the
        # independent IR-derived estimate. Everything downstream -- radial
        # profile, eyewall ring, ML patch -- is positioned from the former,
        # and since 0.91 narrowed the eyewall ring the field has little
        # tolerance for getting it wrong. Drawing both makes a misplaced
        # centre visible on the frame instead of something that has to be
        # inferred afterwards from the output looking odd.
        try:
            fix_lat = getattr(result, "storm_lat", None)
            fix_lon = getattr(result, "storm_lon", None)
            cc = result.diagnostics.get("center_check")
            if fix_lat is None:
                fix_lat = result.diagnostics.get("storm_lat")
                fix_lon = result.diagnostics.get("storm_lon")
            if fix_lat is not None and fix_lon is not None:
                ax.plot(fix_lon, fix_lat, marker="+", color="#00b3ff", markersize=13,
                        markeredgewidth=2.0, linestyle="none", label="fix used")
            if cc and cc.get("method") == "eye" and fix_lat is not None:
                ax.plot(cc["lon"], cc["lat"], marker="x", color="#ff2d2d", markersize=11,
                        markeredgewidth=2.0, linestyle="none", label="IR eye")
                ax.plot([fix_lon, cc["lon"]], [fix_lat, cc["lat"]],
                        color="#ff2d2d", linewidth=1.0, alpha=0.8)
            if fix_lat is not None:
                ax.legend(loc="lower left", fontsize=6, framealpha=0.6)
        except Exception:
            # A marker failing to draw must never take the whole figure
            # down -- the imagery is the point, this is an annotation.
            pass

        # Same color-composite technique as real MW data (mw_composites.py,
        # per Brennan & Cangialosi NHC 2016), not a plain scalar colormap --
        # requires the V/H-producing version of generate_synthetic_mw.
        ax = axes[0, 1]
        try:
            c37 = mw_composites.composite_from_synthetic(result, 37)
            ax.pcolormesh(result.lon, result.lat, c37 / 255.0, shading="auto")
            ax.set_title("Synthetic 37 GHz (real color table)")
        except ValueError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True, fontsize=8)
            ax.set_title("Synthetic 37 GHz (composite unavailable)")

        ax = axes[1, 0]
        try:
            c89 = mw_composites.composite_from_synthetic(result, 89)
            ax.pcolormesh(result.lon, result.lat, c89 / 255.0, shading="auto")
            ax.set_title("Synthetic 89 GHz (real color table)")
        except ValueError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True, fontsize=8)
            ax.set_title("Synthetic 89 GHz (composite unavailable)")

        ax = axes[1, 1]
        # Fourth panel is normally the convective-signal diagnostic, but
        # can be swapped for the ML-free counterpart of the 89 GHz
        # composite. That comparison is exact and comes from the SAME
        # generate (see synthetic_algorithm's ml_free reconstruction), so
        # unlike running strength 1 and strength 0 as two separate frames
        # it isn't contaminated by a re-drawn noise floor or an advanced
        # calibration EMA between the two.
        ml_free = result.diagnostics.get("ml_free_fields") or {}
        show_ml_free = getattr(self, "ml_compare_checkbox", None) is not None and \
            self.ml_compare_checkbox.isChecked()
        if show_ml_free and ml_free:
            try:
                c89_free = mw_composites.build_89_color_composite(ml_free["v89"], ml_free["h89"])
                ax.pcolormesh(result.lon, result.lat, c89_free / 255.0, shading="auto")
                ax.set_title("89 GHz with ML correction OFF (same frame)")
            except Exception as e:
                ax.text(0.5, 0.5, f"ML-free composite unavailable:\n{e}",
                        ha="center", va="center", wrap=True, fontsize=8)
                ax.set_title("89 GHz ML-free (unavailable)")
        else:
            # Prefer the UNCERTAINTY map when there is one.
            #
            # The per-pixel ensemble spread was computed and then reduced
            # to a mean and a max, so the one thing it was good for --
            # saying WHERE the model is guessing -- was discarded, leaving
            # two numbers that say only how much on average. For a
            # synthetic product that is the most honest panel available:
            # it marks the regions the viewer should not trust.
            import numpy as _np      # not imported at module level here
            spread = (result.diagnostics.get("ml_correction", {}) or {}).get(
                "ensemble_spread_field_k")
            conv = result.diagnostics["convective_signal"]
            if spread is not None and _np.isfinite(spread).any():
                im = ax.pcolormesh(result.lon, result.lat, spread,
                                   cmap="magma", shading="auto")
                ax.set_title("Uncertainty: ensemble spread (K, calibrated)")
                self.figure.colorbar(im, ax=ax, shrink=0.8, label="K")
            else:
                im = ax.pcolormesh(result.lon, result.lat, conv,
                                   cmap="viridis", shading="auto")
                ax.set_title("Diagnostic: convective signal (0-1)")
                self.figure.colorbar(im, ax=ax, shrink=0.8)
                if show_ml_free and not ml_free:
                    ax.set_title("Diagnostic: convective signal "
                                 "(no ML correction to compare)")

        # EWRC and WVIR were wired to the progress log only, so on a saved
        # or exported figure they vanished entirely.
        _cycle = []
        _w = result.diagnostics.get("wvir")
        if _w and _w.get("stage") is not None:
            _cycle.append(f"WVIR stage {_w['stage']} ({_w['stage_name']}), "
                          f"conf {_w['confidence']:.2f}")
        _e = result.diagnostics.get("ewrc")
        if _e and _e.get("secondary_radius_km") is not None:
            _cycle.append(f"EWRC {_e['confidence']:.2f} "
                          f"(secondary ring {_e['secondary_radius_km']:.0f} km)")
        elif _e:
            _cycle.append("EWRC: no secondary eyewall")
        if _cycle:
            self.figure.text(0.01, 0.005, "   |   ".join(_cycle),
                             fontsize=7, va="bottom", ha="left", alpha=0.85)

        extrap_h = result.diagnostics.get("fix_extrapolated_hours") or 0.0
        extrap_str = (f", centre PROJECTED {extrap_h:.1f}h past last best track"
                      if extrap_h > 0.05 else "")

        # Independent IR centre check. Reported in the title because it
        # qualifies every structural number below it: an eyewall radius
        # measured from the wrong centre is precise and meaningless.
        cc = result.diagnostics.get("center_check")
        if cc and cc.get("method") == "eye":
            cc_str = (f"\nIR centre check: eye is {cc['offset_km']:.0f} km from the fix "
                      f"(bearing {cc['offset_bearing']:.0f} deg, confidence {cc['confidence']:.2f})")
            if cc["offset_km"] > 40.0:
                cc_str += " -- FIELD IS BUILT AROUND THE WRONG POINT"
        elif cc and cc.get("method") == "cdo_centroid":
            cc_str = ("\nIR centre check: no eye resolved (coldest-cloud centroid "
                      f"{cc['offset_km']:.0f} km away, low confidence)")
        else:
            cc_str = ""

        roci_km = result.diagnostics.get("roci_km")
        roci_str = f", ROCI {roci_km:.0f} km" if roci_km else ", ROCI n/a (used Vmax-based envelope)"
        daynight_str = "day" if result.diagnostics.get("is_daytime") else "night"
        band2_str = ", Band 2 used" if result.diagnostics.get("band2_used") else ""
        persisted = result.diagnostics.get("calibration_state", {})
        persisted_str = ""
        if persisted:
            persisted_str = (
                f" [persisted offset: 37GHz {persisted.get('offset_37', 0):+.1f}K"
                f" ({persisted.get('n_updates_37', 0)} obs), "
                f"89GHz {persisted.get('offset_89', 0):+.1f}K"
                f" ({persisted.get('n_updates_89', 0)} obs)]"
            )
        sources_used = result.diagnostics.get("fusion_sources_used", {"goes": True, "mw": False, "radar": False})
        cov37 = result.diagnostics.get("fusion_coverage_37", {})
        cov89 = result.diagnostics.get("fusion_coverage_89", {})
        fusion_parts = ["GOES"]
        if sources_used.get("mw"):
            fusion_parts.append(f"MW(cov {cov37.get('mw',0)*100:.0f}%/{cov89.get('mw',0)*100:.0f}%)")
        if sources_used.get("radar"):
            fusion_parts.append(f"radar(cov {cov37.get('radar',0)*100:.0f}%/{cov89.get('radar',0)*100:.0f}%)")
        fusion_str = f"\nFused: {' + '.join(fusion_parts)} (37GHz/89GHz coverage shown for non-GOES sources)"

        if result.diagnostics.get("calibration_applied"):
            calib_info = result.diagnostics.get("calibration_info", {})
            m37 = calib_info.get(37, {}).get("bias_k")
            m89 = calib_info.get(89, {}).get("bias_k")
            m37_str = f"{m37:+.1f}K" if m37 is not None else "n/a"
            m89_str = f"{m89:+.1f}K" if m89 is not None else "n/a"
            calib_str = (
                f"\nResidual vs {result.diagnostics.get('calibration_source', '?')}: "
                f"37GHz {m37_str}, 89GHz {m89_str} (fed into persisted baseline, "
                f"not applied to this frame -- already fused above){persisted_str}"
            )
        else:
            calib_str = (
                f"\n{result.diagnostics.get('calibration_reason', 'Fusion/calibration off')}"
                f"{persisted_str}"
            )

        radar = result.diagnostics.get("radar_check")
        if radar and radar.get("available"):
            max_dbz = radar.get("max_dbz")
            max_dbz_str = f"{max_dbz:.0f}dBZ" if max_dbz is not None else "n/a"
            et_str = ""
            if radar.get("max_echo_top_km") is not None:
                et_str = f", top {radar['max_echo_top_km']:.1f}km"
            radar_str = (
                f"\nRadar: {radar['station']} ({radar['distance_mi']:.0f}mi) @ "
                f"{radar['scan_time']:%H:%M} UTC, max {max_dbz_str}{et_str} near storm center"
            )
        elif radar:
            radar_str = f"\nRadar: unavailable ({radar.get('reason', '?')})"
        else:
            radar_str = ""

        # Stamp the ML strength and correction magnitude into the title.
        # Without this, two frames generated at different strengths are
        # visually distinguishable but not identifiable -- an A/B pair
        # saved to disk carries no record of which was which, which makes
        # comparison images unusable as evidence once they leave the app.
        ml = result.diagnostics.get("ml_stats") or {}
        structure = result.diagnostics.get("structure") or {}
        if ml.get("applied"):
            nov = ml.get("novelty_scale", 1.0)
            strength_txt = f"{ml['strength']:g}"
            if isinstance(nov, float) and nov < 1.0:
                strength_txt += (f" x novelty {nov:.2f} ({ml.get('novelty_sigma', 0):.1f}"
                                 f" sigma out) = {ml.get('effective_strength', 0):.2f}")
            ml_str = (
                f"\nML correction: strength {strength_txt}, "
                f"mean |delta| {ml['mean_abs_k']:.1f}K / peak {ml['peak_abs_k']:.1f}K, "
                f"PCT peak 37GHz {ml['pct_peak_37_pctwindow']:.0f}% of colour window "
                f"/ 89GHz {ml['pct_peak_89_pctwindow']:.0f}%"
            )
            if ml.get("physics_mismatch"):
                ml_str += ("\n  WARNING: this checkpoint was trained against the PRE-0.95 "
                           "backbone physics; its 37 GHz residuals are measured from a "
                           "backbone that no longer exists. Retrain, or set strength 0.")
            elif ml.get("frac_at_channel_clamp", 0) > 0.10:
                ml_str += f" [{ml['frac_at_channel_clamp']*100:.0f}% of patch pinned at clamp]"
            # Structural effect, which is the part magnitude cannot show:
            # the same mean |delta| describes both a correction that
            # resolved an eyewall and one that erased a core.
            for freq in (37, 89):
                before = structure.get(f"ml_free_{freq}")
                after = structure.get(f"corrected_{freq}")
                if before and after and after.get("eyewall_radius_km") is not None:
                    ratio_b, ratio_a = before.get("rmw_ratio"), after.get("rmw_ratio")
                    ratio_txt = ""
                    if isinstance(ratio_b, float) and isinstance(ratio_a, float):
                        ratio_txt = f", eyewall/RMW {ratio_b:.2f}->{ratio_a:.2f}"
                    ml_str += (
                        f"\n  {freq}GHz structure: eyewall "
                        f"{before['eyewall_radius_km']:.0f}->{after['eyewall_radius_km']:.0f} km"
                        f"{ratio_txt}, core area "
                        f"{(before.get('core_area_km2') or 0):.0f}->"
                        f"{(after.get('core_area_km2') or 0):.0f} km2"
                    )
        elif ml.get("reason") == "strength 0":
            ml_str = "\nML correction: DISABLED (strength 0) -- uncorrected parametric backbone"
        else:
            ml_str = ""

        self.figure.suptitle(
            f"{result.storm_id} — {result.scene_time:%Y-%m-%d %H:%M} UTC "
            f"(RMW est. {result.diagnostics['rmw_km']:.0f} km{roci_str}, {daynight_str}{band2_str}{extrap_str})"
            f"{fusion_str}{calib_str}{radar_str}{cc_str}{ml_str}",
            fontsize=9,
        )
        self.figure.tight_layout()
        self.canvas.draw()
        self.save_image_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_VERSION)
        self.resize(1200, 800)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        credentials_tab = CredentialsTab()

        tabs.addTab(GenerateTab(credentials_tab), "Generate")
        tabs.addTab(credentials_tab, "MW Data Credentials")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
