"""
Entry point for the ML pipeline: turn cached TC-PRIMED files into
training examples, then train the correction model.

Run interactively and pick a step:
    python run_ml_pipeline.py

Or skip the prompt:
    python run_ml_pipeline.py --step 1     # mine only
    python run_ml_pipeline.py --step 2     # train only (uses existing .npz)
    python run_ml_pipeline.py --step all   # both

Step 1 options (mining):
    python run_ml_pipeline.py --step 1 --start 2018 --end 2025 --stream true
    python run_ml_pipeline.py --step 1 --start 2022 --agency NHC
    python run_ml_pipeline.py --estimate --start 2018 --end 2025

--agency NHC means AL/EP/CP; JTWC means WP/IO/SH. The JTWC basins are
outside GOES coverage, so they will almost entirely skip until
Himawari/Meteosat ingest exists -- the option warns and runs anyway
rather than silently mining nothing.

--estimate lists what would be processed and exits. Listing S3 fetches no
file bodies, so it costs seconds and is the right way to size a run
before committing to it.

Step 3 fits the V/H calibration constants against the mined data. This is
the direct attack on the bias term, which barely moved between 298 and
799 examples (11.09 -> 11.51 K) while spread fell -- the signature of
systematic constant error rather than something more data fixes. It
prints a diff and stops; --apply-fit writes it.

Step 4 runs the whole sequence: mine -> fit -> re-mine -> train. Two
mines is not redundancy: the fit needs data generated under the CURRENT
backbone, and applying it MOVES the backbone, so the first dataset is
measured against something that no longer exists. Without --apply-fit,
step 4 stops at the diff and changes nothing.

Step 2 alone is the common case once mining has already been done --
mining is network-bound and slow, training is not, so there is no reason
to re-mine every time a training setting changes.
"""
from __future__ import annotations

import argparse
import sys

import ml_data_mining
import ml_train
import training_data_export as tde

# --- mining knobs -----------------------------------------------------
MAX_PER_STORM = None
MAX_WORKERS = 10          # concurrent GOES downloads; raise on a fast line

# Link speeds the estimate reports, in Mbps. Override with --mbps to see
# the figure for your actual connection rather than these defaults.
LINK_MBPS = (50, 200)

# Rough per-example CPU cost, single-core: scattered-point regridding of
# the MW swath, gaussian filtering, composites, structure metrics and the
# IR centre check. A guess, not a measurement -- it exists so a fast link
# does not make the estimate promise a runtime the CPU cannot deliver.
# Correct it once a real season has been timed.
SECONDS_PER_EXAMPLE_CPU = 4.0

# Basins with no GOES coverage at all. Skipped before any best-track
# lookup so they cost nothing. Set to () to attempt them anyway once
# Himawari-9 support exists.
EXCLUDE_BASINS = ("WP", "IO", "SH")

# --- training knobs ---------------------------------------------------
# batch_size 4 is an estimate for 256px patches on 6GB, not a measurement.
# If it hits CUDA out-of-memory, drop to 2. If epoch 1 runs comfortably,
# raising it is the biggest remaining training speedup.
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 50

# torch.compile is off by default: on Windows it needs a separate
# version-matched triton-windows package, and a 3050 reports "not enough
# SMs" for its better optimizations anyway. Set True to try it -- the
# code now forces a warmup pass so a failure falls back cleanly instead
# of crashing mid-training.
COMPILE_MODEL = False

# --- correction-quality knobs ----------------------------------------
# Weight on the PCT-space loss term. The per-channel L1 term alone cannot
# see the V/H combination the colour composites render, and that
# combination is amplified up to 3.36x at 37 GHz -- so a model selected
# on L1 alone is selected on a quantity only loosely related to what the
# output looks like. Set to 0 to reproduce the pre-0.88 objective.
PCT_LOSS_WEIGHT = ml_train.PCT_LOSS_WEIGHT

# Probability of blanking the vmax/RMW conditioning layers per sample.
# Counteracts the model leaning on those constant layers instead of the
# imagery -- the suspected cause of a real failure where a 50 kt storm
# with a genuine core had that core suppressed. Set to 0 to disable.
SCALAR_DROPOUT_P = ml_train.SCALAR_DROPOUT_P


# --- Mining source ---------------------------------------------------
# STREAM_FROM_S3 reads TC-PRIMED overpasses in place on the bucket
# (s3_range_reader) and writes no source files to disk. That is what
# lifts the dataset-size ceiling from "how much fits on the laptop" to
# "how long am I willing to leave it running".
#
# With streaming OFF, mining only sees whatever is already in the local
# TC-PRIMED cache -- which is what produced the 298-example dataset, and
# will keep producing roughly that no matter how many times it is re-run.
STREAM_FROM_S3 = True

# Seasons to mine when streaming. REQUIRED for streaming, since there is
# no local cache to enumerate. Start small: one season is a realistic
# first run and tells you the per-storm cost before committing to a
# decade. TC-PRIMED covers 1998-2023ish; the GOES-R ABI era this
# project's IR comes from starts 2017 (GOES-16) / 2018-19 (GOES-17/18),
# so seasons before ~2018 will mostly skip on missing GOES.
SEASONS = (2022,)

# Basins to include. EXCLUDE_BASINS below still applies on top.
BASINS = ("AL", "EP", "CP")

# Agency -> basin mapping, for --agency.
#
# NHC runs the Atlantic and the eastern/central Pacific; JTWC covers the
# western Pacific, north Indian Ocean and southern hemisphere. The split
# is operationally meaningful, but for THIS project it is also a hard
# capability boundary: the JTWC basins sit outside GOES coverage, and the
# IR side of every training example comes from GOES-R ABI. Mining them
# today produces almost nothing but "no GOES coverage" skips. The mapping
# exists so the option is ready when Himawari/Meteosat support lands, and
# --agency warns rather than silently wasting a long run.
AGENCY_BASINS = {
    "NHC": ("AL", "EP", "CP"),
    "JTWC": ("WP", "IO", "SH"),
    "all": ("AL", "EP", "CP", "WP", "IO", "SH"),
}
GOES_COVERED_BASINS = ("AL", "EP", "CP")


def estimate_mining() -> None:
    """Pre-flight: count what a streaming run would process, before it
    starts. Listing S3 is cheap (no file bodies), so this costs seconds
    and answers 'how long is this going to take' with a number rather
    than a guess."""
    import tcprimed_ingest as tp

    total_files = 0
    total_bytes = 0
    storms = 0
    per_instrument: dict = {}
    other: dict = {}
    for season in SEASONS:
        for storm in tp.list_available_storms(season_start=season, season_end=season,
                                              basins=BASINS or None):
            if storm["basin"] in EXCLUDE_BASINS:
                continue
            storms += 1
            for f in tp.list_storm_overpass_files(
                basin=storm["basin"], storm_num=storm["storm_num"],
                season=storm["season"],
            ):
                # Filter to the instruments this project can actually
                # read. Without this the count includes the whole GPM
                # constellation and over-reports several times over.
                inst = f.get("instrument")
                if inst not in tp.SUPPORTED_INSTRUMENTS:
                    other[inst] = other.get(inst, 0) + 1
                    continue
                per_instrument[inst] = per_instrument.get(inst, 0) + 1
                total_files += 1
                total_bytes += f.get("size_bytes", 0)

    print(f"Seasons {list(SEASONS)}, basins {list(BASINS)}")
    print(f"  storms: {storms}")
    print(f"  usable overpasses: {total_files}"
          + (f"  ({', '.join(f'{k} {v}' for k, v in sorted(per_instrument.items()))})"
             if per_instrument else ""))
    if other:
        skipped = sum(other.values())
        print(f"  other instruments present but not read: {skipped} "
              f"({', '.join(f'{k} {v}' for k, v in sorted(other.items(), key=lambda x: -x[1]))})")
    print(f"  total size on S3: {tp.format_bytes(total_bytes)}")
    # MEASURED on real files, replacing an earlier 20-40% estimate that
    # came from a simulated HDF5 access pattern. Across 13 TC PRIMED
    # overpasses the library touched 93% of each file: the variables this
    # project reads span most of a 13-20 MB object, so partial reads save
    # very little. Files under 64 MB are now fetched whole in one request
    # (see s3_range_reader.WHOLE_OBJECT_MAX_BYTES).
    print(f"  expected transfer: ~{tp.format_bytes(int(total_bytes * 0.95))} "
          f"(measured ~93-100% of each file is read)")
    print(f"  written to local disk: 0 bytes of source data")
    # --- The GOES side, quantified -----------------------------------
    # Saying "the GOES fetch dominates" without a number is not useful
    # when that is the thing deciding the runtime. These are ASSUMPTIONS,
    # labelled as such: the survival rate especially varies with basin
    # and season (a storm outside GOES view contributes nothing), and the
    # only way to pin it down is to run one season and look at the skip
    # reasons.
    from ml_constants import EXTRA_IR_FETCH_LIMIT
    bands = 4 + min(EXTRA_IR_FETCH_LIMIT, 3)      # 13/9/7/2 plus extras
    goes_mb_per_band = 12                          # typical ABI mesoscale sector
    for survival, label in ((0.4, "pessimistic"), (0.7, "optimistic")):
        examples = int(total_files * survival)
        goes_gb = examples * bands * goes_mb_per_band / 1000
        print(f"  if {int(survival*100)}% of overpasses yield an example ({label}):")
        print(f"      ~{examples} training examples, ~{goes_gb:.1f} GB of GOES traffic "
              f"({bands} bands each)")
        # Wall-clock, with the assumption stated. Transfer-bound is the
        # right model here: the parametric algorithm is fast next to
        # moving tens of GB, and mining runs several workers in parallel.
        total_gb = goes_gb + total_bytes * 0.3 / 1e9
        for mbps in LINK_MBPS:
            hours = total_gb * 8 * 1000 / mbps / 3600
            print(f"      at {mbps} Mbps sustained: ~{hours:.1f} h of transfer")
        # A transfer-bound estimate stops being the right model on a fast
        # link. Each example also costs CPU -- scattered-point regridding
        # of the MW swath, several gaussian filters, the composites, the
        # structure metrics and the centre check -- which no amount of
        # bandwidth removes. SECONDS_PER_EXAMPLE_CPU is a rough per-core
        # figure; with max_workers in flight the wall clock is that
        # divided by however many cores actually keep up.
        cpu_hours = examples * SECONDS_PER_EXAMPLE_CPU / 3600
        print(f"      CPU floor (~{SECONDS_PER_EXAMPLE_CPU}s/example, serial): "
              f"~{cpu_hours:.1f} h; less with workers, but it does not scale "
              f"with bandwidth")
    print()
    if max(LINK_MBPS) >= 500:
        print("  On a link this fast the transfer is NOT the constraint -- per-example")
        print(f"  CPU is. Raising MAX_WORKERS (currently {MAX_WORKERS}) helps only until")
        print("  the cores saturate, so expect the CPU floor above, not the transfer line.")
    else:
        print("  So the GOES fetch is likely both the slower AND the larger half.")
    print("  TC-PRIMED streaming removed a STORAGE ceiling, not a time one.")
    print()
    print("  Survival rate is the big unknown -- run one season first and read")
    print("  the skip reasons, then extrapolate. Storms outside GOES view, or")
    print("  before GOES-16/17/18 existed, contribute nothing.")


def preflight() -> bool:
    """Prove the generation pipeline works before committing to a long run.

    Runs one synthetic example end to end -- generate, then export -- with
    no network. Two mining runs were lost to errors that would have shown
    up here in under a second: a naive/aware datetime clash, and a missing
    CALIBRATION key. Both failed identically on every one of thousands of
    overpasses, and both were invisible until the summary.

    Deliberately synthetic rather than a real overpass: this is checking
    that the CODE PATH is intact, and a real fetch would confound that
    with network and data-availability problems.
    """
    import numpy as np
    from datetime import datetime, timezone
    from data_types import BandImage, StormFix
    from synthetic_algorithm import generate_synthetic_mw

    try:
        n = 96
        lat, lon = np.mgrid[20:10:n * 1j, -160:-146:n * 1j]
        d = np.sqrt((lat - 15.0) ** 2
                    + ((lon + 153.0) * np.cos(np.radians(15.0))) ** 2) * 111.32
        ir = np.where(d < 20, 282.0, 300.0 - 108.0 * np.exp(-0.5 * (d / 110.0) ** 2))
        t = datetime(2022, 9, 2, 16, 0, tzinfo=timezone.utc)
        mk = lambda v, b: BandImage(band=b, satellite="GOES-18", scene_time=t,
                                    values=v, lat=lat, lon=lon, units="K",
                                    mesoscale_sector="M1")
        fix = StormFix(storm_id="XX012022", valid_time=t, lat=15.0, lon=-153.0,
                       vmax_kt=95.0, mslp_mb=960.0, rmw_nm=20.0, roci_nm=200.0)
        r = generate_synthetic_mw(mk(ir, 13), mk(ir + 3, 9), mk(ir + 6, 7), fix,
                                  real_swath=None)
        for name in ("v37", "h37", "v89", "h89"):
            arr = getattr(r, name)
            if not np.all(np.isfinite(arr)):
                print(f"PREFLIGHT FAILED: {name} contains non-finite values")
                return False
            if not (50.0 < float(np.nanmin(arr)) and float(np.nanmax(arr)) < 350.0):
                print(f"PREFLIGHT FAILED: {name} outside physical range "
                      f"({np.nanmin(arr):.0f}-{np.nanmax(arr):.0f} K)")
                return False
    except Exception as e:
        import traceback
        print(f"PREFLIGHT FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()
        print("The generation path is broken. Mining would fail on EVERY")
        print("overpass with this same error -- fix it before running.")
        return False

    # --- Time-convention check ---------------------------------------
    # Generation alone does NOT cross the boundary that failed twice: the
    # mining path compares a TC-PRIMED scene time against best-track fix
    # times, and against goes_fetch's satellite-era table. Exercise that
    # comparison directly, with mixed awareness, because a synthetic
    # generate happily passes while mining fails on every overpass.
    try:
        from datetime import datetime, timedelta, timezone
        import besttrack
        import goes_fetch
        import solar
        from data_types import StormFix

        aware = datetime(2022, 9, 2, 16, 0, tzinfo=timezone.utc)
        naive = aware.replace(tzinfo=None)  # naive-ok: deliberately constructing the bad input this check exists to detect
        for fixes_tz in (aware, naive):
            fixes = [StormFix(storm_id="XX012022", valid_time=fixes_tz + timedelta(hours=h),
                              lat=25.0, lon=-70.0, vmax_kt=90.0, mslp_mb=960.0,
                              rmw_nm=25.0, roci_nm=200.0) for h in (-6, 0, 6)]
            for target in (aware, naive):
                got = besttrack.interpolate_fix(fixes, target)
                if got is None or got.valid_time.tzinfo is None:
                    print("PREFLIGHT FAILED: interpolate_fix returned a naive time")
                    return False
        for t in (aware, naive):
            goes_fetch.select_satellite(25.0, -70.0, t)
            solar.is_daytime(25.0, -70.0, t)
    except Exception as e:
        print(f"PREFLIGHT FAILED (time convention): {type(e).__name__}: {e}")
        print()
        print("This is the failure that cost two mining runs -- naive and")
        print("tz-aware datetimes meeting at a comparison. See timeutil.py.")
        return False

    print("Preflight OK: generation path intact and time conventions agree "
          "(synthetic, no network).")
    return True


def run_mining() -> int:
    print("=" * 70)
    if STREAM_FROM_S3:
        print("Step 1: streaming TC-PRIMED from S3 into training examples")
        print("=" * 70)
        estimate_mining()
    else:
        print("Step 1: processing LOCAL TC-PRIMED cache into training examples")
        print("       (STREAM_FROM_S3 is False -- this only sees files already")
        print("        on disk, and will not grow the dataset)")
        print("=" * 70)
    if not preflight():
        return 0

    result = ml_data_mining.mine_local_tcprimed_cache(
        progress_callback=print, max_workers=MAX_WORKERS,
        max_per_storm=MAX_PER_STORM,
        exclude_basins=EXCLUDE_BASINS,
        stream=STREAM_FROM_S3,
        seasons=SEASONS if STREAM_FROM_S3 else (),
        basins=BASINS if STREAM_FROM_S3 else (),
    )
    print()
    if result.get("aborted"):
        print()
        print("*** RUN ABORTED EARLY: " + str(result.get("abort_reason")) + " ***")
        print("*** Nothing was saving and one failure dominated. ***")
        print()
    # Where the time actually went. Reported rather than reasoned about:
    # the last two performance assumptions in this project (block size,
    # and which paths were serial) were both wrong until measured.
    _pt = getattr(ml_data_mining, "_phase_totals", {})
    _n = _pt.get("_n", 0)
    if _n:
        print()
        print(f"Time per saved example, averaged over {_n:,} of them:")
        for _k in ("mw_read", "goes", "extra_glm", "generate", "export"):
            if _k in _pt:
                print(f"  {_k:<10} {_pt[_k]/_n:6.2f} s")
        print(f"  {'total':<10} {sum(v for k, v in _pt.items() if k != '_n')/_n:6.2f} s"
              f"   (wall clock is this divided by workers)")
    print()
    print(f"Attempted: {result['attempted']}, saved: {result['saved']}")
    print(f"Storms processed: {result['storms_processed']}")
    if result["skipped_reasons"]:
        print("Skip reasons (not necessarily problems -- most storm-times won't")
        print("have both GOES coverage and a usable pass, that's expected):")
        for reason, count in sorted(result["skipped_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    return result["saved"]


def run_calibration(apply_fit: bool = False) -> bool:
    """Step 3: fit the V/H constants against the mined dataset.

    This is the direct attack on the bias term. Across 298 and then 799
    examples the measured bias barely moved (11.09 -> 11.51 K) while
    spread fell, which is the signature of systematic constant error
    rather than anything more data will fix.

    Prints the diff and stops unless --apply-fit is given. Applying
    changes the backbone, which invalidates every stored residual, so it
    bumps VH_PHYSICS_ID and requires a re-mine -- a decision, not a side
    effect. Returns True if constants were written.
    """
    import calibrate_constants as cc

    print("=" * 70)
    print("Step 3: fitting calibration constants against the mined data")
    print("=" * 70)
    result = cc.fit_from_dataset(progress_callback=print)
    if not result.get("fitted"):
        print(result.get("note", "no fit produced"))
        return False

    print()
    print(cc.format_result(result))
    if not apply_fit:
        print()
        print("NOT applied. Review the diff above, then re-run with --apply-fit.")
        print("Applying rewrites CALIBRATION, bumps VH_PHYSICS_ID and requires")
        print("a re-mine, because every stored residual is measured against the")
        print("backbone these constants define.")
        return False

    out = cc.apply_fitted_constants(result)
    print()
    print(f"Applied: {', '.join(out['applied'])}")
    if out["skipped"]:
        print(f"Skipped (ambiguous in source): {', '.join(out['skipped'])}")
    print(f"Backup:  {out['backup']}")
    print(f"VH_PHYSICS_ID -> {out['physics_id']}")
    return True


def run_training() -> None:
    summary = tde.dataset_summary()
    print()
    print(f"Total training examples available: {summary['n_examples']}")
    print(f"Sensors: {summary['sensors']}")

    if summary["n_examples"] == 0:
        print("\nNo training examples on disk. Run step 1 first.")
        sys.exit(1)
    if summary["n_examples"] < 20:
        print("\nWARNING: small dataset. Training will run, but don't trust the")
        print("resulting corrections yet.")

    print()
    print("=" * 70)
    print("Step 2: training")
    print("=" * 70)
    # Push the module-level knobs through so editing them here actually
    # takes effect (train() reads them from ml_train's namespace).
    ml_train.PCT_LOSS_WEIGHT = PCT_LOSS_WEIGHT
    ml_train.SCALAR_DROPOUT_P = SCALAR_DROPOUT_P
    print(f"PCT loss weight: {PCT_LOSS_WEIGHT}, scalar dropout: {SCALAR_DROPOUT_P}")
    ml_train.train(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                   epochs=EPOCHS, compile_model=COMPILE_MODEL)


def prompt_for_step() -> str:
    print("Which step would you like to run?")
    print("  1) Mine cached TC-PRIMED files into .npz training examples")
    print("  2) Train the correction model on existing .npz files")
    print("  3) Both (mine, then train)")
    try:
        choice = input("Enter 1, 2 or 3 [3]: ").strip() or "3"
    except EOFError:
        return "all"
    return {"1": "1", "2": "2", "3": "all"}.get(choice, "all")


def main():
    # Declared up front: these module-level settings are the single source
    # of truth for mining, and the CLI overrides them in place rather than
    # threading parameters through, so running the file with no arguments
    # behaves exactly as it did before. (Must precede any read of them,
    # including in argparse help strings.)
    global SEASONS, BASINS, STREAM_FROM_S3, EXCLUDE_BASINS, LINK_MBPS

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["1", "2", "3", "4", "all"],
                    help="1=mine, 2=train, 3=fit constants, 4=full sequence "
                         "(mine -> fit -> re-mine -> train), all=1+2. "
                         "Omit to be prompted.")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent overpasses (default 10). This work is "
                         "round-trip bound, so more helps until S3 throttling "
                         "or local CPU takes over.")
    ap.add_argument("--max-per-storm", type=int, default=None,
                    help="cap examples per storm, spread across its lifetime. "
                         "Overpasses of one storm are highly correlated, so "
                         "e.g. 6 cuts runtime a lot for a modest loss of "
                         "diversity.")
    ap.add_argument("--apply-fit", action="store_true",
                    help="let step 3/4 WRITE the fitted constants (rewrites "
                         "CALIBRATION, bumps VH_PHYSICS_ID, forces a re-mine)")
    ap.add_argument("--start", type=int, metavar="YEAR",
                    help=f"first season to mine (default {SEASONS[0]})")
    ap.add_argument("--end", type=int, metavar="YEAR",
                    help="last season to mine, inclusive (default: same as --start)")
    ap.add_argument("--stream", choices=["true", "false"],
                    help=f"read TC-PRIMED off S3 without downloading "
                         f"(default {str(STREAM_FROM_S3).lower()})")
    ap.add_argument("--agency", choices=sorted(AGENCY_BASINS),
                    help="NHC = AL/EP/CP, JTWC = WP/IO/SH, all = both")
    ap.add_argument("--basins", metavar="AL,EP",
                    help="explicit basin list, overrides --agency")
    ap.add_argument("--mbps", type=float, metavar="N",
                    help="your link speed, for the time estimate (e.g. 1050)")
    ap.add_argument("--estimate", action="store_true",
                    help="show what step 1 would process, then exit without mining")
    args = ap.parse_args()

    if args.start is not None:
        end = args.end if args.end is not None else args.start
        if end < args.start:
            ap.error(f"--end {end} is before --start {args.start}")
        SEASONS = tuple(range(args.start, end + 1))
    elif args.end is not None:
        ap.error("--end requires --start")
    if args.mbps:
        LINK_MBPS = (args.mbps,)
    if args.stream is not None:
        STREAM_FROM_S3 = (args.stream == "true")
    if args.basins:
        BASINS = tuple(b.strip().upper() for b in args.basins.split(",") if b.strip())
    elif args.agency:
        BASINS = AGENCY_BASINS[args.agency]
    if args.basins or args.agency:
        # EXCLUDE_BASINS defaults to the non-GOES basins, which would
        # silently cancel an explicit --agency JTWC. An explicit basin
        # choice wins; otherwise the flag would appear to work and mine
        # nothing.
        EXCLUDE_BASINS = tuple(b for b in EXCLUDE_BASINS if b not in BASINS)
        uncovered = [b for b in BASINS if b not in GOES_COVERED_BASINS]
        if uncovered:
            print(f"WARNING: basin(s) {', '.join(uncovered)} are outside GOES coverage.")
            print("  Every training example needs GOES-R ABI infrared, so these will")
            print("  almost all skip with 'no_goes'. They become useful once")
            print("  Himawari/Meteosat ingest exists -- not before.")
            print()

    if args.estimate:
        if not STREAM_FROM_S3:
            print("--estimate only applies to streaming mode (--stream true).")
            return
        estimate_mining()
        return

    global MAX_WORKERS, MAX_PER_STORM
    if args.workers:
        MAX_WORKERS = args.workers
    if args.max_per_storm:
        MAX_PER_STORM = args.max_per_storm
        print(f"Capping at {MAX_PER_STORM} example(s) per storm, spread across "
              f"each storm's lifetime.")

    step = args.step or prompt_for_step()

    if step == "3":
        run_calibration(apply_fit=args.apply_fit)
        return

    if step == "4":
        # The full sequence, with the decision point kept explicit.
        #
        # Two mines are unavoidable: the fit needs data generated under the
        # CURRENT backbone, and applying it moves the backbone, so the
        # first dataset is measured against something that no longer
        # exists. Chaining them without the re-mine would train on a
        # dataset the vintage guard is right to refuse.
        print("Full sequence: mine -> fit constants -> re-mine -> train")
        print()
        if run_mining() == 0:
            print("Mining saved nothing; stopping before the fit.")
            return
        if not run_calibration(apply_fit=args.apply_fit):
            print()
            print("Stopping before the re-mine. Re-run with --apply-fit once the")
            print("fitted constants look right; nothing has been changed.")
            return
        print()
        print("Constants changed, so the mined dataset is now a stale vintage.")
        print("Re-mining under the new backbone.")
        run_mining()
        run_training()
        return

    if step in ("1", "all"):
        saved = run_mining()
        if step == "all" and saved == 0:
            print()
            print("Mining saved nothing new. Continuing to training with whatever")
            print("examples already exist on disk.")

    if step in ("2", "all"):
        run_training()

    if step == "1":
        print()
        print("Mining complete. Run with --step 2 to train.")


if __name__ == "__main__":
    main()
