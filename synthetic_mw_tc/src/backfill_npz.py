"""Add what older NPZs are missing, in place, without re-mining (0.160).

    python backfill_npz.py [data_dir] [--workers N] [--only extra-ir|env] [--force] [--dry-run]

Two things older training examples lack:

1. The supplementary IR bands (ir_band11/10/15). Until 0.159 they were
   never fetched -- BAND_INFO did not list them and the ValueError was
   swallowed -- so every NPZ carries none, and a model trained on them has
   learned those channels are flat.
2. The ERA5 environment (env_* scalars, tcprimed_env.ENV_KEYS), new in
   0.160.

Neither touches the physics: the extra bands feed only the ML correction,
and the environment is metadata. So adding them afterwards gives exactly
what a fresh mine would have written, at a fraction of the cost -- no TC
PRIMED overpass reads, no generation, no GLM.

How each is reconstructed:

- Extra IR: from the SAME scan the NPZ was built from, or not at all.
  `scene_time_iso` is band 13's scan time but the sector is not recorded,
  so band 13 is re-fetched (default choice, then full disk, then each
  mesoscale sector) until it reproduces the stored ir_band13 EXACTLY; the
  extra bands then come from that scan via get_band_image_matching() and
  are regridded with generation's own _regrid_to() onto the NPZ's grid.
  The first version asked only for the right TIME: 2 of 8 live files got
  a mesoscale scan where band 13 was full disk, and 8.4 vs 10.3 um
  correlation fell from 1.000 to 0.857.
- Environment: tcprimed_env.env_at() at the scan time; one ranged read per
  storm, cached.

Files are rewritten atomically (temp file + os.replace), keep every
existing array unchanged, and a file that already has everything is left
alone unless --force.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import goes_fetch
import tcprimed_env
import training_data_export as tde
from ml_constants import MODEL_EXTRA_IR_BANDS

_print_lock = threading.Lock()


def _say(msg):
    with _print_lock:
        print(msg, flush=True)


def _storm_parts(storm_id: str):
    """'AL022024' -> ('AL', 2, 2024)."""
    s = str(storm_id).strip().upper()
    return s[:2], int(s[2:4]), int(s[4:8])


def missing_parts(names) -> dict:
    names = set(names)
    return {
        "extra_ir": [b for b in MODEL_EXTRA_IR_BANDS if f"ir_band{b}" not in names],
        "env": [k for k in tcprimed_env.ENV_KEYS if k not in names],
    }


def reproduce_band13(sat, t, clat, clon, stored):
    """The band-13 image this NPZ was built from, or None.

    The NPZ records the scan TIME but not the sector, and a frame's bands
    are only co-registered if they share one scan. So band 13 is re-fetched
    with the default choice, then full disk, then each mesoscale sector,
    and accepted only when it reproduces the stored ir_band13 EXACTLY."""
    want = np.asarray(stored, dtype=np.float32)
    for sector_only in (None, "F", "M1", "M2"):
        try:
            img = goes_fetch.get_band_image_any_sector(sat, 13, t, clat, clon,
                                                       sector_only=sector_only)
        except Exception:
            img = None
        if img is None:
            continue
        got = np.asarray(img.values, dtype=np.float32)
        if got.shape == want.shape and np.array_equal(got, want, equal_nan=True):
            return img
    return None


def backfill_one(path: str, do_ir=True, do_env=True, force=False, dry_run=False) -> str:
    """Returns a one-word outcome: 'complete', 'updated', 'would-update',
    'partial' or 'failed:<reason>'."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    todo = missing_parts(data)
    need_ir = do_ir and (force or todo["extra_ir"])
    need_env = do_env and (force or todo["env"])
    if not (need_ir or need_env):
        return "complete"
    if dry_run:
        return "would-update"

    t = datetime.fromisoformat(str(data["scene_time_iso"]))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    lat, lon = data["lat"], data["lon"]
    clat, clon = float(data["storm_lat"]), float(data["storm_lon"])
    added, partial = [], False

    if need_ir:
        sat = (str(data["goes_satellite"]) if "goes_satellite" in data
               else goes_fetch.select_satellite(clat, clon, t))
        if sat is None:
            return "failed:no_satellite"
        bands = MODEL_EXTRA_IR_BANDS if force else todo["extra_ir"]
        anchor = reproduce_band13(sat, t, clat, clon, data["ir_band13"])
        if anchor is None:
            # Could not find the scan this NPZ was built from, so any extra
            # band would be from a DIFFERENT scan -- misregistered against
            # band 13. Leave them absent (the trainer reports the mix).
            partial = True
        else:
            from synthetic_algorithm import _regrid_to
            with ThreadPoolExecutor(max_workers=len(bands) or 1) as pool:
                imgs = dict(zip(bands, pool.map(
                    lambda b: goes_fetch.get_band_image_matching(sat, b, anchor, clat, clon),
                    bands)))
            for b in bands:
                img = imgs.get(b)
                if img is None:
                    partial = True
                    continue
                data[f"ir_band{b}"] = np.asarray(_regrid_to(lat, lon, img), dtype=np.float32)
                added.append(f"ir_band{b}")
            if "goes_satellite" not in data:
                data["goes_satellite"] = np.str_(sat)

    if need_env:
        basin, num, season = _storm_parts(data["storm_id"])
        series = tcprimed_env.load_env_series(basin, num, season)
        env = tcprimed_env.env_at(series, t)
        for k, v in env.items():
            data[k] = np.float32(v)
        added.append("env")
        if series is None:
            partial = True

    if not added:
        return "failed:nothing_fetched"
    tmp = path[:-4] + ".backfill_tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    return "partial" if partial else "updated"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", nargs="?", default=tde.DEFAULT_EXPORT_DIR)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", choices=("extra-ir", "env"))
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even where the keys already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    # Temp files left by an interrupted run (this script's, or a mine's).
    # Only ones older than 10 minutes: a run writing RIGHT NOW is not
    # interrupted, and its temp file is about to be renamed into place.
    stale = [f for f in glob.glob(os.path.join(a.data_dir, "*.npz"))
             if (".tmp" in os.path.basename(f) or f.endswith(".backfill_tmp.npz"))
             and time.time() - os.path.getmtime(f) > 600]
    for f in stale:
        try:
            os.remove(f)
        except OSError:
            pass
    if stale:
        print(f"Removed {len(stale)} temp file(s) left by an interrupted run")
    files = tde.list_training_files(a.data_dir)
    if not files:
        print(f"no NPZs in {a.data_dir}")
        return 1
    do_ir, do_env = a.only in (None, "extra-ir"), a.only in (None, "env")
    import ml_data_mining
    workers = ml_data_mining.memory_safe_workers(a.workers, log=print)
    print(f"{len(files)} NPZ(s) in {a.data_dir}; "
          f"{'extra IR' if do_ir else ''}{' + ' if do_ir and do_env else ''}"
          f"{'environment' if do_env else ''}; {workers} worker(s)")

    outcomes: dict = {}
    t0 = time.time()
    done = [0]

    def run(path):
        try:
            r = backfill_one(path, do_ir, do_env, a.force, a.dry_run)
        except Exception as e:
            r = f"failed:{type(e).__name__}: {str(e)[:80]}"
        with _print_lock:
            outcomes[r] = outcomes.get(r, 0) + 1
            done[0] += 1
            n = done[0]
        if r.startswith("failed") or r == "partial":
            _say(f"  {os.path.basename(path)}: {r}")
        if n % 50 == 0 or n == len(files):
            _say(f"  ... {n}/{len(files)} ({time.time() - t0:.0f} s)")
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, files))
    print("Done:", ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items())))
    eb = goes_fetch.EXTRA_BAND_STATS
    if eb["requested"]:
        miss = sum(eb["missing"].values())
        print(f"Extra IR bands: {eb['requested'] - miss} of {eb['requested']} fetched")
    return 0 if not any(k.startswith("failed") for k in outcomes) else 2


if __name__ == "__main__":
    sys.exit(main())
