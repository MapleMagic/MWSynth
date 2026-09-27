"""Offline backbone replay: rerun the physics on stored examples, no network.

Added in 0.163. Every NPZ already holds everything the physics backbone
reads -- the six GOES bands on their grid, lightning, and the storm's
position, intensity, RMW and ROCI -- plus the real MW target. So a
candidate physics change can be scored against every mined example in
minutes, BEFORE anyone re-mines for it.

Why the backbone can be rebuilt without the real swath: export stores
`backbone = physics - ml_delta - baseline_shift`, where baseline_shift is
the only thing the real swath contributes and ml_delta was zero (mining
runs the model off). So generation with real_swath=None and ml_strength=0
must give back exactly the stored backbone -- and `--check` verifies that
it does, per file, before any variant result is trusted.

    python backbone_replay.py --check [data_dir] [--limit N]
    python backbone_replay.py --variant NAME [data_dir] [--limit N] [--workers N]
    python backbone_replay.py --list

A VARIANT is a named set of physics overrides, applied inside the worker
around each generation and then removed (see VARIANTS). Scoring uses the
same measures the h37 diagnostic reported, so a variant's effect reads
directly against that table: residual by radius and intensity for all four
channels, shear quadrants, and the inner-core cold-top slope.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

import training_data_export as tde

CHANNELS = ("v37", "h37", "v89", "h89")
R_EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8])
V_BANDS = ((0, 34), (34, 64), (64, 96), (96, 999))
QUADS = ("DR", "UR", "UL", "DL")
COLD_TOP_K = 208.0
# A reproduction is exact up to the storage quantisation (0.01 K) plus
# float32 storage of the inputs; anything beyond this is a real mismatch.
CHECK_TOL_K = 0.05


# --- rebuilding the inputs ----------------------------------------------------

def example_inputs(z) -> dict:
    """generate_synthetic_mw arguments for one stored example."""
    from data_types import BandImage, StormFix
    import goes_fetch

    t = datetime.fromisoformat(str(z["scene_time_iso"]))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    clat, clon = float(z["storm_lat"]), float(z["storm_lon"])
    sat = str(z["goes_satellite"]) if "goes_satellite" in z.files and str(z["goes_satellite"]) \
        else goes_fetch.select_satellite(clat, clon, t)
    lat = np.asarray(z["lat"], dtype=np.float64)
    lon = np.asarray(z["lon"], dtype=np.float64)

    def band(n, key, units):
        # Export writes an EMPTY array, not a missing key, for a band the
        # frame did not have (band 2 at night / on full disk).
        if key not in z.files or np.asarray(z[key]).size == 0:
            return None
        return BandImage(band=n, satellite=sat or "", scene_time=t,
                         values=np.asarray(z[key], dtype=np.float64), lat=lat, lon=lon,
                         units=units, mesoscale_sector="")

    sid = str(z["storm_id"])
    rmw = float(z["storm_rmw_nm"]) if "storm_rmw_nm" in z.files else np.nan
    roci = float(z["storm_roci_nm"]) if "storm_roci_nm" in z.files else np.nan
    fix = StormFix(storm_id=sid, valid_time=t, lat=clat, lon=clon,
                   vmax_kt=float(z["storm_vmax_kt"]),
                   rmw_nm=rmw if np.isfinite(rmw) else None,
                   roci_nm=roci if np.isfinite(roci) else None, basin=sid[:2])
    return {
        "args": (band(13, "ir_band13", "K"), band(9, "wv_band9", "K"),
                 band(7, "swir_band7", "K"), fix),
        "kwargs": {"band2": band(2, "vis_band2", "reflectance"), "real_swath": None,
                   "flash_density": (np.asarray(z["flash_density"], dtype=np.float64)
                                     if "flash_density" in z.files else None),
                   "ml_strength": 0.0,
                   "psf_sensor": str(z["mw_sensor"]) if "mw_sensor" in z.files else None,
                   # The stored ERA5 environment: the ML ignores it here (model
                   # off) but the 0.167 shear-asymmetry levers read its heading.
                   "env": {k: float(z[k]) for k in z.files if k.startswith("env_")} or None},
    }


def replay_backbone(z, capture: dict = None, stored_surface: bool = False) -> dict:
    """The GOES-only physics backbone for one stored example, as generation
    makes it now. With `capture`, also records the texture field (so the
    check can account for the one swath-dependent term, see texture_terms)
    and the land fraction / elevation generation used. With
    `stored_surface`, generation is handed the land fraction and elevation
    MINING stored instead of recomputing them on this machine."""
    import synthetic_algorithm as sa
    inp = example_inputs(z)
    orig_tex = sa._multispectral_texture_field
    if capture is not None:
        def spy(*a, **k):
            out = orig_tex(*a, **k)
            capture["texture_field"] = np.array(out, copy=True)
            return out
        sa._multispectral_texture_field = spy
    kwargs = dict(inp["kwargs"])
    if stored_surface:
        # Handed over AFTER generation's own smoothing (surface_override),
        # because export stores the smoothed field. 0.164's first version
        # patched surface_type instead -- smoothing it a second time.
        kwargs["surface_override"] = {
            k: (np.asarray(z[k], dtype=np.float64)
                if k in z.files and np.asarray(z[k]).size else None)
            for k in ("land_fraction", "elevation_m")}
    try:
        r = sa.generate_synthetic_mw(*inp["args"], **kwargs)
    finally:
        sa._multispectral_texture_field = orig_tex
    if capture is not None:
        for k in ("land_fraction", "elevation_m"):
            if r.diagnostics.get(k) is not None:
                capture[k] = np.asarray(r.diagnostics[k], dtype=np.float64)
    return {c: np.asarray(r.diagnostics[f"backbone_{c}"], dtype=np.float64) for c in CHANNELS}


# The one term of the STORED backbone that replay cannot rebuild: texture
# amplitude. Mining measures it from the REAL swath when one is present
# (synthetic_algorithm, "tex_amp_* = measured_* if ... else default"), and
# the floor follows at 0.4x; replay has no swath, so it uses the defaults --
# which is also what every GOES-only inference uses. That makes it a small
# target leak in the stored training backbone (to be removed, with a
# physics-ID change, alongside the next physics revision). Both terms are
# LINEAR and pass through the same sensor PSF, so
#     stored - replay = (a_m - a_d) * PSF(T) + (0.4 a_m - f_d) * PSF(N)
# with one unknown per channel, a_m. The check solves for it; what is left
# must be storage rounding if everything else is reproduced exactly.
_TEX_DEFAULT = {"v37": 4.0, "h37": 5.0, "v89": 7.0, "h89": 9.0}
_FLOOR_DEFAULT = {"v37": 2.0, "h37": 2.5, "v89": 1.8, "h89": 2.2}
_FLOOR_FRACTION = 0.4


def texture_terms(z, texture_field) -> dict:
    """PSF(T) and PSF(N) per channel frequency, exactly as generation forms
    them: T is the captured multispectral texture field, N the seeded
    footprint noise (crc32 of storm id and grid shape)."""
    import zlib
    import synthetic_algorithm as sa
    from scipy.ndimage import gaussian_filter
    import mw_psf
    shape = texture_field.shape
    seed = zlib.crc32(f"{str(z['storm_id'])}|{shape[0]}x{shape[1]}".encode())
    noise = sa._normalize_texture(gaussian_filter(
        np.random.default_rng(seed).standard_normal(shape), sigma=1.3))
    lat = np.asarray(z["lat"], dtype=np.float64)
    lon = np.asarray(z["lon"], dtype=np.float64)
    sensor = str(z["mw_sensor"]) if "mw_sensor" in z.files else mw_psf.DEFAULT_SENSOR
    out = {}
    for freq in (37, 89):
        out[freq] = (mw_psf.apply_sensor_psf(texture_field, lat, lon, freq, sensor),
                     mw_psf.apply_sensor_psf(noise, lat, lon, freq, sensor))
    return out


# --- variants -----------------------------------------------------------------
# name -> (description, callable returning a context manager). Filled in as
# physics candidates are written; "current" is the physics as it stands.

@contextlib.contextmanager
def _no_change():
    yield


VARIANTS = {
    "current": ("the physics as it stands (must reproduce the stored backbone)", _no_change),
}


# --- scoring --------------------------------------------------------------------

def _r_over_rmw(z):
    lat = np.asarray(z["lat"], dtype=np.float64)
    lon = np.asarray(z["lon"], dtype=np.float64)
    clat, clon = float(z["storm_lat"]), float(z["storm_lon"])
    dy = (lat - clat) * 111.2
    dx = (lon - clon) * 111.2 * np.cos(np.radians(clat))
    rmw = float(z["storm_rmw_nm"]) if np.isfinite(float(z["storm_rmw_nm"])) else 30.0
    return np.hypot(dx, dy) / max(rmw * 1.852, 5.0), dx, dy


def score_example(z, backbone: dict) -> dict:
    """Per-example summary statistics of (real - backbone): radial means by
    channel, shear-quadrant inner-core means, and cold-top coverage."""
    rr, dx, dy = _r_over_rmw(z)
    idx = np.digitize(rr, R_EDGES) - 1
    out = {"vmax": float(z["storm_vmax_kt"]), "radial": {}, "quad": {}, "cold": {}}
    for c in CHANNELS:
        real = tde.unpack_tb(z[f"target_{c}"]).astype(np.float64)
        res = real - backbone[c]
        out["radial"][c] = [float(np.nanmean(res[(idx == i) & np.isfinite(res)]))
                            if ((idx == i) & np.isfinite(res)).sum() >= 20 else np.nan
                            for i in range(len(R_EDGES) - 1)]
        if c == "h37":
            h37_res = res
    sdir = float(z["env_shear_deep_dir_deg"]) if "env_shear_deep_dir_deg" in z.files else np.nan
    if np.isfinite(sdir):
        rel = (np.degrees(np.arctan2(dx, dy)) - sdir) % 360.0
        quad = (rel // 90).astype(int)
        ir = np.asarray(z["ir_band13"], dtype=np.float64)
        for q in range(4):
            m = (rr < 1.0) & (quad == q)
            ok = m & np.isfinite(h37_res)
            if ok.sum() >= 20:
                out["quad"][QUADS[q]] = float(np.mean(h37_res[ok]))
                okc = m & np.isfinite(ir)
                out["cold"][QUADS[q]] = float(np.mean(ir[okc] < COLD_TOP_K)) if okc.sum() else np.nan
    return out


def _job(args):
    path, variant, check = args
    try:
        with np.load(path, allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}

        class _Z(dict):
            files = property(lambda self: list(self.keys()))
        z = _Z(z)
        cap = {} if check else None
        with VARIANTS[variant][1]():
            bb = replay_backbone(z, capture=cap)
        rec = {"path": path, "ok": True,
               "physics_id": str(z["vh_physics_id"]) if "vh_physics_id" in z.files else "?"}
        if check:
            _check_against_stored(z, bb, cap, rec)
            # Surface fields: mining stored the land fraction and elevation it
            # USED. A difference means mining ran with another land-mask
            # backend (e.g. global_land_mask absent -> "none" or cartopy) --
            # which moves the backbone along coasts and nowhere else.
            for k in ("land_fraction", "elevation_m"):
                if k in z.files and np.asarray(z[k]).size and k in cap:
                    st = np.asarray(z[k], dtype=np.float64)
                    diff = np.abs(cap[k] - st)
                    rec[f"surface_{k}"] = (int((diff > (0.01 if k == "land_fraction" else 1.0)).sum()),
                                           float(np.nanmax(diff)) if diff.size else 0.0)
            if rec["worst"] > CHECK_TOL_K:
                # Replay again WITH mining's own surface fields: if that
                # reproduces, the surface fields are proven to be the cause.
                cap2, rec2 = {}, {}
                bb2 = replay_backbone(z, capture=cap2, stored_surface=True)
                _check_against_stored(z, bb2, cap2, rec2)
                rec["worst_with_stored_surface"] = rec2["worst"]
        rec["score"] = score_example(z, bb)
        return rec
    except Exception as e:
        return {"path": path, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _check_against_stored(z, bb, cap, rec):
    terms = texture_terms(z, cap["texture_field"]) if "texture_field" in cap else None
    worst = 0.0
    for c in CHANNELS:
        stored = tde.unpack_tb(z[f"backbone_{c}"]).astype(np.float64)
        both = np.isfinite(stored) & np.isfinite(bb[c])
        nan_mismatch = int((np.isfinite(stored) != np.isfinite(bb[c])).sum())
        d = stored - bb[c]
        a_m = np.nan
        # From 0.168 the stored backbone carries DEFAULT-amplitude texture
        # (the swath-measured excess is removed at export), so it must match
        # as it is. Only when it does not -- a pre-0.168 file -- is the one
        # swath-measured amplitude solved for. Solving unconditionally broke
        # an exact 0.168 match: its model forces floor = 0.4 x texture, the
        # defaults do not, so it "corrected" a perfect match by up to 5 K.
        raw = float(np.max(np.abs(d[both]))) if both.any() else 0.0
        if raw > CHECK_TOL_K and terms is not None and both.any():
            pT, pN = terms[37 if c.endswith("37") else 89]
            known = _TEX_DEFAULT[c] * pT + _FLOOR_DEFAULT[c] * pN
            basis = pT + _FLOOR_FRACTION * pN
            m = both & np.isfinite(basis) & np.isfinite(known)
            denom = float(np.sum(basis[m] ** 2))
            if denom > 0:
                # stored - replay = a_m * basis - known  ->  solve a_m
                a_m = float(np.sum((d[m] + known[m]) * basis[m]) / denom)
                d = d - (a_m * basis - known)
        res = float(np.max(np.abs(d[both]))) if both.any() else 0.0
        worst = max(worst, res)
        rec[f"maxdiff_{c}"] = res
        rec[f"nanmismatch_{c}"] = nan_mismatch
        rec[f"texamp_{c}"] = a_m
    rec["worst"] = worst


def run_jobs(files, fn, workers=None, log=print) -> list:
    """fn(path) over files in worker processes (spawn), with progress."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import ml_data_mining
    workers = ml_data_mining.memory_safe_workers(
        workers or max(1, (os.cpu_count() or 2) - 1), log=log, per_worker_gb=0.8, base_gb=0.3)
    t0, out = time.time(), []
    if workers <= 1:
        for i, f in enumerate(files):
            out.append(fn(f))
            if (i + 1) % 50 == 0:
                log(f"  ... {i + 1}/{len(files)} ({time.time() - t0:.0f} s)")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for i, rec in enumerate(ex.map(fn, files, chunksize=4)):
                out.append(rec)
                if (i + 1) % 50 == 0:
                    log(f"  ... {i + 1}/{len(files)} ({time.time() - t0:.0f} s)")
    log(f"Processed {len(out)} example(s) in {time.time() - t0:.0f} s on {workers} worker(s)")
    return out


def run(files, variant="current", check=False, workers=None, log=print) -> list:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import ml_data_mining
    workers = ml_data_mining.memory_safe_workers(
        workers or max(1, (os.cpu_count() or 2) - 1), log=log,
        per_worker_gb=0.8, base_gb=0.3)
    jobs = [(f, variant, check) for f in files]
    t0 = time.time()
    out = []
    if workers <= 1:
        for i, j in enumerate(jobs):
            out.append(_job(j))
            if (i + 1) % 50 == 0:
                log(f"  ... {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for i, rec in enumerate(ex.map(_job, jobs, chunksize=4)):
                out.append(rec)
                if (i + 1) % 50 == 0:
                    log(f"  ... {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)")
    log(f"Replayed {len(out)} example(s) with variant '{variant}' in {time.time() - t0:.0f} s "
        f"on {workers} worker(s)")
    return out


def report(recs, log=print):
    """The h37 diagnostic's measures, for a replayed backbone."""
    good = [r for r in recs if r.get("ok")]
    bad = [r for r in recs if not r.get("ok")]
    if bad:
        log(f"  {len(bad)} failed, e.g. {bad[0]['error']}")
    hdr = "".join(f"{R_EDGES[i]:>4.1f}-{R_EDGES[i+1]:<4.1f}" for i in range(len(R_EDGES) - 1))
    for lo, hi in V_BANDS:
        grp = [r["score"] for r in good if lo <= r["score"]["vmax"] < hi]
        if not grp:
            continue
        log(f"\nVmax {lo}-{hi if hi < 999 else '+'} kt ({len(grp)} examples)")
        log(f"  r/RMW   {hdr}")
        for c in CHANNELS:
            cols = np.array([g["radial"][c] for g in grp], dtype=np.float64)
            with np.errstate(all="ignore"):
                m = np.nanmean(cols, axis=0)
            log(f"  {c:5s}  " + "".join(f"{v:>+9.1f}" if np.isfinite(v) else "      ---" for v in m))
    qs = [r["score"] for r in good if r["score"]["quad"]]
    if qs:
        log("\nInner-core h37 by shear quadrant: " + "  ".join(
            f"{q} {np.nanmean([g['quad'].get(q, np.nan) for g in qs]):+.1f}" for q in QUADS))
        pairs = [(g["cold"][q], g["quad"][q]) for g in qs for q in QUADS
                 if q in g["quad"] and np.isfinite(g["cold"].get(q, np.nan)) and g["vmax"] >= 34]
        if len(pairs) > 10:
            x, y = np.array(pairs).T
            none = y[x == 0].mean() if (x == 0).any() else np.nan
            full = y[x > 0.5].mean() if (x > 0.5).any() else np.nan
            log(f"Inner-core h37 vs cold-top coverage (Vmax >= 34 kt): slope "
                f"{np.polyfit(x, y, 1)[0]:+.1f} K per unit; no cold tops {none:+.1f}, "
                f">50% cold {full:+.1f}")


# --- the depression sweep (0.163) ------------------------------------------------
# The backbone is LINEAR in the depression gain (verified to 1e-13 K), and
# the ramp is a per-example scalar. So two replays per example -- gains 0,
# and the depression doubled at every intensity -- give the backbone for
# ANY gain and ANY intensity dependence:  TB(g) = TB0 - g * ramp(vmax) * K,
# K = TB0 - TB(doubled). Per example we keep only sums, from which every
# statistic below is exact.

SWEEP_BANDS = ((0, 34), (34, 50), (50, 64), (64, 83), (83, 96), (96, 113), (113, 999))


# A band's gain is only fitted where the depression actually ACTS: enough
# pixels it moves by > AFFECTED_K, across enough storms. Elsewhere the fit
# explains noise -- on 19 examples an unguarded band returned +1,283, and
# the out-of-sample test duly made h37 RMSE 20 -> 74 K.
AFFECTED_K = 0.5
MIN_AFFECTED_PX = 5000
MIN_AFFECTED_STORMS = 5


# --- the lever sweep (0.167) -------------------------------------------------------
# Seven levers, each a gain on one term; 0 = the fitted physics. Every one is
# linear and they add exactly (verified to ~1e-13 K), so TB(g) = TB0 +
# sum_j g_j K_j. Six need one replay each; "bg" (a constant offset) needs
# none: smoothing and the sensor PSF both preserve a constant, so K_bg = 1.
LEVERS = ("dep", "em", "out", "far", "aem", "adep", "bg")
REPLAYED = ("dep", "em", "out", "far", "aem", "adep")
LEVER_NAMES = {"dep": "depression", "em": "emission", "out": "outer depr 1.5-2.5",
               "far": "far depr 3.5-5", "aem": "asym emission", "adep": "asym depression",
               "bg": "offset (K)"}
LEVER_BOUNDS = {"dep": (-1, 3), "em": (-1, 3), "out": (-1, 3), "far": (-1, 3),
                "aem": (-1, 1), "adep": (-1, 1), "bg": (-20, 20)}
# Radial zones in r/RMW. Scored SEPARATELY: 0.166 scored the whole
# ~1,000 km grid, where everything inside 2 RMW is ~2% of the pixels, so
# no core improvement could show. "env" (> 8 RMW) is reported, never fitted.
ZONES = (("core", 0.0, 2.0), ("bands", 2.0, 4.0), ("outer", 4.0, 8.0), ("env", 8.0, np.inf))
FIT_ZONES = ("core", "bands", "outer")
LAND_FRACTION_MAX = 0.05          # "ocean" pixel; land is scored apart, never fitted
QUAD_ORDER = ("DR", "UR", "UL", "DL")


def _lever_tables(sa):
    return {"dep": sa.DEPRESSION_INTENSITY_GAIN, "em": sa.EMISSION_GAIN,
            "out": sa.OUTER_DEPRESSION_GAIN, "far": sa.FAR_DEPRESSION_GAIN,
            "aem": sa.ASYM_EMISSION_GAIN, "adep": sa.ASYM_DEPRESSION_GAIN}


def _moments(r0, K, m):
    """Sufficient statistics of one pixel set: everything any fit or score
    below needs, so the replays never have to be rerun for new analyses."""
    rf, Kf = r0[m], K[:, m]
    return {"n": int(m.sum()), "r2": float(rf @ rf), "sr": float(rf.sum()),
            "b": Kf @ rf, "A": Kf @ Kf.T, "sk": Kf.sum(axis=1),
            "aff": (np.abs(Kf) > AFFECTED_K).sum(axis=1)}


def _sweep_job(path):
    import synthetic_algorithm as sa
    try:
        with np.load(path, allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}

        class _Z(dict):
            files = property(lambda self: list(self.keys()))
        z = _Z(z)
        if "vis_band2" in z.files and np.asarray(z["vis_band2"]).size \
                and np.asarray(z["vis_band2"]).shape != np.asarray(z["lat"]).shape:
            return {"path": path, "ok": False,
                    "error": "band 2 stored on its own grid (its lat/lon not saved) -- cannot rebuild"}
        tables = _lever_tables(sa)
        saved = ({k: dict(v) for k, v in tables.items()}, dict(sa.DEPRESSION_RAMP_KT))
        tb, cap = {}, {}
        try:
            sa.DEPRESSION_RAMP_KT = {37: (-2.0, -1.0), 89: (-2.0, -1.0)}   # ramp = 1 everywhere
            for lever in (None,) + REPLAYED:
                for t in tables.values():
                    for c in t:
                        t[c] = 0.0
                if lever:
                    for c in tables[lever]:
                        tables[lever][c] = 1.0
                tb[lever] = replay_backbone(z, capture=cap if lever is None else None)
        finally:
            for k, v in saved[0].items():
                tables[k].clear()
                tables[k].update(v)
            sa.DEPRESSION_RAMP_KT = saved[1]
        rr, dx, dy = _r_over_rmw(z)
        land = (np.asarray(z["land_fraction"], dtype=np.float64)
                if "land_fraction" in z.files and np.asarray(z["land_fraction"]).size
                else cap.get("land_fraction", np.zeros_like(rr)))
        ocean = land < LAND_FRACTION_MAX
        idx = np.digitize(rr, R_EDGES) - 1
        sdir = float(z["env_shear_deep_dir_deg"]) if "env_shear_deep_dir_deg" in z.files else np.nan
        quad = None
        if np.isfinite(sdir):
            quad = ((np.degrees(np.arctan2(dx, dy)) - sdir) % 360.0 // 90).astype(int)
        rec = {"path": path, "ok": True, "storm": os.path.basename(path).split("_")[0],
               "vmax": float(z["storm_vmax_kt"]), "z": {}, "radial": {}, "quad": {}}
        for c in CHANNELS:
            real = tde.unpack_tb(z[f"target_{c}"]).astype(np.float64)
            r0 = real - tb[None][c]
            K = np.stack([tb[l][c] - tb[None][c] for l in REPLAYED] + [np.ones_like(r0)])
            ok = np.isfinite(r0) & np.all(np.isfinite(K), axis=0)
            rec["z"][c] = {(zn, sf): _moments(r0, K, ok & (rr >= lo) & (rr < hi) & (om if sf == "ocean" else ~om))
                           for zn, lo, hi in ZONES
                           for sf, om in (("ocean", ocean), ("land", ocean))}
            rec["radial"][c] = [_moments(r0, K, ok & ocean & (idx == i))
                                for i in range(len(R_EDGES) - 1)]
            if quad is not None:
                rec["quad"][c] = {QUAD_ORDER[q]: _moments(r0, K, ok & ocean & (rr < 2.0) & (quad == q))
                                  for q in range(4)}
        return rec
    except Exception as e:
        return {"path": path, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _pool(recs, c, zones, surf="ocean", balanced=False):
    """(A, b, affected px, storms per lever) pooled over examples and zones.
    `balanced`: each zone weighted by 1 / its pooled pixel count, so the
    core's few pixels count as much as the outer region's many."""
    L = len(LEVERS)
    A, b, aff = np.zeros((L, L)), np.zeros(L), np.zeros(L)
    storms = [set() for _ in range(L)]
    for zn in zones:
        n = sum(r["z"][c][(zn, surf)]["n"] for r in recs)
        if n == 0:
            continue
        w = 1.0 / n if balanced else 1.0
        for r in recs:
            m = r["z"][c][(zn, surf)]
            A += w * m["A"]
            b += w * m["b"]
            aff += m["aff"]
            for j in range(L):
                if m["aff"][j] > 0:
                    storms[j].add(r["storm"])
    return A, b, aff, [len(s_) for s_ in storms]


def _fit(recs, c, levers, zones=FIT_ZONES, balanced=True, surf="ocean"):
    """Bounded least-squares gains {lever: gain}; a lever failing the signal
    guard gets NaN and takes no part."""
    from scipy.optimize import minimize
    out = {l: float("nan") for l in levers}
    if not recs:
        return out
    A, b, aff, st = _pool(recs, c, zones, surf, balanced)
    use = [LEVERS.index(l) for l in levers
           if aff[LEVERS.index(l)] >= MIN_AFFECTED_PX and st[LEVERS.index(l)] >= MIN_AFFECTED_STORMS
           and A[LEVERS.index(l), LEVERS.index(l)] > 0]
    if not use:
        return out
    Au, bu = A[np.ix_(use, use)], b[use]
    res = minimize(lambda g: float(g @ Au @ g - 2 * bu @ g), np.zeros(len(use)),
                   jac=lambda g: 2 * (Au @ g) - 2 * bu, method="L-BFGS-B",
                   bounds=[LEVER_BOUNDS[LEVERS[j]] for j in use])
    for j, g in zip(use, res.x):
        out[LEVERS[j]] = float(g)
    return out


def _vec(g):
    return np.array([0.0 if not np.isfinite(g.get(l, np.nan)) else g[l] for l in LEVERS])


def _zone_rmse(recs, c, zone, gains_of, surf="ocean"):
    n = s_ = 0.0
    for r in recs:
        v = _vec(gains_of(r))
        m = r["z"][c][(zone, surf)]
        n += m["n"]
        s_ += m["r2"] - 2 * v @ m["b"] + v @ m["A"] @ v
    return (s_ / n) ** 0.5 if n else float("nan")


def _fitter(train, c, levers, per_band=True, balanced=True, zones=FIT_ZONES):
    if not per_band:
        g = _fit(train, c, levers, zones, balanced)
        return lambda r: g
    table = {(lo, hi): _fit([r for r in train if lo <= r["vmax"] < hi], c, levers, zones, balanced)
             for lo, hi in SWEEP_BANDS}
    return lambda r: next((g for (lo, hi), g in table.items() if lo <= r["vmax"] < hi), {})


def _halves(good):
    storms = sorted({r["storm"] for r in good})
    half_a = {s_ for i, s_ in enumerate(storms) if i % 2 == 0}
    return [r for r in good if r["storm"] in half_a], [r for r in good if r["storm"] not in half_a]


def sweep_report(recs, log=print):
    good = [r for r in recs if r.get("ok")]
    bad = [r for r in recs if not r.get("ok")]
    for r in bad[:5]:
        log(f"  not replayed: {os.path.basename(r['path'])}: {r['error']}")
    if len(bad) > 5:
        log(f"  ... and {len(bad) - 5} more not replayed")
    A, B = _halves(good)
    log(f"\n{len(good)} examples, {len({r['storm'] for r in good})} storms; halves A/B split by "
        f"storm ({len(A)}/{len(B)}). Ocean pixels only (land fraction < {LAND_FRACTION_MAX}) "
        f"unless stated. Zones in r/RMW: " + ", ".join(f"{n} {lo:g}-{hi:g}" for n, lo, hi in ZONES))
    log("Levers: " + "; ".join(f"{l} = {LEVER_NAMES[l]} [{LEVER_BOUNDS[l][0]:+g}..{LEVER_BOUNDS[l][1]:+g}]"
                               for l in LEVERS))
    log("Fits are ZONE-BALANCED over core+bands+outer (each zone weighted by 1/its pixel "
        "count) unless marked 'all-pixel'.")

    def oos_row(c, levers, per_band=True, balanced=True):
        fb, fa = _fitter(A, c, levers, per_band, balanced), _fitter(B, c, levers, per_band, balanced)
        return [(_zone_rmse(B, c, zn, fb), _zone_rmse(A, c, zn, fa)) for zn, _, _ in ZONES[:3]]

    log("\n=== 1. Which levers carry the fix? OUT-OF-SAMPLE RMSE (K) by zone, fit on one half "
        "of the storms, scored on the other (B | A). Per-band gains.")
    for c in CHANNELS:
        log(f"  {c:4s} {'':24s}" + "".join(f"{zn:>17s}" for zn, _, _ in ZONES[:3]))
        base = [(_zone_rmse(B, c, zn, lambda r: {}), _zone_rmse(A, c, zn, lambda r: {}))
                for zn, _, _ in ZONES[:3]]
        rows = [("none (current physics)", base)]
        for l in LEVERS:
            rows.append((f"{l} alone", oos_row(c, (l,))))
        rows.append(("ALL levers", oos_row(c, LEVERS)))
        rows.append(("ALL, all-pixel objective", oos_row(c, LEVERS, balanced=False)))
        rows.append(("ALL, one global gain", oos_row(c, LEVERS, per_band=False)))
        for name, cells in rows:
            log(f"       {name:24s}" + "".join(f"  {b_:6.2f} | {a_:6.2f}" for b_, a_ in cells))

    log("\n=== 2. Robustness: global gains (ALL levers) fitted on everything, and again "
        "without the worst 2% of examples")
    for c in CHANNELS:
        base = {id(r): sum(r["z"][c][(zn, "ocean")]["r2"] for zn in FIT_ZONES)
                / max(1, sum(r["z"][c][(zn, "ocean")]["n"] for zn in FIT_ZONES)) for r in good}
        cut = np.quantile(list(base.values()), 0.98) if base else np.inf
        g_all = _fit(good, c, LEVERS)
        g_trim = _fit([r for r in good if base[id(r)] <= cut], c, LEVERS)
        fmt = lambda v: "  ---" if not np.isfinite(v) else f"{v:+6.2f}"
        log(f"  {c}: " + "  ".join(f"{l} {fmt(g_all[l])}/{fmt(g_trim[l])}" for l in LEVERS))

    log("\n=== 3. Per-band gains, ALL levers, each half: A / B   (--- = too little signal)")
    fmt = lambda v: " ---" if not np.isfinite(v) else f"{v:+.2f}"
    joint = {}
    for c in CHANNELS:
        log(f"  {c}  Vmax kt   " + "".join(f"{l:>13s}" for l in LEVERS) + "     n")
        for lo, hi in SWEEP_BANDS:
            ga = [r for r in A if lo <= r["vmax"] < hi]
            gb = [r for r in B if lo <= r["vmax"] < hi]
            if not ga and not gb:
                continue
            fa, fb = _fit(ga, c, LEVERS), _fit(gb, c, LEVERS)
            joint[(lo, hi, c)] = _fit(ga + gb, c, LEVERS)
            log(f"       {lo:>3}-{hi if hi < 999 else '+':<5}" + "".join(
                f"{fmt(fa[l]) + '/' + fmt(fb[l]):>13s}" for l in LEVERS) + f"  {len(ga) + len(gb):4d}")

    band_gain = lambda r, c: next((joint[(lo, hi, c)] for lo, hi in SWEEP_BANDS
                                   if lo <= r["vmax"] < hi and (lo, hi, c) in joint), {})

    def radial_table(gain_of, title):
        log(f"\n{title}")
        hdr = "".join(f"{R_EDGES[i]:>4.1f}-{R_EDGES[i+1]:<4.1f}" for i in range(len(R_EDGES) - 1))
        for lo, hi in ((0, 34), (34, 64), (64, 96), (96, 999)):
            grp = [r for r in good if lo <= r["vmax"] < hi]
            if not grp:
                continue
            log(f"  Vmax {lo}-{hi if hi < 999 else '+'} kt  r/RMW {hdr}")
            for c in CHANNELS:
                row = []
                for i in range(len(R_EDGES) - 1):
                    # mean over EXAMPLES of each example's bin mean, as the diagnostic does;
                    # residual after the levers is r0 - g.K, so its sum is sr - g.sk
                    vals = [(r["radial"][c][i]["sr"] - _vec(gain_of(r, c)) @ r["radial"][c][i]["sk"])
                            / r["radial"][c][i]["n"] for r in grp if r["radial"][c][i]["n"] >= 20]
                    row.append(f"{np.mean(vals):>+9.1f}" if vals else "      ---")
                log(f"     {c:5s}            " + "".join(row))

    radial_table(lambda r, c: {}, "=== 4. BEFORE (current physics), ocean, mean real - backbone (K):")
    radial_table(band_gain, "=== 5. AFTER, per-band ALL-lever gains fitted on ALL (in-sample):")

    log("\n=== 6. Shear asymmetry inside 2 RMW (ocean): mean real - backbone by quadrant, "
        "before -> after (per-band ALL-lever gains)")
    for c in CHANNELS:
        cells = []
        for q in QUAD_ORDER:
            ms = [r["quad"][c][q] for r in good if c in r["quad"] and r["quad"][c][q]["n"] >= 20]
            gs = [band_gain(r, c) for r in good if c in r["quad"] and r["quad"][c][q]["n"] >= 20]
            if not ms:
                cells.append(f"{q}  ---")
                continue
            before = np.mean([m["sr"] / m["n"] for m in ms])
            after = np.mean([(m["sr"] - _vec(g) @ m["sk"]) / m["n"] for m, g in zip(ms, gs)])
            cells.append(f"{q} {before:+5.1f} -> {after:+5.1f}")
        log(f"  {c}: " + "   ".join(cells))

    log("\n=== 7. Land pixels (never fitted): out-of-sample RMSE by zone, none -> ALL levers")
    for c in CHANNELS:
        fb, fa = _fitter(A, c, LEVERS), _fitter(B, c, LEVERS)
        cells = []
        for zn, _, _ in ZONES[:3]:
            n0 = _zone_rmse(B, c, zn, lambda r: {}, surf="land")
            n1 = _zone_rmse(B, c, zn, fb, surf="land")
            cells.append(f"{zn} {n0:6.2f} -> {n1:6.2f}")
        log(f"  {c}: " + "   ".join(cells))


def _compact(o):
    """float32 arrays: ~7 significant digits, i.e. RMSE differences of
    ~1e-4 K -- and a file small enough to send (it lets the analysis be
    redone, or extended, with no replays)."""
    if isinstance(o, np.ndarray) and o.dtype.kind == "f":
        return o.astype(np.float32)
    if isinstance(o, dict):
        return {k: _compact(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_compact(v) for v in o]
    return o


def save_sweep(recs, path):
    import gzip
    import pickle
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump({"version": 3, "levers": LEVERS, "zones": ZONES,
                     "records": _compact(recs)}, fh, protocol=4)


def _expand(o):
    if isinstance(o, np.ndarray) and o.dtype == np.float32:
        return o.astype(np.float64)
    if isinstance(o, dict):
        return {k: _expand(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_expand(v) for v in o]
    return o


def load_sweep(path):
    import gzip
    import pickle
    with gzip.open(path, "rb") as fh:
        d = pickle.load(fh)
    d["records"] = _expand(d["records"])
    if d.get("levers") != LEVERS:
        raise ValueError(f"{path} was written with levers {d.get('levers')}, this build has {LEVERS}")
    return d["records"]


def audit(files, log=print):
    """What the sweep will be working with -- read from the NPZs alone, no
    replays, a minute or two for a full dataset."""
    rows = []
    for f in files:
        try:
            with np.load(f, allow_pickle=False) as z:
                names = set(z.files)
                vmax = float(z["storm_vmax_kt"])
                rmw = float(z["storm_rmw_nm"]) if "storm_rmw_nm" in names else np.nan
                sdir = float(z["env_shear_deep_dir_deg"]) if "env_shear_deep_dir_deg" in names else np.nan
                lf = np.asarray(z["land_fraction"]) if "land_fraction" in names else np.zeros(1)
                tgt = tde.unpack_tb(z["target_h37"])
                rr, _, _ = _r_over_rmw(z)
                v2 = np.asarray(z["vis_band2"]) if "vis_band2" in names else np.zeros(0)
                rows.append({
                    "storm": os.path.basename(f).split("_")[0], "vmax": vmax,
                    "rmw_missing": not np.isfinite(rmw), "shear": np.isfinite(sdir),
                    "land": float(np.mean(lf >= LAND_FRACTION_MAX)),
                    "cover": float(np.mean(np.isfinite(tgt))),
                    "core_cover": float(np.mean(np.isfinite(tgt[rr < 2.0]))) if (rr < 2.0).any() else 0.0,
                    "rr_max": float(np.nanmax(rr)),
                    "phys": str(z["vh_physics_id"]) if "vh_physics_id" in names else "?",
                    "mask": str(z["land_mask_backend"]) if "land_mask_backend" in names else "unrecorded",
                    "b2_own_grid": bool(v2.size and v2.shape != np.asarray(z["lat"]).shape),
                })
        except Exception as e:
            rows.append({"error": f"{os.path.basename(f)}: {type(e).__name__}: {e}"})
    ok = [r for r in rows if "error" not in r]
    log(f"{len(files)} files, {len(ok)} readable, {len({r['storm'] for r in ok})} storms")
    for r in [r for r in rows if "error" in r][:5]:
        log(f"  UNREADABLE {r['error']}")
    from collections import Counter
    log(f"physics IDs: {dict(Counter(r['phys'] for r in ok))}")
    log(f"land masks:  {dict(Counter(r['mask'] for r in ok))}")
    log(f"ERA5 shear heading present: {sum(r['shear'] for r in ok)}/{len(ok)} "
        f"(needed by the asymmetry levers and quadrant table)")
    log(f"RMW missing (30 nm assumed for radial zones): {sum(r['rmw_missing'] for r in ok)}")
    log(f"band 2 on its own grid (cannot be replayed): {sum(r['b2_own_grid'] for r in ok)}")
    lands = np.array([r["land"] for r in ok]); covers = np.array([r["cover"] for r in ok])
    cores = np.array([r["core_cover"] for r in ok])
    log(f"land share of grid: median {np.median(lands):.0%}, 90th pct {np.quantile(lands, .9):.0%}, "
        f"examples > 25% land: {int((lands > .25).sum())}")
    log(f"real-MW coverage of grid: median {np.median(covers):.0%}; of the core (< 2 RMW): "
        f"median {np.median(cores):.0%}, examples with < 50% core coverage: {int((cores < .5).sum())}")
    log(f"grid reaches r/RMW: median {np.median([r['rr_max'] for r in ok]):.1f} "
        f"(zones beyond 8 RMW are 'env', reported but never fitted)")
    log("\nexamples and storms per intensity band (the signal guard needs >= "
        f"{MIN_AFFECTED_STORMS} storms per band per half):")
    for lo, hi in SWEEP_BANDS:
        g = [r for r in ok if lo <= r["vmax"] < hi]
        log(f"  {lo:>3}-{hi if hi < 999 else '+':<5} {len(g):5d} examples, "
            f"{len({r['storm'] for r in g}):4d} storms")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", nargs="?", default=None)
    ap.add_argument("--check", action="store_true",
                    help="verify the current physics reproduces every stored backbone")
    ap.add_argument("--variant", default="current")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="replay every example with each lever and fit them (saves the results)")
    ap.add_argument("--report", metavar="FILE",
                    help="re-analyse a saved sweep, no replays")
    ap.add_argument("--physics", choices=("current", "0.136"), default="current",
                    help="replay under the pre-0.168 physics (for data mined before it)")
    ap.add_argument("--audit", action="store_true",
                    help="summarise the dataset the sweep will use (no replays)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    if a.physics == "0.136":
        # Set BEFORE synthetic_algorithm is imported anywhere, so this process
        # and every spawned worker (which inherit the environment) agree.
        os.environ["MWSYNTH_PHYSICS"] = "0.136"
        import synthetic_algorithm as _sa
        _sa.USE_ADOPTED_CURVES = False
        _sa.VH_PHYSICS_ID = _sa._compute_vh_physics_id()
    if a.list:
        for k, (desc, _) in VARIANTS.items():
            print(f"  {k:24s} {desc}")
        return 0
    if a.report:
        sweep_report(load_sweep(a.report))
        return 0
    if a.variant not in VARIANTS:
        print(f"unknown variant {a.variant!r}; --list shows them")
        return 2
    files = tde.list_training_files(a.data_dir)
    if a.limit:
        # Spread the subset across the whole (sorted) set rather than the
        # first N, which would be one basin and one season.
        step = max(1, len(files) // a.limit)
        files = files[::step][:a.limit]
    if not files:
        print("no training examples found")
        return 1
    if a.audit:
        audit(files)
        return 0
    if a.sweep:
        recs = run_jobs(files, _sweep_job, a.workers)
        import synthetic_algorithm as sa
        out = f"sweep_{sa.VH_PHYSICS_ID}_{time.strftime('%Y%m%d-%H%M')}.pkl.gz"
        save_sweep(recs, out)
        print(f"Per-example results saved to {os.path.abspath(out)} "
              f"({os.path.getsize(out) / 1e6:.1f} MB) -- rerun the analysis with "
              f"--report {out} (no replays)")
        # Report from what was SAVED, so this output and any later --report
        # of the file are identical.
        sweep_report(load_sweep(out))
        return 0
    recs = run(files, variant="current" if a.check else a.variant, check=a.check,
               workers=a.workers)
    if a.check:
        good = [r for r in recs if r.get("ok")]
        worst = sorted(good, key=lambda r: -r["worst"])
        n_bad = sum(r["worst"] > CHECK_TOL_K or any(r[f"nanmismatch_{c}"] for c in CHANNELS)
                    for r in good)
        print(f"\nReproduction (after solving each channel's swath-measured texture "
              f"amplitude): {len(good) - n_bad} of {len(good)} exact within {CHECK_TOL_K} K "
              f"(storage quantisation is 0.01 K); worst {worst[0]['worst']:.3f} K "
              f"({os.path.basename(worst[0]['path'])})" if good else "nothing replayed")
        import surface_type
        import synthetic_algorithm as _sa
        print(f"  (land mask on this machine: {surface_type.backend_name()}; replaying physics "
              f"{_sa.VH_PHYSICS_ID})")
        from collections import Counter
        vint = Counter(r.get("physics_id", "?") for r in good)
        other = {k: v for k, v in vint.items() if k != _sa.VH_PHYSICS_ID}
        if other:
            print(f"  NOTE: {sum(other.values())} file(s) were made under different physics "
                  f"({', '.join(f'{k}: {v}' for k, v in other.items())}). They cannot reproduce "
                  f"under {_sa.VH_PHYSICS_ID} -- by design. To check them, rerun with "
                  f"--physics 0.136.")
        for r in worst[:5]:
            if r["worst"] > CHECK_TOL_K:
                print(f"  {os.path.basename(r['path'])}: " + ", ".join(
                    f"{c} {r[f'maxdiff_{c}']:.2f} K / {r[f'nanmismatch_{c}']} NaN-px" for c in CHANNELS))
                lf, el = r.get("surface_land_fraction"), r.get("surface_elevation_m")
                if lf or el:
                    print(f"    surface fields vs mining: land fraction differs at "
                          f"{lf[0] if lf else '?'} px (max {lf[1] if lf else 0:.2f}), elevation at "
                          f"{el[0] if el else '?'} px (max {el[1] if el else 0:.0f} m)")
                w2 = r.get("worst_with_stored_surface")
                if w2 is not None:
                    print(f"    replayed with MINING's surface fields: worst {w2:.3f} K -> "
                          + ("CAUSE FOUND: mining used different land/elevation data"
                             if w2 <= CHECK_TOL_K else "still differs: not the surface fields"))
        if n_bad:
            # Severity, not a blanket verdict: a least-squares fit pooled over
            # thousands of examples is not moved by 1% of files differing by
            # a few kelvin on a few pixels; a systematic failure is fatal.
            frac = n_bad / max(len(good), 1)
            if frac <= 0.02 and worst[0]["worst"] < 5.0:
                print(f"  -> {n_bad} file(s) ({frac:.0%}), worst {worst[0]['worst']:.1f} K: too few "
                      f"and too small to bias pooled fits -- sweep results are usable. "
                      f"The lines above say what differs; worth understanding, not blocking.")
            else:
                print(f"  -> {n_bad} file(s) ({frac:.0%}) differ, worst {worst[0]['worst']:.1f} K: "
                      f"variant scores are NOT trustworthy until the cause is found and fixed.")
    report(recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
