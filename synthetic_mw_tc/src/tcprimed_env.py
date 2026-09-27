"""ERA5 environment for each overpass, from TC PRIMED's per-storm env file.

Added in 0.160 (plan items #2, #11, #13, #14, #15, #18 all start here).

Every TC PRIMED storm directory holds one environmental file,
`..._env_s<start>_e<end>.nc`: 6-hourly ERA5 diagnostics computed the SHIPS
way, plus 3-D cylindrical and rectilinear fields. What is in it was read
from a real file (AL022024) rather than assumed:

- It is ~220 MB, nearly all 3-D fields. Only small arrays are needed, so
  it is read RANGED: ~0.7 MB in ~10 requests instead of 220 MB.
- Values are packed int16 with scale_factor / add_offset / _FillValue,
  which h5py does not apply. `_unpack` does.
- The `layer` and `regions` dimensions have NO coordinate values; their
  meaning is only in each variable's attributes:
      shear  layer   = 850_to_500_hPa, 850_to_200_hPa
      regions        = 0_to_300_km, 0_to_500_km, 0_to_800_km, 200_to_800_km
  Those labels are CHECKED on every read. Indexing by position alone
  would silently use the wrong layer if a future version reordered them.
- shear_direction is where the shear vector points TO, compass degrees:
  "westerly shear has a value of 90 deg".
- overpass_storm_metadata has one row per overpass FILE (all sensors),
  with storm motion and intensity change at the overpass time. Its
  periods are -24..+24 h and NEGATIVE MEANS PAST. Only past columns are
  exposed here: the future ones are the answer, not an input.
- diagnostics/sst has fewer times than the rest (39 vs 53 on AL022024),
  so core SST comes from cylindrical/sst at r <= 50 km, which is aligned
  to every time.

Fields exposed per overpass (NaN when unavailable):

    env_shear_deep_ms, env_shear_deep_dir_deg   850-200 hPa, 0-500 km
    env_shear_mid_ms,  env_shear_mid_dir_deg    850-500 hPa, 0-500 km
    env_rh_mid_pct     700-500 hPa mean, 200-800 km   (SHIPS RHMD-like)
    env_rh_low_pct     850-700 hPa mean, 200-800 km
    env_sst_k          cylindrical SST, r <= 50 km, azimuthal mean
    env_pi_kt          theoretical potential intensity, 200-800 km
    env_motion_u_ms, env_motion_v_ms            best-track motion
    env_dvmax_past12_kt, env_dvmax_past24_kt    PAST intensity change
"""
from __future__ import annotations

import math
import threading
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np

ENV_KEYS = (
    "env_shear_deep_ms", "env_shear_deep_dir_deg",
    "env_shear_mid_ms", "env_shear_mid_dir_deg",
    "env_rh_mid_pct", "env_rh_low_pct",
    "env_sst_k", "env_pi_kt",
    "env_motion_u_ms", "env_motion_v_ms",
    "env_dvmax_past12_kt", "env_dvmax_past24_kt",
)

SHEAR_LAYERS = ("850_to_500_hPa", "850_to_200_hPa")
REGIONS = ("0_to_300_km", "0_to_500_km", "0_to_800_km", "200_to_800_km")
_DEEP, _MID = SHEAR_LAYERS.index("850_to_200_hPa"), SHEAR_LAYERS.index("850_to_500_hPa")
_R500, _R200_800 = REGIONS.index("0_to_500_km"), REGIONS.index("200_to_800_km")

# Diagnostics are 6-hourly. An overpass more than this far outside the
# series gets NaN rather than an extrapolated value.
MAX_EDGE_GAP_S = 6 * 3600


class EnvFormatError(RuntimeError):
    """The file's layer/region labels are not the ones this module indexes."""


class EnvSeries:
    """One storm's environment: 6-hourly diagnostics plus per-overpass rows."""

    def __init__(self, key, times, fields, sst_times, sst, overpass):
        self.key = key
        self.times = times            # epoch seconds, float64, ascending
        self.fields = fields          # name -> (N,) float64
        self.sst_times = sst_times
        self.sst = sst
        self.overpass = overpass      # filename -> {u, v, dv12, dv24, t}


def _attr_str(ds, name):
    v = ds.attrs.get(name)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO":
        v = v.ravel()[0]
        return v.decode() if isinstance(v, bytes) else str(v)
    return str(v)


def _unpack(ds) -> np.ndarray:
    """Packed int16 -> float64 with fill as NaN, THEN scale and offset
    (the fill value is defined on the packed integers)."""
    raw = np.asarray(ds[()])
    out = raw.astype(np.float64)
    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        out[raw == np.asarray(fill).ravel()[0]] = np.nan
    sf = ds.attrs.get("scale_factor")
    ao = ds.attrs.get("add_offset")
    if sf is not None:
        out *= float(np.asarray(sf).ravel()[0])
    if ao is not None:
        out += float(np.asarray(ao).ravel()[0])
    return out


def _unpack_slice(ds, sel) -> np.ndarray:
    """_unpack for a slice: HDF5 then reads only the chunks it touches --
    the full cylindrical SST array cost 4.5 MB for three radii."""
    raw = np.asarray(ds[sel])
    out = raw.astype(np.float64)
    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        out[raw == np.asarray(fill).ravel()[0]] = np.nan
    sf, ao = ds.attrs.get("scale_factor"), ds.attrs.get("add_offset")
    if sf is not None:
        out *= float(np.asarray(sf).ravel()[0])
    if ao is not None:
        out += float(np.asarray(ao).ravel()[0])
    return out


def _check_labels(ds, attr, expected):
    got = _attr_str(ds, attr)
    if got is None or tuple(got.split()) != tuple(expected):
        raise EnvFormatError(
            f"{ds.name}.{attr} is {got!r}, expected {' '.join(expected)!r} -- "
            f"refusing to index by position into an unrecognised layout")


def parse_env(h5, key: str = "") -> EnvSeries:
    """Build an EnvSeries from an open TC PRIMED env file (any h5py-like)."""
    d, cy, op = h5["diagnostics"], h5["cylindrical"], h5["overpass_storm_metadata"]

    for name in ("shear_magnitude", "shear_direction"):
        _check_labels(d[name], "layer", SHEAR_LAYERS)
        _check_labels(d[name], "regions", REGIONS)
    _check_labels(d["relative_humidity"], "regions", REGIONS)

    times = np.asarray(d["time"][()], dtype=np.float64)
    mag, dirn = _unpack(d["shear_magnitude"]), _unpack(d["shear_direction"])
    levels = np.asarray(d["level"][()], dtype=np.float64)
    rh = _unpack(d["relative_humidity"])[:, :, _R200_800]            # (N, level)
    mid = (levels >= 500) & (levels <= 700)
    low = (levels >= 700) & (levels <= 850)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rh_mid = np.nanmean(rh[:, mid], axis=1) if mid.any() else np.full(len(times), np.nan)
        rh_low = np.nanmean(rh[:, low], axis=1) if low.any() else np.full(len(times), np.nan)
    pi = _unpack(d["potential_intensity_theoretical"]).reshape(len(times), -1)[:, 0]

    fields = {
        "shear_deep_ms": mag[:, _DEEP, _R500], "shear_deep_dir_deg": dirn[:, _DEEP, _R500],
        "shear_mid_ms": mag[:, _MID, _R500], "shear_mid_dir_deg": dirn[:, _MID, _R500],
        "rh_mid_pct": rh_mid, "rh_low_pct": rh_low, "pi_kt": pi,
    }

    # Core SST from the cylindrical grid (aligned to every time).
    sst_times = np.asarray(cy["time"][()], dtype=np.float64)
    radius = _unpack(cy["radius"])
    core = np.isfinite(radius) & (radius <= 50.0)
    n_core = int(np.argmax(~core)) if (~core).any() else len(radius)   # radii ascend
    sst3 = _unpack_slice(cy["sst"], np.s_[:, :, :max(1, n_core)])     # (N, az, r<=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN over land
        sst = np.nanmean(sst3.reshape(len(sst_times), -1), axis=1)

    # Per-overpass rows: motion and PAST intensity change only.
    periods = [int(p) for p in np.asarray(op["intensity_change_periods"][()]).ravel()]
    ic = _unpack(op["intensity_change"])
    u = _unpack(op["storm_speed_zonal_component"])
    v = _unpack(op["storm_speed_meridional_component"])
    ot = np.asarray(op["time"][()], dtype=np.float64)
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in np.asarray(op["filename"][()]).ravel()]
    i12 = periods.index(-12) if -12 in periods else None
    i24 = periods.index(-24) if -24 in periods else None
    overpass = {}
    for j, n in enumerate(names):
        overpass[n] = {
            "t": ot[j], "u": u[j], "v": v[j],
            "dv12": ic[j, i12] if i12 is not None else np.nan,
            "dv24": ic[j, i24] if i24 is not None else np.nan,
        }
    return EnvSeries(key, times, fields, sst_times, sst, overpass)


def _interp(times, values, t):
    ok = np.isfinite(values)
    if ok.sum() == 0:
        return np.nan
    ts, vs = times[ok], values[ok]
    if t < ts[0] - MAX_EDGE_GAP_S or t > ts[-1] + MAX_EDGE_GAP_S:
        return np.nan
    return float(np.interp(t, ts, vs))


def _interp_vector(times, mag, dirn, t):
    """Interpolate a (magnitude, toward-direction) vector through its
    components, so 350 deg -> 10 deg passes through 0, not 180."""
    rad = np.radians(dirn)
    ux = _interp(times, mag * np.sin(rad), t)
    vy = _interp(times, mag * np.cos(rad), t)
    if not (np.isfinite(ux) and np.isfinite(vy)):
        return np.nan, np.nan
    return float(math.hypot(ux, vy)), float(math.degrees(math.atan2(ux, vy)) % 360.0)


def env_at(series: Optional[EnvSeries], when, overpass_filename: str = None) -> dict:
    """All ENV_KEYS for one overpass. NaN where unavailable; never raises."""
    out = {k: np.nan for k in ENV_KEYS}
    if series is None:
        return out
    t = when.timestamp() if hasattr(when, "timestamp") else float(when)
    f = series.fields
    out["env_shear_deep_ms"], out["env_shear_deep_dir_deg"] = _interp_vector(
        series.times, f["shear_deep_ms"], f["shear_deep_dir_deg"], t)
    out["env_shear_mid_ms"], out["env_shear_mid_dir_deg"] = _interp_vector(
        series.times, f["shear_mid_ms"], f["shear_mid_dir_deg"], t)
    out["env_rh_mid_pct"] = _interp(series.times, f["rh_mid_pct"], t)
    out["env_rh_low_pct"] = _interp(series.times, f["rh_low_pct"], t)
    out["env_pi_kt"] = _interp(series.times, f["pi_kt"], t)
    out["env_sst_k"] = _interp(series.sst_times, series.sst, t)

    row = series.overpass.get(overpass_filename) if overpass_filename else None
    if row is None and series.overpass:
        # Nearest overpass row within 30 min (same storm, same time) if the
        # filename itself is not listed.
        best = min(series.overpass.values(), key=lambda r: abs(r["t"] - t))
        row = best if abs(best["t"] - t) <= 1800 else None
    if row is not None:
        out["env_motion_u_ms"], out["env_motion_v_ms"] = float(row["u"]), float(row["v"])
        out["env_dvmax_past12_kt"] = float(row["dv12"])
        out["env_dvmax_past24_kt"] = float(row["dv24"])
    return out


# --- fetching ---------------------------------------------------------

_cache: dict = {}
_cache_lock = threading.Lock()
_storm_locks: dict = {}
CACHE_MAX_STORMS = 64


def find_env_key(basin: str, storm_num: int, season: int,
                 version: str = "v01r01", version_type: str = "final"):
    """(key, size) of the storm's env file, or (None, None)."""
    import tcprimed_ingest as tp
    prefix = f"{version}/{version_type}/{season}/{basin.upper()}/{storm_num:02d}/"
    s3 = tp._get_s3_client()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=tp.BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if "_env_" in obj["Key"] and obj["Key"].endswith(".nc"):
                return obj["Key"], obj["Size"]
    return None, None


def load_env_series(basin: str, storm_num: int, season: int,
                    progress_callback=None) -> Optional[EnvSeries]:
    """The storm's EnvSeries, fetched once per process (ranged, ~0.7 MB)
    and cached. None if the storm has no env file or it cannot be read --
    logged, never raised: the environment is an input, not a gate."""
    sid = (basin.upper(), int(storm_num), int(season))
    with _cache_lock:
        if sid in _cache:
            return _cache[sid]
        lock = _storm_locks.setdefault(sid, threading.Lock())
    with lock:                                # one fetch per storm, even under a pool
        with _cache_lock:
            if sid in _cache:
                return _cache[sid]
        series = None
        try:
            import tcprimed_ingest as tp
            from s3_range_reader import open_s3_hdf5
            key, size = find_env_key(*sid)
            if key is None:
                if progress_callback:
                    progress_callback(f"  env: no environmental file for "
                                      f"{sid[0]}{sid[1]:02d}{sid[2]}")
            else:
                h5, reader = open_s3_hdf5(tp._get_s3_client(), tp.BUCKET, key,
                                          size=size, prefer_ranged=True)
                try:
                    series = parse_env(h5, key)
                finally:
                    h5.close()
                if progress_callback:
                    st = reader.stats()
                    progress_callback(
                        f"  env: {sid[0]}{sid[1]:02d}{sid[2]} "
                        f"{st['bytes_fetched']/1e6:.2f} MB of {size/1e6:.0f} MB")
        except Exception as e:
            if progress_callback:
                progress_callback(f"  env unavailable for {sid[0]}{sid[1]:02d}{sid[2]} "
                                  f"({type(e).__name__}: {e})")
            series = None
        with _cache_lock:
            if len(_cache) >= CACHE_MAX_STORMS:
                _cache.pop(next(iter(_cache)))
            _cache[sid] = series
        return series


# --- model encoding (0.160) --------------------------------------------
# (value - centre) / scale for each ERA5 channel. Chosen for O(1) planes
# over the observed range, not fitted: shear 0-30 m/s, RH 40-90 %, SST
# 297-305 K, past-24 h change -60..+60 kt, motion 0-15 m/s.
ENV_ENCODING = {
    "rh_mid": (60.0, 15.0), "sst": (301.0, 2.0), "dv24": (0.0, 20.0),
    "shear": (0.0, 10.0), "motion": (0.0, 5.0),
}


def encode_env(env) -> np.ndarray:
    """Model channel values, in ml_constants.ENV_CHANNELS order, for one
    example -- the ONLY encoder, used by training and inference alike.
    None, an empty dict or all-NaN input gives zeros with present = 0; a
    single NaN field becomes 0 (its mean) and leaves present = 1."""
    out = np.zeros(8, dtype=np.float32)
    if not env:
        return out
    g = lambda k: float(env.get(k, np.nan)) if env.get(k) is not None else np.nan
    mag, d = g("env_shear_deep_ms"), g("env_shear_deep_dir_deg")
    vals = [g(k) for k in ENV_KEYS]
    if not any(np.isfinite(v) for v in vals):
        return out
    c, sc = ENV_ENCODING["shear"]
    if np.isfinite(mag) and np.isfinite(d):
        out[0] = mag * math.sin(math.radians(d)) / sc
        out[1] = mag * math.cos(math.radians(d)) / sc
    for i, (key, enc) in ((2, ("env_rh_mid_pct", "rh_mid")), (3, ("env_sst_k", "sst")),
                          (4, ("env_dvmax_past24_kt", "dv24"))):
        v = g(key)
        if np.isfinite(v):
            c, sc = ENV_ENCODING[enc]
            out[i] = (v - c) / sc
    c, sc = ENV_ENCODING["motion"]
    for i, key in ((5, "env_motion_u_ms"), (6, "env_motion_v_ms")):
        v = g(key)
        if np.isfinite(v):
            out[i] = v / sc
    out[7] = 1.0
    return out
