"""
Surface type (land vs ocean) for the synthetic MW backbone.

WHY THIS EXISTS: every background brightness temperature in CALIBRATION
is an open-ocean value, and before this module they were applied
uniformly across the grid regardless of what was underneath. That is
wrong wherever land is in view, and wrong in a way that inverts the
signature the composites are built on.

At 37 and 89 GHz, ocean is a poor emitter and strongly polarizing --
H-pol emissivity is far below V-pol, so V-H runs roughly 55-80 K. Land
is a near-blackbody at these frequencies with emissivity around 0.9-0.95
in BOTH polarizations, so land sits close to its physical temperature
(~285-300 K) with V-H of only a few K.

The consequence is not a small offset. A warm, unpolarized surface is
exactly what heavy precipitation looks like over ocean: it is the
signature the emission model exists to produce. So an unmasked land
background is not merely mis-levelled, it is actively imitating the
feature being detected -- and it does so over the coast, which is where
a landfalling storm matters most. This is also the reason
polarization-corrected temperature was devised in the first place
(Spencer et al. 1989; Cecil & Chronis 2018): to suppress surface
emissivity differences so the ice-scattering signal survives across a
coastline.

BACKENDS. There is no land/sea dataset bundled here, and downloading one
at generation time is not acceptable for a real-time tool. Instead this
tries, in order:
  1. `global_land_mask` -- a small pure-Python package with an embedded
     mask, no network at runtime. The recommended option.
  2. `cartopy`'s Natural Earth land geometries, if it happens to be
     installed for other reasons.
  3. All-ocean, with a single explicit warning.

The fallback is honest rather than silent: an all-ocean assumption is
correct for the large majority of TC frames and merely restores the
previous behaviour, but the caller is told, once, that it is in force.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Nominal elevation assigned to land pixels by surface_elevation_m() until
# a real DEM is wired in. Deliberately modest and flat: a wrong constant is
# honest, whereas synthesised terrain would look like information.
NOMINAL_LAND_ELEVATION_M = 200.0

_BACKEND = None          # resolved lazily: "packed" | "global_land_mask" | "cartopy" | "none"
_WARNED = False

# --- Bit-packed, memory-mapped land mask (0.161) ---------------------------
# global_land_mask decompresses its whole (21600, 43200) mask AT IMPORT as
# one byte per cell: 933 MB, in every process that touches land. With
# mining moving to one PROCESS per worker, that is 933 MB x workers for one
# bit of information per cell. Packed to bits it is 117 MB, and as a
# memory-mapped .npy the OS reads only the pages a lookup touches and
# shares them between every process on the machine.
#
# Built once from global_land_mask (in a short-lived subprocess, so the
# 933 MB is released), then identical in every lookup -- verified against
# globe.is_land on random points, poles, the dateline and real grids.
import json as _json
import os as _os
import threading as _threading

PACKED_MASK_DIR = _os.path.expanduser("~/.synthetic_mw_tc/cache")
PACKED_MASK_PATH = _os.path.join(PACKED_MASK_DIR, "globe_land_bits_v1.npy")
PACKED_META_PATH = _os.path.join(PACKED_MASK_DIR, "globe_land_bits_v1.json")
_packed = None
_packed_meta = None
_packed_lock = _threading.Lock()


def build_packed_land_mask(path: str = None, meta_path: str = None) -> str:
    """Write the bit-packed LAND mask derived from global_land_mask.
    Holds the 933 MB array while it runs: call it in a short-lived process
    (ensure_packed_land_mask does)."""
    path = path or PACKED_MASK_PATH
    meta_path = meta_path or PACKED_META_PATH
    from global_land_mask import globe
    land = np.logical_not(globe._mask)
    packed = np.packbits(land, axis=1)
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{_os.getpid()}.npy"     # unique: builders may race
    np.save(tmp, packed)
    meta = {
        "shape": list(land.shape),
        "lat0": float(globe._lat[0]), "dlat": float(globe._lat[1] - globe._lat[0]),
        "lat_min": float(globe._lat.min()), "lat_max": float(globe._lat.max()),
        "lon0": float(globe._lon[0]), "dlon": float(globe._lon[1] - globe._lon[0]),
        "lon_min": float(globe._lon.min()), "lon_max": float(globe._lon.max()),
    }
    tmp_meta = f"{meta_path}.tmp{_os.getpid()}"
    with open(tmp_meta, "w") as fh:
        _json.dump(meta, fh)
    # Mask first, metadata last: a reader that finds the metadata finds a
    # complete mask. os.replace is atomic, so racing builders just
    # overwrite one identical file with another.
    _os.replace(tmp, path)
    _os.replace(tmp_meta, meta_path)
    return path


def ensure_packed_land_mask() -> bool:
    """True once the packed mask exists. Builds it in a SUBPROCESS if
    missing, so the caller never holds global_land_mask's 933 MB."""
    if _os.path.exists(PACKED_MASK_PATH) and _os.path.exists(PACKED_META_PATH):
        return True
    try:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1, mp_context=_mp.get_context("spawn")) as ex:
            ex.submit(build_packed_land_mask).result()
    except Exception:
        try:
            build_packed_land_mask()           # in-process fallback
        except Exception:
            return False
    return _os.path.exists(PACKED_MASK_PATH)


def _load_packed():
    global _packed, _packed_meta
    if _packed is not None:
        return _packed, _packed_meta
    with _packed_lock:
        if _packed is None:
            with open(PACKED_META_PATH) as fh:
                meta = _json.load(fh)
            arr = np.load(PACKED_MASK_PATH, mmap_mode="r")
            if list(arr.shape) != [meta["shape"][0], (meta["shape"][1] + 7) // 8]:
                raise ValueError("packed land mask does not match its metadata")
            _packed_meta, _packed = meta, arr
    return _packed, _packed_meta


def packed_is_land(lat, lon) -> np.ndarray:
    """global_land_mask.globe.is_land, from the packed memory-mapped copy:
    same index arithmetic (clamp, then truncate), one bit per cell.
    Non-finite coordinates are ocean rather than an IndexError."""
    packed, m = _load_packed()
    lat = np.asarray(lat, dtype=np.float64)
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat_c = np.clip(np.where(ok, lat, 0.0), m["lat_min"], m["lat_max"])
    lon_c = np.clip(np.where(ok, lon, 0.0), m["lon_min"], m["lon_max"])
    li = ((lat_c - m["lat0"]) / m["dlat"]).astype(int)
    lo = ((lon_c - m["lon0"]) / m["dlon"]).astype(int)
    byte = packed[li, lo >> 3]
    land = ((byte >> (7 - (lo & 7))) & 1).astype(bool)
    return land & ok


def _resolve_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import importlib.util
        if importlib.util.find_spec("global_land_mask") is not None:
            # Prefer the packed copy; fall back to the package itself if
            # the copy cannot be built (e.g. an unwritable home directory).
            _BACKEND = "packed" if ensure_packed_land_mask() else "global_land_mask"
            return _BACKEND
    except Exception:
        pass
    try:
        import cartopy.io.shapereader  # noqa: F401
        from shapely.geometry import Point  # noqa: F401
        _BACKEND = "cartopy"
        return _BACKEND
    except ImportError:
        pass
    _BACKEND = "none"
    return _BACKEND


def land_fraction(lat: np.ndarray, lon: np.ndarray, progress_callback=None) -> np.ndarray:
    """Return a 0-1 land fraction on the given grid.

    0 is open ocean, 1 is land. Intermediate values only arise from the
    smoothing applied by the caller; the underlying masks are binary.

    Never raises. If no backend is available it returns all-zeros (all
    ocean) and warns once, which reproduces the pre-existing behaviour
    rather than failing a generate.
    """
    global _WARNED
    backend = _resolve_backend()
    shape = np.asarray(lat).shape

    if backend == "packed":
        try:
            return packed_is_land(lat, lon).astype(np.float32)
        except Exception:
            backend = "global_land_mask"

    if backend == "global_land_mask":
        try:
            from global_land_mask import globe
            # The package expects longitudes in [-180, 180).
            lon_w = (np.asarray(lon) + 180.0) % 360.0 - 180.0
            return globe.is_land(np.asarray(lat), lon_w).astype(np.float32)
        except Exception:
            pass

    if backend == "cartopy":
        try:
            return _cartopy_land_fraction(lat, lon)
        except Exception:
            pass

    if not _WARNED:
        _WARNED = True
        if progress_callback:
            progress_callback(
                "No land/sea mask backend available -- treating the whole grid as ocean. "
                "Land in view will be rendered with ocean background emissivity, which at "
                "37/89 GHz mimics heavy precipitation. Install 'global_land_mask' to fix."
            )
    return np.zeros(shape, dtype=np.float32)


def _cartopy_land_fraction(lat, lon) -> np.ndarray:
    """Rasterize Natural Earth land polygons onto the grid. Slower than
    the embedded-mask backend, so it is only the fallback."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import MultiPoint
    from shapely.prepared import prep
    from shapely.ops import unary_union

    reader = shpreader.Reader(
        shpreader.natural_earth(resolution="110m", category="physical", name="land")
    )
    land = prep(unary_union(list(reader.geometries())))
    lat_a = np.asarray(lat)
    lon_a = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    out = np.zeros(lat_a.shape, dtype=np.float32)
    flat_lat, flat_lon = lat_a.ravel(), lon_a.ravel()
    pts = MultiPoint(np.column_stack([flat_lon, flat_lat]))
    flags = np.fromiter((land.contains(p) for p in pts.geoms), dtype=bool,
                        count=len(flat_lat))
    out.ravel()[:] = flags.astype(np.float32)
    return out


def surface_elevation_m(lat: np.ndarray, lon: np.ndarray,
                        land_frac: Optional[np.ndarray] = None) -> np.ndarray:
    """Coarse surface elevation in metres, for use as a model conditioning
    channel (Li et al. 2025 found adding elevation improved their results).

    There is no bundled terrain dataset and downloading one at generation
    time is not acceptable for a real-time tool, so this currently derives
    a nominal land elevation from the land mask rather than pretending to
    real topography. That captures the part which actually matters for
    passive MW -- that the surface underneath is land at all, and roughly
    how much of the atmosphere sits above it -- without inventing terrain
    detail the project cannot support.

    Drop in ETOPO (or any gridded DEM) here when a real one is available;
    the channel and its normalization already exist, so nothing downstream
    needs to change.
    """
    if land_frac is None:
        land_frac = land_fraction(lat, lon)
    return (np.asarray(land_frac, dtype=np.float32) * NOMINAL_LAND_ELEVATION_M)


def backend_name() -> str:
    """Which backend is in use, for diagnostics."""
    return _resolve_backend()
