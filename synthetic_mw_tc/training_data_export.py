"""
Paired (GOES input, real MW output) training example export, for
training a model to learn a CORRECTION on top of the existing parametric
synthetic algorithm -- not to replace it. Per direct guidance: the model
should learn (real_MW - parametric_backbone), the systematic error the
current algorithm makes, rather than the full MW field from scratch --
more data-efficient, integrates naturally with the existing fusion/
calibration architecture, and stays debuggable.

SENSOR RESTRICTION, per direct guidance: only GMI and AMSR2 examples are
saved. Both have genuinely higher native resolution than WSFM/SSMIS, and
both have operational eras (GMI 2014-present, AMSR2 2012-present) that
cleanly overlap the GOES-R ABI series (GOES-16 onward, 2016-present) this
project's inputs come from -- TMI/AMSR-E were considered and deliberately
excluded, since their eras mostly or entirely predate ABI, meaning
training on them would pair real MW ground truth with a different
generation of geostationary imagery than this model will ever see at
inference time.

FORMAT: one compressed .npz file per example (numpy's native format --
no new dependency, easy to load later with np.load). Each file has the
GOES bands (input), the parametric backbone output (what the existing
algorithm produces from GOES alone, before any real MW is blended in --
the OTHER half of the residual target), and the regridded, QC'd real MW
V/H fields (the ground truth used to compute what correction the model
should have learned to add).
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np

DEFAULT_EXPORT_DIR = os.path.expanduser("~/.synthetic_mw_tc/training_data")

# Sensor name prefixes treated as "GMI family" or "AMSR2 family" for
# filtering purposes -- covers both the NRT and archive access paths for
# GMI (same physical instrument, just different latency/route), which
# should both count as equally valid training targets.
GMI_SENSOR_PREFIXES = ("GMI",)
AMSR2_SENSOR_PREFIXES = ("AMSR2",)


def _sensor_is_allowed(sensor: str | None) -> bool:
    if not sensor:
        return False
    return sensor.startswith(GMI_SENSOR_PREFIXES) or sensor.startswith(AMSR2_SENSOR_PREFIXES)


# Optional fields written when the caller supplies them (0.98). Older
# exports lack these; ml_train fills the corresponding channels with a
# neutral plane rather than dropping them, so mixed-vintage datasets stay
# trainable against the current architecture.
_OPTIONAL_EXPORT_KEYS = ("land_fraction", "elevation_m")


# --- Compact storage ------------------------------------------------
# Brightness temperatures are stored as scaled uint16 rather than
# float32. This is not premature optimisation: streaming TC PRIMED off S3
# removes the source-file storage problem and immediately makes the
# EXPORT the binding constraint. At float32 and a 512x512 footprint an
# example is ~23 MB, so 10,000 of them is ~115 GB compressed -- more than
# the GOES archive this was meant to avoid.
#
# uint16 at 0.01 K resolution spans 0-655 K, which covers every physical
# Tb with precision far finer than sensor noise (~0.5-1 K) and better
# than float16 would give in this range (float16's ~0.15 K at 300 K comes
# from its relative precision). Halves the file for no meaningful loss.
TB_SCALE = 0.01          # K per stored unit
TB_OFFSET = 0.0
_TB_SENTINEL = 0         # reserved for NaN


def pack_tb(arr):
    """float Tb array -> scaled uint16, NaN preserved as a sentinel."""
    a = np.asarray(arr, dtype=np.float64)
    out = np.full(a.shape, _TB_SENTINEL, dtype=np.uint16)
    good = np.isfinite(a)
    q = np.round((a[good] - TB_OFFSET) / TB_SCALE)
    # Clamp into the representable range, keeping 0 free as the NaN flag.
    out[good] = np.clip(q, 1, np.iinfo(np.uint16).max).astype(np.uint16)
    return out


def unpack_tb(arr):
    """Inverse of pack_tb; the sentinel comes back as NaN."""
    a = np.asarray(arr)
    out = a.astype(np.float32) * TB_SCALE + TB_OFFSET
    out[a == _TB_SENTINEL] = np.nan
    return out


def _vh_physics_id() -> str:
    """Backbone physics identifier in force at export time."""
    try:
        from synthetic_algorithm import VH_PHYSICS_ID
        return VH_PHYSICS_ID
    except Exception:
        return "unknown"


def check_physics_consistency(file_paths: list, strict: bool = True) -> dict:
    """Group the dataset by the backbone physics each example was
    generated under, and refuse to train on a mixture.

    WHY THIS IS FATAL RATHER THAN A WARNING: the learning target is
    (real_MW - backbone), measured against the backbone in the file. When
    0.95 inverted the 37 GHz eyewall from a scattering depression to an
    emission signature -- about 120 K at the core -- examples from before
    and after that change stopped describing the same quantity. Training
    across both does not average two views of one thing; it fits a model
    to two different things at once and converges toward neither, while
    every loss curve looks entirely healthy.

    Old exports carry no identifier and are reported as
    'pre-0.99 (unversioned)'.
    """
    groups = {}
    for f in file_paths:
        vid = "pre-0.99 (unversioned)"
        try:
            with np.load(f, allow_pickle=True) as d:
                if "vh_physics_id" in getattr(d, "files", ()):
                    vid = str(d["vh_physics_id"])
        except Exception:
            vid = "unreadable"
        groups.setdefault(vid, []).append(f)

    if len(groups) > 1 and strict:
        summary = ", ".join(f"{k}: {len(v)} file(s)" for k, v in sorted(groups.items()))
        raise RuntimeError(
            "Training set mixes backbone physics vintages -- " + summary + ".\n"
            "The residual target is measured against the backbone, so these files "
            "describe different quantities and training across them converges toward "
            "neither. Re-mine under the current physics, or train on one vintage only "
            "(pass strict=False to override deliberately)."
        )
    return groups


def export_training_example(
    band13, band9, band7, band2, storm_fix, result, output_dir: str = DEFAULT_EXPORT_DIR,
    extra_ir: dict | None = None,   # override; defaults to the regridded
                                    # arrays the model actually saw
) -> str | None:
    """Save one paired training example, if real MW data was actually
    used in this generation AND it came from GMI or AMSR2 specifically
    (see module docstring for why those two and not WSFM/SSMIS/AMSR3).
    Returns the saved file path, or None if there was no real MW data
    this run, or it came from a sensor outside the allowed set -- neither
    is an error, both are silently skipped, since most runs won't have
    real GMI/AMSR2 data available and that's expected/fine.

    band13/band9/band7/band2: the same BandImage objects passed into
        generate_synthetic_mw (band2 may be None -- night, or not fetched).
    storm_fix: the StormFix used for this generation.
    result: the SyntheticMWResult returned by generate_synthetic_mw.
    """
    diag = result.diagnostics
    if not diag.get("fusion_sources_used", {}).get("mw"):
        return None
    if not _sensor_is_allowed(diag.get("mw_sensor")):
        return None

    os.makedirs(output_dir, exist_ok=True)

    scene_time = band13.scene_time
    sensor_tag = diag["mw_sensor"].replace("-", "")
    filename = f"{result.storm_id}_{scene_time:%Y%m%d%H%M}_{sensor_tag}.npz"
    path = os.path.join(output_dir, filename)

    # Resumability (0.117). The filename is fully determined by storm,
    # scene time and sensor, so an existing file is the same example. A
    # multi-season mine is hours long and any interruption -- a dropped
    # connection, a laptop lid -- previously meant starting over and
    # re-downloading every GOES band already processed. Skipping work
    # already on disk makes a resumed run cost only what is left.
    #
    # Only skips when the file was written under the CURRENT backbone
    # physics. A stale-vintage file must be redone, not reused, or the
    # dataset silently mixes vintages and the 0.99 guard refuses to train
    # on it later.
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=True) as _d:
                _vid = (str(_d["vh_physics_id"])
                        if "vh_physics_id" in _d.files else None)
            if _vid == _vh_physics_id():
                return path
        except Exception:
            pass    # unreadable or half-written -- fall through and rewrite

    def _tb(x):
        """Brightness-temperature array -> scaled uint16.

        pack_tb existed from 0.104 and was NEVER CALLED -- the exporter
        kept writing float32, so the compact-storage work was dead and
        every file was twice the intended size. Wired in here.

        Backward compatible by construction: the loader keys on dtype, so
        float32 files already on disk stay readable and a dataset may
        contain both.
        """
        if x is None:
            return np.array([], dtype=np.uint16)
        return pack_tb(x)

    def _f32(x):
        return x.astype(np.float32) if x is not None else np.array([], dtype=np.float32)

    np.savez_compressed(
        path,
        # --- input: GOES ---
        ir_band13=band13.values.astype(np.float32),
        wv_band9=band9.values.astype(np.float32),
        swir_band7=band7.values.astype(np.float32),
        vis_band2=(band2.values.astype(np.float32) if band2 is not None else np.array([], dtype=np.float32)),
        # --- input: supplementary ABI IR bands (0.99) ---
        # Stored under the exact key ml_train reads (ir_band<N>). Bands
        # absent from a given frame are simply not written; the trainer
        # substitutes a neutral plane, so mixed-vintage datasets stay
        # usable rather than shifting every channel index.
        **{f"ir_band{b}": np.asarray(arr, dtype=np.float32)
           for b, arr in (extra_ir if extra_ir is not None
                          else result.diagnostics.get("extra_ir_regridded", {})).items()
           if arr is not None},
        # --- input: surface conditioning (0.99) ---
        # Land fraction and elevation, resolved at generation time.
        # Ocean-only frames still write these (all zeros) so the presence
        # of the key means "this was computed", not "there was land".
        **({"land_fraction": np.asarray(result.diagnostics["land_fraction"], dtype=np.float32)}
           if result.diagnostics.get("land_fraction") is not None else {}),
        **({"elevation_m": np.asarray(result.diagnostics["elevation_m"], dtype=np.float32)}
           if result.diagnostics.get("elevation_m") is not None else {}),
        # --- provenance ---
        # Which backbone radiative physics produced the backbone_* arrays
        # in this file. ml_train refuses to mix vintages: the residual
        # target is measured against the backbone, so examples generated
        # before and after a physics change describe different quantities
        # and averaging them trains toward neither.
        **({"flash_density": np.asarray(result.diagnostics["flash_density"],
                                       dtype=np.float32)}
           if result.diagnostics.get("flash_density") is not None else {}),
        vh_physics_id=np.str_(_vh_physics_id()),
        lat=result.lat.astype(np.float32),
        lon=result.lon.astype(np.float32),
        # --- input: storm state (RMW/vortex geometry matters a lot for
        # the current parametric model, and is cheap context for a
        # learned model too) ---
        storm_lat=np.float32(storm_fix.lat),
        storm_lon=np.float32(storm_fix.lon),
        storm_vmax_kt=np.float32(storm_fix.vmax_kt),
        storm_rmw_nm=np.float32(storm_fix.rmw_nm) if storm_fix.rmw_nm else np.float32(np.nan),
        storm_roci_nm=np.float32(storm_fix.roci_nm) if storm_fix.roci_nm else np.float32(np.nan),
        # --- input: what the EXISTING parametric algorithm produces from
        # GOES alone, before any real MW is blended in. This is the
        # OTHER half of the residual-learning target: a training script
        # computes (target_v37 - backbone_v37) as the correction the
        # model should learn to predict, not target_v37 in isolation. ---
        backbone_v37=_tb(diag.get("backbone_v37")),
        backbone_h37=_tb(diag.get("backbone_h37")),
        backbone_v89=_tb(diag.get("backbone_v89")),
        backbone_h89=_tb(diag.get("backbone_h89")),
        # --- target: real MW, regridded onto the SAME grid as the GOES
        # input above, already QC'd (physical-bounds masked, despeckled
        # where applicable) -- NaN where there was no real coverage at
        # that pixel, which a training pipeline should treat as "no
        # supervision signal here," not zero ---
        target_v37=_tb(diag.get("mw_regridded_v37")),
        target_h37=_tb(diag.get("mw_regridded_h37")),
        target_v89=_tb(diag.get("mw_regridded_v89")),
        target_h89=_tb(diag.get("mw_regridded_h89")),
        # --- metadata ---
        storm_id=result.storm_id,
        scene_time_iso=scene_time.isoformat(),
        mw_sensor=diag["mw_sensor"],
        sensor_note=str(diag.get("calibration_source", "")),
    )
    return path


def dataset_summary(output_dir: str = DEFAULT_EXPORT_DIR) -> dict:
    """Quick inventory of what's been accumulated so far -- how many
    examples, which storms, which sensors, date range -- useful for
    checking whether there's "enough" to actually attempt training yet."""
    if not os.path.isdir(output_dir):
        return {"n_examples": 0, "storms": [], "sensors": {}, "date_range": None}

    files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
    storms = sorted(set(f.split("_")[0] for f in files))
    times = []
    sensors: dict = {}
    for f in files:
        try:
            data = np.load(os.path.join(output_dir, f), allow_pickle=True)
            times.append(datetime.fromisoformat(str(data["scene_time_iso"])))
            sensor = str(data["mw_sensor"]) if "mw_sensor" in data else "unknown"
            sensors[sensor] = sensors.get(sensor, 0) + 1
        except Exception:
            continue

    return {
        "n_examples": len(files),
        "storms": storms,
        "sensors": sensors,
        "date_range": (min(times), max(times)) if times else None,
    }

