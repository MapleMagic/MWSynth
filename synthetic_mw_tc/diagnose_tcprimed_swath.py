"""
Inspect a raw cached TC-PRIMED overpass file as this project actually
reads it, to find out why the regridded MW ends up absent at the storm.

diagnose_training_data.py established that every storm-centered training
patch has zero supervised pixels. That means the real MW never lands at
the storm on the GOES grid. There are only a few ways that happens:

  1. Longitude convention mismatch -- TC-PRIMED using 0..360 while GOES
     uses -180..180. A storm at 60W would then sit at 300 in the swath
     and never overlap the GOES scene at all.
  2. lat/lon arrays whose shape does not match the brightness-temperature
     array they are supposed to geolocate (easy to get wrong when 37GHz
     and 89GHz live in different groups with different resolutions).
  3. The swath being geolocated somewhere else entirely -- wrong group's
     latitude/longitude paired with the right group's Tc.

This prints the actual numbers for each, next to the storm position the
file itself reports, so the cause is visible rather than inferred.

Run: python diagnose_tcprimed_swath.py [optional_path_to_file.nc]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

import tcprimed_ingest as tp


def diagnose_file(path: str):
    name = os.path.basename(path)
    print(f"=== {name} ===")

    parsed = tp._parse_overpass_filename(name)
    if parsed is None:
        print("  filename not recognised as a TC-PRIMED overpass file")
        return
    instrument = parsed["instrument"]
    print(f"  instrument={instrument} basin={parsed['basin']} "
          f"storm={parsed['storm_num']:02d} season={parsed['season']}")

    import h5py

    # What the file itself says about where the storm was
    with h5py.File(path, "r") as f:
        meta = {}
        if "overpass_storm_metadata" in f:
            g = f["overpass_storm_metadata"]
            for key in ("storm_latitude", "storm_longitude", "latitude", "longitude",
                        "intensity", "time"):
                if key in g:
                    try:
                        meta[key] = float(np.asarray(g[key][()]).flat[0])
                    except Exception:
                        pass
            print(f"  overpass_storm_metadata keys: {sorted(g.keys())}")
        print(f"  storm metadata values: {meta}")

    # Now read it exactly the way the pipeline does
    try:
        swath = tp.read_overpass_as_swath(path, instrument)
    except Exception as e:
        print(f"  read_overpass_as_swath FAILED: {type(e).__name__}: {e}")
        return

    print(f"  scene_time={swath.scene_time}")
    for freq, (la, lo) in ((37, swath.grid_for(37)), (89, swath.grid_for(89))):
        tb = swath.v37 if freq == 37 else swath.v89
        print(f"  --- {freq}GHz ---")
        print(f"    Tc shape={tb.shape}  lat shape={la.shape}  lon shape={lo.shape}")
        if la.shape != tb.shape:
            print("    *** SHAPE MISMATCH: lat/lon do not geolocate this Tc array ***")
        print(f"    lat range [{np.nanmin(la):.2f}, {np.nanmax(la):.2f}]")
        print(f"    lon range [{np.nanmin(lo):.2f}, {np.nanmax(lo):.2f}]")
        if np.nanmax(lo) > 180.0:
            print("    *** LONGITUDE > 180: file uses 0..360, project expects -180..180 ***")
        finite = int(np.isfinite(tb).sum())
        print(f"    Tc finite {finite}/{tb.size} ({100.0*finite/tb.size:.1f}%)")
        if finite:
            print(f"    Tc range [{np.nanmin(tb):.1f}, {np.nanmax(tb):.1f}] K")

    # Does the swath actually contain the storm the metadata reports?
    slat = meta.get("storm_latitude", meta.get("latitude"))
    slon = meta.get("storm_longitude", meta.get("longitude"))
    if slat is not None and slon is not None:
        la, lo = swath.grid_for(89)
        inside = (np.nanmin(la) <= slat <= np.nanmax(la)
                  and np.nanmin(lo) <= slon <= np.nanmax(lo))
        print(f"  storm ({slat:.2f}, {slon:.2f}) inside 89GHz swath bounds: {inside}")
        if not inside:
            print("  *** the swath does not cover its own storm -- geolocation is wrong ***")
    print()


def main():
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = sorted(glob.glob(os.path.join(tp.DEFAULT_LOCAL_DIR, "*.nc")))
        if not files:
            print(f"No .nc files in {tp.DEFAULT_LOCAL_DIR}")
            return
        # one GMI and one AMSR2 if available, since they use different groups
        picked, seen = [], set()
        for f in files:
            p = tp._parse_overpass_filename(os.path.basename(f))
            if p and p["instrument"] not in seen:
                seen.add(p["instrument"])
                picked.append(f)
            if len(picked) >= 2:
                break
        files = picked or files[:1]

    for f in files:
        diagnose_file(f)


if __name__ == "__main__":
    main()
