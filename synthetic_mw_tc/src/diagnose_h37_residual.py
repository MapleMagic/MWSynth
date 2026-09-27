"""Storm-relative residual composites: (real - backbone) by r/RMW and intensity.

Plan item #1 (0.158). Before changing any h37 physics, find the SHAPE of
the error that constant-fitting could not remove (h37 improved only ~6% in
both fits): a ring at 1-2 RMW points at eyewall emission, a broad outer
excess at stratiform/warm rain, a flat offset at the background. All four
channels are reported so h37 can be read against the others.

0.160 adds, for NPZs carrying the ERA5 environment (mined by 0.160+, or
run through backfill_npz.py):

- SHEAR-RELATIVE quadrants (downshear-left/right, upshear-left/right),
  using the 850-200 hPa, 0-500 km shear vector's heading. Northern
  Hemisphere convention; every basin mined so far (AL/EP/CP) is NH.
- the radial h37 profile split by shear strength;
- convective-burst metrics (plan #16): cold-top (< 208 K) fraction and
  GLM flash density inside the RMW, per shear quadrant, for examples that
  had just intensified rapidly (past 24 h >= +30 kt) against the rest.
  Past-only intensity change: nothing here looks ahead of the overpass.

    python diagnose_h37_residual.py [data_dir] [--csv out.csv]
"""
import glob
import os
import sys

import numpy as np

import training_data_export as tde

CHANNELS = ("h37", "v37", "v89", "h89")
R_EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8])
V_BANDS = ((0, 34), (34, 64), (64, 96), (96, 999))


def _tb(z, key):
    a = z[key]
    if a.dtype == np.uint16:
        unpack = getattr(tde, "unpack_tb", None)
        if unpack is None:
            raise RuntimeError("training_data_export.unpack_tb missing")
        a = unpack(a)
    a = np.asarray(a, dtype=np.float64)
    return np.where((a > 50) & (a < 350), a, np.nan)


def r_over_rmw(z):
    lat, lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    clat, clon = float(z["storm_lat"]), float(z["storm_lon"])
    dy = (lat - clat) * 111.2
    dx = (lon - clon) * 111.2 * np.cos(np.radians(clat))
    rmw_km = max(float(z["storm_rmw_nm"]) * 1.852, 5.0)
    return np.hypot(dx, dy) / rmw_km


QUADS = ("DR", "UR", "UL", "DL")      # clockwise from downshear
Q_BINS = ((0, 1), (1, 2), (2, 4))      # r/RMW
SHEAR_STRATA = ((0, 5), (5, 10), (10, 99))    # m/s
COLD_TOP_K = 208.0


def shear_quadrant(z, shear_dir_deg):
    """Per-pixel quadrant index into QUADS: angle measured CLOCKWISE from
    the downshear heading, 0-90 = downshear-right ... 270-360 =
    downshear-left (looking downshear)."""
    lat, lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    clat, clon = float(z["storm_lat"]), float(z["storm_lon"])
    dy = (lat - clat) * 111.2
    dx = (lon - clon) * 111.2 * np.cos(np.radians(clat))
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    rel = (bearing - shear_dir_deg) % 360.0
    return (rel // 90).astype(int)


def _env(z, k):
    return float(z[k]) if k in z.files else np.nan


def main(data_dir=None, csv_path=None):
    data_dir = data_dir or tde.DEFAULT_EXPORT_DIR
    files = tde.list_training_files(data_dir)
    if not files:
        print(f"no NPZs in {data_dir}")
        return 1
    nb = len(R_EDGES) - 1
    # [band][channel] -> per-radial-bin sums over EXAMPLES (each example's
    # bin mean counts once, so one huge-grid storm cannot dominate)
    acc = {b: {c: [[] for _ in range(nb)] for c in CHANNELS} for b in V_BANDS}
    ids = set()
    for f in files:
        z = np.load(f)
        ids.add(str(z["vh_physics_id"]))
        vmax = float(z["storm_vmax_kt"])
        band = next(b for b in V_BANDS if b[0] <= vmax < b[1])
        rr = r_over_rmw(z)
        idx = np.digitize(rr, R_EDGES) - 1
        for c in CHANNELS:
            res = _tb(z, f"target_{c}") - _tb(z, f"backbone_{c}")
            for i in range(nb):
                v = res[(idx == i) & np.isfinite(res)]
                if v.size >= 20:
                    acc[band][c][i].append(float(np.mean(v)))
    print(f"{len(files)} examples, physics id(s): {sorted(ids)}")
    print("mean (real - backbone) K per radial bin; n = examples contributing\n")
    hdr = "".join(f"{R_EDGES[i]:>4.1f}-{R_EDGES[i+1]:<4.1f}" for i in range(nb))
    for b in V_BANDS:
        n_ex = max(len(acc[b]["h37"][i]) for i in range(nb))
        if n_ex == 0:
            continue
        print(f"Vmax {b[0]}-{b[1] if b[1] < 999 else '+'} kt  (up to {n_ex} examples)")
        print(f"  r/RMW  {hdr}")
        for c in CHANNELS:
            row = "".join(f"{np.mean(v):>+9.1f}" if v else "      ---"
                          for v in acc[b][c])
            print(f"  {c:5s}  {row}")
        print()
    print("Read h37 against v37: a feature in h37 only is polarisation-")
    print("specific (surface emissivity / wind roughening); one in both is")
    print("hydrometeor emission or scattering.")
    shear_relative(files, csv_path)
    return 0


def shear_relative(files, csv_path=None):
    """Shear-quadrant composites, shear strata, burst metrics (0.160)."""
    rows = []
    qacc = {c: {(i, q): [] for i in range(len(Q_BINS)) for q in range(4)}
            for c in ("h37", "v37", "v89")}
    strata = {st: [[] for _ in range(len(R_EDGES) - 1)] for st in SHEAR_STRATA}
    burst = {grp: {q: {"cold": [], "flash": []} for q in range(4)}
             for grp in ("RI (past 24 h >= +30 kt)", "other")}
    n_env = 0
    for f in files:
        z = np.load(f)
        sdir, smag = _env(z, "env_shear_deep_dir_deg"), _env(z, "env_shear_deep_ms")
        if not (np.isfinite(sdir) and np.isfinite(smag)):
            continue
        n_env += 1
        rr = r_over_rmw(z)
        quad = shear_quadrant(z, sdir)
        res = {c: _tb(z, f"target_{c}") - _tb(z, f"backbone_{c}") for c in ("h37", "v37", "v89")}
        row = {"file": os.path.basename(f), "vmax_kt": float(z["storm_vmax_kt"]),
               "shear_ms": smag, "shear_dir_deg": sdir,
               "rh_mid_pct": _env(z, "env_rh_mid_pct"), "sst_k": _env(z, "env_sst_k"),
               "dv_past24_kt": _env(z, "env_dvmax_past24_kt")}
        for c, r in res.items():
            for i, (a, b) in enumerate(Q_BINS):
                for q in range(4):
                    m = (rr >= a) & (rr < b) & (quad == q) & np.isfinite(r)
                    if m.sum() >= 20:
                        v = float(np.mean(r[m]))
                        qacc[c][(i, q)].append(v)
                        if c == "h37" and i == 0:
                            row[f"h37_res_r0-1_{QUADS[q]}"] = v
        st = next(x for x in SHEAR_STRATA if x[0] <= smag < x[1])
        idx = np.digitize(rr, R_EDGES) - 1
        for i in range(len(R_EDGES) - 1):
            m = (idx == i) & np.isfinite(res["h37"])
            if m.sum() >= 20:
                strata[st][i].append(float(np.mean(res["h37"][m])))
        ir = z["ir_band13"].astype(np.float64)
        fl = z["flash_density"].astype(np.float64) if "flash_density" in z.files else None
        dv = row["dv_past24_kt"]
        grp = "RI (past 24 h >= +30 kt)" if np.isfinite(dv) and dv >= 30 else "other"
        for q in range(4):
            m = (rr < 1.0) & (quad == q) & np.isfinite(ir)
            if m.sum() >= 20:
                cold = float(np.mean(ir[m] < COLD_TOP_K))
                burst[grp][q]["cold"].append(cold)
                row[f"cold_frac_rmw_{QUADS[q]}"] = cold
                if fl is not None:
                    fv = float(np.mean(fl[m]))
                    burst[grp][q]["flash"].append(fv)
                    row[f"flash_rmw_{QUADS[q]}"] = fv
        rows.append(row)

    print()
    if n_env == 0:
        print("No NPZ carries the ERA5 environment yet -- run backfill_npz.py "
              "(or mine with 0.160+) for the shear-relative sections.")
        return
    print(f"=== SHEAR-RELATIVE ({n_env} examples with environment) ===")
    print("mean (real - backbone) K; quadrants clockwise from downshear, NH")
    for c in ("h37", "v37", "v89"):
        print(f"  {c}   r/RMW    " + "".join(f"{q:>8s}" for q in QUADS))
        for i, (a, b) in enumerate(Q_BINS):
            cells = [qacc[c][(i, q)] for q in range(4)]
            print(f"        {a}-{b:<5}  " + "".join(
                f"{np.mean(v):>+8.1f}" if v else "     ---" for v in cells)
                + f"   (n {min(len(v) for v in cells)}-{max(len(v) for v in cells)})")
    print()
    print("h37 radial profile by deep-layer shear:")
    hdr = "".join(f"{R_EDGES[i]:>4.1f}-{R_EDGES[i+1]:<4.1f}" for i in range(len(R_EDGES) - 1))
    print(f"  shear m/s   {hdr}")
    for st, bins in strata.items():
        n = max((len(b) for b in bins), default=0)
        if n:
            print(f"  {st[0]:>3}-{st[1] if st[1] < 99 else '+':<5}   " + "".join(
                f"{np.mean(b):>+9.1f}" if b else "      ---" for b in bins) + f"   (n<={n})")
    print()
    print(f"Convective bursts inside 1 RMW: cold-top (<{COLD_TOP_K:.0f} K) fraction / "
          f"mean flash density, by shear quadrant")
    for grp, qs in burst.items():
        n = max(len(qs[q]["cold"]) for q in range(4))
        if not n:
            continue
        cells = []
        for q in range(4):
            c, fl = qs[q]["cold"], qs[q]["flash"]
            cells.append(f"{QUADS[q]} {np.mean(c):.2f}/{np.mean(fl) if fl else float('nan'):.3f}"
                         if c else f"{QUADS[q]} ---")
        print(f"  {grp:26s} n<={n}:  " + "   ".join(cells))
    if csv_path:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-example table: {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    sys.exit(main(a.data_dir, a.csv))
