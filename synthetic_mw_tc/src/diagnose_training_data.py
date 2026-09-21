"""
Inspect real exported .npz training examples to find out why training
produced a loss of exactly 0.0000.

A zero loss with a zero-initialized output layer means the model's
prediction (0) already equals the target everywhere the loss actually
counts. That leaves exactly two possibilities, and this script
distinguishes them:

  A) The supervision MASK is empty -- target_* is NaN everywhere inside
     the storm-centered patch, so masked_loss() divides a sum of zero by
     a clamped denominator and returns 0. The model gets no gradient at
     all and cannot learn anything.

  B) The RESIDUAL is genuinely zero -- target_* equals backbone_*, so
     there is nothing to correct. This would mean the exported backbone
     and target are the same array, which would be an export bug.

Run: python diagnose_training_data.py
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ml_constants import PATCH_SIZE, TB_STD

DEFAULT_DATA_DIR = os.path.expanduser("~/.synthetic_mw_tc/training_data")


def _centered_patch(arr, center_rc, ps):
    h, w = arr.shape[:2]
    r, c = center_rc
    half = ps // 2
    r0, r1 = r - half, r - half + ps
    c0, c1 = c - half, c - half + ps
    pad_top, pad_left = max(0, -r0), max(0, -c0)
    pad_bot, pad_right = max(0, r1 - h), max(0, c1 - w)
    cropped = arr[max(0, r0):min(h, r1), max(0, c0):min(w, c1)]
    if pad_top or pad_bot or pad_left or pad_right:
        cropped = np.pad(cropped, [(pad_top, pad_bot), (pad_left, pad_right)],
                         mode="constant", constant_values=np.nan)
    return cropped


def diagnose(data_dir: str = DEFAULT_DATA_DIR, max_files: int = 10):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(files)} example(s). Inspecting up to {max_files}.\n")

    totals = {"empty_mask": 0, "zero_residual": 0, "healthy": 0}
    whole_grid_coverage = []

    for path in files[:max_files]:
        d = np.load(path, allow_pickle=True)
        name = os.path.basename(path)
        print(f"=== {name} ===")

        lat, lon = d["lat"], d["lon"]
        slat, slon = float(d["storm_lat"]), float(d["storm_lon"])
        print(f"  grid {lat.shape}  storm ({slat:.2f}, {slon:.2f})")
        print(f"  grid lat range [{np.nanmin(lat):.2f}, {np.nanmax(lat):.2f}] "
              f"lon range [{np.nanmin(lon):.2f}, {np.nanmax(lon):.2f}]")

        in_grid = (np.nanmin(lat) <= slat <= np.nanmax(lat)
                   and np.nanmin(lon) <= slon <= np.nanmax(lon))
        print(f"  storm inside grid bounds: {in_grid}")

        # Whole-array coverage first -- is there ANY real MW in this file?
        for ch in ("v37", "h37", "v89", "h89"):
            tgt = d[f"target_{ch}"]
            bb = d[f"backbone_{ch}"]
            if tgt.size == 0 or bb.size == 0:
                print(f"  {ch}: EMPTY ARRAY saved (target size={tgt.size}, backbone size={bb.size})")
                continue
            fin = int(np.isfinite(tgt).sum())
            pct = 100.0 * fin / tgt.size
            if ch == "v37":
                whole_grid_coverage.append(pct)
            print(f"  {ch}: target finite {fin}/{tgt.size} ({pct:.1f}% of whole grid)")

        # Now the storm-centered patch the trainer actually uses
        dist2 = (lat - slat) ** 2 + (lon - slon) ** 2
        center_rc = np.unravel_index(np.argmin(dist2), dist2.shape)
        print(f"  patch center pixel: {center_rc}  (patch {PATCH_SIZE}x{PATCH_SIZE})")

        tstack, bstack = [], []
        skip = False
        for ch in ("v37", "h37", "v89", "h89"):
            tgt, bb = d[f"target_{ch}"], d[f"backbone_{ch}"]
            if tgt.size == 0 or bb.size == 0:
                skip = True
                break
            tstack.append(_centered_patch(tgt, center_rc, PATCH_SIZE))
            bstack.append(_centered_patch(bb, center_rc, PATCH_SIZE))
        if skip:
            print("  -> cannot evaluate patch (empty arrays)\n")
            continue

        tstack, bstack = np.stack(tstack), np.stack(bstack)
        mask = np.isfinite(tstack)
        n_sup = int(mask.sum())
        print(f"  SUPERVISED PIXELS IN PATCH: {n_sup} / {mask.size} "
              f"({100.0*n_sup/mask.size:.2f}%)")

        if n_sup == 0:
            print("  -> EMPTY MASK. This file contributes NOTHING to the loss.")
            totals["empty_mask"] += 1
        else:
            resid = (tstack - bstack)[mask]
            print(f"  residual (K): mean={np.nanmean(resid):+.2f} "
                  f"std={np.nanstd(resid):.2f} "
                  f"min={np.nanmin(resid):+.2f} max={np.nanmax(resid):+.2f}")
            print(f"  normalized target the model sees: mean="
                  f"{np.nanmean(resid)/TB_STD:+.4f}")
            if np.allclose(resid, 0.0, atol=1e-6):
                print("  -> RESIDUAL IS ZERO. backbone == target; nothing to learn.")
                totals["zero_residual"] += 1
            else:
                print("  -> healthy: real supervision with a nonzero residual.")
                totals["healthy"] += 1
        print()

    print("=" * 60)
    print(f"empty mask (no supervision): {totals['empty_mask']}")
    print(f"zero residual (nothing to learn): {totals['zero_residual']}")
    print(f"healthy: {totals['healthy']}")
    if whole_grid_coverage:
        avg = sum(whole_grid_coverage) / len(whole_grid_coverage)
        print(f"\naverage MW coverage over the WHOLE GOES grid: {avg:.2f}%")
        if avg == 0.0:
            print("  -> the regridded MW is empty across the ENTIRE scene, not just")
            print("     at the storm. The swath never lands on the GOES grid at all,")
            print("     which points at swath geolocation (coordinate convention or")
            print("     lat/lon shape), NOT at patch placement.")
            print("     Run: python diagnose_tcprimed_swath.py")
        else:
            print("  -> MW does land on the grid, just not at the storm. That points")
            print("     at patch placement / storm position rather than geolocation.")

    if totals["empty_mask"] and not totals["healthy"]:
        print("\nDIAGNOSIS: the storm-centered patch contains no real MW coverage.")
    elif totals["zero_residual"] and not totals["healthy"]:
        print("\nDIAGNOSIS: backbone and target are identical -- an export bug.")


if __name__ == "__main__":
    diagnose()
