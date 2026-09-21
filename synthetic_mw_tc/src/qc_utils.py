"""
Small shared quality-control helpers. Split out from goes_fetch.py /
synthetic_algorithm.py so both can sanitize data the same way without
duplicating logic.

The core problem this addresses: GOES ABI L1b files always contain some
fraction of QC-flagged, off-limb, or fill-value pixels. Converting those
raw radiances straight to brightness temperature via the Planck inversion
produces NaN/inf (log of a non-positive number, or division by ~0), and
those non-finite values then silently propagate through any nearest-
neighbor regridding downstream -- which is what caused
"data must be finite, check for nan or inf values" from matplotlib.

Fix: mask bad pixels using the file's own DQF (data quality flag) array
plus a finite-value check, then fill masked pixels with the nearest valid
pixel's value (standard "nearest-neighbor inpainting"). This keeps the
field usable for aggregate/algorithmic purposes without inventing physics,
since a QC-flagged pixel in a real product would typically just be masked
and interpolated by other consumers of the data too.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata


def fill_invalid_nearest(arr: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """Replace arr[invalid_mask] with the value of the nearest valid pixel.
    If everything is invalid (degenerate/empty scene), returns arr unchanged
    (caller should treat that as a hard failure, not silently plot garbage).

    NOTE: for wide contiguous invalid regions (e.g. a several-pixel-wide
    dropped-detector stripe, a real GOES ABI characteristic), this
    produces a visible duplicated-column artifact -- confirmed directly:
    a 6-pixel-wide gap fills as two flat halves (nearest-from-left,
    nearest-from-right), not a smooth blend, because every pixel just
    copies its single closest valid neighbor regardless of direction.
    sanitize_field() below uses fill_invalid_smooth() instead for exactly
    this reason; this function is kept for callers that specifically want
    the cheaper nearest behavior (e.g. small/sparse invalid regions where
    the difference is invisible).
    """
    if not invalid_mask.any():
        return arr
    if invalid_mask.all():
        return arr
    # distance_transform_edt with return_indices gives, for every pixel, the
    # index of the nearest pixel where invalid_mask is False.
    idx = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def fill_invalid_smooth(arr: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """Fill arr[invalid_mask] via linear interpolation from surrounding
    valid pixels in ALL directions (not just the single nearest one) --
    smoothly blends across wide gaps (e.g. a dropped-detector stripe)
    instead of nearest-fill's duplicated-column artifact. Falls back to
    nearest-fill only for pixels outside the valid points' convex hull
    (typically just the outer domain edge, where "smoothly blend from
    surrounding data" isn't well-defined anyway) -- unlike the radar
    regrid case elsewhere in this project, this fallback is safe here:
    GOES data is a dense regular grid with genuinely well-defined
    interior neighbors on all sides, not sparse angular rays, so there's
    no equivalent risk of reintroducing a directional artifact.
    """
    if not invalid_mask.any():
        return arr
    if invalid_mask.all():
        return arr

    ys, xs = np.indices(arr.shape)
    valid_mask = ~invalid_mask
    points = np.column_stack([ys[valid_mask], xs[valid_mask]])
    values = arr[valid_mask]
    query_points = np.column_stack([ys[invalid_mask], xs[invalid_mask]])

    filled_values = griddata(points, values, query_points, method="linear")

    still_nan = ~np.isfinite(filled_values)
    if still_nan.any():
        nearest_values = griddata(points, values, query_points[still_nan], method="nearest")
        filled_values[still_nan] = nearest_values

    result = arr.copy()
    result[invalid_mask] = filled_values
    return result


def sanitize_field(values: np.ndarray, dqf: np.ndarray | None = None, extra_invalid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Mask (DQF != 0) OR non-finite OR extra_invalid, then smoothly fill.

    Returns (clean_array, bad_fraction). Caller should treat a very high
    bad_fraction (e.g. > 0.5) as a sign the scene/sector is mostly missing
    data rather than trusting a heavily-inpainted field.
    """
    values = np.asarray(values, dtype=np.float64)
    invalid = ~np.isfinite(values)
    if dqf is not None:
        invalid = invalid | (np.asarray(dqf) != 0)
    if extra_invalid is not None:
        invalid = invalid | extra_invalid

    bad_fraction = float(invalid.sum()) / invalid.size if invalid.size else 0.0

    if invalid.all():
        raise ValueError("Entire field is invalid/missing (all QC-flagged or non-finite). "
                          "Try a different scene time or mesoscale sector.")

    clean = fill_invalid_smooth(values, invalid)
    return clean, bad_fraction


def assert_finite(name: str, arr: np.ndarray) -> None:
    """Last-resort guard: raise a clear error instead of letting a NaN/inf
    silently reach matplotlib or downstream math."""
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"'{name}' still has {n_bad} non-finite value(s) after QC. "
            "This shouldn't happen if sanitize_field() was applied upstream -- "
            "check that every input array went through it."
        )
