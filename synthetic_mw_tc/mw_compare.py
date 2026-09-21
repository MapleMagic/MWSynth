"""
First step toward bridging the synthetic MW algorithm and real ingested
MW data (mw_ingest.py / mw_composites.py): regrids a real MWSwath's V-pol
and PCT fields onto a SyntheticMWResult's grid and computes basic
difference statistics, so the two can start being used together instead
of living in separate tabs with no connection between them.

This is intentionally a starting point, not a finished calibration
pipeline. synthetic_algorithm.py currently outputs ONE scalar Tb per
frequency (not separate V/H), so there's no single obviously-correct real
channel to compare it against:
  - V-pol Tb is the more commonly cited "representative" channel when
    people informally talk about "the 37 GHz Tb" or "the 89 GHz Tb."
  - PCT is the ice-scattering-sensitive quantity the synthetic algorithm's
    scattering-index logic is most directly trying to approximate.
Rather than silently pick one, both are computed -- use whichever framing
makes more sense for what you're trying to check, or look at both.

apply_bias_calibration() below closes the loop automatically: shifts the
synthetic V/H/scalar fields by the mean bias against real V-pol data
(a uniform additive correction per frequency, the simplest defensible
calibration -- not a fit, not spatially-varying, just "make the average
level match"). This is what the Generate tab now runs automatically when
real MW data is available. A spatially-varying or regression-based
calibration is a reasonable future step; this is deliberately the
simplest thing that could plausibly help, not a finished approach.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.interpolate import griddata

import mw_composites


def _regrid_nearest(src_lat, src_lon, src_values, dst_lat, dst_lon):
    """Nearest-neighbor regrid of a (possibly irregular/conically-scanned)
    real-swath field onto the synthetic algorithm's rectangular grid."""
    points = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    values = src_values.ravel()
    return griddata(points, values, (dst_lat, dst_lon), method="nearest")


def _stats(synthetic: np.ndarray, real_on_synth_grid: np.ndarray) -> dict:
    mask = np.isfinite(synthetic) & np.isfinite(real_on_synth_grid)
    if not mask.any():
        return {"bias_k": None, "rmse_k": None, "n_pixels": 0}
    diff = synthetic[mask] - real_on_synth_grid[mask]  # positive = synthetic warmer than real
    return {
        "bias_k": float(np.mean(diff)),
        "rmse_k": float(np.sqrt(np.mean(diff**2))),
        "n_pixels": int(mask.sum()),
    }


def compare_frequency(synthetic_result, real_swath, freq: int) -> dict:
    """Compare synthetic_result's freq-GHz field (37 or 89) against
    real_swath's same-frequency V-pol and PCT fields, regridded onto the
    synthetic result's grid.

    Returns a dict:
        {
          "real_v_on_synth_grid": ndarray (same shape as synthetic field),
          "real_pct_on_synth_grid": ndarray,
          "vs_v": {"bias_k":..., "rmse_k":..., "n_pixels":...},
          "vs_pct": {...},
        }
    bias_k > 0 means the synthetic field runs warmer than the real one on
    average over the overlapping area; negative means colder.
    """
    if freq == 37:
        synthetic_field = synthetic_result.freq_37ghz
        real_v, real_h = real_swath.v37, real_swath.h37
        real_lat, real_lon = real_swath.grid_for(37)
    elif freq == 89:
        synthetic_field = synthetic_result.freq_89ghz
        real_v, real_h = real_swath.v89, real_swath.h89
        real_lat, real_lon = real_swath.grid_for(89)
    else:
        raise ValueError("freq must be 37 or 89")

    real_pct = mw_composites.compute_pct(real_v, real_h, freq)

    v_on_synth = _regrid_nearest(real_lat, real_lon, real_v, synthetic_result.lat, synthetic_result.lon)
    pct_on_synth = _regrid_nearest(real_lat, real_lon, real_pct, synthetic_result.lat, synthetic_result.lon)

    return {
        "real_v_on_synth_grid": v_on_synth,
        "real_pct_on_synth_grid": pct_on_synth,
        "vs_v": _stats(synthetic_field, v_on_synth),
        "vs_pct": _stats(synthetic_field, pct_on_synth),
    }


def compare_both_frequencies(synthetic_result, real_swath) -> dict:
    """Convenience wrapper: compare_frequency for both 37 and 89 GHz."""
    return {
        37: compare_frequency(synthetic_result, real_swath, 37),
        89: compare_frequency(synthetic_result, real_swath, 89),
    }


def format_stats_summary(comparison: dict) -> str:
    """Human-readable summary of compare_both_frequencies' output, for
    display in the GUI or a log."""
    lines = []
    for freq in (37, 89):
        c = comparison[freq]
        lines.append(f"{freq} GHz:")
        for label, key in [("vs real V-pol", "vs_v"), ("vs real PCT", "vs_pct")]:
            s = c[key]
            if s["n_pixels"] == 0:
                lines.append(f"  {label}: no overlapping finite pixels")
            else:
                lines.append(
                    f"  {label}: bias={s['bias_k']:+.1f}K, RMSE={s['rmse_k']:.1f}K, n={s['n_pixels']}"
                )
    return "\n".join(lines)


def apply_bias_calibration(
    synthetic_result, real_swath, max_shift_k: float = 40.0, max_vh_shift_k: float = 5.0
):
    """Shift synthetic_result's scalar (freq_37ghz/freq_89ghz) fields by
    the mean bias against real_swath's V-pol data, so the calibrated
    scalar's average level matches the real pass -- this is the simplest
    defensible auto-calibration, not a spatial fit, just "make the mean
    match." Used automatically by the Generate tab when real MW data is
    available.

    max_shift_k: hard cap on the SCALAR shift magnitude per frequency.
    max_vh_shift_k: a SEPARATE, much tighter cap on the V/H shift.

    WHY TWO DIFFERENT CAPS: a real run showed that shifting V/H by the
    same amount as the scalar can push the color composite into a
    saturating, texture-losing mess. The concrete cause: NRL's 89 GHz
    blue channel range is only 20K wide ([270,290]) -- a real run showed
    that even a small +5.1K persisted-offset nudge alone (nowhere near
    the scalar's own +-40K cap) was enough to push V89 to 65%+ of that
    narrow window and render as visibly cyan. No baseline placement
    inside a 20K-wide window survives a +-40K shift; the ranges are
    simply too narrow relative to what a full quantitative correction
    needs to move by. Rather than accept either "colors are visually
    broken" or "the scalar comparison is quantitatively wrong," these are
    decoupled: the scalar (used by mw_compare.py's stats, meant to be
    accurate) gets the full clamped correction; V/H (used only for
    coloring) gets a much smaller, separately-clamped nudge so the
    composite stays in a usable part of NRL's designed color range. This
    means the SCALAR and the V/H-derived color won't always imply exactly
    the same brightness temperature after calibration -- that's an
    accepted tradeoff, not an oversight.

    Returns (calibrated_result, calibration_info) where calibration_info
    is {37: {"bias_k":..., "rmse_k":..., "n_pixels":...}, 89: {...}} (the
    "vs_v" stats from compare_both_frequencies -- bias_k here is the
    UNCLAMPED measured bias; applied_bias_k/applied_vh_bias_k report what
    was actually applied to the scalar vs. V/H respectively).
    If a frequency has no overlapping finite pixels (bias_k is None),
    that frequency's fields are left uncalibrated (unshifted).
    """
    comparison = compare_both_frequencies(synthetic_result, real_swath)
    calibration_info = {freq: comparison[freq]["vs_v"] for freq in (37, 89)}

    updates = {"diagnostics": dict(synthetic_result.diagnostics)}  # copy, avoid aliasing the original

    bias37 = calibration_info[37]["bias_k"]
    if bias37 is not None:
        applied37 = max(-max_shift_k, min(max_shift_k, bias37))
        applied37_vh = max(-max_vh_shift_k, min(max_vh_shift_k, bias37))
        calibration_info[37]["applied_bias_k"] = applied37
        calibration_info[37]["applied_vh_bias_k"] = applied37_vh
        updates["freq_37ghz"] = synthetic_result.freq_37ghz - applied37
        if synthetic_result.v37 is not None:
            updates["v37"] = synthetic_result.v37 - applied37_vh
            updates["h37"] = synthetic_result.h37 - applied37_vh

    bias89 = calibration_info[89]["bias_k"]
    if bias89 is not None:
        applied89 = max(-max_shift_k, min(max_shift_k, bias89))
        applied89_vh = max(-max_vh_shift_k, min(max_vh_shift_k, bias89))
        calibration_info[89]["applied_bias_k"] = applied89
        calibration_info[89]["applied_vh_bias_k"] = applied89_vh
        updates["freq_89ghz"] = synthetic_result.freq_89ghz - applied89
        if synthetic_result.v89 is not None:
            updates["v89"] = synthetic_result.v89 - applied89_vh
            updates["h89"] = synthetic_result.h89 - applied89_vh

    calibrated = dataclasses.replace(synthetic_result, **updates)
    return calibrated, calibration_info
