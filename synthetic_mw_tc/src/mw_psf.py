"""
Sensor point-spread function (antenna pattern) for the synthetic fields.

WHY: the backbone renders V/H at the GOES grid spacing (~2 km) with no
antenna pattern at all, so its 37 GHz field carries detail no 37 GHz
radiometer could resolve. Real instantaneous fields of view are large and
frequency-dependent -- GMI is 4.4 x 7.2 km at 89 GHz but 8.6 x 14.0 km at
37 -- which is a factor of four in area between the two channels of the
same instrument.

This also fixes a structural wrong called out in the README's known
limitations: before this, the 37 and 89 GHz SPATIAL responses were
identical by construction, with all frequency dependence living in the
calibration constants. No real sensor pair behaves that way, and it is a
large part of why synthetic 37 GHz has looked implausibly sharp next to
real imagery.

IFOV figures are the -3 dB footprint dimensions published for each
instrument (Li et al. 2026, Table 1; matching the TC PRIMED
documentation):

    GMI     89.0 GHz  4.4 x 7.2 km      36.5 GHz  8.6 x 14.0 km
    AMSR2   89.0 GHz  3.0 x 5.0 km      36.5 GHz  7.0 x 12.0 km
    SSMIS   91.7 GHz  13.1 x 14.4 km    37.0 GHz  27.5 x 44.2 km

APPLIED TO THE BACKBONE ONLY, never to fused output. Real MW arrives
already carrying its own sensor's PSF; blurring the fused field would
convolve it a second time and smear genuine observations. The backbone is
the only part that is artificially sharp, so it is the only part that
needs this.

DELIBERATE SIMPLIFICATION: the real footprint ellipse is oriented
along-scan/along-track, which rotates across a swath and depends on scan
geometry this project does not model. The ellipse is applied axis-aligned
in grid space instead. That is right on the scale of the blur (the
footprint is elongated by roughly 1.6-2x, and getting the magnitude right
matters far more than the angle) but it is an approximation, and a real
scan-geometry treatment would be an improvement rather than a rewrite.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# (across, along) -3 dB footprint in km, per sensor per frequency.
SENSOR_IFOV_KM = {
    "GMI":     {37: (8.6, 14.0), 89: (4.4, 7.2)},
    "AMSR2":   {37: (7.0, 12.0), 89: (3.0, 5.0)},
    "AMSR3":   {37: (7.0, 12.0), 89: (3.0, 5.0)},   # AMSR2-like pending specs
    "SSMIS":   {37: (27.5, 44.2), 89: (13.1, 14.4)},
    "WSFM":    {37: (27.5, 44.2), 89: (13.1, 14.4)},  # SSMIS-like
}

# Sensor assumed when none is known. GMI is the reasonable default: it is
# the reference instrument for most TC MW imagery, and sits between the
# sharper AMSR2 and the much coarser SSMIS, so a wrong guess errs
# modestly in both directions rather than badly in one.
DEFAULT_SENSOR = "GMI"

# A -3 dB footprint dimension is a full width at half maximum, so the
# equivalent Gaussian sigma is FWHM / (2 * sqrt(2 * ln 2)).
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def _grid_spacing_km(lat: np.ndarray, lon: np.ndarray) -> tuple:
    """(km per row, km per column) for the working grid."""
    if lat.shape[0] < 2 or lat.shape[1] < 2:
        return 2.0, 2.0
    dlat_km = abs(float(np.nanmedian(np.diff(lat, axis=0)))) * 111.32
    mean_lat = float(np.nanmean(lat))
    dlon_km = abs(float(np.nanmedian(np.diff(lon, axis=1)))) * 111.32 * np.cos(
        np.radians(mean_lat))
    return max(dlat_km, 1e-6), max(dlon_km, 1e-6)


def psf_sigma_px(lat, lon, freq: int, sensor: str = DEFAULT_SENSOR) -> tuple:
    """Gaussian sigma in PIXELS, (row, col), for this sensor/frequency on
    this grid. Returns (0, 0) when the footprint is already finer than the
    grid, in which case blurring would only destroy information."""
    ifov = SENSOR_IFOV_KM.get(sensor, SENSOR_IFOV_KM[DEFAULT_SENSOR]).get(freq)
    if not ifov:
        return 0.0, 0.0
    across_km, along_km = ifov
    km_row, km_col = _grid_spacing_km(lat, lon)
    # Map the smaller footprint dimension to rows and the larger to
    # columns. Axis-aligned, per the module note.
    sigma_row = (across_km * FWHM_TO_SIGMA) / km_row
    sigma_col = (along_km * FWHM_TO_SIGMA) / km_col
    # Below ~0.5 px the kernel is a no-op numerically and only costs time.
    return (sigma_row if sigma_row > 0.5 else 0.0,
            sigma_col if sigma_col > 0.5 else 0.0)


def apply_sensor_psf(field, lat, lon, freq: int, sensor: str = DEFAULT_SENSOR):
    """Convolve a synthetic field with the sensor footprint for `freq`.

    NaN-aware: blurring a field with NaN holes would otherwise spread the
    NaN across the whole kernel footprint. Uses the standard normalized-
    convolution trick -- filter the zero-filled data and the validity
    mask, then divide -- so valid data near a gap is weighted by how much
    valid data actually contributed.
    """
    sigma = psf_sigma_px(lat, lon, freq, sensor)
    if sigma == (0.0, 0.0):
        return field

    arr = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return field
    if valid.all():
        return gaussian_filter(arr, sigma=sigma, mode="nearest")

    filled = np.where(valid, arr, 0.0)
    num = gaussian_filter(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    # Do not invent data where there was none.
    out[~valid] = np.nan
    return out


def describe(lat, lon, sensor: str = DEFAULT_SENSOR) -> str:
    """One-line summary for the log, so the applied blur is visible
    rather than an invisible change to every field."""
    parts = []
    for freq in (37, 89):
        sr, sc = psf_sigma_px(lat, lon, freq, sensor)
        ifov = SENSOR_IFOV_KM.get(sensor, {}).get(freq)
        if ifov:
            parts.append(f"{freq}GHz {ifov[0]:.1f}x{ifov[1]:.1f}km "
                         f"(sigma {sr:.1f}x{sc:.1f}px)")
    return f"Sensor PSF [{sensor}]: " + ", ".join(parts)
