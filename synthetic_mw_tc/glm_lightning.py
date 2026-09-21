"""
GOES GLM (Geostationary Lightning Mapper) flash density as a model input.

WHY THIS IS THE MOST VALUABLE NEW OBSERVABLE AVAILABLE HERE
-----------------------------------------------------------
Both Li et al. papers, and this project's own measurements, run into the
same wall: cloud-top IR under-determines the sub-cloud hydrometeor
column. A single cloud-top temperature corresponds to a RANGE of internal
ice-scattering intensities, and no amount of architecture removes that.
The measured consequence is a bias of 11.09 K out of a 15.85 K residual
and a skill ceiling near +0.30.

Lightning is the obvious way past it. Flashes are produced by charge
separation in the mixed-phase region, which requires graupel and
supercooled water colliding in strong updrafts -- which is very close to
being a direct observation of the thing 89 GHz scattering measures.
Crucially it is information IR does NOT carry: a thick cirrus canopy and
an active convective core can share a cloud-top temperature and differ
completely in flash rate.

Neither paper uses it. GLM flies on the same GOES platform this project
already fetches from, in the same public S3 buckets, so the marginal cost
is one more product per frame.

WHAT IT CANNOT DO. Lightning is sparse, intermittent and biased toward
the most vigorous convection. Many raining regions never produce a flash,
and TC eyewalls in particular are often electrically quiet even when
convectively intense -- lightning in mature eyewalls tends to come in
bursts around intensity change rather than steadily. So this is a strong
positive indicator and a weak negative one: flashes mean deep convection,
but no flashes mean very little. It is added as a model INPUT, where the
network can learn that asymmetry, and deliberately NOT wired into the
parametric convective signal, where it would act as a hard vote.

ACCUMULATION WINDOW. Flash rate is meaningful over minutes, not
instants. GLM files cover 20 seconds each, so a window of several minutes
either side of the frame is accumulated -- long enough for a rate to
mean something, short enough not to smear a moving storm.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np

# Minutes either side of the frame time to accumulate flashes over.
# GLM granules are 20 s, so +/-5 min is 30 files -- enough for a stable
# rate without smearing a storm moving at 10-15 kt (which travels only a
# few km in that time, well inside one grid cell).
ACCUM_MINUTES = 5.0

# Gaussian smoothing applied to the flash-density grid, in km. Individual
# flashes are point events with location errors of several km; without
# smoothing the field is a sparse scatter of spikes that convolves badly
# with everything downstream and gives the network nothing learnable.
SMOOTH_KM = 15.0

# Normalization for the model input channel. Flash density is heavily
# skewed -- most pixels are zero, a few are very active -- so the channel
# is log1p'd before scaling rather than fed raw, which would leave the
# network a spike train with almost no dynamic range in between.
FLASH_LOG_SCALE = 3.0


def _accumulate_flashes(satellite: str, target_time, progress_callback=None):
    """Fetch GLM flash points within ACCUM_MINUTES of target_time.

    Returns (lat, lon, energy) arrays, or None if GLM is unavailable.
    Never raises: GLM is an enhancement, and a frame without it must
    still generate.
    """
    try:
        import goes_fetch
    except ImportError:
        return None

    fetch = getattr(goes_fetch, "get_glm_flashes", None)
    if fetch is None:
        # goes_fetch has no GLM reader yet. Returning None rather than
        # raising keeps this importable and testable ahead of that, and
        # makes the missing piece explicit at the call site.
        if progress_callback:
            progress_callback("GLM: goes_fetch.get_glm_flashes() not implemented -- "
                              "lightning channel will be empty.")
        return None

    try:
        return fetch(satellite, target_time - timedelta(minutes=ACCUM_MINUTES),
                     target_time + timedelta(minutes=ACCUM_MINUTES))
    except Exception as e:
        if progress_callback:
            progress_callback(f"GLM unavailable ({type(e).__name__}) -- "
                              f"lightning channel will be empty.")
        return None


def flash_density_grid(lat, lon, satellite: str, target_time,
                       progress_callback=None) -> np.ndarray:
    """Flash density on the working grid, in flashes per grid cell per
    accumulation window. All zeros when GLM is unavailable, which is the
    same value a genuinely lightning-free scene produces -- deliberately,
    so the model's input distribution does not shift depending on whether
    the product happened to be reachable.
    """
    from scipy.ndimage import gaussian_filter

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    out = np.zeros(lat.shape, dtype=np.float32)

    flashes = _accumulate_flashes(satellite, target_time, progress_callback)
    if flashes is None:
        return out
    flat, flon = np.asarray(flashes[0]), np.asarray(flashes[1])
    if flat.size == 0:
        return out

    # Bin onto the grid by nearest cell. The grid is a smoothly varying
    # curvilinear crop, so digitize against its own row/column coordinate
    # ranges rather than assuming a regular projection.
    lat0, lat1 = np.nanmin(lat), np.nanmax(lat)
    lon0, lon1 = np.nanmin(lon), np.nanmax(lon)
    rows, cols = lat.shape
    inside = ((flat >= lat0) & (flat <= lat1) & (flon >= lon0) & (flon <= lon1))
    if not inside.any():
        return out
    # Rows follow decreasing latitude in this project's grids; derive the
    # direction from the data rather than assuming it.
    lat_desc = lat[0, 0] > lat[-1, 0]
    fr = (lat1 - flat[inside]) / max(lat1 - lat0, 1e-9)
    if not lat_desc:
        fr = 1.0 - fr
    fc = (flon[inside] - lon0) / max(lon1 - lon0, 1e-9)
    ri = np.clip((fr * (rows - 1)).astype(int), 0, rows - 1)
    ci = np.clip((fc * (cols - 1)).astype(int), 0, cols - 1)
    np.add.at(out, (ri, ci), 1.0)

    # Smooth to a density. Kernel in pixels from the grid spacing.
    try:
        import mw_psf
        km_row, km_col = mw_psf._grid_spacing_km(lat, lon)
    except Exception:
        km_row = km_col = 2.0
    sigma = (SMOOTH_KM / max(km_row, 1e-6), SMOOTH_KM / max(km_col, 1e-6))
    out = gaussian_filter(out, sigma=sigma, mode="constant")

    if progress_callback:
        n = int(inside.sum())
        progress_callback(f"GLM: {n} flash(es) in +/-{ACCUM_MINUTES:.0f} min, "
                          f"peak density {out.max():.3f}/cell")
    return out.astype(np.float32)


def normalize_for_model(flash_density) -> np.ndarray:
    """Scale flash density into the range the model's other channels use.

    log1p first: the raw distribution is mostly zeros with a long tail, so
    a linear scaling would put essentially all pixels at one end and give
    the network no gradient to work with across the range that matters.
    """
    a = np.asarray(flash_density, dtype=np.float32)
    return np.log1p(np.maximum(a, 0.0)) / FLASH_LOG_SCALE
