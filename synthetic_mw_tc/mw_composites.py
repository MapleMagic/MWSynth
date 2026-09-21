"""
Polarization-corrected temperature (PCT) and 37/89 GHz color composites.

This is a direct, faithful port of the ACTUAL NRL GeoIPS production
source code (not a reverse-engineered approximation) -- NRLMMD-GEOIPS is
the real, open-source repository NRL Monterey uses to generate these
products operationally:
  https://github.com/NRLMMD-GEOIPS/geoips/tree/main/geoips/plugins/classes/algorithms/pmw_tb
  https://github.com/NRLMMD-GEOIPS/geoips/tree/main/geoips/plugins/classes/colormappers/pmw_tb
Specifically ported from:
  pmw_37pct.py, pmw_89pct.py     (standalone PCT algorithms)
  pmw_color37.py, pmw_color89.py (RGB composite algorithms)
This supersedes two earlier versions of this module: one that guessed at
a single shared technique for both frequencies, and one that extracted
approximate colors by sampling pixels from reference images. Both were
reasonable given what was available at the time, but this is the actual
source, not an approximation of it.

IMPORTANT: NRL uses genuinely DIFFERENT PCT coefficients for the
standalone PCT product vs. the same quantity computed inside the color
composite's red channel. This isn't a mistake to reconcile -- both are
copied exactly as found:
  - Standalone PCT37 = 2.15*V37 - 1.15*H37   (pmw_37pct.py)
  - color37's red channel  = 2.181*V37 - 1.181*H37   (pmw_color37.py)
  - Standalone PCT89 = 1.7*V89 - 0.7*H89     (pmw_89pct.py)
  - color89's red channel  = 1.818*V89 - 0.818*H89   (pmw_color89.py)

COLOR37 (pmw_color37.py, verbatim logic):
    R = normalize(2.181*V37 - 1.181*H37, range=[260,280], inverted)
    G = normalize(V37, range=[180,300])
    B = normalize(H37, range=[160,300])

COLOR89 (pmw_color89.py, verbatim logic):
    R = normalize(1.818*V89 - 0.818*H89, range=[220,310], inverted)
    G = normalize(H89, range=[240,300])      <- H, not V, at 89 GHz
    B = normalize(V89, range=[270,290])      <- V, not H, at 89 GHz
This confirms (with exact source now, not inference) what the Brennan &
Cangialosi NHC slide deck said in words: V and H swap between which
channel they drive at 37 vs. 85/89 GHz.

"inverted" means: at the LOW end of the range (more ice scattering) the
normalized value is 1 (full channel intensity); at the HIGH end (clear/
warm) it's 0. This is what makes "colder PCT = more red" work, since raw
PCT decreases with more scattering. NRL's apply_gamma(x, 1.0) calls are a
no-op (gamma=1.0 is the identity), so they're omitted here -- nothing is
lost by not implementing a generic gamma function for an exponent that's
always called with 1.0 in this source.
"""
from __future__ import annotations

import numpy as np

# PCT coefficients, standalone product (NRL pmw_37pct.py / pmw_89pct.py).
# PCT = (1+Theta)*V - Theta*H
PCT_THETA = {
    37: 1.15,   # PCT37 = 2.15*V37 - 1.15*H37
    89: 0.70,   # PCT89 = 1.70*V89 - 0.70*H89
}

# PCT-like coefficients used specifically inside the color composite's red
# channel (NRL pmw_color37.py / pmw_color89.py) -- deliberately different
# from PCT_THETA above, see module docstring.
COLOR_RED_THETA = {
    37: 1.181,  # 2.181*V37 - 1.181*H37
    89: 0.818,  # 1.818*V89 - 0.818*H89
}

# (red_range, green_source, green_range, blue_source, blue_range) per
# frequency, exactly as coded in pmw_color37.py / pmw_color89.py.
COLOR_COMPOSITE_SPEC = {
    37: {"red_range": (260.0, 280.0), "green": "V", "green_range": (180.0, 300.0),
         "blue": "H", "blue_range": (160.0, 300.0)},
    89: {"red_range": (220.0, 310.0), "green": "H", "green_range": (240.0, 300.0),
         "blue": "V", "blue_range": (270.0, 290.0)},
}


def compute_pct(tb_v: np.ndarray, tb_h: np.ndarray, freq: int) -> np.ndarray:
    """Standalone PCT product (NRL pmw_37pct.py / pmw_89pct.py coefficients).
    freq must be 37 or 89."""
    theta = PCT_THETA[freq]
    return (1 + theta) * tb_v - theta * tb_h


def _normalize(values: np.ndarray, vmin: float, vmax: float, inverse: bool = False) -> np.ndarray:
    """Port of NRL's apply_data_range(..., min_outbounds='crop',
    max_outbounds='crop', norm=True): crop to [vmin,vmax], then normalize
    to 0-1, optionally inverted."""
    cropped = np.clip(values, vmin, vmax)
    x = (cropped - vmin) / (vmax - vmin)
    if inverse:
        x = 1.0 - x
    return x


def build_37_color_composite(v37: np.ndarray, h37: np.ndarray, pct37: np.ndarray = None) -> np.ndarray:
    """Direct port of pmw_color37.py. pct37 argument is accepted for
    backward-compatible call signatures but ignored -- the composite's
    red channel uses its own PCT-like coefficient (COLOR_RED_THETA),
    computed internally from v37/h37 directly, exactly as NRL's source
    does (it doesn't take a precomputed PCT as input either)."""
    theta = COLOR_RED_THETA[37]
    red_raw = (1 + theta) * v37 - theta * h37
    r = _normalize(red_raw, *COLOR_COMPOSITE_SPEC[37]["red_range"], inverse=True)
    g = _normalize(v37, *COLOR_COMPOSITE_SPEC[37]["green_range"], inverse=False)
    b = _normalize(h37, *COLOR_COMPOSITE_SPEC[37]["blue_range"], inverse=False)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def build_89_color_composite(v89: np.ndarray, h89: np.ndarray, pct89: np.ndarray = None) -> np.ndarray:
    """Direct port of pmw_color89.py. pct89 argument accepted for
    backward-compatible call signatures but ignored -- same reasoning as
    build_37_color_composite above."""
    theta = COLOR_RED_THETA[89]
    red_raw = (1 + theta) * v89 - theta * h89
    r = _normalize(red_raw, *COLOR_COMPOSITE_SPEC[89]["red_range"], inverse=True)
    g = _normalize(h89, *COLOR_COMPOSITE_SPEC[89]["green_range"], inverse=False)
    b = _normalize(v89, *COLOR_COMPOSITE_SPEC[89]["blue_range"], inverse=False)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def composite_from_swath(swath, freq: int) -> np.ndarray:
    """Convenience wrapper: build the color composite directly from an
    MWSwath (data_types.MWSwath) for the given frequency (37 or 89)."""
    if freq == 37:
        return build_37_color_composite(swath.v37, swath.h37)
    elif freq == 89:
        return build_89_color_composite(swath.v89, swath.h89)
    else:
        raise ValueError("freq must be 37 or 89")


def composite_from_synthetic(result, freq: int) -> np.ndarray:
    """Same as composite_from_swath, but for a SyntheticMWResult
    (synthetic_algorithm.py) instead of a real MWSwath -- uses the exact
    same build_37_color_composite/build_89_color_composite functions, so
    synthetic and real composites are colored identically."""
    if freq == 37:
        if result.v37 is None or result.h37 is None:
            raise ValueError(
                "SyntheticMWResult has no v37/h37 -- was it generated with an "
                "older version of generate_synthetic_mw that only produced a "
                "scalar freq_37ghz?"
            )
        return build_37_color_composite(result.v37, result.h37)
    elif freq == 89:
        if result.v89 is None or result.h89 is None:
            raise ValueError(
                "SyntheticMWResult has no v89/h89 -- was it generated with an "
                "older version of generate_synthetic_mw that only produced a "
                "scalar freq_89ghz?"
            )
        return build_89_color_composite(result.v89, result.h89)
    else:
        raise ValueError("freq must be 37 or 89")
