"""
Single-channel colorized products: 37H, 89H, 37pct, 89pct.

Direct port of the ACTUAL NRL GeoIPS production colormaps (not
approximated -- see mw_composites.py's docstring for the same point about
the composite algorithms):
  https://github.com/NRLMMD-GEOIPS/geoips/blob/main/geoips/plugins/classes/colormappers/pmw_tb/cmap_37H.py
  https://github.com/NRLMMD-GEOIPS/geoips/blob/main/geoips/plugins/classes/colormappers/pmw_tb/cmap_89H.py
  https://github.com/NRLMMD-GEOIPS/geoips/blob/main/geoips/plugins/classes/colormappers/pmw_tb/cmap_37pct.py
  https://github.com/NRLMMD-GEOIPS/geoips/blob/main/geoips/plugins/classes/colormappers/pmw_tb/cmap_89pct.py

This supersedes an earlier version of this module that extracted
approximate colors by sampling pixels from reference images -- that
extraction turned out to be qualitatively accurate (same overall color
progression, same "wrap to a second palette at the coldest extreme" for
89H) but is now replaced with the exact transition breakpoints and exact
hex/named colors from the real source, not a pixel-sampled approximation
of it.

37pct and 89pct are NEW here -- standalone colorized PCT products (using
the PCT_THETA coefficients in mw_composites.py) that weren't available in
the pixel-extraction-only version of this module, since there was no
image to sample colors from for that channel specifically.

Each colormap is implemented as a list of (value_range, color_range)
segments, exactly matching each source file's `transition_vals` /
`transition_colors` lists, with per-pixel linear interpolation within
each segment (equivalent to what matplotlib's LinearSegmentedColormap
would produce when applied to normalized data, but done directly against
the physical Kelvin values here rather than building a Colormap object).
"""
from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors

import mw_composites


def _colorize_segments(values: np.ndarray, transition_vals: list, transition_colors: list) -> np.ndarray:
    """Per-pixel linear interpolation across a list of (v0,v1) segments
    with (color0,color1) endpoints -- a direct evaluation of the same
    piecewise-linear colormap NRL's create_linear_segmented_colormap
    builds, applied straight to the physical values rather than via a
    matplotlib Colormap+Normalize pair. Values outside the full range
    clamp to the nearest endpoint color."""
    flat = values.ravel().astype(np.float64)
    out = np.zeros((flat.size, 3))
    filled = np.zeros(flat.size, dtype=bool)

    for (v0, v1), (c0, c1) in zip(transition_vals, transition_colors):
        rgb0 = np.array(mcolors.to_rgb(c0))
        rgb1 = np.array(mcolors.to_rgb(c1))
        mask = (flat >= v0) & (flat <= v1) & ~filled
        if not mask.any():
            continue
        if v1 > v0:
            frac = (flat[mask] - v0) / (v1 - v0)
        else:
            frac = np.zeros(int(mask.sum()))
        out[mask] = rgb0[None, :] + frac[:, None] * (rgb1 - rgb0)[None, :]
        filled |= mask

    v_min = transition_vals[0][0]
    v_max = transition_vals[-1][1]
    below = (flat < v_min) & ~filled
    above = (flat > v_max) & ~filled
    if below.any():
        out[below] = mcolors.to_rgb(transition_colors[0][0])
    if above.any():
        out[above] = mcolors.to_rgb(transition_colors[-1][1])

    return np.clip(out, 0, 1).reshape(values.shape + (3,))


# ---------------------------------------------------------------------------
# 37H (cmap_37H.py, default data_range=[125,310], "MUST include 125 and 300")
# ---------------------------------------------------------------------------
H37_TRANSITION_VALS = [
    (125, 180), (180, 195), (195, 210), (210, 220), (220, 230),
    (230, 240), (240, 260), (260, 280), (280, 310),
]
H37_TRANSITION_COLORS = [
    ("lightyellow", "darkmagenta"), ("#80007F", "#0080FF"), ("#0080FF", "#3AB9FF"),
    ("#3AB9FF", "#7DFDFF"), ("#7DFDFF", "#80FF82"), ("#80FF82", "#FFFF80"),
    ("#FFFF80", "#FF8000"), ("#FF8000", "#800000"), ("silver", "black"),
]

# ---------------------------------------------------------------------------
# 89H (cmap_89H.py, default data_range=[105,305], "MUST include 180 and 280")
# ---------------------------------------------------------------------------
H89_TRANSITION_VALS = [
    (105, 180), (180, 212), (212, 228), (228.1, 254), (254.1, 280), (280, 305),
]
H89_TRANSITION_COLORS = [
    ("white", "black"), ("#A4641A", "#FC0603"), ("#F4CD03", "#F2F403"),
    ("#8CF303", "#0FB503"), ("#06DCFD", "#0708B5"), ("navy", "white"),
]

# ---------------------------------------------------------------------------
# 37pct (cmap_37pct.py, default data_range=[230,280], "MUST include 230,280")
# ---------------------------------------------------------------------------
PCT37_TRANSITION_VALS = [(230, 240), (240, 260), (260, 280)]
PCT37_TRANSITION_COLORS = [("cyan", "yellow"), ("yellow", "red"), ("red", "darkred")]

# ---------------------------------------------------------------------------
# 89pct (cmap_89pct.py, default data_range=[105,280], "MUST include 125,265")
# ---------------------------------------------------------------------------
PCT89_TRANSITION_VALS = [
    (105, 125), (125, 150), (150, 175), (175, 212),
    (212, 230), (230, 250), (250, 265), (265, 280),
]
PCT89_TRANSITION_COLORS = [
    ("orange", "chocolate"), ("chocolate", "indianred"), ("indianred", "firebrick"),
    ("firebrick", "red"), ("gold", "yellow"), ("lime", "limegreen"),
    ("deepskyblue", "blue"), ("navy", "slateblue"),
]


def colorize_37h(h37: np.ndarray) -> np.ndarray:
    rgb01 = _colorize_segments(h37, H37_TRANSITION_VALS, H37_TRANSITION_COLORS)
    return np.clip(rgb01 * 255, 0, 255).astype(np.uint8)


def colorize_89h(h89: np.ndarray) -> np.ndarray:
    rgb01 = _colorize_segments(h89, H89_TRANSITION_VALS, H89_TRANSITION_COLORS)
    return np.clip(rgb01 * 255, 0, 255).astype(np.uint8)


def colorize_37pct(v37: np.ndarray, h37: np.ndarray) -> np.ndarray:
    """PCT37 (NRL pmw_37pct.py coefficients, mw_composites.PCT_THETA[37]),
    colorized with the exact NRL cmap_37pct.py palette."""
    pct = mw_composites.compute_pct(v37, h37, 37)
    rgb01 = _colorize_segments(pct, PCT37_TRANSITION_VALS, PCT37_TRANSITION_COLORS)
    return np.clip(rgb01 * 255, 0, 255).astype(np.uint8)


def colorize_89pct(v89: np.ndarray, h89: np.ndarray) -> np.ndarray:
    """PCT89 (NRL pmw_89pct.py coefficients, mw_composites.PCT_THETA[89]),
    colorized with the exact NRL cmap_89pct.py palette."""
    pct = mw_composites.compute_pct(v89, h89, 89)
    rgb01 = _colorize_segments(pct, PCT89_TRANSITION_VALS, PCT89_TRANSITION_COLORS)
    return np.clip(rgb01 * 255, 0, 255).astype(np.uint8)


def h_channel_from_swath(swath, freq: int) -> np.ndarray:
    """Convenience wrapper: colorize the H channel directly from an
    MWSwath (data_types.MWSwath) or SyntheticMWResult for the given
    frequency (37 or 89) -- works for either since both objects expose
    h37/h89 attributes the same way."""
    if freq == 37:
        return colorize_37h(swath.h37)
    elif freq == 89:
        return colorize_89h(swath.h89)
    else:
        raise ValueError("freq must be 37 or 89")


def pct_channel_from_swath(swath, freq: int) -> np.ndarray:
    """Same idea as h_channel_from_swath, for the standalone PCT
    colorized products instead of raw H."""
    if freq == 37:
        return colorize_37pct(swath.v37, swath.h37)
    elif freq == 89:
        return colorize_89pct(swath.v89, swath.h89)
    else:
        raise ValueError("freq must be 37 or 89")
