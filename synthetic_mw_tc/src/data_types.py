"""
Shared data structures used across the ingestion, algorithm, and GUI layers.

Kept dependency-light (stdlib + numpy only) so every other module can import
this without pulling in boto3/xarray/etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class BandImage:
    """A single ABI band's data, already converted to physical units
    (brightness temperature in Kelvin for IR/WV bands, reflectance factor
    for the visible band 2)."""

    band: int
    satellite: str          # "GOES-18" or "GOES-19"
    scene_time: datetime     # scan start time
    values: np.ndarray       # 2D array, physical units
    lat: np.ndarray          # 2D array, matches values.shape
    lon: np.ndarray          # 2D array, matches values.shape
    units: str                # "K" or "reflectance"
    mesoscale_sector: str     # "M1" or "M2"
    source_key: str = ""      # S3 key this came from, for provenance/debug


@dataclass
class StormFix:
    """One best-track (or interpolated) fix for a storm at a given time."""

    storm_id: str            # ATCF ID, e.g. "AL092023"
    valid_time: datetime
    lat: float
    lon: float
    vmax_kt: float
    mslp_mb: Optional[float] = None
    rmw_nm: Optional[float] = None       # radius of max wind, if available
    roci_nm: Optional[float] = None      # radius of outermost closed isobar, if available
    storm_type: str = ""                  # TD/TS/HU/etc.
    basin: str = ""
    # How many hours this fix's POSITION was extrapolated beyond the last
    # real best-track entry (0.0 for a real or interpolated fix). Purely
    # informational -- lets the GUI say the centre is a projection rather
    # than an observation, since a silently-projected centre is
    # indistinguishable from a real one and produces a confidently
    # misplaced image.
    extrapolated_hours: float = 0.0


@dataclass
class SyntheticMWResult:
    """Output of the synthetic-microwave algorithm for one scene."""

    scene_time: datetime
    storm_id: str
    freq_37ghz: np.ndarray     # synthetic brightness temp field, Kelvin (scalar blend)
    freq_89ghz: np.ndarray     # synthetic brightness temp field, Kelvin (scalar blend)
    lat: np.ndarray
    lon: np.ndarray
    diagnostics: dict = field(default_factory=dict)  # intermediate fields for debugging/plotting
    # Separate synthetic V/H fields (same grid as freq_37ghz/freq_89ghz),
    # added so mw_composites.py's real color-composite technique (which
    # needs V and H separately, not one blended scalar) can be reused
    # directly on synthetic output. freq_37ghz/freq_89ghz above are KEPT
    # as-is (unchanged scalar blend) since mw_compare.py's real-vs-
    # synthetic stats compare against that scalar specifically.
    v37: Optional[np.ndarray] = None
    h37: Optional[np.ndarray] = None
    v89: Optional[np.ndarray] = None
    h89: Optional[np.ndarray] = None


@dataclass
class MWSwath:
    """A real passive-microwave swath, cropped to a region of interest,
    with the four raw channels needed for 37/89 GHz color composites plus
    the derived PCT values. Produced by mw_ingest.py.

    lat/lon/*_tb arrays share the same shape (they're on the sensor's
    native swath grid, NOT regridded to a rectangular lat/lon grid --
    unlike GOES's fixed grid, PMW conically-scanning swaths are naturally
    irregular, so composites are built and plotted directly on this native
    grid via pcolormesh rather than forcing a resample).

    Some sensors (AMSR2) sample different frequencies at different native
    along-scan resolutions, so 37 GHz and 89 GHz can legitimately be on
    DIFFERENT grids within the same overpass. lat/lon is the default/
    primary grid (v89/h89's grid where the two differ); lat37/lon37,
    lat89/lon89 are optional per-frequency overrides -- use grid_for()
    rather than accessing lat/lon directly if you need the grid that
    actually matches a specific frequency's array shape.
    """

    sensor: str               # "GMI", "AMSR2", or "SSMIS"
    scene_time: datetime
    lat: np.ndarray
    lon: np.ndarray
    v37: np.ndarray
    h37: np.ndarray
    v89: np.ndarray
    h89: np.ndarray
    source_note: str = ""     # provenance string, e.g. granule ID or file name
    lat37: Optional[np.ndarray] = None
    lon37: Optional[np.ndarray] = None
    lat89: Optional[np.ndarray] = None
    lon89: Optional[np.ndarray] = None

    def grid_for(self, freq: int):
        """Returns (lat, lon) for the given frequency (37 or 89), using
        the frequency-specific grid if one was set, otherwise falling
        back to the shared lat/lon."""
        if freq == 37 and self.lat37 is not None:
            return self.lat37, self.lon37
        if freq == 89 and self.lat89 is not None:
            return self.lat89, self.lon89
        return self.lat, self.lon
