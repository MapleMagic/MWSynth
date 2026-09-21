"""
Solar elevation angle and day/night determination, used to decide whether
GOES Band 2 (0.64 um visible) carries real signal or just night-time noise
at the storm's location and time.

Uses the standard NOAA Solar Calculator formulas (fractional-year based
approximations for solar declination and the equation of time) -- these
are well-established, accurate to a small fraction of a degree, and need
no external data files or network access (unlike an ephemeris-based
approach), which matters here since this needs to run as a quick gating
check before deciding whether to bother fetching Band 2 at all.

Reference: https://gml.noaa.gov/grad/solcalc/solareqns.PDF (NOAA ESRL
Global Monitoring Laboratory's published solar position equations).
"""
from __future__ import annotations

import timeutil

import math
from datetime import datetime

# Visible imagery quality degrades near the terminator (long slant path,
# low sun angle shadows/glint) well before the sun is literally below the
# horizon -- default to requiring a modest positive elevation, not just
# "not technically nighttime," so Band 2 isn't used on marginal twilight
# scenes either.
DEFAULT_MIN_ELEVATION_DEG = 5.0


def solar_elevation_deg(lat: float, lon: float, dt_utc: datetime) -> float:
    """Approximate solar elevation angle (degrees above the horizon) at
    (lat, lon) and the given UTC datetime. Positive = sun above horizon.
    lon is degrees East (negative for West, matching this project's
    convention elsewhere)."""
    dt_utc = timeutil.as_utc(dt_utc)
    day_of_year = dt_utc.timetuple().tm_yday
    hour_frac = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # Fractional year (radians), per NOAA's formulation.
    gamma = 2 * math.pi / 365.0 * (day_of_year - 1 + (hour_frac - 12) / 24.0)

    # Equation of time (minutes) -- corrects mean solar time to true solar time.
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Solar declination (radians).
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4 * lon  # minutes
    true_solar_time = hour_frac * 60 + time_offset  # minutes since midnight
    hour_angle_deg = (true_solar_time / 4.0) - 180.0  # degrees

    lat_rad = math.radians(lat)
    hour_angle_rad = math.radians(hour_angle_deg)

    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(
        hour_angle_rad
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith_deg = math.degrees(math.acos(cos_zenith))

    return 90.0 - zenith_deg


def is_daytime(
    lat: float, lon: float, dt_utc: datetime, min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG
) -> bool:
    """True if the sun is at least min_elevation_deg above the horizon at
    (lat, lon, dt_utc) -- the gate used to decide whether GOES Band 2
    (visible) is usable, not just whether it's technically after sunrise."""
    return solar_elevation_deg(lat, lon, dt_utc) >= min_elevation_deg
