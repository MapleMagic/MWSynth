"""
NEXRAD (WSR-88D) radar station coordinates, embedded locally so lookups
don't need a network call.

Source: user-supplied NCEI station list (nexrad-stations.txt), parsed
with the file's own fixed-width column boundaries (taken from its dash-
underline header row, not guessed/hardcoded) and filtered to STNTYPE ==
'NEXRAD' only -- the source file also includes 47 TDWR (Terminal Doppler
Weather Radar) stations, a different network entirely that is NOT part of
the unidata-nexrad-level2 S3 archive, so those are deliberately excluded.
163 real NEXRAD stations parsed, including non-CONUS sites relevant to
tropical cyclones (Guam, Puerto Rico, South Korea, Japan, Azores) that a
CONUS-only assumption would have missed.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional


class RadarStation(NamedTuple):
    icao: str
    name: str
    state: str
    lat: float
    lon: float


# Auto-generated from nexrad-stations.txt -- NEXRAD-only (TDWR excluded), (icao, name, state, lat, lon)
_RAW_STATIONS = [
    ('KABR', 'ABERDEEN', 'SD', 45.455833, -98.413333),
    ('KABX', 'ALBUQUERQUE', 'NM', 35.149722, -106.82388),
    ('KAKQ', 'NORFOLK RICH', 'VA', 36.98405, -77.007361),
    ('KAMA', 'AMARILLO', 'TX', 35.233333, -101.70927),
    ('KAMX', 'MIAMI', 'FL', 25.611083, -80.412667),
    ('KAPX', 'GAYLORD', 'MI', 44.90635, -84.719533),
    ('KARX', 'LA CROSSE', 'WI', 43.822778, -91.191111),
    ('KATX', 'SEATTLE', 'WA', 48.194611, -122.49569),
    ('KBBX', 'BEALE AFB', 'CA', 39.495639, -121.63161),
    ('KBGM', 'BINGHAMTON', 'NY', 42.199694, -75.984722),
    ('KBHX', 'EUREKA', 'CA', 40.498583, -124.29216),
    ('KBIS', 'BISMARCK', 'ND', 46.770833, -100.76055),
    ('KBLX', 'BILLINGS', 'MT', 45.853778, -108.6068),
    ('KBMX', 'BIRMINGHAM', 'AL', 33.172417, -86.770167),
    ('KBOX', 'BOSTON', 'MA', 41.955778, -71.136861),
    ('KBRO', 'BROWNSVILLE', 'TX', 25.916, -97.418967),
    ('KBUF', 'BUFFALO', 'NY', 42.948789, -78.736781),
    ('KBYX', 'KEY WEST', 'FL', 24.5975, -81.703167),
    ('KCAE', 'COLUMBIA', 'SC', 33.948722, -81.118278),
    ('KCBW', 'HOULTON', 'ME', 46.03925, -67.806431),
    ('KCBX', 'BOISE', 'ID', 43.490217, -116.23603),
    ('KCCX', 'STATE COLLEGE', 'PA', 40.923167, -78.003722),
    ('KCLE', 'CLEVELAND', 'OH', 41.413217, -81.859867),
    ('KCLX', 'CHARLESTON', 'SC', 32.655528, -81.042194),
    ('KCRI', 'ROC FAA REDUNDANT RDA 1', 'OK', 35.238333, -97.46),
    ('KCRP', 'CORPUS CHRISTI', 'TX', 27.784017, -97.51125),
    ('KCXX', 'BURLINGTON', 'VT', 44.511, -73.166431),
    ('KCYS', 'CHEYENNE', 'WY', 41.151919, -104.80603),
    ('KDAX', 'SACRAMENTO', 'CA', 38.501111, -121.67783),
    ('KDDC', 'DODGE CITY', 'KS', 37.760833, -99.968889),
    ('KDFX', 'LAUGHLIN AFB', 'TX', 29.273139, -100.28033),
    ('KDGX', 'JACKSON BRANDON', 'MS', 32.279944, -89.984444),
    ('KDIX', 'PHILADELPHIA', 'NJ', 39.947089, -74.410731),
    ('KDLH', 'DULUTH', 'MN', 46.836944, -92.209722),
    ('KDMX', 'DES MOINES', 'IA', 41.7312, -93.722869),
    ('KDOX', 'DOVER AFB', 'DE', 38.825767, -75.440117),
    ('KDTX', 'DETROIT', 'MI', 42.7, -83.471667),
    ('KDVN', 'DAVENPORT', 'IA', 41.611667, -90.580833),
    ('KDYX', 'DYESS AFB', 'TX', 32.5385, -99.254333),
    ('KEAX', 'KANSAS CITY', 'MO', 38.81025, -94.264472),
    ('KEMX', 'TUCSON', 'AZ', 31.89365, -110.63025),
    ('KENX', 'ALBANY', 'NY', 42.586556, -74.064083),
    ('KEOX', 'FORT RUCKER', 'AL', 31.460556, -85.459389),
    ('KEPZ', 'EL PASO', 'NM', 31.873056, -106.698),
    ('KESX', 'LAS VEGAS', 'NV', 35.70135, -114.89165),
    ('KEVX', 'EGLIN AFB', 'FL', 30.565033, -85.921667),
    ('KEWX', 'AUSTIN SAN ANTONIO', 'TX', 29.704056, -98.028611),
    ('KEYX', 'EDWARDS', 'CA', 35.09785, -117.56075),
    ('KFCX', 'ROANOKE', 'VA', 37.0244, -80.273969),
    ('KFDR', 'ALTUS AFB', 'OK', 34.362194, -98.976667),
    ('KFDX', 'CANNON AFB', 'NM', 34.634167, -103.61888),
    ('KFFC', 'ATLANTA', 'GA', 33.36355, -84.56595),
    ('KFSD', 'SIOUX FALLS', 'SD', 43.587778, -96.729444),
    ('KFSX', 'FLAGSTAFF', 'AZ', 34.574333, -111.19844),
    ('KFTG', 'DENVER FRONT RANGE AP', 'CO', 39.786639, -104.5458),
    ('KFWS', 'DALLAS', 'TX', 32.573, -97.30315),
    ('KGGW', 'GLASGOW', 'MT', 48.206361, -106.62469),
    ('KGJX', 'GRAND JUNCTION', 'CO', 39.062169, -108.21376),
    ('KGLD', 'GOODLAND', 'KS', 39.366944, -101.70027),
    ('KGRB', 'GREEN BAY', 'WI', 44.498633, -88.111111),
    ('KGRK', 'FORT HOOD', 'TX', 30.721833, -97.382944),
    ('KGRR', 'GRAND RAPIDS', 'MI', 42.893889, -85.544889),
    ('KGSP', 'GREER', 'SC', 34.883306, -82.219833),
    ('KGWX', 'COLUMBUS AFB', 'MS', 33.896917, -88.329194),
    ('KGYX', 'PORTLAND', 'ME', 43.891306, -70.256361),
    ('KHDC', 'HAMMOND MUNICIPAL AIRPORT', 'LA', 30.5193, -90.4074),
    ('KHDX', 'HOLLOMAN AFB', 'NM', 33.077, -106.12003),
    ('KHGX', 'HOUSTON', 'TX', 29.4719, -95.078733),
    ('KHNX', 'SAN JOAQUIN VALLEY', 'CA', 36.314181, -119.63213),
    ('KHPX', 'FORT CAMPBELL', 'KY', 36.736972, -87.285583),
    ('KHTX', 'HUNTSVILLE', 'AL', 34.930556, -86.083611),
    ('KICT', 'WICHITA', 'KS', 37.654444, -97.443056),
    ('KICX', 'CEDAR CITY', 'UT', 37.59105, -112.86218),
    ('KILN', 'CINCINNATI', 'OH', 39.420483, -83.82145),
    ('KILX', 'LINCOLN', 'IL', 40.1505, -89.336792),
    ('KIND', 'INDIANAPOLIS', 'IN', 39.7075, -86.280278),
    ('KINX', 'TULSA', 'OK', 36.175131, -95.564161),
    ('KIWA', 'PHOENIX', 'AZ', 33.289233, -111.66991),
    ('KIWX', 'FORT WAYNE', 'IN', 41.358611, -85.7),
    ('KJAX', 'JACKSONVILLE', 'FL', 30.484633, -81.7019),
    ('KJGX', 'ROBINS AFB', 'GA', 32.675683, -83.350833),
    ('KJKL', 'JACKSON', 'KY', 37.590833, -83.313056),
    ('KLBB', 'LUBBOCK', 'TX', 33.654139, -101.81416),
    ('KLCH', 'LAKE CHARLES', 'LA', 30.125306, -93.215889),
    ('KLGX', 'LANGLEY HILL NW WASHINGTON', 'WA', 47.116944, -124.10666),
    ('KLNX', 'NORTH PLATTE', 'NE', 41.957944, -100.57622),
    ('KLOT', 'CHICAGO', 'IL', 41.604444, -88.084444),
    ('KLRX', 'ELKO', 'NV', 40.73955, -116.8027),
    ('KLSX', 'ST LOUIS', 'MO', 38.698611, -90.682778),
    ('KLTX', 'WILMINGTON', 'NC', 33.98915, -78.429108),
    ('KLVX', 'LOUISVILLE', 'KY', 37.975278, -85.943889),
    ('KLWX', 'STERLING', 'VA', 38.976111, -77.4875),
    ('KLZK', 'LITTLE ROCK', 'AR', 34.8365, -92.262194),
    ('KMAF', 'MIDLAND ODESSA', 'TX', 31.943461, -102.18925),
    ('KMAX', 'MEDFORD', 'OR', 42.081169, -122.71736),
    ('KMBX', 'MINOT AFB', 'ND', 48.393056, -100.86444),
    ('KMHX', 'MOREHEAD CITY', 'NC', 34.775908, -76.876189),
    ('KMKX', 'MILWAUKEE', 'WI', 42.9679, -88.550667),
    ('KMLB', 'MELBOURNE', 'FL', 28.113194, -80.654083),
    ('KMOB', 'MOBILE', 'AL', 30.679444, -88.24),
    ('KMPX', 'MINNEAPOLIS', 'MN', 44.848889, -93.565528),
    ('KMQT', 'MARQUETTE', 'MI', 46.531111, -87.548333),
    ('KMRX', 'KNOXVILLE', 'TN', 36.168611, -83.401944),
    ('KMSX', 'MISSOULA', 'MT', 47.041, -113.98622),
    ('KMTX', 'SALT LAKE CITY', 'UT', 41.262778, -112.44777),
    ('KMUX', 'SAN FRANCISCO', 'CA', 37.155222, -121.89844),
    ('KMVX', 'GRAND FORKS', 'ND', 47.527778, -97.325556),
    ('KMXX', 'MAXWELL AFB', 'AL', 32.53665, -85.78975),
    ('KNKX', 'SAN DIEGO', 'CA', 32.919017, -117.0418),
    ('KNQA', 'MEMPHIS', 'TN', 35.344722, -89.873333),
    ('KOAX', 'OMAHA', 'NE', 41.320369, -96.366819),
    ('KOHX', 'NASHVILLE', 'TN', 36.247222, -86.5625),
    ('KOKX', 'NEW YORK CITY', 'NY', 40.865528, -72.863917),
    ('KOTX', 'SPOKANE', 'WA', 47.680417, -117.62677),
    ('KOUN', 'NORMAN NSSL', 'OK', 35.236058, -97.46235),
    ('KPAH', 'PADUCAH', 'KY', 37.068333, -88.771944),
    ('KPBZ', 'PITTSBURGH', 'PA', 40.531717, -80.217967),
    ('KPDT', 'PENDLETON', 'OR', 45.69065, -118.85293),
    ('KPOE', 'FORT POLK', 'LA', 31.155278, -92.976111),
    ('KPUX', 'PUEBLO', 'CO', 38.45955, -104.18135),
    ('KRAX', 'RALEIGH DURHAM', 'NC', 35.665519, -78.48975),
    ('KRGX', 'RENO', 'NV', 39.754056, -119.46202),
    ('KRIW', 'RIVERTON', 'WY', 43.066089, -108.4773),
    ('KRLX', 'CHARLESTON', 'WV', 38.311111, -81.722778),
    ('KRTX', 'PORTLAND', 'OR', 45.715039, -122.965),
    ('KSFX', 'POCATELLO', 'ID', 43.1056, -112.68613),
    ('KSGF', 'SPRINGFIELD', 'MO', 37.235239, -93.400419),
    ('KSHV', 'SHREVEPORT', 'LA', 32.450833, -93.84125),
    ('KSJT', 'SAN ANGELO', 'TX', 31.371278, -100.4925),
    ('KSOX', 'SANTA ANA MOUNTAINS', 'CA', 33.817733, -117.636),
    ('KSRX', 'FORT SMITH', 'AR', 35.290417, -94.361889),
    ('KTBW', 'TAMPA', 'FL', 27.7055, -82.401778),
    ('KTFX', 'GREAT FALLS', 'MT', 47.459583, -111.38533),
    ('KTLH', 'TALLAHASSEE', 'FL', 30.397583, -84.328944),
    ('KTLX', 'OKLAHOMA CITY', 'OK', 35.333361, -97.277761),
    ('KTWX', 'TOPEKA', 'KS', 38.99695, -96.23255),
    ('KTYX', 'FORT DRUM', 'NY', 43.755694, -75.679861),
    ('KUDX', 'RAPID CITY', 'SD', 44.124722, -102.83),
    ('KUEX', 'HASTINGS', 'NE', 40.320833, -98.441944),
    ('KVAX', 'MOODY AFB', 'GA', 30.890278, -83.001806),
    ('KVBX', 'VANDENBERG AFB', 'CA', 34.83855, -120.39791),
    ('KVNX', 'VANCE AFB', 'OK', 36.740617, -98.127717),
    ('KVTX', 'LOS ANGELES', 'CA', 34.412017, -119.17875),
    ('KVWX', 'EVANSVILLE', 'IN', 38.26025, -87.724528),
    ('KYUX', 'YUMA', 'AZ', 32.495281, -114.65671),
    ('LPLA', 'LAJES AB', '', 38.73028, -27.32167),
    ('PABC', 'BETHEL FAA', 'AK', 60.791944, -161.87638),
    ('PACG', 'SITKA', 'AK', 56.852778, -135.52916),
    ('PAEC', 'NOME', 'AK', 64.511389, -165.295),
    ('PAHG', 'ANCHORAGE', 'AK', 60.725914, -151.35146),
    ('PAIH', 'MIDDLETON ISLAND', 'AK', 59.460767, -146.30344),
    ('PAKC', 'KING SALMON', 'AK', 58.679444, -156.62944),
    ('PAPD', 'FAIRBANKS', 'AK', 65.035114, -147.50143),
    ('PGUA', 'ANDERSEN AFB AGANA', 'GU', 13.455833, 144.811111),
    ('PHKI', 'SOUTH KAUAI', 'HI', 21.893889, -159.5525),
    ('PHKM', 'KAMUELA', 'HI', 20.125278, -155.77777),
    ('PHMO', 'MOLOKAI', 'HI', 21.132778, -157.18027),
    ('PHWA', 'SOUTH SHORE', 'HI', 19.095, -155.56888),
    ('RKJK', 'KUNSAN', '', 35.924167, 126.622222),
    ('RKSG', 'CAMP HUMPHREYS', '', 37.207569, 127.285561),
    ('RODN', 'KADENA', '', 26.3078, 127.903469),
    ('TJUA', 'SAN JUAN', 'PR', 18.115667, -66.078167),
]

STATIONS = [RadarStation(icao=r[0], name=r[1], state=r[2], lat=r[3], lon=r[4]) for r in _RAW_STATIONS]
STATIONS_BY_ICAO = {s.icao: s for s in STATIONS}

EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    lat1r, lon1r, lat2r, lon2r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(min(1.0, a)))


def find_nearest_station_in_range(
    lat: float, lon: float, max_range_mi: float = 200.0
) -> tuple[Optional[RadarStation], Optional[float]]:
    """Returns (nearest_station, distance_mi), or (None, None) if no
    station is within max_range_mi. Deliberately does NOT just return the
    globally-nearest station regardless of distance -- per spec, a storm
    far from any radar should get no radar source at all, not the
    nearest-but-still-very-far one."""
    best_station = None
    best_dist = None
    for station in STATIONS:
        dist = haversine_miles(lat, lon, station.lat, station.lon)
        if dist <= max_range_mi and (best_dist is None or dist < best_dist):
            best_station = station
            best_dist = dist
    return best_station, best_dist
