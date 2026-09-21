# Changelog

Chronological development log, oldest first. Kept in full because most
entries record *why* a decision was made or how a bug was diagnosed, not
just what changed.

---

## QC / the NaN crash fix

Real GOES scenes always contain some QC-flagged, off-limb, or fill-value
pixels. Converting those straight to brightness temperature produced
NaN/Inf, which then silently propagated through the nearest-neighbor
regrid into matplotlib — that's what threw `data must be finite, check
for nan or inf values`.

Fixed in `src/qc_utils.py` (`sanitize_field`): every band's DQF (data
quality flag) array is used to mask bad pixels, radiance is checked for
non-positive values before the Planck inversion, and masked/non-finite
pixels are filled with the nearest valid pixel (`fill_invalid_nearest`,
standard nearest-neighbor inpainting). This is applied at ingestion
(`goes_fetch.py`), again defensively at the top of the algorithm and after
every regrid (`synthetic_algorithm.py`), and as a final `assert_finite`
guard right before the result is returned — so a non-finite value now
either gets cleanly filled or raises a clear, specific error instead of
crashing deep in a plotting call. If a scene is more than 50% bad pixels,
`get_band_image` raises a `UserWarning` so you know a frame was mostly
inpainted rather than trusting it blindly.

## ROCI

Best-track ROCI (radius of outermost closed isobar) is now parsed from
field 19 (0-indexed) of the ATCF record and fed into the radial envelope
model as the rainband/CDO size, instead of only inferring size from Vmax.
It's typically populated on just the 34kt-radii row per valid time, not
every row, so the parser keeps the largest non-zero ROCI/RMW value seen
across all rows sharing a timestamp. When ROCI isn't present on a given
best-track entry (common for weaker/older systems), it falls back to the
original Vmax-based envelope estimate.

## Fixes from live testing round 2

**Longitude/latitude were wrong (the "-300 to 0" axis bug):**
`fixed_grid_to_latlon` had two bugs in the ABI fixed-grid navigation math:
1. The `x`/`y` scan-angle arrays (radians) were incorrectly multiplied by
   `perspective_point_height` before use. They must stay in radians —
   scaling them fed huge arguments into `sin`/`cos`, which wrap
   chaotically and produced nonsense lat/lon ranges.
2. The longitude formula had a sign error (`lambda0 + arctan(...)`
   instead of the correct `lambda0 - arctan(...)`).

Both are fixed now and verified against the one case with a known exact
answer: at the sub-satellite point (x=y=0), lat must be exactly 0 and lon
must be exactly `longitude_of_projection_origin`. A synthetic mesoscale-
sized window now maps to a realistic ~9° footprint centered correctly,
instead of the old full-globe-scale garbage range.

**Timezone display bug:** `QDateTimeEdit` doesn't lock its internal time
spec by default, so once you interacted with the widget it silently
started treating the value as local time — then `.toUTC()` shifted it by
your UTC offset even though the field is labeled UTC and you'd entered it
as UTC. Fixed by explicitly locking the widget to `Qt.TimeSpec.UTC`. Also
added a note under the field: the AWS S3 console's "Last modified" column
shows *your browser's* local time zone, not UTC — the actual scan time is
only reliable from the timestamp embedded in the object key name itself
(the `s...` field). The app now also logs the exact UTC time it's
requesting when you hit Generate, so you can cross-check it directly.

**Important:** the GOES and best-track fetch code was written against the
documented file/bucket formats and has now been exercised against your
live run (which surfaced and fixed the QC bug above), but only lightly —
keep an eye out for other edge cases as you use it more.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd src
python main.py
```

You'll need outbound HTTPS access to `*.s3.amazonaws.com` and
`ftp.nhc.noaa.gov`. On the **Generate** tab: pick a target UTC time,
satellite (GOES-18 or GOES-19), mesoscale sector (or Auto to search both),
and the storm's ATCF basin/number/season (e.g. AL, 09, 2024 for AL092024)
so the best-track file can be located.

## MW Data Credentials tab

A second tab holds username/password fields for the three services
`mw_ingest.py` will eventually use (NASA Earthdata for GMI, JAXA G-Portal
for AMSR2, NOAA CLASS for SSMIS). Storage behavior (`src/credentials.py`):

- **Off by default.** Credentials only live in memory for the current
  session unless you check "Save credentials to disk."
- When enabled, they're written as **plaintext JSON** to
  `~/.synthetic_mw_tc/credentials.json` (file permissions set to owner-only
  where the OS supports it). This is a convenience store for a personal
  tool, not a secrets manager — there's no encryption at rest.
- **Clear saved credentials** deletes that file immediately, with no
  confirmation dialog, by design (this is sensitive account info and
  shouldn't be behind an extra alert to click through).

## Project layout

```
src/
  data_types.py         # shared dataclasses (BandImage, StormFix, SyntheticMWResult)
  qc_utils.py             # NaN/Inf masking + nearest-fill, shared by ingestion + algorithm
  goes_fetch.py             # GOES-18/19 RadM S3 ingestion + calibration to K/reflectance
  besttrack.py                # NHC ATCF best-track parser + interpolation (incl. ROCI/RMW)
  synthetic_algorithm.py        # the core non-ML algorithm (read the docstring)
  credentials.py                  # local credential storage for MW services
  mw_ingest.py                      # real GMI/AMSR2 ingestion, SSMIS documented-not-yet
  mw_composites.py                    # PCT + NRL-style 37/89 GHz color composites
  main.py                                 # entry point
  gui/
    main_window.py                          # PyQt6 GUI, Generate + Real MW tabs
    credentials_tab.py                        # PyQt6 GUI, MW Data Credentials tab
```

## Known risk areas to check on first live run

- **GOES key parsing regex** (`goes_fetch._parse_key`) — written against the
  documented `OR_ABI-L1b-RadM{1,2}-M6C{band}_G{sat}_s{timestamp}...` pattern.
  If NOAA changes scan mode (M6 -> M3/M4) or filename conventions, the regex
  will need a tweak.
- **Fixed-grid lat/lon conversion** (`goes_fetch.fixed_grid_to_latlon`) — the
  standard ABI geostationary projection math; double check against a known
  reference pixel the first time you run it for real, since a sign error
  here would silently misplace the storm.
- **ATCF field indices** (`besttrack.parse_best_track_text`) — b-deck format
  is comma-delimited but the FTP files sometimes have extra whitespace
  around fields; already handled with `.strip()`, but wind-radii lines vs.
  the core BEST/TAU=0 lines should be double-checked against a real file.
- **`radiance_to_physical`** brightness-temp formula uses the standard ABI
  L1b Planck-function inversion — this is well-established, low risk, but
  worth a spot check against a known scene.

## Calibration

`synthetic_algorithm.CALIBRATION` holds first-guess constants (background
clear-sky Tb, max scattering depression, etc.) pulled from general published
TC microwave Tb ranges, **not fit to real data yet**. Once `mw_ingest.py` is
built, the intended workflow is: generate a synthetic frame and a real GMI/
AMSR2 frame for the same storm/time, compare, and adjust the constants (or
graduate to a proper regression/fit) to close the gap.

## Real MW ingestion (GMI / AMSR2 / SSMIS) + 37/89 GHz color composites

New in this round: `mw_ingest.py` and `mw_composites.py`, plus a "Real MW"
GUI tab.

**Status by sensor** (deliberately uneven — see `mw_ingest.py`'s
docstring for the full reasoning):
- **GMI: fully working**, via the real `earthaccess` package against
  NASA's GES DISC archive (`GPM_1CGPMGMI`, swath S1). Pulls 37V/37H/89V/89H
  directly from the product's fixed channel order.
- **AMSR2: implemented**, via the real `gportal` package (JAXA's own
  Python client). The dataset ID lookup and HDF5 variable names are
  discovered by pattern-matching rather than hardcoded, since I couldn't
  verify the exact catalog path/naming without live access — if it can't
  find the right dataset or variables on your first run, it'll raise a
  clear error telling you to call `mw_ingest.list_amsr2_datasets()` and
  adjust.
- **SSMIS: not automated for real-time use.** The only SSMIS source with
  a genuinely simple, public, scriptable access pattern is NOAA/NCEI's
  Climate Data Record archive — but that carries **~1 month latency**,
  so it's a research/backtesting source, not something you'd use on an
  active storm. NOAA CLASS is the actual near-real-time SSMIS path, but
  it's an order-and-retrieve system without a simple documented REST API
  I could verify, so rather than guess at a request format and risk
  silently submitting something wrong, `fetch_ssmis_swath()` raises
  `NotImplementedError` with an explanation instead of pretending to work.

**Search behavior:** rather than a fixed time window, the Real MW tab
searches the **past 6 hours** of imagery up through the time you set; if
nothing's found there, it automatically widens to the **past 9h**, then
**past 12h**, before giving up (`mw_ingest.fetch_recent_swath`). Polar-
orbiting sensors don't cover every location every orbit, so this gives you
the best chance of finding *something* usable without manually retrying.
If your Python environment is missing `earthaccess` or `gportal`, you'll
get a clear message pointing at `pip install -r requirements.txt` instead
of a raw `ModuleNotFoundError`.

### Fixes from live testing

**GMI not finding a pass you could see elsewhere:** this queries the
standard GES DISC science archive, not the near-real-time PPS feed --
that archive can lag by several hours, so "not found in the past 12h" can
mean the overpass just hasn't been processed into the archive yet, not
that there's no coverage. Also added a fallback: it now tries both
`GPM_1CGPMGMI` and `GPM_1CGMI` as the short_name (I couldn't verify with
full confidence which is CMR's actual registered name without live
access), logging which one returned results, plus per-attempt diagnostic
logging (exact time window, bbox, and granule count tried) so a
still-empty result is easier to diagnose.

**AMSR2 dataset lookup failing:** the original lookup only checked 2
levels of the G-Portal category tree with a couple of guessed key
spellings, which was too fragile -- rewritten to recursively search the
*entire* tree (`_walk_dataset_tree`) for any path resembling "L1B", so it
should now find the right dataset regardless of how deep it's nested or
exactly how the category is spelled. If it still can't find one, the
error message now includes an actual summary of the tree it saw. You can
also bypass the lookup entirely by setting the
`MW_INGEST_AMSR2_DATASET_ID` environment variable once you've identified
the correct ID via `list_amsr2_datasets()`. Also fixed the `requirements.txt`
pin (`gportal>=0.4`, matching what actually installed — the earlier `>=0.5`
was a guess that didn't match the real available version).

**37/89 GHz color composites** (`mw_composites.py`): built using the
actual documented NRL technique (Lee et al. 2002), not an invented scheme:
- PCT37 = 2.18×V37 − 1.18×H37 (Cecil et al. 2002 coefficients, as cited by
  Kieper & Jiang 2012 for NRL's own "37color" product)
- PCT89 = 1.818×V89 − 0.818×H89 (Spencer et al. 1989, as cited by NRL's
  own unified TC microwave calibration paper)
- RGB assignment: **R = H, G = PCT, B = V**. This isn't arbitrary — it's
  the assignment that actually reproduces the outcomes Lee et al. (2002)
  describe (ocean = green, deep convection = pink, light rain/low cloud =
  cyan), verified by hand in the module docstring and numerically (ocean-
  like inputs come out green-dominant, convection-like inputs come out
  red/blue-dominant with suppressed green).
- The exact per-channel stretch ranges (Kelvin → 0–255) are the one part
  that's a reasonable approximation rather than a verified NRL lookup
  table (that internal table isn't published) — see `CHANNEL_STRETCH` in
  `mw_composites.py`, documented as the tunable part, the same way
  `synthetic_algorithm.CALIBRATION` is.
- **Not yet connected to the synthetic algorithm** — `synthetic_algorithm.py`
  currently outputs a single scalar Tb per frequency, not separate V/H, so
  there's no synthetic *color* composite yet. Seeing this module's output
  against real GMI passes is exactly the next step toward extending the
  algorithm to produce plausible V/H pairs instead of a scalar.

### AMSR2 HDF5 variable naming — fixed with real file confirmation

Thanks to a real error output, we now have the exact key names JAXA uses
in the `L1SGBTBR` (resampled L1B) product:
- 37 GHz is stored as **36.5GHz**, not 37 — `Brightness Temperature (36.5GHz,V/H)`.
- 89 GHz has two sub-swaths from AMSR2's two 89GHz feedhorns:
  `Brightness Temperature (89.0GHz-A/B,V/H)`. We use **A** (the primary
  sample; B is a secondary offset sample, standard practice to skip it).
- There's no plain `Latitude`/`Longitude` — only
  `Latitude/Longitude of Observation Point for 89A/89B`. Since `BTBR`
  means every channel is resampled onto a common grid, we use 89A's
  geolocation for all channels (6.9 GHz through 89 GHz alike).
- Added a shape-consistency check after reading all four channels plus
  lat/lon: if a future file format doesn't actually share one grid across
  channels (breaking that assumption), this now raises a clear error
  immediately instead of failing confusingly downstream during cropping.

All six key lookups (lat, lon, 36.5V, 36.5H, 89A-V, 89A-H) were verified
directly against the real key list from the error output and resolve
correctly.

### Two more real-data bugs fixed

**AMSR2: 36.5GHz and 89GHz are genuinely different grids, not a
resampling gap.** Your error showed lat/lon and 89A at width 486, but
36.5GHz at width 243 -- exactly half. That's real AMSR2 hardware (lower
frequencies have larger antenna footprints, so they're natively sampled
at half the along-scan rate of 89GHz), not something this file failed to
resample. Rather than erroring out, `MWSwath` now supports genuinely
different grids per frequency (`lat37/lon37` vs `lat89/lon89`, via a new
`grid_for(freq)` accessor) -- the 36.5GHz grid is built by pairwise-
averaging adjacent 89A geolocation samples (486 → 243, an exact 2x
decimation matching the real sample spacing), each frequency is cropped
to the target box independently, and the GUI now asks for each panel's
own grid instead of assuming one shared grid. Verified end-to-end against
the exact shapes from your error.

**GMI: short_name is now resolved dynamically instead of guessed.**
Zero granules across both hardcoded short_name guesses and all three
lookback windows (6/9/12h) over a 12°-wide tropical Pacific box was too
consistent to be a real coverage gap -- that pointed at the short_name
itself not matching what's actually registered in CMR. `fetch_gmi_swath`
now runs a keyword search (`earthaccess.search_datasets`) against GES DISC
first to discover the real registered short_name(s), tries those before
falling back to the static guesses, and logs which one actually worked.
There's also a new standalone diagnostic,
`mw_ingest.diagnose_gmi_availability()`, you can run from a Python shell
to check whether a given short_name has *any* granules recently with no
location filter -- useful for telling "wrong product name" apart from
"real latency/coverage gap" if this still comes up empty.

### GMI still returning zero — temporal format fix

The dynamic short_name resolution confirmed `GPM_1CGPMGMI` (and `_R`) are
real, CMR-registered collections — so the identifier was never the
problem. Zero granules across every name AND every window (6/9/12h) over
a 12°×12° tropical box is essentially impossible as a genuine coverage
gap (GPM revisits the deep tropics multiple times a day), which pointed
at the query itself, not the product.

The likely culprit: `temporal=(start, end)` was passed as raw Python
`datetime` objects. Every working earthaccess example in the docs passes
`temporal` as ISO-8601 strings instead — if `search_data` doesn't coerce
datetime objects the same way internally, the temporal filter can fail to
parse without raising an error, silently excluding everything. Fixed by
explicitly formatting both bounds as `%Y-%m-%dT%H:%M:%SZ` strings before
the call. `diagnose_gmi_availability()` was also upgraded to check two
things separately: whether a short_name has *any* granules at all (no
filters), and whether it has granules in the recent past with no location
filter — so if this still comes up empty, the diagnostic will tell you
specifically whether it's a temporal-filter problem or a real gap, rather
than one ambiguous number.

### GMI still zero after the temporal-string fix — this is likely archive latency, not a bug

With the short_name confirmed real and the temporal format fixed, the
search window's own end time (essentially "right now" -- the GUI defaults
the target time to current UTC) still returned zero across every window
up to 12h. At this point the most evidence-backed explanation is the one
flagged in `mw_ingest.py`'s original docstring: GES DISC's *standard*
archive (not the NRT/PPS feed) has undocumented processing latency, and
it may simply exceed 12 hours for this product.

Two changes to let you test that directly instead of guessing further:
- **"Search depth" selector** on the Real MW tab: "Recent (6/9/12h)" or
  "Extended (6/9/12/24/48h)". If Extended finds something, that's strong
  confirmation it was a latency issue, not a coverage or query problem.
- `diagnose_gmi_availability()` now separately reports "any granules at
  all" (no filters) vs. "granules in the recent global window" per
  short_name -- run it from a Python shell for a direct, concrete answer
  rather than another empty search log.

If Extended search still comes back empty for a location/time you're
confident had a real pass, that would point back at something in the
query rather than latency -- worth reporting back with the
`diagnose_gmi_availability()` output if so, since that'd be genuinely new
evidence to work from.

### GMI mystery resolved — it was archive latency, plus a EULA step

Going back a few days finally got a granule match, which settles the
earlier back-and-forth: the search/query logic was correct all along
(short_name, temporal format, bbox) -- the "zero results in the past
6-12h" was genuinely archive latency on GES DISC's non-NRT feed, not a
bug, exactly as flagged early on.

The next error (`Eula Acceptance Failure`) is a one-time, account-level
step, not a code issue: your Earthdata account needs to authorize the
"NASA GESDISC DATA ARCHIVE" application and accept its End User License
Agreement before downloads succeed -- do this once at
https://urs.earthdata.nasa.gov (Applications -> Authorized Apps ->
"APPROVE MORE APPLICATIONS"), and it covers all GES DISC-hosted products
going forward. `fetch_gmi_swath` now recognizes this specific error and
raises a clear message with those exact steps instead of surfacing NASA's
raw error text.

## GMI near-real-time option (GMI-NRT) — added, unverified

You asked whether there's a source for more recent GMI than the GES DISC
archive (which, per the earlier testing, can lag by multiple days). There
is: NASA's PPS operates a separate near-real-time feed
(`jsimpsonhttps.pps.eosdis.nasa.gov`), typically available within a few
hours rather than days. It's a genuinely different system, not just a
flag on the same API:
- **Requires separate registration** at
  https://registration.pps.eosdis.nasa.gov/registration/, explicitly
  opting into NRT access — a normal Earthdata/GES DISC login alone won't
  work here.
- Different file naming (`.RT-H5` suffix) and a different, PPS-specific
  listing mechanism (`/text/...` endpoint with server-side wildcard
  matching, per NASA's own retrieval documentation).

New "GMI-NRT" option on the Real MW sensor dropdown, reusing your saved
Earthdata credentials (assuming your account has NRT access enabled — the
credential system is shared, only the access grant is separate). **This
is the one ingestion path in the project that hasn't been tested against
a live server** (no network access in the environment this was built in)
— the filename-parsing regex was verified against NASA's own documented
example filename, but the auth mechanism and listing response format are
implemented to spec rather than confirmed. If it errors, the message
should at least be specific about what failed (auth vs. no matching
files vs. download failure) rather than opaque.

## Color composites rewritten against real reference images

You provided two real GPM GMI color37/color89 images (cyclonicwx.com,
Tropical Storm Lala, 2026-08-14 2039Z). Sampling actual pixel colors from
them (not just eyeballing) showed the original composite implementation
was wrong in a more fundamental way than bad stretch ranges:

**89 GHz was never a 3-channel composite** — the real product is a
single-channel colormap. Sampled pixels formed a clean progression:
background/clear = teal-cyan with R pinned near a constant ~60, moderate
convection = pure `(R,G,0)` yellow-to-orange (B exactly 0, not just low),
intense core = pure `(R,0,0)` red. `mw_composites.py` now implements this
as `build_89_color_composite()`: PCT89 pushed through a 6-stop hand-
calibrated lookup table, with the actual RGB values at each stop taken
directly from sampled reference pixels.

**37 GHz's channel assignment was backwards.** R turned out to be exactly
zero across both open ocean *and* moderate rain in the reference image —
it only activates for the most extreme convective pixels — while G and B
rise together with rain intensity (B catching up to G faster, consistent
with H37/V37 polarization collapsing as rain thickens). That's not
compatible with three independently linear-stretched channels; R needed
to be a thresholded ice-scattering indicator instead. New mapping:
`R = scattering_index(PCT37)` (0 until PCT37 drops below ~245K, ramps to
full by ~180K), `G = stretch(V37)`, `B = stretch(H37)`. Checked by hand
against the sampled ocean/rain/core colors and landed close on all three
(see the module docstring for the exact numbers).

Both are visually validated against a synthetic test storm (green→cyan
for 37 GHz, teal→yellow→orange→red for 89 GHz) rather than just asserted
— the images are what real cyclonicwx.com output looks like now, not a
guess. The Kelvin breakpoints in `PCT89_COLOR_STOPS` are the one
remaining approximation (no source data file was available for the
reference image, only its rendered colors, so those breakpoints are
estimated from typical published PCT89 thresholds) — the *colors* at each
stop are measured, not guessed, which is the part that mattered most for
getting the look right.

## PPS registration for GMI-NRT — clarified, and a credential-routing bug fixed

Confirmed against current NASA documentation: **PPS is a completely
separate account system from Earthdata Login**, not an "authorize this
app" step the way the GES DISC EULA was. To get GMI-NRT working:

1. Register at https://registration.pps.eosdis.nasa.gov/registration/
   (separate from urs.earthdata.nasa.gov — registering with one doesn't
   register you with the other).
2. Check the "Near-Realtime Products" box on that form — easy to miss,
   and without it your account only gets the multi-day-latency research
   archive, not the fast `jsimpson` NRT server. Already registered without
   it? Use "Verify Email or Update Info" on the same site to add NRT
   access rather than re-registering.
3. Verify your email via the confirmation link PPS sends.
4. PPS has no separate password — your registered email serves as **both**
   the username and password for `jsimpsonhttps` access.

This exposed a real bug: the GUI's "GMI-NRT" option was reusing your
**Earthdata** credentials, which would never have worked since they're
different accounts. Fixed — the credentials tab now has a dedicated PPS
entry (a single "registered email" field, mirrored internally into both
the username and password slots to match how PPS actually authenticates),
and GMI-NRT now correctly reads from that instead of Earthdata's entry.

## GMI-NRT path/format corrected against a real, live-confirmed URL

You checked the actual PPS server directly and found the real structure,
which differs from the 2020-era documentation this code was originally
built against:

- **Directory is `/1CR/`, not `/1C/GMI/`.** Fixed.
- **File extension is `.RT-NC` (NetCDF), not `.RT-H5` (HDF5)** as the old
  docs described. The filename-timestamp regex still matches fine (it
  only looks for the embedded date/time, not the extension), but opening
  the file now goes through a new `_open_gmi_swath1_group()` helper that
  tries the expected `S1` group first and, if that fails, inspects the
  file directly via `netCDF4` and raises an error listing the actual
  groups/variables found — so if the NRT repackaging changed the internal
  structure too, you'll get a specific, actionable error instead of an
  opaque xarray `KeyError`.
- Added a fallback listing path: if the server-side wildcard-glob query
  (documented for the old path, unconfirmed at `/1CR/`) returns nothing,
  it now falls back to a full directory listing filtered client-side by
  date before giving up.
- Hardened the download-URL construction to handle three possible listing
  response formats (bare filename, path relative to `1CR/`, or already-
  absolute path) rather than assuming one — verified all three normalize
  to the correct URL against your real example filename.

## SSMIS-NRT and WSFM-NRT added

You found two more directories on the same PPS NRT server: `/1C/SSMIS/`
(solves the SSMIS latency problem flagged earlier — this is real
near-real-time SSMIS, not the ~1-month-latency NCEI archive) and
`/1C/WSFM/` (a bonus sensor not previously in scope — WSF-M/MWI, a newer
DoD satellite launched 2024).

**SSMIS-NRT**: channel layout confirmed directly against GES DISC's own
1C-SSMIS documentation (the same standardized swath convention already
verified working for GMI) — S2 = `(37V, 37H)`, S4 = `(91V, 91H)`. S2 and
S4 are separate swaths with potentially different native resolution/
geolocation (the same situation as AMSR2's 36.5/89 GHz grids), so this
uses `MWSwath`'s per-frequency `lat37/lon37` vs `lat89/lon89` fields the
same way. Filename parsing and URL construction verified directly against
your real example filename.

**WSFM-NRT**: MWI is genuinely new enough (2024 launch) that there's no
publicly confirmed channel-order reference for its 1C product the way
GMI/SSMIS have — it has several fully-polarimetric channels (V, H, plus
3rd/4th Stokes components at multiple frequencies) which makes guessing
particularly risky. Rather than hardcode a possibly-wrong index, this
function tries to auto-discover the 36.75/89 GHz V/H channel indices from
frequency metadata actually present in the downloaded file, and raises a
detailed diagnostic error (dumping what metadata and variables it did
find) if that fails, instead of silently proceeding on a guess. Verified
both paths work correctly: successful discovery against well-formed
metadata, and a clean refusal-with-diagnostics when metadata is absent.

Both new sensors reuse the shared PPS listing/download logic that GMI-NRT
already used (refactored out into `_pps_nrt_find_and_download()` so the
three don't duplicate that flow), and both route through the same PPS
credentials-tab entry as GMI-NRT, since it's the same account.

## NRT cache location, the "false permissions" bug, and TLE overpass prediction

Three fixes/additions from this round:

**Cache location**: NRT downloads (GMI-NRT, SSMIS-NRT, WSFM-NRT) now go to
`~/.synthetic_mw_tc/NRT/<GMI|SSMIS|WSFM>/` instead of `/tmp/mw_cache/...`
— `/tmp`'s permission and lifetime behavior varies by OS and isn't a great
place for files you actually want to keep and re-inspect.

**The "false permissions error"**: you correctly diagnosed the root cause
— `/1C/SSMIS/` and `/1C/WSFM/` are large FLAT folders (no date
subdirectories, unlike what was assumed), spanning the whole ~2-week NRT
retention window across multiple satellites. An unnarrowed wildcard/full
listing there is expensive enough that PPS apparently rejects it with a
permissions-flavored error, which the old code then misdiagnosed as "not
registered for NRT access." Real fix, not a workaround: **TLE-based
overpass prediction** (`tle_predict.py`, new module) now narrows every
PPS query to only the specific hour(s) — and for SSMIS, the specific
satellite (F16/F17/F18) — the real orbit actually passed near the target
location, using live TLEs from Celestrak and `skyfield` for propagation.
This avoids the expensive scan entirely rather than catching and working
around the error after the fact.

**TLE overpass prediction, using your NORAD IDs** (GMI 39574, AMSR2
38337, WSF-M 59483, SSMIS F16/F17/F18 28054/29522/36032): also solves the
original ask directly — distinguishing a genuine coverage miss from a
data-latency/access problem. If TLE prediction finds no overpass within a
sensor's approximate swath width of the target in the search window, the
fetch functions now skip the PPS query entirely (confirmed via test: zero
network calls made in that case). If every lookback tier still comes up
empty, `mw_ingest.diagnose_overpass_prediction()` runs as a final check
and appends either "TLE says a pass DID happen — likely a latency/access
problem" or "TLE says no pass happened — likely a genuine miss" to the
error message, so you're not left guessing. Swath half-widths are
approximate/published values for GMI/AMSR2/SSMIS; WSF-M/MWI's isn't
publicly confirmed and is assumed similar to AMSR2/SSMIS (flagged as an
assumption in the module docstring, same situation as its channel ID).

Also caught and fixed in testing: both `fetch_ssmis_swath_nrt` and
`fetch_wsfm_swath_nrt` imported `xarray` unconditionally at the top of
the function, before the new TLE early-return check — meaning the
early-skip-on-predicted-miss path (which doesn't need xarray at all)
would still crash with `ModuleNotFoundError` if xarray weren't installed.
Moved both imports to after the point where they're actually needed.

## PPS listing approach rewritten (404s), WSFM channel discovery fixed

**The 404 errors**: two things were wrong. First, a real bug — when
`name_filter` was empty (the GMI-NRT case), the wildcard pattern
`*{name_part}*{token}*` collapsed to a literal double-asterisk `**`,
which the server likely rejected as malformed. Second, and more
fundamentally: the "/text/" wildcard-query endpoint this all relied on
(from 2020-era PPS documentation) returned 404 consistently across
multiple different subdirectories and query patterns — strong evidence
it doesn't exist on the current server, not a one-off glitch. Rather than
patch around a possibly-dead endpoint, `_pps_nrt_find_and_download` now
fetches the real Apache-style directory listing directly (the same HTML
you get browsing `/1CR/?C=M;O=D` in a browser) and filters client-side —
verified against a realistic autoindex HTML snippet, correctly extracting
only real data files and excluding sort-header/parent-directory links.
Results are cached in-process for a few minutes so the expanding
lookback-tier search doesn't refetch the same large listing repeatedly.

**No date folders, confirmed**: nothing in the current code assumes date
subdirectories for any PPS path — flagged explicitly in
`_pps_nrt_find_and_download`'s docstring now so this doesn't regress.

**WSFM channel discovery, fixed with real evidence**: your file's actual
metadata showed `S1`'s `Tc` LongName reads "channels 1) 10.85 GHz V-Pol
and 2) 10.85 GHz H-Pol" — meaning S1 only holds the 10.85 GHz pair, not
all 17 channels as previously assumed. MWI's channels are split across
multiple swath groups, the same pattern GMI/SSMIS use. `_discover_wsfm_channels()`
now scans groups S1 through S8, parses each one's `Tc` LongName text for
frequency/polarization info (regex-matched against the exact real string
format your file produced), and identifies which group+index holds each
of 37V/37H/89V/89H — rather than assuming everything lives in one group.
Verified against your exact real LongName string, a full mocked 5-group
file structure (correctly locates 37/89 GHz in different groups from
10.85/18.85/23.8 GHz), and confirmed it still fails loudly with
diagnostic info (not a silent wrong guess) when a needed channel genuinely
isn't found anywhere.

## Day/night gating for Band 2 (solar.py)

New `solar.py` module: computes solar elevation angle at the storm's
lat/lon and scene time using the standard NOAA Solar Calculator formulas
(fractional-year-based declination + equation of time) — no ephemeris
file or network access needed, just math, so it's cheap to check before
deciding whether to even fetch Band 2. Validated against known reference
cases (near-90° elevation at solar noon on the equinox at the equator,
negative elevation at local midnight).

- `generate_synthetic_mw()` now checks day/night at `(storm_fix.lat,
  storm_fix.lon, band13.scene_time)` and discards `band2` if it's night,
  even if one was passed in — Band 2 at night is just near-zero-reflectance
  sensor noise, not signal, and using it would inject noise into the
  convective-signal blend rather than sharpen it. Both `is_daytime` and
  `band2_used` are recorded in the result's `diagnostics` dict.
- The GUI's Generate tab now checks day/night *before* fetching Band 2 at
  all, skipping the fetch entirely at night rather than downloading it and
  discarding it — saves a request and shows a clear log line either way.
  The plot title now shows "day"/"night" and whether Band 2 was used.
- Daytime is defined as solar elevation ≥ 5° (not just "sun above the
  horizon") since visible imagery quality degrades near the terminator —
  long slant paths, low-angle shadows/glint — well before actual sunset,
  so a small positive threshold avoids using marginal twilight scenes too.

## Compare tab — bridging synthetic and real MW (first step)

New `mw_compare.py` module and "Compare" GUI tab, delivering on the
"start bridging the gap" ask. What it does: runs the synthetic algorithm
and a real MW fetch (any sensor, including Local file) for the same
storm/time, regrids the real V-pol and PCT fields onto the synthetic
result's grid (nearest-neighbor, same technique used elsewhere), and
computes bias/RMSE between them. Displays a 2×2 grid (synthetic 37 GHz |
real 37 GHz composite, synthetic 89 GHz | real 89 GHz composite) with the
stats in the figure title. Tested against synthetic data on a deliberately
irregular/sheared grid (mimicking a real swath) to confirm the regridding
and stats work correctly.

Deliberately scoped as a starting point, not a finished calibration
pipeline: `synthetic_algorithm.py` outputs one scalar Tb per frequency
(not separate V/H), so there's no single obviously-correct real channel
to compare against. Both V-pol and PCT comparisons are computed rather
than silently picking one — V-pol is the more commonly cited
"representative" channel, PCT is the ice-scattering quantity the
synthetic algorithm's scattering-index logic is most directly trying to
approximate. The natural next step, once there's real bias/RMSE data from
actual storms, is feeding that back into tuning
`synthetic_algorithm.CALIBRATION` — that's a manual/iterative loop this
tab is meant to support, not automate away.

## Color composites rebuilt against an authoritative NHC source

You supplied the real source: Brennan & Cangialosi (NASA/NOAA NHC), WMO
RA-IV Workshop 2016 — https://severeweather.wmo.int/TCFW/RAIV_Workshop2016/07_Microwave_MichaelBrennan.pdf.
Slide 35 states the channel assignment directly and explicitly:

    37 color composite:    PCT (red), V (green), H (blue)
    85 color composite:    PCT (red), V (blue),  H (green)

This settles something my earlier pixel-sampling approach got wrong: **89
GHz uses the same 3-channel technique as 37 GHz**, not the single-channel
PCT colormap I'd built from limited evidence in one reference image. Also
important — V and H are **swapped between the two frequencies** (V=green/
H=blue at 37 GHz, but H=green/V=blue at 85/89 GHz), confirmed twice in
the source, not a typo. `mw_composites.py` is rewritten around this:

- `build_37_color_composite`: R=scattering-index(PCT37), G=V37, B=H37
- `build_89_color_composite`: R=scattering-index(PCT89), G=H89, B=V89
- R is a scattering-index transform of PCT (colder PCT → higher R), not a
  direct stretch — required to make "colder = more scattering = redder"
  work, since raw PCT numerically *decreases* with more scattering.

**On "too bright / lacking pink":** widened the 37 GHz R threshold
slightly, but conservatively — a wider first attempt (260K/200K) was
tested and rejected because it bled red into areas that earlier pixel-
sampled evidence confirmed should be *exactly* zero (open ocean and
moderate rain). Settled on a smaller widening (252K/190K), verified by
testing the full radial progression from storm edge to center: R stays
exactly 0 at 40-60km out and only appears where PCT has genuinely dropped
close to the center, consistent with the confirmed-zero constraint.
Worth being honest about a real possibility here too: the pass that
prompted this was Tropical Storm strength, not a major hurricane — it
may simply not have had ice scattering intense enough to show any pink at
all, which would be correct behavior, not a bug. If pink is still absent
on a genuinely intense hurricane pass, that's the stronger signal these
thresholds need more work.

The 89 GHz thresholds/stretch ranges are a first-guess starting point
(no pixel-sampled reference for this specific 3-channel scheme yet,
unlike 37 GHz) — the channel *assignment* is now authoritative, but the
exact Kelvin breakpoints will likely need tuning against a real pass.

## Synthetic composites now use the same color table as real data

`synthetic_algorithm.py` now produces separate `v37`/`h37`/`v89`/`h89`
fields alongside the original scalar `freq_37ghz`/`freq_89ghz` — computed
in parallel via new `CALIBRATION` entries (`bg_v_37`, `emission_boost_h37`,
etc.), using the same response/scattering fields the scalar already used,
just with per-channel baselines and rates (V starts warmer than H over
clear ocean; H rises faster with rain, converging with V; both depressed
similarly by ice scattering). The original scalar fields are **unchanged**
— `mw_compare.py`'s bias/RMSE stats still compare against them specifically.

New `mw_composites.composite_from_synthetic(result, freq)` mirrors
`composite_from_swath()` and calls the *exact same*
`build_37_color_composite`/`build_89_color_composite` functions used for
real data — so synthetic and real composites are colored identically, not
just similarly. Both the Generate tab and the Compare tab now render
synthetic output this way instead of a generic `turbo` scalar colormap.
Verified end-to-end: the synthetic V/H fields feed cleanly through the
real composite functions, and a synthetic test storm renders a sensible
green-ocean/orange-eyewall (37 GHz) and teal-ocean/dark-scattering-ring
(89 GHz) pattern — see the attached test render.

Worth being direct about calibration status: the V/H baseline constants
are first-guess estimates like the rest of `CALIBRATION`, not fit to real
data. This is exactly what the Compare tab (bias/RMSE against real V-pol)
is for going forward — now that both synthetic and real use the same
color logic, differences in the *pictures* should track differences in
the *numbers* mw_compare.py reports, which is the actual bridge asked for
a few rounds back.

## 37H / 89H single-channel products (new, precisely calibrated)

You offered a fallback given the composite-technique back-and-forth:
"use 37H and 89H instead, as those are not composites... and there ARE
known color tables for both." New `mw_h_channel.py` implements exactly
this — and unlike the RGB composites, the color tables aren't
approximated or reverse-engineered from limited evidence, they're
**extracted pixel-by-pixel** from your two reference images ("GPM GMI
37H/89H Brightness Temperature [K]").

Method: located each colorbar's tick-label rows by detecting text
pixels, confirmed the K-to-pixel mapping is exactly linear (37H: 6.0 px/K
across all 9 labeled ticks; 89H: ~5.4 px/K across all 8 — both verified,
not assumed), then densely sampled the colorbar column and converted
each sample's position to a Kelvin value via that confirmed mapping.
Caught and fixed one edge-of-image artifact (a stray black border pixel)
during extraction by cross-checking against the surrounding gradient.

Both extracted tables turned out to be legitimate, recognizable
meteorological color-table designs, not arbitrary gradients: 37H sweeps
smoothly from cream (coldest) through purple/blue/cyan/green/yellow/
orange/red to near-black (warmest); 89H goes from pale gray (warmest)
through navy/green/yellow/red to pure black at ~180K marking the most
intense ice scattering, then wraps to a distinct brown/orange band for
the even-colder tail beyond that — a real, known convention some
operational enhancement curves use to make the most extreme pixels stand
out rather than just going "more black."

New "Display" selector on both the Real MW and Compare tabs switches
between "Color composite (RGB, PCT/V/H)" and "H-channel only (37H/89H)"
for both frequencies. Works for real data (any sensor, including Local
file) and synthetic data alike, via `mw_h_channel.h_channel_from_swath()`
and direct calls on `SyntheticMWResult`'s `h37`/`h89` fields — both object
types expose the same attribute names, so the same functions work on either.

Verified end-to-end: LUT interpolation reproduces exact extracted colors
at the sampled control points, clamps correctly outside the data range,
and a full side-by-side render (composite vs. H-channel, both
frequencies) confirms the extracted 89H table's distinctive black-core/
orange-ring pattern renders exactly as expected from the source colorbar.

## Real MW and Compare tabs removed — ingestion moved to automatic background calibration

Per direct request: this project's focus is synthetic MW imagery, with
real MW ingestion as backing infrastructure, not a first-class feature of
its own. The **Real MW** and **Compare** tabs (and their `RealMWTab`,
`MWFetchWorker`, `CompareTab`, `CompareWorker` GUI classes) are removed.
`mw_ingest.py`, `mw_composites.py`, `mw_compare.py`, and `mw_h_channel.py`
are all still fully intact and functional — only their dedicated UI tabs
are gone.

**What replaced them:** the Generate tab now automatically ingests real
MW data and calibrates against it in the background, via a new
"Auto-ingest real MW + calibrate" checkbox (on by default). After running
the synthetic algorithm, `GenerateWorker._auto_calibrate()`:

1. Tries **GMI-NRT**, then **AMSR2**, then the **GMI archive** (in that
   order — recent/fast sensors first, the reliable-but-slow archive as a
   last resort), skipping any sensor with no credentials configured in
   the MW Data Credentials tab (no wasted network calls).
2. Search is bounded to the past 6h/12h only (not the full 6/9/12/24/48h
   "Extended" tiers the old Real MW tab offered) — keeps an ordinary
   Generate click from taking as long as a deliberate deep search would.
3. If a real pass is found, `mw_compare.apply_bias_calibration()` (new
   function) shifts the synthetic V/H/scalar fields by the mean bias
   against the real pass's V-pol data — a uniform additive correction
   (not spatially-varying, not a fit; the simplest defensible thing that
   could plausibly help). The applied bias is shown in the plot title.
4. Any failure at any point (missing credentials, no overpass, network
   error) is caught and logged, falling through to the next sensor or, if
   all three fail, to the plain uncalibrated synthetic result — this
   never blocks or crashes a Generate click.

Verified directly (mocking the network layer, since this sandbox has no
network access): sensor priority order is respected, sensors without
credentials are skipped without wasting a call, the loop stops at the
first successful ingest rather than continuing needlessly, and the
graceful "nothing found anywhere" fallback correctly returns an
uncalibrated result with a clear reason logged rather than crashing.

`mw_ingest.load_local_swath()` (the local-file failsafe) and the H-channel
color tables (`mw_h_channel.py`) remain available for direct Python use
even though their GUI toggles are gone with the removed tabs — reasonable
if you want to script something ad hoc, just not exposed as a button
anymore.

## Persistent calibration state — the missing link, now connected

Found `calibration_state.py` already present with a well-designed
persistent EMA-offset system (documented in its own docstring: why an
additive offset rather than refitting `CALIBRATION`'s dozen coupled
constants from one comparison, why EMA rather than overwrite). Its *read*
side was already wired into `generate_synthetic_mw()` — every generation
automatically applies the current persisted offset to `freq_37ghz`/
`freq_89ghz`/`v37`/`h37`/`v89`/`h89` alike (uniform per-frequency shift,
so PCT is unaffected — shifting V and H by the same amount shifts PCT by
that same amount too, keeping the V/H/PCT relationship internally
consistent). Verified end-to-end: setting a known offset and diffing
against a zero-offset baseline reproduces exactly the expected shift.

What was missing was the *write* side: nothing was actually calling
`calibration_state.update_offset()`, so the persisted state could never
change. `GenerateWorker._auto_calibrate()` now does both things when it
finds a real pass:

1. **Immediate, frame-specific correction** via `mw_compare.apply_bias_calibration()`
   (the one-shot approach) — the most accurate result for *this* frame,
   since it's computed directly against real data available right now.
2. **Persisted EMA update** via `calibration_state.update_offset()` — feeds
   the same bias into the running offset, so *future* Generate runs
   benefit too, even when no real data happens to be available for that
   specific frame (which is most of the time, given the bounded 6h/12h
   auto-calibration search window).

These two layers compound sensibly rather than conflicting: the persisted
offset provides a slow-learning baseline correction applied to literally
every generation; the one-shot correction, when real data is available,
applies on top of that for extra precision on that specific frame.

Added to the Generate tab: a status label showing the current persisted
offset and how many observations it's based on, and a "Reset persisted
calibration" button (no confirmation dialog — it's just a numeric offset,
not sensitive data) for if it ever drifts somewhere clearly wrong. Tested
directly: reset correctly zeros both the offset and observation count.

## NEXRAD radar added as an automatic ingest/calibration cross-check

New `nexrad_stations.py` (embedded station database, 163 real NEXRAD
sites parsed from your uploaded file) and `radar_ingest.py`.

**Station parsing**: the uploaded file is fixed-width; column boundaries
were taken directly from the file's own dash-underline header row (not
guessed), which correctly handles rows with missing fields (some entries
have no WBAN or state). Filtered to `STNTYPE == 'NEXRAD'` only — the file
also contains 47 TDWR (Terminal Doppler Weather Radar) stations, a
different network not part of the `unidata-nexrad-level2` archive, which
would have silently produced wrong station lookups if left in. Kept
non-CONUS sites relevant to tropical cyclones (Guam, Puerto Rico, South
Korea, Japan, Azores) rather than assuming CONUS-only.

**Range constraint, exactly as specified**: `find_nearest_station_in_range()`
returns `(None, None)` — not "the nearest station regardless of distance"
— when nothing is within 200 miles (default, adjustable) of the storm's
best-track center. Verified directly: a storm near Miami correctly finds
KAMX (~27mi); a storm in the open Pacific correctly finds nothing, rather
than returning some distant, useless CONUS station.

**Data source, verified against two authoritative sources** (Unidata's
own bucket-migration announcement plus the official awslabs/open-data-docs
README, not assumed): `s3://unidata-nexrad-level2/<Y>/<M>/<D>/<STATION>/
<STATION><YYYYMMDD>_<HHMMSS>_V06`. Confirmed this is the *current* bucket
name (renamed from `noaa-nexrad-level2` in 2025, format unchanged) and
that files after mid-2016 aren't gzip-compressed while older ones are —
`radar_ingest.py` handles both. Filename-timestamp parsing tested against
all four documented real example formats. Nearest-scan selection logic
tested against a realistic mocked S3 listing.

**Important caveat**: reading the actual Level 2 data uses Py-ART
(`arm_pyart`), the standard library for this — but Py-ART isn't installed
in the environment this was built in (no network to install/test it
live). The reflectivity-extraction code is written against Py-ART's
documented, long-stable API (`read_nexrad_archive`, `get_slice`,
`gate_latitude`/`gate_longitude`) but hasn't been run against a real
file. Same situation as GMI-NRT earlier in this project — written
correctly against the spec, first live run is the real test.

**Deliberately a diagnostic, not a Tb calibration**: radar reflectivity
and microwave brightness temperature are physically different
quantities with no direct conversion between them, so this doesn't
feed into `calibration_state.py`'s bias-correction pipeline the way real
MW passes do. It's a qualitative cross-check — "does radar show intense
convection where the synthetic algorithm's response field says it
should" — reported in the plot title and diagnostics
(`result.diagnostics["radar_check"]`), not a number that adjusts the
synthetic output. New "Auto-check nearby NEXRAD radar" checkbox on the
Generate tab (on by default), same non-blocking pattern as the MW
auto-calibration — any failure (out of range, no recent scan, Py-ART
missing) is caught and logged, never crashes a Generate click.

## Color tables replaced with the actual NRL GeoIPS production source

You found the real thing — NRLMMD-GEOIPS is NRL Monterey's actual
open-source repository for generating these products operationally, not
a third party's interpretation of them. `mw_composites.py` and
`mw_h_channel.py` are now direct, faithful ports of:
- `pmw_color37.py` / `pmw_color89.py` (RGB composite algorithms)
- `pmw_37pct.py` / `pmw_89pct.py` (standalone PCT algorithms)
- `cmap_37H.py` / `cmap_89H.py` (37H/89H colormaps)
- `cmap_37pct.py` / `cmap_89pct.py` (37pct/89pct colormaps)

This supersedes two earlier versions of this code — one that guessed at
a shared technique for both frequencies, one that extracted approximate
colors by sampling reference-image pixels. Both were reasonable given
what was available at the time, but this is the actual source now, not
an approximation of it.

**What was actually different from my best prior guess:**
- **89 GHz composite channel assignment confirmed exactly**: R=PCT-like
  (inverted), G=**H**, B=**V** — matching what the Brennan/Cangialosi NHC
  slides said in words, now with exact coefficients and ranges instead of
  inferred ones.
- **NRL uses genuinely different PCT coefficients for different purposes**,
  not one universal formula: standalone PCT37 = 2.15V−1.15H, but
  color37's internal red channel uses 2.181V−1.181H specifically; standalone
  PCT89 = 1.70V−0.70H (the original 1989 Spencer coefficient), while
  color89's red channel uses 1.818V−0.818H. Both pairs are preserved
  exactly as found, not reconciled into one number.
- **Exact display ranges**, which turned out to matter a lot: color37's
  red channel crops/normalizes over **[260,280]** (not the much colder,
  wider window I'd been using) — meaning scattering onset should register
  as red far more readily than my earlier version showed. color89's red
  channel uses **[220,310]**.

**A real bug this surfaced and fixed**: testing the exact algorithm
against the synthetic algorithm's existing clear-ocean V/H baselines
(190K/150K for 37 GHz, 245K/190K for 89 GHz) produced a jarring red/dark-red
"ocean" — those baselines fell outside NRL's expected clear-sky range for
the red channel to read near-zero. Retuned to 225K/155K (37 GHz) and
275K/250K (89 GHz), verified directly to render green (37 GHz) and dark
gray-teal (89 GHz) — matching the NRL slide deck's documented clear-sky
appearance ("sea surface = green" for 37 GHz; "relatively cloud-free =
gray or black" for 85/89 GHz). A full synthetic-storm render now shows a
believable green ocean with a subtle eyewall signature at 37 GHz, and the
classic red-ring/black-eye "donut" signature at 89 GHz that's the
hallmark of real intense-TC 89 GHz imagery.

**New capability**: 37pct/89pct standalone colorized products (using the
exact NRL cmap_37pct.py/cmap_89pct.py palettes) are now available in
`mw_h_channel.py` (`colorize_37pct`, `colorize_89pct`,
`pct_channel_from_swath`) — these didn't exist in the pixel-extraction
version since there was no reference image to sample colors from for
that specific channel. Not yet wired into the GUI (consistent with the
current Generate-tab-only scope), but available for direct use.

Verified throughout: exact hand-calculated formula match for both
composite functions, exact color match at colormap segment boundaries,
`mw_compare.py`'s real-vs-synthetic stats confirmed still working with
the updated PCT coefficients, and full visual renders of all six product
types (37H, 37pct, color37, 89H, 89pct, color89).

## Baseline tuning against a real NRL-Monterey reference for the same storm

You shared two real NRL-Monterey color37/color89 images (WSF-M/MWI pass,
actual AMSR2-family data) for the same storm (CP01 LALA) the app was
already tracking — genuinely useful, since it's a direct apples-to-apples
comparison rather than a different storm/sensor/time.

**37 GHz — darker green, done.** The previous fix (V=225,H=155 →
RGB(0,95,0)) still read as too bright next to the real product. Retuned
to V=208,H=125 → RGB(0,59,0), verified to keep a comfortable safety
margin (red_raw=306K vs. the 280K threshold where red starts bleeding
in) so it stays a clean, pure, darker green rather than picking up an
unwanted red tint.

**89 GHz — checked, not changed, here's why.** Before touching anything,
computed the current synthetic baseline (V=275,H=250) directly: it's
dark navy-slate `(41,42,63)`, not cyan, and a full response-gradient
check (baseline through max response) shows it moves directly toward red
with no cyan waypoint anywhere in between. So there wasn't a concrete bug
in the synthetic model's own baseline to fix. Worth noting plainly: the
real reference image itself shows a *very* cyan-dominant background across
most of the frame — that appears to be a genuine property of how real
V/H combinations render under the exact NRL formula for actual moist
tropical air, not something either version of this code was over-
producing. If the *actual program output* (as opposed to this hand
-verified baseline check) still shows more cyan than expected — especially
once real ingested V/H data is driving the auto-calibration bias rather
than the synthetic model's own parametric baseline — a screenshot of that
specific run would make it possible to calibrate precisely instead of
guessing further.

## Real bug found and fixed: unclamped calibration was saturating colors

Your screenshot's title bar was the key — it showed the persisted offset
(-31.9K/-15.7K) and a fresh one-shot correction (-39.0K/-19.2K) stacking
to a combined ~+71K (37 GHz) / ~+35K (89 GHz) shift. Verified by hand:
applying that full shift to the baseline lands at RGB(0,210,65) for 37 GHz
and RGB(0,190,255) for 89 GHz — both deep in the saturating tail of the
NRL color ranges, exactly matching "practically any non-red pixel was
saturated with cyan."

Two real things were going on, not one:

1. **`mw_compare.apply_bias_calibration` had no safety clamp.** The
   persisted offset in `calibration_state.py` was always capped at ±40K,
   but the fresh per-frame correction on top of it had no limit at all —
   it fully closed whatever gap remained, however large, in a single
   shot. Added a `max_shift_k=40.0` clamp (matching the persisted state's
   own cap), verified directly: an artificially extreme -125K measured
   gap now correctly clamps to -40K applied. The GUI now shows both the
   raw measured bias and the actually-applied (possibly clamped) bias
   separately, rather than implying the full measured gap was applied.

2. **The previous "darker green" baseline fix was chasing a picture
   instead of physics.** Real ingested AMSR2 data, across 7 separate
   comparisons for an actual storm, consistently measured a large gap
   between that baseline and reality — meaning matching a reference
   image's look had pushed the raw baseline further from what real data
   actually shows, not closer. Moved the baseline partway toward what the
   calibration data suggests (not all the way — this is one storm's
   history, not a validated dataset): V=250,H=170 for 37 GHz →
   RGB(0,148,18), a moderate green with visible texture, not the
   near-maxed RGB(0,210,65) that prompted this fix. 89 GHz similarly
   moved to V=283,H=260 → RGB(23,85,165), a genuine blue rather than
   saturated cyan.

**If you've run an earlier version**: click "Reset persisted calibration"
on the Generate tab once after updating — the old learned offset was
computed against the previous baseline and is no longer meaningful
relative to this one; leaving it in place would effectively double-apply
a correction that's already partly baked into the new baseline.

## Real bug found: 37 GHz structurally could never show pink/red

Your two screenshots were the key diagnostic — comparing them ruled out
calibration as the cause. Image 2 had auto-calibration **disabled**
(only a leftover 1-observation persisted offset of -10.4K/-5.1K) and
still showed the same flat green/cyan appearance as the heavily-
calibrated image 1. That pointed straight at the core synthetic V/H
generation, not the calibration layer.

Traced it exactly: the red channel is driven by `2.181×V − 1.181×H`
(weighted ~2x toward V), needing to drop from its baseline (~344K) below
280K to show any red at all, below 260K for full saturation. With the
previous **equal** depression on V and H (both -90K max), that weighted
quantity only drops ~58K even at a realistic storm's peak response
(~0.64) — never enough to cross 280K. **37 GHz could not show pink/red
at any storm intensity, structurally**, regardless of calibration —
exactly matching what both screenshots showed (a slightly darker green
blob, never red).

Fixed with **asymmetric** depression — V drops much more than H
(physically motivated: ice scattering pulls both toward a similar cold
floor, and V starts higher so it has further to fall) — retuned to
(110, 70), verified against the exact same realistic storm scene to
produce a proper graded transition: pure green until moderate response
(~0.4), a genuine partial-red zone (~0.45-0.5), full red only at the
most intense pixels (~0.55+). A full visual render now shows a clear red
core surrounded by green, matching the "sea surface green / deep
convection pink" pattern from both the NRL slides and the RAMMB
synthetic-imagery description you shared. Also verified this holds up
with a calibration shift stacked on top (the actual scenario from your
screenshots) — the core still shows a distinct signature rather than
being washed back to flat green.

89 GHz wasn't touched — its existing depression magnitudes (130, 150)
were already large enough to cross its own red-threshold at realistic
response levels, which is exactly why your screenshots showed a working
red core there but not at 37 GHz.

## Major architecture change: real weighted fusion of GOES + real MW + radar

Per direct request: the pipeline previously generated from GOES alone,
then applied real MW as an after-the-fact bias correction, then reported
radar as a separate diagnostic — three sequential, independent steps,
"building one onto another" rather than genuinely combining them. That's
almost certainly what was behind the "too flat, doesn't look like a real
MW pass" complaints from recent rounds, and it's also what caused the
saturation bug a few rounds back (a large correction applied on top of
an already-generated result, rather than real data properly informing
the generation itself).

**New design**: `generate_synthetic_mw()` now accepts `real_swath` and
gridded `radar_lat/radar_lon/radar_dbz` directly, and fuses all
available sources as **weighted equals per pixel** — not sequential
layering. Priority weights default to `{"goes": 0.1, "mw": 0.3, "radar":
0.6}` (radar > MW > GOES, exactly as requested), but critically these are
**renormalized per pixel based on which sources actually have data
there** — radar only covers a small area (default 150km radius, CONUS/
territory-only), real MW swaths cover a modest area (default 300km),
GOES covers the whole mesoscale sector. Where all three overlap, radar
dominates; where only MW+GOES overlap, MW dominates; where neither MW nor
radar reach, it's GOES alone — smoothly, not as a hard cutoff.

**How the fusion works physically**: real MW V/H data and radar
reflectivity are each converted into a comparable 0-1 "response index"
(how much scattering/convective signal is present) — real MW via the
same PCT-based scattering-index transform used elsewhere, radar via a
simple 20-55 dBZ normalization — then regridded onto the GOES analysis
grid (nearest-neighbor, with a hard distance mask so sparse real data
doesn't get incorrectly extrapolated across the whole sector). These
per-source response fields are combined via the new `_weighted_fuse()`,
and the FUSED response (not the GOES-only one) drives the same V/H
reconstruction formulas as before — so real/radar data now shapes the
storm's structure directly, not just its overall brightness level.

**Verified extensively**, since this touches the core generation path:
- `_weighted_fuse`: 4 explicit test cases (full overlap, one source
  missing, GOES-only, and spatially-partial radar coverage with correct
  per-pixel weight redistribution) — all exact matches to hand-computed
  expected values.
- `_regrid_external_with_mask`: confirmed a sparse source's coverage
  correctly stays localized (only ~3.5% of a test grid covered, matching
  the source cluster's actual footprint) rather than incorrectly
  extrapolating everywhere.
- Full end-to-end runs: GOES-only, GOES+real-MW, and GOES+real-MW+radar
  all produce sensible composite imagery (confirmed via rendered test
  images) with proper red/pink cores where a real depression signal
  exists.

**What changed in the GUI**: `GenerateWorker` now fetches real MW and
radar *before* calling the synthetic algorithm (previously: after).
`mw_compare.apply_bias_calibration`'s one-shot correction is **no longer
auto-applied** — fusion already incorporates real data more directly and
more precisely (per-pixel, not a single frame-wide shift), so that
mechanism is redundant and was also the direct cause of the earlier
saturation bug. The persisted `calibration_state` EMA baseline is
**kept** (still applied automatically to every generation, still updated
via measured residual bias after fusion) — it serves a different,
complementary purpose: a slow-learning correction for the GOES-only
baseline specifically, useful in the majority of a typical mesoscale
sector that no real MW/radar pass ever reaches. Checkboxes renamed
("Fuse real MW data into generation", "Fuse nearby NEXRAD radar") and
the plot title now reports actual fusion coverage percentages per source
instead of an "applied bias" figure that no longer describes what's
happening.

## Radar spoke artifact fixed, echo tops added, multi-frame loop + GIF export

**Radar spoke artifact (confirmed and fixed)**: traced the mechanism
directly from a real screenshot showing a storm 91mi from its radar —
computed the actual gap between adjacent NEXRAD rays at that range
(~2.5km), which exceeds typical GOES pixel spacing, explaining exactly
why nearest-neighbor regridding left visible unfilled gaps between rays.
`_regrid_external_with_mask` now supports a `method` parameter; radar
fusion uses `"linear"` (smoothly interpolating between rays, with a
nearest-neighbor fallback only at the outer edge where linear's
convex-hull limitation would otherwise leave gaps) instead of `"nearest"`
(hard cell boundaries). Real MW swath fusion stays on `"nearest"` (denser,
more uniform native sampling, no equivalent gap problem).

**Echo tops added** (`radar_ingest.get_echo_top_gates_near_storm`) — a
genuinely different convective-intensity signal from base reflectivity,
useful for overshooting tops/VHTs specifically. Reads *all* elevation
sweeps in the volume (not just the lowest, unlike base reflectivity),
since echo top is inherently a 3D-volume property — different sweeps hit
different lat/lon at the same range due to beam height, so this bins
gates meeting a dBZ threshold into a coarse lat/lon grid and takes the
genuine per-column maximum altitude (tested against known expected
values via the groupby-max logic). Combined with reflectivity in the
fusion via `np.fmax`, not averaging — either signal alone indicates real
intense convection, and averaging would dilute whichever one is actually
diagnostic at a given pixel (verified `np.fmax`'s NaN handling: uses
whichever source is present, only NaN when both are absent).
`fetch_radar_for_fusion()` now fetches both together, with echo-top
failures logged but non-fatal (reflectivity alone still useful). Also
caught and fixed a real bug during testing: the fusion diagnostics
weren't correctly reporting radar as "used" when only echo-top data (no
reflectivity) was present.

**Multi-frame loop + GIF export**: new "Frames" dropdown (Single/5/10/15),
generating frames at 10-minute intervals ending at the selected target
time (verified: a 6-frame request ending at 19:00 UTC correctly spans
18:10-19:00 in chronological order, matching your stated example).
Deliberately GOES-only for multi-frame runs — fetching real MW/radar
separately for every frame in a loop would be slow and often futile
anyway (their revisit times rarely line up with an arbitrary historical
10-minute step), so fusion stays exclusive to single-frame generation
where getting one frame as accurate as possible actually matters. New
frame slider scrubs through generated frames; "Export GIF..." button with
an FPS spinner (1-60) renders each frame and assembles them via Pillow
(verified end-to-end: correct frame count and correctly FPS-derived
per-frame duration in the resulting GIF file).

## Real fix: MW fusion now preserves actual texture, not just intensity level

Traced your exact complaint to a real architectural gap: even with real MW
data weighted ~29-75% within its coverage, it was only ever contributing a
*derived scalar index* (a single 0-1 "how much scattering" number per
pixel) that then got pushed back through the same smooth parametric
formula used for GOES-only generation. The real data's actual measured
Kelvin values — and all their genuine grainy, textured character — never
survived into the output, only a rough intensity nudge. That's exactly
"a new color table on a satellite image."

**Fixed**: V/H generation now builds three candidate VALUE fields (not
index fields) per frequency — GOES's parametric backbone, real MW's
*actual* regridded V/H values (radar still uses a modeled backbone, since
reflectivity isn't V/H) — and fuses them with the same per-pixel priority
weighting as before, but operating on real Kelvin values this time.
Verified directly: injected deliberately grainy synthetic noise into a
test MW pass and measured local pixel-to-pixel variance in the output —
initially ~4x higher than GOES-only (confirming texture actually
transfers through now), which settled to ~2x after a necessary follow-up
fix (below) while still clearly exceeding GOES-only's smoothness.

**A real bug this caught**: the above test also revealed an axis-aligned
stripe/moiré artifact — traced to nearest-neighbor regridding between two
*perfectly regular* grids at different resolutions (an artifact of the
synthetic test setup, not real satellite data, which is naturally
irregular/sheared). Confirmed this by re-testing with a properly-sized,
naturally sheared swath matching real GMI dimensions (~885km wide) — the
artifact disappeared, leaving genuine organic grain. Added a light
NaN-aware smoothing pass (`_nan_aware_light_smooth`, sigma=0.6 — much
lighter than GOES's own 1.2) on the MW value fields specifically as a
safety net regardless, since shipping code that *can* produce that
artifact under some real-world grid alignment wasn't acceptable even if
unlikely.

## Multi-frame loops now support MW/radar fusion, with threading for speed

Per direct request, accepting the speed cost: `GenerateLoopWorker` now
takes the same credentials/auto-calibrate/radar-check parameters as the
single-frame path, and fetches real MW + radar for *every* frame in a
loop, not just GOES. To make the added cost more bearable, frames
generate **concurrently** via `ThreadPoolExecutor` (up to 4 at once)
whenever fusion is enabled — each frame's fetch is largely independent,
I/O-bound work (multiple separate network round-trips once MW/radar
searches are involved), so threading gives a genuine speedup rather than
fighting Python's GIL. GOES-only loops stay sequential (already fast
enough, keeps progress messages in readable chronological order).

Shared fetch logic (`fetch_real_mw_for_fusion`, `fetch_radar_for_fusion_data`)
was refactored out of `GenerateWorker` into module-level functions with
no shared mutable state, specifically so they're safe to call
concurrently from multiple threads — verified the threading pattern
directly: out-of-order concurrent completion correctly reassembles into
chronological frame order (via a future-to-index mapping), and a single
frame's failure correctly propagates as an exception rather than being
silently swallowed or corrupting other frames' results.

One deliberate scope limit: the persisted `calibration_state` baseline is
**not** updated per-frame in multi-frame loops (unlike single-frame runs),
to avoid concurrent-write races on its JSON file from multiple threads.
Multi-frame fusion directly informs each frame's output; it just doesn't
feed the long-term learned baseline the way a single-frame run does.

## Three real bugs found and fixed from your LALA/Bertha comparison

**1. Spoke pattern reintroduced by my own earlier fix.** The "linear
interpolation for radar" fix from last round had a nearest-neighbor
fallback for points outside linear's convex hull — which reintroduced the
exact spoke artifact it was meant to eliminate, specifically in the
boundary zone between good coverage and the distance cutoff. That zone
can be large exactly where it matters most: at long range, or in genuine
beam-blocked sectors (PHWA specifically has real, well-documented
blockage from the Big Island's volcanic terrain — your instinct there was
right). Fixed by removing the fallback entirely: points outside the
convex hull are now honestly NaN ("no radar data here"), falling back to
whatever other sources are available, rather than a crude invented value.

**2. The "bubble" artifact in the Bertha run — a real design flaw, not a
rendering glitch.** Wherever radar happened to cover a pixel, its
priority weight (0.6) competed *directly* against real MW's actual
texture (0.3) in the final value blend — collapsing MW's real
contribution from 75% (when only GOES+MW were present) down to 30%
purely because radar also reached that pixel, even though radar has no
genuine texture of its own to justify outweighing real measured data.
That's exactly why you saw a smoother, differently-textured circular
patch with a crisp edge at radar's exact coverage radius. Fixed by
merging GOES+radar into one combined "backbone" *before* computing the
parametric V/H fields, so real MW's weight relative to that backbone
stays constant (0.3 vs 0.7) regardless of whether radar also happens to
cover a given pixel — radar still fully shapes the backbone's structure
wherever it has data, it just no longer separately out-competes MW's real
detail on top of that. Verified directly: re-ran the same
MW-covers-most-of-domain + radar-covers-small-circle scenario that
exposed the bug, and the circular seam is gone — uniform texture
character across the whole MW-covered area.

**3. Vertical striping — present in the raw GOES IR input itself, not a
fusion issue at all.** Traced this to `qc_utils.py`'s bad-pixel inpainting:
`fill_invalid_nearest` (nearest-neighbor fill via `distance_transform_edt`)
handles small/sparse gaps fine, but for a *wide* contiguous invalid
region — like a real GOES ABI dropped-detector stripe, a genuine sensor
characteristic — it splits the gap into two flat halves (nearest-from-
left, nearest-from-right) rather than blending, creating a visible
duplicated-column artifact. Confirmed directly: a 6-pixel-wide test gap
filled as two distinct flat values, not a gradient. Added
`fill_invalid_smooth()` (linear interpolation from surrounding valid
pixels in all directions, via `scipy.interpolate.griddata`, with a
nearest-neighbor fallback only at the genuine domain edge — a safe
fallback here, unlike the radar case, since GOES data is a dense regular
grid with well-defined neighbors on all sides). `sanitize_field()` now
uses this by default. Verified directly: the same 6-pixel gap now fills
with a smooth, genuinely monotonic gradient instead of a hard split.
`fill_invalid_nearest` is kept available for any caller that specifically
wants the cheaper behavior, but nothing currently uses it directly.

All three fixes verified end-to-end: a full pipeline run with a
deliberately-injected wide invalid stripe now completes cleanly with no
propagated NaN, and the radar/MW backbone-merge fix confirmed via a
realistic-scale re-run of the exact scenario that exposed the bubble
artifact.

## Big validation, plus one more radar refinement

Your comparison against the real GMI pass is genuinely the best evidence
this project has had — **without radar, the 89 GHz composite is
"practically spot on" and 37 GHz is "pretty close"** against real GPM GMI
imagery for the same storm, reportedly outperforming even RAMMB's
ML-based synthetic product. That validates the direct-value MW fusion,
the backbone-merge fix, and the retuned baselines from recent rounds —
genuinely good news.

**The remaining radar-specific issue, fixed**: with radar included, 37 GHz
showed a small isolated color anomaly not present in the no-radar run.
The radar site (PHMO) was 89mi/143km from the storm — right at the edge
of the 150km fusion radius, the least reliable part of its range (beam
diameter is measurably ~2.2km at that distance vs ~1km close-in — a
genuinely coarser, blurrier sample). With only 5% coverage, that
long-range data wasn't blending with anything else — it was an isolated
patch contributing at full strength right up to a hard cutoff.

Added `_regrid_confidence_taper`: radar's influence now fades gracefully
based on true distance from the **radar site itself** (not from the
sparse gate data, which would be circular) — full confidence within 60%
of the fusion radius, tapering linearly to zero at the radius itself.
Verified directly against the exact real scenario: at 143km the taper is
effectively 0 (that pixel now falls back cleanly to the GOES+MW
backbone instead of an isolated radar-driven spike), while close-range
radar data keeps full confidence, unaffected.

This required threading the radar station's actual coordinates through
the pipeline (`radar_ingest.fetch_radar_for_fusion` now includes
`station_lat`/`station_lon`, passed to `generate_synthetic_mw` via new
`radar_site_lat`/`radar_site_lon` parameters in both single-frame and
multi-frame paths). Fully backward compatible — calling without these
new parameters works exactly as before, just without the taper.

## Three more real bugs from a broader LALA/Bertha comparison

**1. MW's coverage edge had the same hard-seam problem radar had.** The
radar confidence taper from last round only covered radar's edge — real
MW swaths also have a hard distance cutoff with no feathering, and a real
Bertha run showed exactly the same symptom: a crisp, visible straight-line
boundary where MW-textured data met the pure-GOES backbone. Generalized
`_weighted_fuse` to accept per-pixel weight *arrays*, not just scalars
(tested both scalar backward-compatibility and the new array behavior
directly), then added `_edge_feather_taper` — distance-to-nearest-
real-data-point based (the *correct* metric for a swath, unlike radar's
site-distance taper: a swath has no single transmitter to measure range
from, what matters is local data density). MW's edge now fades gracefully
into the backbone instead of a hard seam — verified with a partial-
coverage swath scenario matching Bertha's ~26% coverage case.

**2. Switched MW's regridding to linear interpolation.** A Bertha run
with radar included showed diagonal striping in the 89 GHz panel — same
family of artifact as the radar spoke issue fixed earlier, just from MW's
own scan geometry this time rather than radar's angular rays. MW value
fields now use `method="linear"` (matching radar's fix) instead of
`"nearest"`, for the same reason: linear interpolation smoothly bridges
between source points instead of creating hard nearest-neighbor cell
boundaries that can alias into visible patterns.

**3. Isolated radar-gate "spikes" — a genuine QC gap, now fixed.** LALA's
radar run (PHMO, 89mi) showed a thin, isolated linear artifact even after
last round's distance-taper fix — the taper reduces a far-range reading's
*weight*, but doesn't touch the *underlying value* if it's a genuine
statistical outlier (ground clutter, anomalous propagation, or biological
scatterers like birds/insects are all real, well-documented sources of
isolated single-gate radar spikes with no meteorological meaning).
Added `_despeckle_gates`: for gates above 35 dBZ, checks whether at least
3 other gates exist within 3km — isolated readings with no spatial
support get discarded, exactly matching how real radar QC pipelines
distinguish weather from clutter/noise. Verified with two direct tests:
an injected isolated 58 dBZ spike (the exact scenario from the report) is
correctly removed, while a genuine spatially-coherent 15-gate convective
cluster at the same intensity is fully preserved — this doesn't touch
real weather, only genuinely unsupported single-point anomalies.

All three verified together in a full GOES+MW+radar integration test
combining every fix from this and recent rounds.

## Physical-bounds QC for real MW data, dead code cleanup, and a first step toward ML

**Physical-bounds QC (fixes the Chantal 2025 wedge artifact)**: a run
with NO radar involved at all — deliberately isolating the cause — showed
a sharp, geometrically coherent wedge of anomalous color in a real
swath's corner. This is a different failure mode from anything fixed so
far: not an isolated single-point spike (radar's despeckle filter), not
a hard coverage-edge seam (the feathering fix) — a *region* of
implausible values, most consistent with the known real phenomenon of
conically-scanning radiometers having degraded data quality at the
outermost edge of their scan (viewing geometry changes measurably near
a swath's edge). Added `MW_PHYSICAL_BOUNDS` + `_apply_physical_bounds`:
real MW V/H values outside per-channel physically-plausible Kelvin
ranges are masked before they ever reach regridding or the response-index
calculation, applied consistently everywhere real_swath data is used, not
just one code path. Verified directly: an injected implausible edge value
(480K) is correctly masked while normal values pass through untouched,
and a full end-to-end test with a corrupted swath corner (matching the
reported artifact's shape) confirms the wedge no longer appears — the
corner cleanly falls back to the GOES+MW backbone instead.

**Dead code found and removed**: while wiring the training-data export
below, discovered `GenerateWorker` had *two* `run()` methods — Python
silently uses the second, so the first was calling `self._auto_calibrate()`
and `self._auto_radar_check()`, methods that don't exist anywhere in the
class anymore (leftover from before the fusion-architecture rewrite
several rounds back). Never executed, so no behavioral impact, but real
clutter and risk — worth being upfront that I nearly edited the wrong
one before catching it. Removed.

**On "how partial microwave passes or misses can mess up the image"**:
this round's three bugs (a swath-edge QC gap, on top of last round's
edge-feathering and despeckling) are a direct illustration of exactly
that concern — heterogeneous partial-coverage data fusion keeps
surfacing new edge cases test-by-test. Worth being honest about this
rather than implying each fix is "the last one": hand-tuned fusion
heuristics are inherently reactive to whatever specific failure mode the
next real storm happens to expose.

**On the CNN/U-Net suggestion**: agreed this is a legitimate, quite
possibly better long-term direction — RAMMB's own synthetic MW product is
diffusion-model based, so there's real precedent for a learned approach
outperforming hand-crafted physical/fusion heuristics on exactly this
problem, including probably handling partial-coverage cases more
gracefully by learning spatially-varying trust implicitly rather than
needing an explicit taper/feather/despeckle/bounds-check for every new
failure mode discovered by hand. That said, actually training a U-Net
needs a real paired dataset (hundreds-to-thousands of GOES/real-MW
examples across many storms) and real training infrastructure (GPU
compute, a training loop, validation) that doesn't exist in this
environment — not something to fake or overpromise here.

What's actually implemented instead: `training_data_export.py`, a
genuinely useful, correctly-scoped first step. New "Save paired
GOES+real-MW training example" checkbox on the Generate tab (off by
default) — whenever a run actually has real MW data fused in, saves the
GOES input bands + the regridded, QC'd real MW V/H (already computed by
generate_synthetic_mw as part of fusion, now exposed via
`diagnostics["mw_regridded_v37"]` etc rather than duplicating that work)
as one `.npz` example. This means every normal use of the app with real
data available quietly builds toward an actual training dataset, without
requiring any workflow change or claiming a model exists yet. Includes
`dataset_summary()` to check accumulated inventory (example count, storm
coverage, date range) — useful for judging when there's "enough" to
actually attempt training. Verified end-to-end: correctly skips runs with
no real MW data, correctly saves paired examples when MW is present, and
the saved file's metadata matches the real StormFix used for that run.

## Major feature: MIMIC-TC-style morphing, combined with synthetic tracking

Per direct request: real MW data now behaves like MIMIC-TC (advecting the
most recent pass that actually observed the storm, filling gaps between
overpasses) *combined with* continuous synthetic GOES-driven tracking
(like RAMMB's product) and radar, all inside the same weighted-fusion
architecture already built — not three separate systems bolted together.

**"Most recent pass that actually HIT the storm"** (`mw_ingest.find_swath_that_hit_storm`):
previously, the nearest pass in time was used even if its swath only
barely clipped the search box without ever crossing the storm's actual
circulation center — exactly the "missed microwave pass" case you
flagged with Genevieve. Now searches progressively further back until it
finds a pass whose data comes within 75km of the storm's *own position
at that pass's observation time* (not the current time — the storm has
usually moved). A near-miss doesn't get treated as good data anymore.

**Advection/morphing** (`mw_ingest.morph_swath_to_time`): once a real hit
is found, its lat/lon grid is shifted by the storm's own displacement
between the pass's observation time and now — the actual MIMIC-TC
technique. Verified directly: a pass observed at the storm's old
position gets correctly repositioned to the storm's current position,
with the V/H data itself and the original `scene_time` left untouched.

**Age-based confidence decay** (`synthetic_algorithm._morph_age_confidence`):
advection captures storm translation, not genuine structural evolution
(intensification, eyewall replacement) — so an aging morphed pass
gradually cedes weight back to the GOES/radar backbone rather than being
trusted exactly as much as a fresh pass forever. Full confidence under
1.5h, decaying to a 0.15 floor by 6h (never literally zero — aging real
structure is still worth something). Verified end-to-end: a low-confidence
morphed pass pulls the fused result measurably closer to the pure
GOES-only baseline than a fresh pass does. The persisted calibration
baseline is guarded separately — only fresh/high-confidence passes
(≥0.5) feed it, so a stale morphed pass's measured bias doesn't pollute
the long-term learned offset, even though it's still fully used for that
specific frame's fusion.

**A real infinite-loop bug, found and fixed via direct testing, not left
for you to find**: the backward-search's "jump further back when stuck"
fallback was computed relative to the *returned swath's* scene_time —
which doesn't change if the same swath keeps coming back from a
barely-shifted search window, so the jump target recomputed itself
identically forever. A direct test with a mocked "most recent pass
misses, an earlier one hits" scenario caught this immediately (timed out
rather than returning). Fixed by computing the jump relative to the
search cursor itself instead, which is guaranteed to always make forward
progress. Re-verified with the same test afterward: 6 progress messages
instead of an infinite loop, correctly finds and morphs the real earlier
hit. Also verified the genuine "no hit exists anywhere" case still
terminates cleanly (hits the iteration safety cap, returns the best
available fallback with `was_hit=False`).

All three pieces (hit-finding, morphing, age-confidence) verified
together in one final integration test through the actual
`generate_synthetic_mw` call.

## GIF verified directly, and one more real bug found in its titles

Actually opened and inspected the uploaded GIF file itself (not just the
static comparison images) — confirmed correct: 10 frames, exactly 200ms
per frame (matching the requested 5fps), consistent 906×741 size
throughout.

**Found a real bug while inspecting individual frames**: every frame's
title showed a direct contradiction — "Fused: GOES + MW(cov 99%/99%)"
(accurate; real MW genuinely was fused into that frame) right next to
"Fusion/calibration off" (misleading; a hardcoded fallback string).
Traced it exactly: `GenerateLoopWorker` deliberately never touches
`calibration_state` per-frame (avoids concurrent-write races across
threads, a decision from the multi-threading round) — correct — but as a
side effect, it also never set `calibration_applied`/`calibration_reason`
in each frame's diagnostics at all, so the title code fell back to its
generic default text instead of describing what actually happened for
that frame. Fixed: loop-generated frames now explicitly report real MW
usage (sensor + confidence) when it happened, distinguishing "no pass
found" from "fusion disabled" — without changing the underlying
deliberate choice not to update the persisted baseline per-frame.
Verified directly against a mocked frame generation.

**On the with/without-radar LALA images**: both look meaningfully
cleaner than the versions from recent rounds — no visible spoke/wedge
artifacts in either panel. That's consistent with the taper, despeckle,
and edge-feathering fixes from the last two rounds doing what they were
meant to. Worth being appropriately calibrated about this, though: visual
absence of the *specific* artifact patterns already found and fixed is a
good sign, not proof there's nothing else there — the pattern this whole
stretch of rounds has followed is that each fix closes one real gap
without guaranteeing there isn't another.

## Major fix: the "moving MW patch" problem — backbone now carries real texture everywhere

This was a sharp architectural point, not a bug report: no amount of
edge-feathering can hide a boundary between a smooth region and a
textured region — the transition can be gradual, but the *character
mismatch* on either side of it is still obvious, which is exactly why
the real MW swath kept reading as a visibly distinct "patch" moving
across the frame even after the feathering fix. The real problem was
that the GOES-only "backbone" was fundamentally too smooth (a parametric
formula driven by one aggregated response scalar, then gaussian-blurred)
to ever look like real MW texture, no matter how the transition into it
was handled.

**The fix**: `_multispectral_texture_field()` extracts genuine fine-scale
spatial texture directly from the actual GOES imagery — high-pass
filtering each of band 13 (IR), band 9 (WV), band 7 (SWIR), and band 2
(VIS, when available) and combining them, capturing real small-scale
cloud structure (individual convective towers, banding gradients) that
the existing coarse `convective_signal` blend collapses away. This is
injected into the backbone's V/H fields *after* the main smoothing pass
(so it isn't immediately blurred back out), giving the backbone plausible
MW-like grain across the *entire* mesoscale sector — not just wherever
real data happens to reach. 89 GHz gets a larger injection than 37 GHz
(matching real passive microwave behavior — 89 GHz is more sensitive to
fine-scale scattering variability) and H-channel a modestly larger
injection than V within each frequency, consistent with H's greater
sensitivity elsewhere in this codebase.

This directly answers "not just band 13" — bands 9, 7, and 2 were already
feeding the coarse `convective_signal` calculation, but their genuine
fine-scale texture was being discarded in that aggregation; this recovers
it specifically for visual character rather than intensity.

**Verified with the actual scenario that matters**, not just an isolated
unit check: a real MW swath covering only the right half of a domain,
compared against the pure-backbone left half. Before this fix, the
MW-covered side's texture would have measured several times higher than
the backbone-only side (the visible "patch" effect). After: the ratio is
1.00 — statistically indistinguishable background texture on both sides,
confirmed both numerically and with a rendered image where the actual
swath boundary (marked with a reference line) is genuinely impossible to
locate from the surrounding texture alone. Only the storm's own core
shows any asymmetry, which reflects real structural differences between
the two data sources, not a smooth/grainy seam. Full regression suite
(GOES-only path, `mw_compare` scalar stats) confirmed unaffected.

Injection magnitudes (`TEXTURE_INJECTION_V37_K`, `_H37_K`, `_V89_K`,
`_H89_K`) are first-guess Kelvin scales, same "needs real-data
calibration" caveat as the rest of `CALIBRATION` — tune these if the
backbone ever reads as too smooth or too noisy once compared against
more real passes.

## Real texture injection: fixing the "MW pass overlaid on GOES" look at its root

Direct, well-articulated critique: even with edge-feathering (weight
tapering) fixing the *transition*, the two regions still looked visually
distinct, because the GOES-only backbone was a perfectly smooth
parametric model everywhere real MW data didn't happen to cover, while
real MW is naturally grainy. A soft weight transition between two
*statistically different-looking* things still reads as "layered," not
truly blended — you correctly identified that weighting alone couldn't
fix this, since it was never a transition-sharpness problem.

**Root cause, precisely identified**: `_texture_signal` (using GOES bands
7 and 9, both already fetched) already computes real convective texture
from actual satellite data — but only as a single 0-1 *scalar magnitude*
per pixel, which just modulates how strong the smooth parametric response
is. The actual spatial *pattern* of that real texture — where exactly the
fine structure sits, its shape, its granularity — was discarded entirely,
every single generation.

**Fixed**: new `_goes_texture_pattern()` extracts the genuine fine-scale
spatial pattern (high-pass filtering: raw band minus a smoothed version,
keeping real cloud-top structure — convective edges, banding) from bands
7 and 9. This real pattern is now injected directly into the backbone's
V/H fields across the **entire domain** (not just where MW happens to
cover), with amplitude scaling by local convective response (subtle over
clear background, matching how real ocean MW imagery isn't perfectly
flat either; much stronger near genuine convection). V and H draw from
overlapping-but-different blends of the two bands (70/30 vs 40/60), so
they're correlated like real V/H are, without being identical.

**Verified this actually addresses the complaint, not just the mechanism**:
1. Correlation test: synthetic output measurably correlates (r≈0.18) with
   a real fine-scale pattern injected into test GOES bands — confirming
   the texture is genuinely *derived from* real satellite structure, not
   independent random noise dressed up to look textured.
2. The actual seam test: built a scene with real MW covering only half
   the domain (the classic boundary scenario) and rendered the full
   composite. Compared to before this fix (where a smooth-left/
   grainy-right split would have been obvious), **no visible boundary is
   discernible at all** — both halves show consistent, comparable
   graininess. This is the direct visual evidence the fix does what was
   asked, not just a plausible-sounding mechanism.
3. Confirmed no regressions: `mw_compare.py`'s scalar-based bias/RMSE
   stats (which don't touch V/H texture at all) still compute correctly,
   full pipeline still returns all-finite output.

## The seam fix from last round didn't actually work on real data — here's why, and the real fix

Your real screenshots showed the exact same flat-left/textured-right
seam my last-round fix was supposed to eliminate, despite my own test
passing. Investigated rather than assuming the report was wrong.

**First discovery**: there was already a *better*, more complete texture
injection implementation sitting in the codebase (`_multispectral_texture_field`)
that I hadn't noticed and duplicated with my own simpler version — this
one already used all 4 GOES bands including VIS (band 2) when available,
directly matching "not just band 13" from two rounds ago. Removed my
redundant duplicate.

**The real bug, found by testing the actual failure condition instead of
my original artificial one**: my last-round test injected *uniform*
synthetic noise across the whole test domain, which accidentally masked
a real problem. Both texture implementations normalize by the
*whole-domain* magnitude (correct — it's what keeps real storm-scale
structure dominant where it genuinely exists) — but that means a
genuinely calm, smooth clear-sky region sitting in the same domain as an
intense storm collapses to near-zero relative texture. Confirmed
directly: a realistic combined test (intense storm + genuinely calm
far-field, matching actual GOES behavior for a real major hurricane in
open ocean) showed the calm region's normalized texture at ~80x smaller
than the storm region's. Real GOES IR/WV over calm ocean genuinely can be
almost perfectly smooth — but real MW imagery has *inherent*
footprint-to-footprint sensor noise regardless of the underlying weather,
which is exactly what was missing.

**Fixed** with a small, independent noise floor — spatially correlated at
roughly MW-footprint scale (not raw per-pixel white noise), deliberately
*not* derived from GOES's own local relative texture magnitude, so it
doesn't collapse the same way. Added on top of (not replacing) the real
GOES-derived texture, so genuine storm structure still dominates where it
exists; the floor only matters where GOES itself shows little.

**Verified against the actual failure condition, not a generic scene**:
built a realistic intense-hurricane test (pinhole eye, warm far-field,
matching the real Genevieve case that exposed this) and measured local
texture directly — far-field texture went from effectively flat to a
clearly non-trivial, consistent level, confirmed visually in a full
render with no discernible flat region anywhere across the domain.
Confirmed no regressions in `mw_compare.py` or the overall pipeline.

## Precisely isolating the remaining seam with your 3-way comparison

Your three-image comparison (GOES+realMW single frame, GOES+realMW loop,
GOES-only loop) was exactly the right test to isolate what was actually
left. The GOES-only loop confirmed the flat-region bug from last round
is genuinely fixed — consistent grain across the entire domain, no dead
spots, verified directly again as a regression check. But the two
real-MW frames still showed a faint residual boundary — a *different*,
more specific problem: not flat-vs-textured anymore, but a texture-STYLE
mismatch, because the injected synthetic texture used fixed guessed
Kelvin amplitudes (4-9K) that won't necessarily match what any given
real MW pass's actual texture amplitude looks like.

**Fixed**: new `_measure_real_texture_amplitude()` measures the ACTUAL
local texture amplitude directly from real MW data (high-pass filter +
std, per channel) whenever it's available, and uses that measured value
to scale the backbone's injected texture instead of the fixed guesses —
so the synthetic texture's amplitude is calibrated to match *this specific
storm's actual pass*, not a one-size-fits-all constant. Falls back to the
original fixed defaults when no real data is available (unaffected —
that's exactly the GOES-only path your image 3 already confirmed works).

**Verified this actually adapts, not just theoretically should**: ran the
same storm through two real MW datasets with deliberately different
texture amplitudes (low-noise vs. high-noise), and measured the resulting
*backbone's* texture far from the real MW coverage in both cases — the
high-amplitude case produced ~6x more backbone texture than the
low-amplitude case, confirming the calibration genuinely tracks the real
data rather than converging on some fixed value regardless of input.
Also re-confirmed the GOES-only path is completely unaffected (still
shows the same non-trivial far-field texture as before this change).

## Baseline color calibration — the 89 GHz seam was "tolerable," but 37 GHz needed the same treatment for color, not just texture

Direct, specific feedback: with the texture-amplitude fix in place, 89
GHz's residual seam was acceptable, but 37 GHz still showed a real
mismatch — GOES-only regions rendered a noticeably more saturated, vivid
green than the muted tone actually seen where real MW was blended in.
This is a genuinely different problem from texture: not local variance,
but the baseline Kelvin *level* the color composite starts from.

**Fixed the same way as texture, applied to the right thing this time**:
new `_measure_real_baseline_value()` measures real MW data's typical
background level directly (median, robust to the storm core being a
small fraction of a wide-coverage swath) and shifts the backbone's
baseline to match — instead of relying purely on the fixed guessed
`CALIBRATION` constants (`bg_v_37=250` etc). Applied as a clamped
correction (±20K) *before* texture injection, so it shifts the whole
backbone's color character, not just individual pixels.

**Deliberately clamped, learning from the earlier saturation incident**:
an unclamped baseline shift is exactly the mechanism that caused a real
saturation bug several rounds back (a large one-shot correction pushed
V/H past the color table's usable range). This clamp is more generous
than `calibration_state`'s existing V/H clamp (±5K, tuned very
conservatively after that incident) since this is a direct per-pass
measurement rather than a compounding EMA correction — but still bounded.
Verified directly: fed in a deliberately extreme (100K off) real
measurement and confirmed the combined shift (this clamp + the existing
persisted-offset clamp) stayed safely bounded (~22K total), nowhere near
the ~70K compounding that caused the original bug.

**Verified this actually shifts the right thing**: ran the same storm
with real MW data whose background was deliberately 20K cooler than the
assumed baseline, and measured the backbone's color *far from any real MW
coverage* — it shifted from `(0,150,19)` to `(0,104,0)`, tracking the
real measurement (230.5K measured vs. 230.0K actual) almost exactly.
Confirmed the GOES-only path (no real data at all) is completely
unaffected — stays at the original fixed baseline, exactly as before.

## Band 2 slowness likely explained (and fixed), plus SSMIS/WSFM added to auto-fusion

**Band 2 slowness**: traced this to a real, physical explanation rather
than a code hang — GOES Band 2 (visible) is natively 0.5km resolution,
4x finer per dimension than bands 7/9/13 (2km), meaning up to 16x more
pixels for the same physical area. That's genuinely far more data to
download and process (fixed-grid lat/lon computation, QC masking, all
scale with pixel count) — and critically, all of that extra detail gets
thrown away anyway, since `synthetic_algorithm.py` regrids band2 down
onto band13's coarser grid before ever using it. Added a downsample step
(`xarray`'s `coarsen()`, block-mean by 4x4) applied immediately after
loading band2, before the expensive per-pixel operations, cutting its
processing cost substantially for no loss of usable detail. **Important
caveat**: no network access in this environment means I couldn't install
xarray or test this against a real file — wrapped in a try/except
specifically so this optimization can't become a *new* way for band2 to
fail; if `coarsen()` doesn't behave as expected against a real file's
actual structure, it falls back to the original (slower, but confirmed
working) full-resolution path rather than raising.

**SSMIS-NRT and WSFM-NRT added to automatic fusion**: these were already
fully implemented (PPS NRT ingestion, TLE-based overpass prediction, the
same infrastructure GMI-NRT/AMSR2 use) but never actually wired into the
`AUTO_CALIBRATE_SENSOR_ORDER` list either worker searches — a real gap,
not something needing new infrastructure. Now tried in the order
GMI-NRT → SSMIS-NRT → AMSR2 → WSFM-NRT → GMI (archive, slow fallback),
with SSMIS-NRT and WSFM-NRT correctly routed to the same `pps_nrt`
credential as GMI-NRT (all three are PPS NRT feeds). Verified directly:
mocked a search where nothing is found anywhere, and confirmed all four
NRT-capable sensors get tried in exactly the intended order before
falling through to the archive.

## MW fusion is no longer optional — it's the default, not a toggle

Direct architectural feedback, and a fair point: a "GOES only" mode
undermines what this tool is actually for. Removed the "Fuse real MW
data into generation" checkbox entirely — real MW fusion is now always
attempted on every Generate click and every frame of a loop, replaced
with an informational label explaining what happens automatically
(GMI-NRT → SSMIS-NRT → AMSR2 → WSFM-NRT → GMI archive) and that it's not
a toggle by design. NEXRAD radar stays as the one remaining optional
checkbox, since it's genuinely more situational (CONUS/territory-limited
coverage, not always relevant).

The graceful fallback behavior is unchanged and still does the real work
here: if no real MW pass is found (no credentials configured, genuine
network failure, or a true coverage gap), generation still proceeds
GOES-only automatically — that fallback was always there for robustness.
The difference is it's no longer something a user can *deliberately*
select; it's what happens automatically when real data genuinely isn't
available, not a normal, equally-weighted option sitting next to it.

On the remaining 37 GHz baseline color difference between the MW-covered
and backbone-only regions in a single frame: agreed this is plausibly an
inherent data characteristic rather than something to keep chasing —
real MW baseline genuinely varies sub-regionally (humidity, wind speed,
sea state) in ways a single measured shift can't fully replicate, and
last round's fix already closes the gap as much as a single scalar
correction reasonably can.

Verified the removal is complete: confirmed via direct source inspection
that `GenerateTab` no longer references the removed checkbox anywhere,
and that both the single-frame and loop generation paths correctly
hardcode `auto_calibrate = True` rather than reading from a widget that
no longer exists.

## NEXRAD station fix: KLIX removed (physically relocated and renamed to KHDC)

Per direct correction: the radar formerly known as KLIX (New Orleans) was
physically relocated component-by-component and renamed KHDC. `KHDC`
was already present in the station list at the exact coordinates given
(30.5193°N, 90.4074°W) — the actual bug was that the old `KLIX` entry was
*also* still present at its former location, so a storm near New Orleans
could incorrectly resolve to the defunct `KLIX` callsign and fail to find
real data (the S3 archive uses the current station identifier, not the
historical one) — exactly matching the reported Bertha issue. Removed
`KLIX` entirely. Verified directly: 162 stations now (was 163), and a
storm near the old KLIX location correctly resolves to KHDC instead.

## Confirmed requests-based (not boto3) for PPS NRT, fixed a real GMI-NRT path bug, added AMSR3

**On the boto3 question**: checked directly — `mw_ingest.py`'s PPS NRT
code (`_pps_html_listing`, `_pps_nrt_find_and_download`) already uses
`requests` with HTTP Basic Auth against the real Apache directory listing
HTML, exactly as it should for this non-AWS server. Confirmed zero
`boto3`/`botocore` references anywhere in the file. Good instinct to
double-check, but this part was already correct.

**A real bug found while verifying, though**: GMI-NRT was the only one
of the four PPS NRT sensors NOT following the confirmed "/1C/<SENSOR>/"
pattern — it used a special-cased "/1CR/" path while SSMIS, WSFM (and
now AMSR2, per your reminder links) all consistently use "/1C/SSMIS/",
"/1C/WSFM/", etc. Fixed to try "1C/GMI" first (matching the confirmed
live pattern), falling back to "1CR" if that finds nothing, for safety
in case the older path is still needed for some date ranges. Verified
directly: the fallback logic tries the new path first and correctly
falls through to the old one when needed.

**AMSR3-NRT added**, per the newly-created `/1C/AMSR3/` directory.
Important, direct caveat: unlike GMI/SSMIS/WSFM (all confirmed against
real downloaded files at some point in this project), AMSR3's actual
file structure has never been seen here — no real file has been
available to inspect, and I don't have a confirmed NORAD ID for its
satellite platform either. Rather than guess at a fixed channel layout
(risking silently mislabeling data), this reuses the exact same
channel-discovery mechanism built for WSFM (`_discover_wsfm_channels` —
genuinely generic despite its name, since it just scans groups for Tc
variables with frequency metadata near 37/89 GHz, the standard
GPM-constellation convention, not anything WSFM-specific). If AMSR3's
real file doesn't follow that convention, this fails with a clear
diagnostic (every group and LongName it found) rather than silently
returning wrong data. TLE prediction simply fails closed for AMSR3 (same
graceful fallback used for any sensor without a known NORAD ID) until
one is added. Added to the automatic fusion sensor priority list
(GMI-NRT → SSMIS-NRT → AMSR2 → AMSR3-NRT → WSFM-NRT → GMI archive) and
routed to the same shared `pps_nrt` credential as the other three NRT
feeds — verified directly that it's now actually tried in the intended
order.

## A real structural bug found via direct real-data comparison, plus a serious pre-existing bug caught while fixing it

**The core complaint, traced mathematically, not guessed at**: GOES-only
37 GHz output showed almost no cyan or pink — just green background and
red core, missing the rich intermediate progression real AMSR2/GMI
imagery (and the program's own MW-blended regions) clearly show. Traced
this to the actual formula: `scat_pot` (which gates the emission-vs-
depression split in the V/H formula) is the *same* signal that drives
overall response magnitude (`convective_signal` is ~85% built from
`scat_pot` terms) — so there's no regime where "moderate rain is
happening" without scattering also being elevated, which suppresses the
emission term almost exactly where it would otherwise matter. Confirmed
by hand-tracing the formula across the full response range: H37's
B-channel contribution (needed for cyan) never exceeded ~39/255
anywhere — cyan was structurally unreachable, not just poorly tuned.

**Fixed** by decoupling the two roles: added `vh_scatter_warm_bound`
(230K), a separate, much colder-onset threshold used only for the V/H
emission/depression split (`scat_pot_vh`), while `convective_signal`
keeps using the original wider-range threshold unchanged. This lets
moderate response (real warm-rain/rainband regions, common and
physically real) stay emission-dominant and sustain genuine H-warming,
while only the most extreme, coldest pixels trigger real depression.
Verified mathematically before implementing (peak B-channel contribution
rose from ~39/255 to ~92/255 across a sustained range, not a narrow
spike) and confirmed the scalar `tb37`/`tb89` path (used by
`mw_compare.py`'s stats) is deliberately untouched.

**A serious pre-existing bug found while verifying this**: initial
testing showed physically impossible negative-Kelvin V/H values (H37 as
low as -386K). Traced to `_multispectral_texture_field` (from an earlier
round) — its normalization uses the 95th-*percentile* magnitude as scale
(deliberately, to be robust to a few extreme pixels), but nothing
afterward actually clips the output, so a genuinely sharp gradient (the
storm's own eye, exactly where structure is sharpest) produced a texture
value of -95 — 95x the intended "roughly unit-scaled" range — which
silently produced impossible negative-Kelvin output once multiplied by
the injection amplitude. This has apparently been sitting in the
codebase since that round, hidden because final RGB rendering clips to
0-255, so an extreme-but-clipped pixel just looked like ordinary
saturated red rather than obviously broken. Added a hard clip
(`TEXTURE_FIELD_CLIP = 3.0`) after the percentile-based normalization.
Verified directly: the same test scene that previously produced -386K
now stays in a physically sane range (H37 min 151K, no negatives
anywhere).

**Honest assessment of where this leaves things**: re-rendered a
corrected, more realistic test scene (a genuinely wide rainband region,
not just a narrow ring, since the first test scene turned out to be too
narrow to properly exercise the fix) and confirmed a real, visible cyan
ring now appears around the core where before there was only a hard
green-to-red boundary — a measurable, verified improvement to a
genuinely broken mechanism. That said, it's still a fairly thin
transitional zone compared to the broad, extensive cyan bands in the
real reference imagery — this is a real structural fix to a confirmed
root cause, not a claim that the output now fully matches real MW
imagery's richness. Further tuning of `vh_scatter_warm_bound` or the
overall response shape may still be warranted once compared against
more real passes.

## GMI-NRT and SSMIS-NRT fixed — brought in line with what actually works

Directly actionable signal: WSFM-NRT and AMSR3-NRT (both using dynamic
channel discovery) successfully retrieve real data; GMI-NRT and
SSMIS-NRT (both using hardcoded group names + fixed channel indices,
based on the GES DISC *archive* convention) didn't. Since all four share
the same underlying listing/download mechanism (`_pps_nrt_find_and_download`),
a difference in success could only come from something downstream of
that — the channel-extraction logic, which is exactly where the two
approaches diverged.

**Rewrote both to use the same dynamic discovery mechanism** as
WSFM-NRT/AMSR3-NRT (`_discover_wsfm_channels` — generic despite the
name), replacing the hardcoded `group="S1"`/fixed-index assumptions.
Both hardcoded versions' own docstrings already acknowledged the risk
("the NRT '.RT-NC' repackaging may not match the GES DISC archive
convention this code assumes") — this replaces that guess with the same
scan-and-parse-LongName-metadata approach already confirmed working for
two of the four sensors, rather than trying to guess a *different* fixed
structure.

**Found and fixed a second, compounding bug while doing this**: the
discovery function's frequency-matching tolerance for the "89 GHz"
channel family was `< 2.0` (covering 87-91 GHz) — but SSMIS's actual
channel sits at 91.665 GHz, just outside that window by 0.665 GHz. This
project's own `PCT_THETA[89]` coefficient is already documented as
covering the broader 85-91 GHz sensor family, not literally 89.0 GHz —
the discovery tolerance just hadn't been widened to match that existing
acknowledgment. Widened to `< 4.0` (85-93 GHz), verified directly against
SSMIS's real channel frequency (91.665 GHz now correctly matches) without
over-widening into unrelated frequencies (confirmed 183 GHz WV channels
still don't match).

**Scope check**: confirmed the separate GES DISC/NCEI *archive* paths
(`fetch_gmi_swath`, `fetch_ssmis_swath` — no "-NRT" suffix, different
latency/product) still correctly use their own hardcoded structure and
were deliberately left untouched, since the archive convention is a
different, more stable product than the NRT repackaging that was
actually reported broken. Also confirmed AMSR2 uses a separate JAXA
G-Portal path entirely, not PPS NRT, so it wasn't affected by or
relevant to this specific bug.

Verified end-to-end: the LongName-parsing plus label-matching pipeline
correctly identifies SSMIS's real 91.665 GHz channels as the 89V/89H
equivalents needed for the fusion pipeline.

## Regression fixed: isolated the SSMIS fix instead of a shared change that broke WSFM/AMSR3

Important correction, and the right instinct: WSFM and AMSR3 (confirmed
working previously) stopped working after last round's fix — because
that fix widened `_discover_wsfm_channels`' frequency-matching tolerance
as a *shared* change (2.0→4.0 GHz) affecting every sensor that calls it,
not just SSMIS (the one that actually needed it). That's exactly the
kind of change that can silently break something working: if WSFM or
AMSR3 has some other channel within the newly-widened window, it could
get matched first (this function keeps only the first match per label),
silently replacing a previously-correct channel with a wrong one.

**Fixed by scoping the change properly**: `_discover_wsfm_channels` now
takes `target_89_ghz`/`tolerance_89` parameters, defaulting to the
*original* (89.0, 2.0) values — restoring GMI-NRT, WSFM-NRT, and
AMSR3-NRT to exactly the behavior that was confirmed working before.
Only `fetch_ssmis_swath_nrt`'s own call site now passes SSMIS's specific
real frequency (91.665 GHz) with a narrow ±1.0 GHz window around it —
precise to what SSMIS actually needs, instead of a blanket widening that
risked every other sensor. Verified directly: SSMIS's real frequency
does NOT match the restored default window (confirming GMI/WSFM/AMSR3
are protected), and does match its own targeted window.

**On GMI 1C vs. 1CR**: this is a genuinely useful scientific point, not
just a path/directory question. 1CR specifically intercalibrates and
co-registers GMI's low-frequency channels (10-89 GHz) with its
high-frequency channels (166-183 GHz) — resolving a footprint-size
mismatch between those two groups. Since this project only ever uses 37
and 89 GHz (both already within the low-frequency group), 1CR's specific
benefit doesn't actually apply to anything used here — plain 1C is the
more directly appropriate product, not just a fallback-of-convenience.
Updated the docstring to reflect this reasoning; 1CR stays as a fallback
for resilience only, not because it's expected to matter for this
project's channel selection.

**Honest caveat, same as always**: I have no way to run this against the
live PPS server from this environment. This is my best diagnosis given
the specific WSFM/AMSR3-worked-then-broke pattern reported, verified
against the actual matching logic directly — but the real test is
whether AMSR3/WSFM come back and SSMIS starts working on an actual run.

## SSMIS: implemented PPS's own documented text/wildcard endpoint, revisiting an earlier assumption

The shared PDF (PPS's own "Accessing the PPS Near Real Time Data using
HTTPS" guide) directly contradicted a note already sitting in this
project's code: an earlier round concluded the documented `/text/`
wildcard endpoint "returns 404 consistently" and switched to parsing the
full HTML directory listing instead. The PDF shows a concrete, working
example against `/text/1C/*/*20200605*S12*` — including real matched
results from `/1C/SSMIS/` specifically, the exact directory reported
still failing after the last round's fix. Revisited that old conclusion
rather than trusting it — the most likely explanation is a URL/wildcard
construction difference in whatever was tried before, not that the
endpoint doesn't exist.

**Implemented `_pps_text_wildcard_listing`**, following the PDF's
documented pattern exactly: query `/text/1C/<SENSOR>/*<name>*<date>-S<hour>*`
and get back matching full paths as plain text, one per line — with the
*server* doing the filtering, rather than fetching and parsing a
potentially enormous full directory listing client-side. This matters
specifically for SSMIS: its directory is already documented elsewhere in
this project as an unusually large, flat folder spanning the whole
~2-week NRT retention window across three satellites — exactly the kind
of directory a full-listing approach would struggle with at scale, while
WSFM/AMSR3/GMI's presumably smaller or better-organized listings don't
have the same problem.

**Scoped to SSMIS only, learning from last round's mistake**: added
`use_wildcard_text_query` as an opt-in parameter, defaulting to `False`
— so GMI-NRT, WSFM-NRT, and AMSR3-NRT (all confirmed working with the
existing full-listing approach) are completely unaffected. Only
`fetch_ssmis_swath_nrt` passes `use_wildcard_text_query=True`. If the
wildcard query fails for any reason (network error, unexpected 404,
empty result), the code falls straight through to the original
full-listing approach as a safety net — this is additive, not a
replacement that could itself become a new single point of failure.

Verified the actual mechanics directly: the constructed wildcard pattern
matches the PDF's documented shape, and fed the PDF's own exact example
response text through the full parsing pipeline (path-stripping,
filename regex, timestamp extraction) end-to-end successfully.

**On the latency figures provided** (AMSR2 ~22-24min, AMSR3 ~20-23min,
GMI ~5-8min, SSMIS ~21-32min after pass time, WSFM in ~2hr batches with
shared server "last modified" timestamps but ~10min-apart embedded scene
times): confirmed directly that scene-time extraction already comes
entirely from each file's own embedded S/E timestamp in its filename,
never from server "last modified" metadata — which matters specifically
given WSFM's described batching behavior (many files sharing one
last-modified stamp would otherwise be indistinguishable by arrival
time alone). No change needed there; noted for future reference in case
lookback-window tuning is warranted once real data confirms SSMIS.

**Same honest caveat as always**: no live network access here to
actually confirm this against the real server. This is the best-founded
fix I can make given the PDF's concrete documentation and direct testing
of the parsing logic — the real test is whether SSMIS data starts coming
through on an actual run, without WSFM/AMSR3/GMI regressing again.

## Verified against real filenames, and a new NRT cache-clearing feature

**Checked the real filename examples directly against the code** — good
news: no bugs found. The extension mismatch was worth checking carefully
(the 2020 PPS PDF showed `.RT-H5` examples, but your real files are all
`.RT-NC`) — confirmed the listing/matching logic doesn't filter by
extension at all, so this was never at risk of silently excluding
anything. Ran the regex parsing against all 7 real filenames you
provided and it matched every one correctly, including confirming the
exact duration pattern you described: GMI ~5.0 min, WSFM ~8.7 min,
AMSR2/AMSR3/SSMIS ~98-112 min (essentially full-orbit swaths). The
existing crop-to-box-around-the-storm step already handles "load the
full swath, most of it ends up off-screen" correctly for all four —
that was already the design, not something needing a fix.

**New feature: NRT cache clearing**, on the MW Data Credentials tab.
`mw_ingest.clear_nrt_cache()` deletes everything under
`~/.synthetic_mw_tc/NRT/` and reports files-deleted/MB-freed — verified
directly against a real filesystem test (correct counts, correctly
clears the in-process listing cache too, handles a not-yet-existing
cache directory gracefully). The button shows a confirmation dialog
with a "Don't ask me again" checkbox; checking it persists to
`~/.synthetic_mw_tc/nrt_cache_prefs.json` and skips the dialog on future
clicks (verified the persistence logic directly: defaults to always-
confirm, correctly remembers the choice once made).

Also fixed a stale code comment (from several rounds back) that
described the `/text/` wildcard endpoint as this project's primary PPS
NRT listing method — it's actually now the exception, used only for
SSMIS; GMI/WSFM/AMSR3 use the full-listing HTML approach. Updated to
accurately describe the current state rather than an outdated one.

## The real explanation: TLE narrowing itself, not a filename/XCAL difference

Your instinct that WSFM/AMSR3 and the others "shouldn't be different" was
right — but the actual mechanism turned out to be more interesting than
the XCAL version string. Checked directly: `NORAD_IDS` (used for
TLE-based overpass prediction, which narrows the PPS query before it's
even sent) has **no entry for AMSR3 at all**. Since the lookup is a bare
`NORAD_IDS[sensor_key]` with no fallback, this raises a `KeyError` —
caught by the surrounding `try/except`, silently falling back to a
broad, unnarrowed search. AMSR3 has been "working" by accident: it never
actually uses TLE narrowing at all.

GMI and SSMIS, by contrast, **do** have valid NORAD IDs, so their TLE
geometry math actually runs. If that math has any bug — something I
can't verify without live testing — it could confidently predict "no
overpass here" when a real one occurred, triggering an immediate
`return None` that never even reaches PPS. That's a precise, falsifiable
explanation for "GMI/SSMIS just don't work," and it isn't about filenames
at all.

**Fixed** by no longer trusting an empty TLE prediction with the same
confidence as a real negative: all four sensors now treat "TLE found
nothing" the same as "TLE unavailable" (fall back to a broader search),
rather than hard-stopping. This change can only ever *widen* a search,
never cause a wrong match — a different risk profile from a tolerance
change, so safe to apply across all four without repeating last round's
mistake of a change that could silently break something working.

**SSMIS got special handling**, because its own docstring already
documents that a truly unnarrowed query previously triggered a real
server-side rejection (not just slowness) on its unusually large,
multi-satellite flat directory. So instead of falling through to the
plain full-listing approach the other three safely use, SSMIS now sweeps
hourly wildcard queries across the whole lookback window when TLE gives
nothing specific — each query still individually server-side-filtered
(avoiding the documented rejection), just more of them than an ideal
TLE-narrowed 1-2. Verified directly: the hour-sweep generates one token
per hour across the full window as intended.

Also documented the missing AMSR3 NORAD ID honestly in `tle_predict.py`
rather than guessing at one — a wrong ID would silently track the wrong
satellite, which is worse than the current explicit gap. If GOSAT-GW's
real NORAD ID becomes available, adding it will let AMSR3 benefit from
genuine narrowing instead of relying on this fallback indefinitely.

Same honest caveat as every round on this: no live network access here
to confirm against the real server. This is the most precise diagnosis
I can offer given the reported pattern, verified at the logic level —
the real test is whether GMI/SSMIS/AMSR2 start coming through without
WSFM/AMSR3 regressing.

## Major architecture change: multi-pass MIMIC-TC crossfade across GMI/AMSR3/WSFM

Per direct strategic guidance: AMSR2-NRT stops transmitting after Aug 31
2026 (AMSR3 is its already-active successor) and SSMIS shuts down
entirely in September 2026 — investing further effort in either has only
weeks of remaining value. Reoriented the whole real-MW pipeline around
the three sensors worth building on long-term: GMI-NRT, AMSR3-NRT, and
WSFM-NRT (`mw_ingest.PRIMARY_NRT_SENSORS`), with the higher-latency GMI
archive kept as a fallback for when none of the three have anything
within the search window (12h back, 3h ahead) — either a genuine multi-
sensor gap, or a request for data older than NRT's ~7-day retention.

**This is a genuinely new fusion mode, not just a bug fix**: previously,
real MW fusion only ever found the single most recent pass and morphed
it forward. Now the system searches for TWO passes — the best "before"
(most recent past pass, across all three sensors) and the best "after"
(soonest future pass, valid for historical/archived generation where
that pass has already happened and its data already exists) — and
crossfades between them:

- `mw_ingest.find_swath_that_hit_storm` was generalized to search and
  retry in either time direction, not just backward — verified with a
  direct test confirming forward search correctly finds a later hit
  after an initial miss, while backward search's existing behavior is
  completely unaffected (regression-tested).
- `mw_ingest.find_mw_pair_for_crossfade` searches all three sensors in
  both directions and selects the best before/after pair — verified
  directly with a mocked scenario where different sensors provide the
  best before vs. after candidate.
- `synthetic_algorithm._crossfade_confidence_toward_after` ramps an
  upcoming pass's confidence from 0 (60+ minutes before its own time) to
  1.0 (at its own time) — the mirror-image complement of the existing
  `_morph_age_confidence` (which fades an *aging* pass out). Verified at
  all boundary points.
- `generate_synthetic_mw` now accepts `real_swath_after`/
  `mw_confidence_after`, blending backbone/before-pass/after-pass as
  THREE weighted sources via the same `_weighted_fuse` mechanism already
  used throughout this project — deliberately scoped down relative to
  the before-pass: the after-pass only feeds the direct-value V/H blend,
  not the response-index/backbone-shaping computation, so it smooths the
  handoff to the next real pass without reshaping the overall storm
  structure model. Verified end-to-end with distinguishable warm/cold
  test signatures: the blend measurably shifts from before-dominant
  early in the crossfade window to after-dominant late in it, exactly
  as intended.
- Wired through the entire GUI call chain (`fetch_real_mw_for_fusion`,
  both `GenerateWorker` and `GenerateLoopWorker`), including the GMI
  archive fallback path and correctly updated diagnostics text for the
  case where only an after-pass is found. Verified with a full mocked
  end-to-end test from the GUI layer down through sensor search,
  morphing, and confidence computation.

**On AMSR2**: did a bounded check as requested — confirmed it's a
completely separate code path (JAXA G-Portal, not PPS NRT) with its own
self-diagnosing error handling already in place (raises a clear error
listing the actual G-Portal catalog tree if its dataset lookup fails).
Given the explicit reprioritization toward the three-sensor architecture
above, didn't sink further effort into deep AMSR2 debugging without
being able to see the actual error output from a live run.

**Honest scope note**: this is a large, newly-built architecture,
verified thoroughly at the logic level (every new function has a direct
test with mocked data, and the full chain was tested end-to-end), but
none of it has been run against the real PPS server. The core building
blocks (search-in-either-direction, pair selection, crossfade math,
three-way fusion) are each individually well-tested; what hasn't been
tested is how it all behaves against the real timing/availability
patterns of live GMI/AMSR3/WSFM data.

## New feature: "Skip every" for GIF/frame loops

Added a "Skip every" spinner (1x-12x) next to the Frames dropdown — same
frame count, but spaced further apart in time, so a 15-frame GIF can
cover a much longer span without needing more frames (slower to
generate, larger file). 1x is the exact original behavior (consecutive
10-minute steps); e.g. 3x means the same N frames but at 30-minute steps,
covering 3x the time span. Verified directly: confirmed the default
(1x) case produces byte-identical frame timing to before this change,
and a 3x case covers exactly 3x the time span with the same 5 frames.
Also tested `GenerateLoopWorker`'s own internal frame-time computation
with `skip_every` set, confirming the worker itself (not just the
standalone logic) computes the right times.

Also worth noting from the shared run: the crossfade architecture from
last round is confirmed working on a real run — WSFM-NRT was found,
correctly reported as a 1-minute-old pass with confidence 1.00, and
successfully fused. Good first real-world validation of that whole
pipeline actually functioning end-to-end, not just in mocked tests.

## Real MW swath despeckling — a genuine gap, found from a real artifact

A real run showed two artifacts near the storm core: a thin isolated
spike cutting across an otherwise smooth 89 GHz field, and a
desaturated gray/white patch that doesn't correspond to any color the
NRL table should ever produce. Traced this to a real, confirmed gap:
`_despeckle_gates` (radar_ingest.py) already handles exactly this class
of problem for radar — isolated, spatially-unsupported gate spikes from
ground clutter, AP, or biological scatterers — but no equivalent check
existed for real MW swath pixel data. `_apply_physical_bounds` only
catches values outside a physically *possible* range; it does nothing
for a value that's individually plausible but wildly inconsistent with
every pixel around it (sensor noise, RFI, footprint contamination) —
exactly the gap an isolated bad GMI pixel would fall through.

Added `_despeckle_mw_field`: a local-median-filter comparison (robust to
a single bad pixel skewing a plain mean) that masks any pixel deviating
more than 15K from its own local neighborhood, applied to both the
before-pass and after-pass real MW data right after physical-bounds
checking. Deliberately conservative (5x5 window, 15K threshold) since
real MW imagery has genuine fine-scale texture that this project has
spent real effort preserving elsewhere — verified directly that it:
(1) correctly removes an injected severe isolated spike (70K off from
its neighborhood) while leaving every surrounding normal-noise pixel
untouched, (2) leaves a genuine smooth gradient plus realistic fine
texture almost completely alone (only 3 pixels out of 3600 touched in a
test with no actual defects), and (3) end-to-end, recreates the exact
reported scenario (an isolated corrupted pixel near a storm core in a
realistic GMI-like swath) and confirms the resulting composite renders
cleanly with no spike or gray patch. Also confirmed no regression in
the normal (no-artifact) fusion path.

## Checkpoint-based MW searching for multi-frame loops, plus two small caps

**Major optimization, per direct request**: multi-frame loops previously
searched independently for every single frame (up to 15 separate
multi-sensor searches for a 15-frame GIF). Now searches only at a
handful of evenly-spaced "checkpoint" frames —
`mw_ingest.compute_checkpoint_frame_indices` gives 5 frames → 2
checkpoints (first, last), 10 frames → 3 (first, ~halfway, last), 15
frames → 4 (first, ~1/3, ~2/3, last) — verified directly against all
three stated examples. Every frame (checkpoint or not) still gets its
own exact MIMIC-TC morph and crossfade confidence computed for its own
specific time — only the expensive network *search* is shared between
checkpoints, not the per-frame math, so precision isn't traded away,
just redundant searching.

The reuse logic itself required some care: a frame between two
checkpoints uses the earlier checkpoint's own "before" search result as
its before-pass, and the *later* checkpoint's own "before" result as its
after-pass (since that later checkpoint's most-recent-pass search is,
from an earlier frame's perspective, itself a future/"after" pass) — this
naturally reuses exactly what's already been found rather than needing
any extra searching. The very last checkpoint has no next checkpoint to
borrow from, so it uses its own dedicated after-search instead. Verified
this bracketing logic against seven test cases (both checkpoint frames
and in-between frames, plus the special last-checkpoint case), and
verified the per-frame morph+confidence computation genuinely varies
correctly across frames sharing the same raw checkpoint swath (near-zero
confidence for an after-pass 3 hours away, ~0.83 confidence for the same
pass when a later frame is only 10 minutes before it). Confirmed
end-to-end with a full mocked 15-frame loop run: exactly 4 MW searches
performed, not 15.

**Two capacity changes, also per direct request**: "Skip every" is now
capped at 3x (30 minutes/frame, down from 12x); frame generation
concurrency (`MAX_CONCURRENT_FRAMES`) raised from 4 to 5.

## Ditched search-ahead (real-time focus), and a concrete speed win alongside it

Per direct guidance: removed forward/"after"-pass searching from the
default real-time pipeline. It only ever had value for archived/
historical generation (a real-time run has no future data to find), and
even then NRT retention is only ~7 days — too narrow a use case to
justify the extra network round-trip on every real-time search.
`find_mw_pair_for_crossfade` now takes `search_after=False` by default;
the underlying forward-search capability (`search_direction="forward"`
throughout `mw_ingest.py`) wasn't deleted, just no longer invoked unless
explicitly requested — kept in case a future archive-specific mode wants
it. `GenerateLoopWorker`'s checkpoint-reuse logic simplified accordingly
(no more bracketing between two checkpoints, just "nearest preceding
checkpoint" per frame). Re-verified end-to-end: a 15-frame loop still
performs exactly 4 searches, not 15, after the simplification.

**While in there, found a real concrete speed win**: the 3 sensors
(GMI-NRT, AMSR3-NRT, WSFM-NRT) were being queried one at a time in
sequence at each checkpoint — independent network round-trips that
parallelize well. Now queried concurrently via a small thread pool.
Verified directly with simulated latency: 3 sensors at 0.3s each took
0.30s total (concurrent) instead of ~0.9s+ (sequential) — a genuine 3x
speedup on the per-checkpoint search cost, not just a theoretical one.

## Brainstorm: further speed optimization for MW data pulling/processing

Checked what's already in place before suggesting more: both the PPS
directory-listing cache (5-min TTL) and the TLE cache (refreshed daily)
already exist and are wired up correctly — not dead/unused code, genuine
existing optimizations. Ideas for further work, roughly in order of
expected impact vs. effort:

1. **Parallelize the checkpoint searches themselves, not just the
   sensors within each one.** Right now, a 15-frame loop's 4 checkpoints
   are still searched one after another (each internally 3x parallel
   per the fix above) — nesting the concurrency so all 4 checkpoints'
   sensor-queries run at once (up to 12 concurrent requests) could
   shrink the whole search phase to roughly one sensor's slowest single
   query, instead of 4 sequential rounds of it.

2. **Overlap the search phase with GOES fetching, instead of doing them
   as two separate sequential stages.** Currently every frame's GOES
   band fetch waits for *all* checkpoint searches to finish, even though
   frame 0 only actually depends on checkpoint 0's result (available
   almost immediately). Restructuring so a frame can start its GOES
   fetch the moment its own bracketing checkpoint is ready — not all of
   them — would let the two phases overlap instead of running back to
   back. More involved than #1 (needs real dependency tracking between
   checkpoint futures and frame futures in the same pool), but probably
   the single biggest remaining win for multi-frame loop wall-clock time.

3. **Lazy-crop large swath files before triggering a full data read.**
   AMSR2/AMSR3/SSMIS/GMI-archive publish long swaths (some ~100 minutes
   of orbit per file) that get cropped to a small box around the storm
   immediately after loading — if the current code reads the entire
   Tc array into memory before cropping (worth confirming directly
   against xarray's actual lazy-loading behavior here, rather than
   assuming), reading just the lat/lon coordinates first to determine
   which scan lines actually fall in the box, then slicing *before*
   triggering the real data load, could cut a lot of wasted I/O for the
   largest files specifically.

Didn't implement #2 or #3 this round — flagging them as concrete,
reasoned next steps rather than doing a large speculative restructure
without being asked to go that far this time.

## MWSynth 0.53 — checkpoint parallelization, AMSR3 channel discovery fallback, storm dropdown

**Checkpoint-level parallelization** (per direct request, holding off on
the xarray lazy-loading idea until it can be verified rather than
assumed): checkpoints in a multi-frame loop now search concurrently
instead of one after another — nesting on top of the existing per-sensor
concurrency within each checkpoint. Verified directly with simulated
per-checkpoint latency: a 15-frame loop (4 checkpoints) completed in
~0.52s total including all frame generation, versus the 1.2s+ that 4
sequential checkpoint searches alone would have taken.

**AMSR3 channel discovery — a real fix from a real error.** The shared
error log showed `_discover_wsfm_channels` scanning zero usable groups
in a real AMSR3 file ("Groups scanned: {}") — confirming the "reuse
WSFM's discovery logic" assumption from an earlier round was wrong for
AMSR3 specifically, even though it's genuinely correct for WSFM (per
direct confirmation that WSFM-NRT pulled real data successfully, e.g.
Dolly/AL042026 today). Added a fallback: when the `S1..S8` naming
assumption finds nothing at all, `_discover_channels_via_h5py_fallback`
walks the file's actual real structure via h5py's `visititems()` —
finding every dataset regardless of what it's actually named or how
deep it's nested — and applies the same frequency-matching logic to
whatever is really there. Tested the callback logic directly against a
mocked file structure entirely different from `S1..S8` and confirmed it
correctly discovers and labels the channels. **Honest caveat**: h5py
isn't installable in this sandbox (no network access) despite already
being a listed requirement, so this could only be tested with mocked
data, not a real AMSR3 file — the error message itself still reports
exactly what h5py *did* find if this fallback also comes up empty,
which is the fastest path to manually hardcoding the right group/index
per the error's own advice.

**Storm selection is now a dropdown**, not three boxes. New
`besttrack.fetch_recent_btk_storms()` parses NHC's live btk directory
listing (https://ftp.nhc.noaa.gov/atcf/btk/) for `.dat` files modified in
the last 24 hours, extracting basin/storm#/year from each ATCF-standard
filename. Tested the parsing/filtering logic directly against a
synthetic Apache-style listing (correctly keeps recent files, filters
out a 48-hour-old one, ignores non-matching filenames). **Honest
caveat, same as always**: written against the standard Apache
mod_autoindex format (the same style already confirmed working for
PPS's NRT server elsewhere in this project) — not verified against
NHC's actual live page, since there's no network access here to check
directly. If the real format differs, this returns an empty list (a
visible "no storms found," not a crash) rather than silently
mis-parsing. The dropdown auto-populates on startup and has a manual
Refresh button; fetched synchronously (a single quick HTTP request) with
the known tradeoff of briefly blocking the UI if NHC's server is slow.

**Renamed to MWSynth**, version tracked as one clearly-findable constant
(`APP_VERSION` in `gui/main_window.py`) rather than a bare string, to
make it easy to remember to bump next time. This build is 0.53.

## Real AMSR3/GMI/WSFM files inspected directly — found and fixed the actual bug

Three real NRT files were provided for direct inspection. Neither h5py
nor netCDF4 could be installed in this sandbox (no network access,
despite both being listed requirements), but raw binary string
extraction (HDF5 embeds attribute text as readable ASCII even in binary
form) was enough to get real, concrete answers rather than more
guessing.

**Confirmed the AMSR3 file genuinely has the needed data**: extracted
real `LongName` strings directly from the file, including
`"Intercalibrated Tb for channels 1) 89 GHz V-Pol A-Scan and 2) 89 GHz
H-Pol A-Scan"` and `"...36.42 GHz V-Pol and 2) 36.42 GHz H-Pol"` — fed
these exact real strings through the existing parsing regex and
confirmed it extracts and labels all four needed channels perfectly.
So the frequency-parsing logic was never the bug.

**Found the real bug**: `nscan6,npixel6,nchannel6` dimension names near
the 89 GHz data strongly suggested the group-naming scheme (`S1`..`S8`)
was probably fine — meaning the failure was in the group*-opening* step
itself, not group naming. Checked directly: none of the 5 places this
project calls `xr.open_dataset(path, group=...)` ever specify an
explicit `engine=` parameter — every single one relies entirely on
xarray's auto-detection, a known-inconsistent behavior for grouped/
nested HDF5-in-netCDF4 files, especially large ones (this AMSR3 file is
147MB). A failed auto-detected engine raises an exception that the
existing scan loop was silently swallowing — indistinguishable from
"this group doesn't exist," exactly matching the reported symptom.

**Fixed**: new `_open_group_robust()` explicitly tries `h5netcdf` then
`netcdf4` (both already-listed dependencies) before falling back to
auto-detection, used at all 5 call sites. Also added per-attempt error
capture in the scan loop, so if every engine still fails for some
group, the resulting error message shows exactly which exception
occurred for each one — turning any future silent "found nothing" into
an immediately diagnosable message, instead of another round of
guessing. Confirmed the full real pipeline (parse real extracted
LongName text → label by frequency) end-to-end.

## Storm dropdown — fixed after a real failure, using a more robust design

The live server showed zero storms, confirming the original date-column
regex didn't match NHC's actual page format. Rather than guess a
different date format and risk the same silent failure again,
`fetch_recent_btk_storms` was redesigned to decouple the two concerns:
get filenames from a simple `href="..."` regex (much harder to break),
then get each file's real modification time from its own HTTP
`Last-Modified` response header via a HEAD request — a standardized
field completely independent of how NHC's page happens to display dates
for humans. HEAD requests run concurrently (up to 16 at once) to stay
fast despite needing one per candidate file. Tested directly against a
realistic mocked scenario, including confirming the `?C=M;O=D` sort-link
href (present on the real page you showed) is correctly ignored rather
than mis-parsed as a storm file.

## Output files now named MWSynth_0.xx.zip

Per direct request — the delivered zip now uses the same version
nomenclature as the app itself.

## MWSynth 0.54 — the real AMSR3 bug: a truncated cached file, not a code logic bug

Last round's diagnostic improvements paid off immediately: instead of
another mysterious "found zero groups," the very next real run showed
h5py's actual error directly — `"truncated file: eof = 48, sblock->
base_addr = 0, stored_eof = 2048"`. That's h5py reporting the file on
disk is incomplete, not a group-naming or engine problem at all.

Traced this to a real, confirmed bug in `_pps_nrt_find_and_download`:
the caching check was `if not os.path.exists(local_path)`, which only
asks "does *anything* exist at this path" — never "did the download
actually finish." A download interrupted partway (network drop, timeout,
anything) leaves a truncated file sitting at the exact final filename,
and every future run trusts it forever, since nothing ever re-verifies
it. That's exactly what happened: a real AMSR3 file stuck at 48 bytes,
causing the identical failure on every subsequent attempt.

**Fixed with two changes working together**:
1. Before trusting an existing cached file, its size is now compared
   against the server's real `Content-Length` (via a HEAD request) —
   any mismatch forces a re-download.
2. Downloads now write to a temp path and atomically rename
   (`os.replace`) onto the final filename, rather than writing directly
   to it — so a *future* interruption leaves an orphaned `.part` file
   instead of a broken "real" one ever appearing at the cached path.

Tested all three cases that matter directly: the exact reported bug (a
48-byte truncated cache, server reporting 2048 bytes) correctly
triggers a re-download and ends up with the complete file; a valid,
already-complete cached file is correctly left alone with zero wasted
network calls; and a HEAD request failure (e.g. offline) correctly falls
back to trusting the existing cache rather than breaking usability when
there's no way to verify.

Worth noting separately: the shared run log shows the overall pipeline
already degrading gracefully around this bug — AMSR3 failed, the search
correctly widened and eventually found a good WSFM-NRT pass instead,
and fusion completed successfully. The resilience design held up even
while this specific bug existed; this fix means AMSR3 shouldn't need
that fallback going forward, once the bad cached file is naturally
replaced (or cleared manually via the existing "Clear NRT cache"
button, for an immediate fix).

## MWSynth 0.55 — simplified search window, and honest uncertainty on a reported miss

**Search simplification, per direct request and confirmed sound
reasoning**: `fetch_recent_swath` no longer tries progressively wider
windows (6h, then 9h, then 12h) one at a time. Every tier's search
already selects the *most recent* candidate within whatever window it
searches, and anything found within 6h or 9h is also within 12h by
definition — so a single 12h search finds exactly the same result the
tiered version would have, just without up to 2 wasted intermediate
queries per sensor. Verified directly: a simulated 6-hour-old pass is
now found in exactly 1 query instead of up to 3, with the correct
result.

While implementing this, found and removed a related piece of dead
code: `lookback_options`/`AUTO_CALIBRATE_LOOKBACK_HOURS` was still being
threaded through `fetch_real_mw_for_fusion`'s signature and both GUI
worker classes, but a previous round's rewrite had already hardcoded
`lookback_hours=12.0` directly at the actual call site — meaning this
parameter did nothing for a while and nobody caught it. Removed cleanly
from both definitions and both call sites.

**On the reported missing AMSR3 pass**: unlike the truncated-file bug
from last round, this report didn't come with a console log, so I don't
have the same kind of direct evidence to trace a specific bug from. My
best-reasoned guess, based on what the code actually does: pass
discovery isn't just "does something exist near this time" —
`find_swath_that_hit_storm` also requires the swath to physically cross
within 75km of the storm's *interpolated best-track position* at that
pass's own observation time, and keeps searching (doesn't accept) a
pass that misses that check. A large, weak, disorganized system (this
report's storm: RMW 74km, ROCI 463km, night) is exactly the kind of case
where a real pass could still get rejected if the best-track center is
uncertain or the circulation too broad to pin down precisely — but I
want to be direct that this is a plausible explanation, not a confirmed
one. If this recurs, the actual console log (the way previous rounds'
real bugs were found) would let this be checked directly instead of
reasoned about secondhand.

## MWSynth 0.56 — found via direct 0.51-vs-current comparison: a real regression

Given a report that NRT "still didn't work" even after 0.55's fixes,
did a systematic diff of 0.51 (the last version with GMI-NRT/WSFM-NRT
confirmed fully working) against the current codebase, rather than
guessing again. `tle_predict.py` is byte-identical between versions --
ruled out entirely. The full `mw_ingest.py` diff pointed to one clear,
concrete regression candidate.

**Found it**: `_open_group_robust` (added specifically to fix AMSR3)
tried explicit engines (`h5netcdf`, then `netcdf4`) *before* falling
back to plain auto-detection. But 0.51's GMI-NRT and WSFM-NRT -- both
confirmed working -- used *only* auto-detection, no `engine=` parameter
at all. Forcing a different engine ahead of whatever auto-detection
would have picked is exactly the kind of change that can silently
behave differently for files that were already opening fine, applied
here to every sensor's group-opening, not just AMSR3's. This is the
same category of mistake as an earlier regression in this project (a
shared matching tolerance widened for one sensor's benefit, which
silently affected two others that didn't need it) -- a fix for one
confirmed-broken case changed the code path for cases that were already
confirmed working.

**Fixed** by reordering: auto-detection is tried first now, exactly
matching 0.51's behavior for GMI-NRT/WSFM-NRT, with the explicit-engine
fallback only reached if that first attempt genuinely fails (AMSR3's
actual confirmed symptom). Verified both directions directly: when
auto-detection succeeds, it's used immediately with zero interference
from the fallback logic (engine attempt order: `[None]` only); when
auto-detection genuinely fails, the fallback correctly engages in order
(`[None, 'h5netcdf']`).

**Honest scope of this claim**: this is the most concrete, best-evidenced
regression the diff turned up, and it's now fixed and tested at the
logic level. I can't promise it's the *only* thing wrong, since there's
no fresh run log to confirm against yet -- if NRT still doesn't come
through after this, the actual console output would again be the
fastest way to find out why, the same way it was for the truncated-file
bug and this engine-ordering issue both.

## MWSynth 0.57 — a real, confirmed diagnostic gap, and honesty about what's not yet solved

Given this run's log: all three primary sensors (GMI-NRT, AMSR3-NRT,
WSFM-NRT) reported "no NRT pass found," with zero error messages and
zero intermediate diagnostic output between "Searching Xh past SENSOR
imagery..." and the final failure. Worth noting directly: AMSR3 still
has no NORAD ID configured (a documented, pre-existing gap), so its TLE
prediction always takes the broad-fallback path rather than real
narrowing -- meaning it isn't vulnerable to a TLE-geometry bug the way
GMI-NRT/WSFM-NRT theoretically could be. All three still failing anyway
pointed away from TLE-specific causes and toward something shared, more
likely in the listing/matching stage itself.

**Found a real, confirmed gap while investigating**: none of
`fetch_gmi_swath_nrt`, `fetch_wsfm_swath_nrt`, or `fetch_amsr3_swath_nrt`
ever accepted a `progress_callback` parameter at all -- and
`fetch_recent_swath`'s kwargs construction only ever passed one through
for the `"GMI"` archive sensor specifically. This means these three
sensors -- the ones this project is actually built around now -- have
never had ANY path for a diagnostic message to reach the user, which is
exactly why this run's log showed nothing useful between "searching"
and "not found."

**Fixed**: added `progress_callback` support to all three fetch
functions, actually threaded it through in `fetch_recent_swath`, and
added a detailed stage-by-stage diagnostic directly in
`_pps_nrt_find_and_download` -- when nothing is found, it now reports
exactly how many files were in the raw listing, how many matched the
name/token filter, how many had a parseable timestamp, and how many
fell in the search window. This mirrors the exact approach that found
the previous two real bugs (the truncated file, the engine-ordering
regression): get specific enough information that the next failure is
immediately diagnosable rather than requiring another round of
guessing. Verified the full chain directly with a mocked empty-listing
scenario -- the diagnostic message now genuinely propagates all the way
from `_pps_nrt_find_and_download` up through `fetch_recent_swath` to
wherever the GUI's progress callback is listening, for all three
sensors (previously only the GMI archive path could ever surface
anything at all).

**Being direct about scope**: this round does NOT claim to have found
or fixed the actual reason all three sensors returned empty this time.
It fixes the fact that there was no way to find out why. The next time
this happens, the log should show specifically which stage failed
(empty listing vs. filter mismatch vs. date-window mismatch), which is
the difference between guessing and actually knowing.

## MWSynth 0.58 — a direct comparison against 0.51, and an important finding

Given the request to fall back to 0.51's NRT-pulling code: did a
byte-level diff of the actual core search/listing/matching logic
(`_pps_html_listing`, `_pps_nrt_find_and_download`'s candidate-selection)
between 0.51 and current, not just a general review. **The core
mechanism is functionally identical.** The only differences are
additive: the truncated-file integrity check, the engine-ordering fix
(already corrected last round to try auto-detection first, matching
0.51 exactly), and diagnostic logging. Reverting to raw 0.51 would not
change GMI-NRT/WSFM-NRT/AMSR3-NRT's actual search behavior, because
that behavior was never actually changed in the first place.

That finding raised an important question: last round's diagnostic
fix — which I verified end-to-end propagates correctly through the
full chain — should have produced a "MW search debug [...]" message in
this run's log showing exactly which stage came up empty. It didn't
appear at all. Checked the actual saved code directly (not just what I
intended to write) and confirmed every function signature and call site
genuinely has the fix in place. That combination — code confirmed
correct, but its output absent from a real run — points most plausibly
at a version mismatch: this run may not actually have been on 0.57.

**To make that unambiguous going forward**: every run now logs
`[MWSynth 0.xx]` as the literal first thing printed, both for single-
frame and loop-mode requests. This isn't a fix to the NRT problem
itself — it's removing the possibility of diagnosing against the wrong
version's behavior, which is a real risk once several rounds of fixes
have accumulated.

**Direct and honest bottom line**: I don't have a new root-cause finding
this round. I have confirmation that the code I already shipped is
correctly in place, and a tool (the version stamp) to settle whether
this run was actually testing it. The next log — ideally starting with
`[MWSynth 0.58]` visibly at the top — will say definitively whether the
stage-by-stage diagnostic from last round actually fires, which is the
next real fork in this investigation.

## MWSynth 0.59 — found the real silent-failure path

0.58's version stamp did its job: it confirmed the right build was
running, which meant the missing debug message was a genuinely
meaningful signal rather than a stale-build red herring. That redirected
the search to "what returns None *without* going through the listing
diagnostic at all" -- and there it was: `if lat37.size == 0 or
lat89.size == 0: return None`, present identically in all four NRT
fetch functions (GMI-NRT, SSMIS-NRT, WSFM-NRT, AMSR3-NRT), with zero
logging.

**What this means concretely**: the search is fundamentally TIME-based
-- it finds the most recent file whose *observation timestamp* falls in
the search window. It does not verify that satellite's orbital swath
actually passed near the storm's location before committing to that
file. A satellite covers the whole globe over a day; "most recent by
time" and "most recent that's actually nearby" are only the same thing
if TLE-based geographic narrowing genuinely worked -- and this project's
own code already documents that narrowing as unreliable enough that it
deliberately falls back to a broad, unnarrowed search rather than trust
a negative TLE prediction. In that broad-search mode, a time-matching
candidate can easily be geographically irrelevant, and until now, that
was discovered only after downloading and opening the file, then
silently discarded.

**Fixed**: all four occurrences now report exactly this when it happens
-- which file was found, what time it was observed, and that its
coverage didn't include the requested location. Verified directly by
reconstructing the exact real-world scenario (a file found by time
match, opened successfully, but with geographic coverage far from the
target storm) and confirming the diagnostic fires with a clear,
specific message instead of silently returning `None`.

**Natural next step, not yet done**: right now, if this happens,
`find_swath_that_hit_storm` gives up immediately rather than retrying at
an earlier time -- it only retries when a candidate is found but misses
the storm's *core* (a different, already-handled case), not when no
usable candidate is found at all. Making a geographic-coverage miss
retry the same way a core-miss already does would be the logical
architectural fix once the diagnostic above confirms this is indeed
what's happening on a real run. Deliberately not doing that larger
change blind, in the same run as finding the bug -- confirming the
diagnosis first is safer than layering another unverified fix on top of
a fix that hasn't been checked against a real run yet.

## MWSynth 0.60 — confirmed by the real log, now actually fixed

0.59's diagnostic came back exactly as predicted: all three primary
sensors reported "found and opened [file]... but its actual swath
coverage doesn't include the requested box." Diagnosis confirmed by a
real run, not just reasoned about.

**Implemented the fix flagged as the natural next step**: all four NRT
fetch functions (GMI-NRT, SSMIS-NRT, WSFM-NRT, AMSR3-NRT) now retry
against progressively older candidates from the same already-fetched
listing when the most recent one's geographic coverage doesn't include
the target -- up to 5 attempts -- instead of giving up after the single
most-recent-by-time candidate. `_pps_nrt_find_and_download` gained an
`exclude_filenames` parameter so each retry asks for "the best candidate
that isn't one we already know doesn't cover this location," reusing the
cached listing rather than re-fetching it from the network each time.

Verified this thoroughly, not just at the unit level:
- Each of GMI-NRT's and AMSR3-NRT's retry loops tested directly with a
  mocked "most recent file misses geographically, older file hits"
  scenario -- both correctly skip the bad candidate and return the good
  one.
- The full real call chain (`find_mw_pair_for_crossfade`, exactly what
  the GUI calls) tested end-to-end reproducing the precise real symptom
  across all three sensors simultaneously -- confirmed it now recovers
  and finds a usable pass instead of falling through to the archive.

This closes the loop that started with 0.57's diagnostic logging: found
where the search was failing silently, confirmed it against a real run,
and implemented the fix that was explicitly deferred until that
confirmation came in.

## MWSynth 0.61 — removed the retry loop, confirmed 0.53 never had it either

**Confirmed directly against the uploaded 0.53 build**: its
`fetch_wsfm_swath_nrt` has the exact same single-attempt structure —
find the best candidate by time, silently return `None` if the crop
comes up empty. No retry logic existed in 0.53 either. So "0.53 worked
for WSFM" wasn't a code difference at all; it means the specific storm/
time tested back then happened to have a genuinely nearby pass. The
code hasn't regressed here — the underlying luck of whether an orbit
crosses near a given storm at a given moment varies test to test, and
always has.

**Removed the retry loop from 0.60**, per direct feedback, and the log
that prompted it explained exactly why it was the wrong approach: GMI's
five retry candidates were all within a 20-minute span -- 5-minute
granules of a single orbital pass, not different passes. If one misses
the storm geographically, they all do, since they're slices of the same
swath. The retry loop was burning all 5 attempts cycling through
functionally-identical geography with zero real chance of finding
something different, while generating a lot of repetitive log noise in
the process.

All four NRT fetch functions are back to a single attempt per sensor,
cut off cleanly by `lookback_hours` (still 12h, from the earlier
simplification) -- no candidate multiplication, no "trying the next
most recent candidate" messages. The diagnostic message from 0.59 is
kept, so a geographic miss is still clearly explained, just without
implying a retry that wasn't actually productive. Verified directly:
a single geo-miss now produces exactly one search call and one clear
diagnostic message per sensor, and the full pipeline test that
previously generated 15+ lines of repetitive retry messages now
produces 6 clean ones.

## MWSynth 0.62 — full mw_ingest.py analysis, and the real structural fix

Per direct request, did a systematic read through the entire file rather
than another reactive patch. This also resolved an apparent
contradiction: 0.60's retry loop found a WSFM pass after one failure;
0.61 (single-attempt) didn't. Both observations were correct, and both
pointed at the same real bug, which 0.61 didn't actually fix -- it just
removed a bad workaround for it.

**The real bug**: `find_swath_that_hit_storm` already has a
sophisticated, well-tested retry/push-back mechanism -- but it only ever
engaged when a swath was *found* and missed the storm's core.
When the sensor's own fetch function returned `None` (nothing usable
found at all, which is exactly what a geographic-coverage miss produces),
this gave up immediately:
```
if swath is None:
    return last_swath, last_lookback, False
```
completely bypassing the retry logic below it. Since a geo-miss returns
plain `None` (not a swath object that happened to miss), the well-tested
retry mechanism never got a chance to run for what the diagnostics from
last round confirmed is the *more common* failure mode. 0.60's retry
loop (inside each sensor's own fetch function, excluding one filename at
a time) was a workaround bolted onto the wrong layer -- and a poorly
scoped one, since GMI's ~5-minute granules meant several "different"
retry candidates were often just slices of the identical useless pass.

**Fixed at the right layer**: removed the immediate give-up. A `None`
result now retries the same way a core-miss already did, jumping the
search time by a full ~90-minute orbital period (not the 6-minute nudge
used for a known core-miss, and not "the next file in the listing") --
enough to reliably reach a genuinely different orbital opportunity
rather than another slice of the same failed pass. This reuses the
existing, already-correct budget/safety-cap logic
(`max_total_lookback_hours`, `MAX_ITERATIONS`) instead of adding a
second, separate retry mechanism duplicated across four sensor
functions.

Verified thoroughly:
- Reconstructed the exact scenario you reported (miss, then a real hit
  after retrying) -- confirms the fix finds it, via a properly-bounded
  orbital-period jump instead of naive filename exclusion.
- Confirmed the 12-hour cutoff is genuinely respected when nothing is
  ever found (bounded to 8 attempts per sensor, doesn't run away).
- Re-verified the original core-miss retry path (found a swath, but it's
  not close enough to the storm) still works correctly -- this touches
  shared logic, so it needed re-checking, not just assuming it still held.
- Checked output volume directly and trimmed a redundant log line (the
  per-attempt "MW search debug" message from inside each sensor function
  already explains a geo-miss; a second "searching further back" line
  for the same event added noise without new information) -- worst-case
  log volume dropped from 72 to 48 lines for a scenario where nothing is
  ever found across all three sensors, and every one of those 48 lines
  now represents genuinely different search coverage, not a repeat.

**Also cleaned up while reading through the full file**, per the
request to actually analyze it rather than spot-fix: the module-level
docstring at the top of `mw_ingest.py` was badly stale (still describing
SSMIS as "not solved" and GMI-NRT as "not implemented," years of actual
work later) -- rewritten to reflect the current, accurate state and the
AMSR2/SSMIS sunset context. `_pps_html_listing`'s docstring still
referenced the tiered 6/9/12/24/48h search removed several rounds ago --
corrected. Noted but did not touch `load_local_swath` (a separate manual-
file-loading feature using an older hardcoded channel-index approach) --
out of scope here since it's not part of the active NRT search pipeline
that's actually been failing.

## MWSynth 0.63 — ML pivot, phase 1: data pipeline, model, training script

Per direct decisions: the model learns a CORRECTION on top of the
existing parametric algorithm's output (not a replacement), trained only
on GMI and AMSR2 (both higher native resolution than WSFM/SSMIS, both
with operational eras that cleanly overlap the GOES-R ABI series this
project already uses as input -- TMI/AMSR-E deliberately excluded, since
their eras mostly or entirely predate ABI).

**Data pipeline (tested)**: `generate_synthetic_mw` now exposes the
parametric backbone (v37/h37/v89/h89 before any real MW is blended in)
and which sensor supplied the fusion target, in its diagnostics dict.
`training_data_export.py` was extended to filter to GMI/AMSR2 only and
save the backbone alongside the existing real-MW target, so a training
script can compute (target - backbone) as the actual learning signal.
Verified end-to-end: the sensor filter correctly allows GMI/AMSR2 and
rejects WSFM/SSMIS/AMSR3, and the saved file's residual is directly
computable and finite.

**New: `ml_data_mining.py`** -- batch-mines historical training examples
across whole storms instead of relying only on slow organic accumulation
through normal app use. Deliberately scoped to one storm or storm-list
at a time (not "download a decade" in one call), given real storage
constraints -- each example is written to disk as soon as it's mined, so
an interrupted run keeps everything it found so far. Built directly on
top of this project's existing, already-tested fetch functions
(`besttrack`, `goes_fetch`, `mw_ingest`, `generate_synthetic_mw`) rather
than adding new untested network logic -- but the orchestration loop
itself needs a real run against live archives to confirm, same caveat as
every other network-touching piece of this project.

**New: `ml_model.py`** -- a compact U-Net (4 encoder/decoder levels,
GroupNorm, ~32 base channels) sized for a 6GB-VRAM laptop GPU. Output
layer is zero-initialized so an untrained model starts by predicting
*no* correction (output ≈ backbone), rather than risking large, wrong
corrections early in training. **Important, different-in-kind caveat**:
PyTorch isn't installable in this sandbox (no network access), so unlike
nearly everything else in this project, this code has never actually
been executed -- no confirmed forward pass, no confirmed gradient step.
Written against standard, well-established U-Net patterns, but "written
carefully" is a meaningfully weaker claim than "tested," and this is the
one place in the whole project where that gap is real. A `__main__`
shape-check block is included specifically so this can be quickly
verified the moment torch is available, before trusting it further.

**New: `ml_train.py`** -- training script: storm-centered patch
extraction, storm-level (not example-level) train/val split to avoid
leakage between correlated nearby-in-time examples, masked L1 loss
(only over pixels with real coverage), mixed-precision training,
best-checkpoint saving. The non-torch parts (patch extraction with edge
padding, the storm-split logic) were tested directly and confirmed
correct — in particular, confirmed zero storm overlap between train and
validation splits. The torch-dependent training loop itself carries the
same "written carefully, never executed" caveat as the model.

**Added `torch>=2.2` to requirements.txt**, with a note to install the
CUDA-matched build for the RTX 3050 specifically rather than assume the
default `pip install torch` command pulls the right one.

**Honest state of things**: this is real, tested infrastructure for
collecting and preparing training data, and a real, carefully-written
but unverified model and training script. It is not yet wired into the
main generation pipeline at all — there's no trained checkpoint to
apply, and building the inference-side integration (loading a checkpoint
and blending its correction into `generate_synthetic_mw`'s output) is
appropriately a later step, once there's an actual model worth applying.
Next steps, roughly in order: accumulate real GMI/AMSR2 training
examples (via the export checkbox during normal use, and/or
`ml_data_mining.py` against specific past storms), install torch and run
`ml_model.py`'s shape check, then attempt real training once there's a
meaningful amount of data — dataset_summary() in
training_data_export.py is worth checking periodically to see whether
there's "enough" yet.

## MWSynth 0.64 — TC-PRIMED integration via boto3 (no new dependency)

Read TC-PRIMED's official documentation (the PDF you linked) directly
before writing any code, plus confirmed the bucket structure against
multiple independent sources (the official products page, the AWS Open
Data Registry entry) rather than assuming a layout from the PDF alone.

**Why this genuinely matters for this project, not just "more data"**:
TC-PRIMED's own curation process already validates that each overpass
file actually covers the storm -- an areal coverage fraction check
within 750km of the interpolated storm center, falling back to 250km
(documented in Section 2.2 of their PDF). That's exactly the class of
problem -- a file matching by time but not actually covering the storm
geographically -- that caused a long chain of real, confirmed bugs in
this project's own live NRT search over the last several rounds. Mining
historical training data through TC-PRIMED sidesteps that whole failure
mode instead of needing to work around it.

**New: `tcprimed_ingest.py`** -- direct boto3 access to the public,
unsigned S3 bucket, reusing the exact same unsigned-access pattern
already established in `goes_fetch.py` for the GOES buckets (no new
library, no new style). Confirmed bucket/key structure directly:
`v01r01/final/<season>/<basin>/<number>/`, with filenames like
`TCPRIMED_v01r01-final_AL062018_GMI_GPM_025795_20180912184512.nc`.
GMI and AMSR2 channel group/variable names are hardcoded directly from
the documentation's own Table 3 (a confirmed, versioned, stable format
-- unlike the live NRT feeds elsewhere in this project, which needed
dynamic channel discovery specifically because their real structure
didn't match what was documented/assumed). Reuses `mw_ingest.py`'s
existing robust multi-engine group opener rather than a third, separate
NetCDF-opening pattern. Returns data as this project's own `MWSwath`
type, so it slots directly into the existing architecture.

**Tested what's testable**: filename parsing verified against the real
example filenames from TC-PRIMED's own products page (GMI, ATMS, SSMIS
overpass files, plus confirming the separate "era5" environmental file
is correctly excluded), and the S3 key-prefix construction verified
character-for-character against the confirmed documented directory
structure. The actual S3 list/download calls themselves carry the same
"never run against the live bucket from this sandbox" caveat as every
other network-touching piece of this project.

**Wired into `ml_data_mining.py`** as `mine_storm_via_tcprimed()` -- a
new, preferred alternative to the existing `mine_storm()` for historical
data collection specifically. Deliberately still fetches GOES imagery
through this project's own `goes_fetch.py` rather than using TC-PRIMED's
bundled infrared data, since TC-PRIMED's IR is a single "clean window"
channel from a separate archive (TC IRAR/HURSAT), not raw GOES ABI --
using it would mean training on input features that don't match what
the operational algorithm actually sees at inference time. TC-PRIMED is
used for exactly one thing: a pre-validated real-MW ground truth, not
the whole pipeline.

## MWSynth 0.65 — direct answer on training status, plus the download button

**Direct answer to "has training started"**: no, and nothing in this
project auto-triggers it. Data collection (the training-data-export
checkbox during normal use, or the mining scripts) and training
(`ml_train.py`) all require explicit, manual action. Seeing empty
training-data/checkpoint folders was the expected state, not a sign of
something broken.

**New: TC-PRIMED download button**, in the MW Data Credentials tab,
right below the existing NRT cache section (same visual pattern). A
year-onward spinner, an "Estimate download size" button (background
thread -- pure S3 LIST calls, no downloads, but can take real time
across a wide year range), and a "Start download" button gated on
having run the estimate first, so the confirmation dialog always shows
a real number rather than a generic warning. The actual download also
runs on a background thread so it doesn't freeze the GUI for what could
be a long time.

**Found and fixed a real bug while testing this, before it shipped**:
`tcprimed_ingest.py` ended up with two different, redundant
implementations of the same size-estimation logic from across this and
last round's work. Testing directly (not just syntax-checking) caught
it immediately -- `format_bytes(500)` returned different output than
expected, which traced back to two competing `format_bytes` definitions
silently shadowing each other. Removed the incomplete one; kept the
more correct one, which turned out to matter for a real reason: it
included the "SL" (South Atlantic) basin in its basin list, which the
other one was silently missing entirely. Re-tested the full discovery
plus size-estimation pipeline against a realistic mocked S3 response
shape (multi-storm discovery via S3's Delimiter listing, correct
per-file size summing, SSMIS correctly excluded) and confirmed it now
behaves correctly end to end.

**Honest caveat, consistent with the rest of TC-PRIMED integration**:
the discovery/estimation/download logic is tested against realistic
mocked S3 responses matching the confirmed real API shape, but has
never touched the live bucket from this sandbox. The first real run is
what confirms this works against actual TC-PRIMED data, the same as
every other network-touching piece of this project.

## MWSynth 0.66 — a real gap found from a good question: intensity was saved, never used

Direct answer, confirmed against the actual documentation rather than
assumed: TC-PRIMED's `overpass_storm_metadata` group explicitly states
`intensity` is "linearly interpolated from the best-track to the
passive microwave observation time" -- so yes, TC-PRIMED links
intensity to the exact MW observation time.

More importantly, tracing the actual code (not just the concept) found
that this project's own pipeline was *already* doing the equivalent
correctly at the data level, independent of TC-PRIMED's own field:
`mine_storm_via_tcprimed` interpolates this project's own best-track to
the swath's own `scene_time`, and `training_data_export.py` already
saves that as `storm_vmax_kt` (and `storm_rmw_nm`) in every training
example, both from TC-PRIMED-sourced and live-pipeline-sourced data.

**But the real, confirmed bug**: that correctly-computed value was
saved into every `.npz` file and then never read again anywhere.
Checked directly which fields `ml_train.py` actually loads -- `storm_lat`/
`storm_lon` (for centering the patch), never `storm_vmax_kt` or
`storm_rmw_nm`. The model had no way to distinguish a 65kt system from
a 115kt one from imagery alone, which matters directly here since the
existing parametric algorithm being corrected already leans heavily on
both intensity and RMW for its own calibration -- a correction model
without the same context can't learn anything intensity-dependent.

**Fixed**: `MWCorrectionUNet`'s input channels increased from 7 to 9,
adding intensity and RMW, each normalized and broadcast to a constant-
value spatial layer (the standard way to inject scalar/global context
into a convolutional architecture, so every pixel in the patch sees the
same "this is a 115kt storm" signal alongside the per-pixel imagery).
`ml_train.py`'s dataset loader now actually reads and broadcasts both,
with RMW's real-world missingness (not every best-track entry includes
it) handled by imputing to the normalization mean rather than
propagating NaN into the tensor.

Tested end-to-end against a realistic fake `.npz` matching the real
export format: confirmed a 115kt storm produces the exact expected
normalized constant layer, and a missing RMW correctly imputes to zero
(post-normalization) rather than corrupting anything. This is exactly
the kind of gap that's easy to miss without training data to look at --
the pipeline looked complete because every individual piece worked,
but nothing had ever traced whether the saved fields were actually
reaching the model.

## MWSynth 0.67 — ML correction wired into generation, always on

Implemented per direct design guidance: the correction is applied to
the parametric BACKBONE (GOES-only output), before the existing real-MW
fusion blend -- not to the final output directly, and not as a toggle.
This is deliberate, not just a simplification: the existing fusion
already weights backbone-vs-real-MW by confidence, so correcting the
backbone itself means that same weighting automatically makes the ML
correction's influence on the final output inversely related to
real-MW confidence, with no new blending logic needed. A fresh, high-
confidence real pass still dominates and the correction's visible
effect shrinks accordingly; when confidence is low or there's no real
MW at all, the (now ML-corrected) backbone carries the output --
exactly the "most useful when confidence is lower" behavior described.

**New: `ml_constants.py`** -- normalization constants (TB/intensity/RMW
scaling, patch size) pulled into their own shared module so training
and inference can't silently drift apart. `ml_train.py` updated to
import from it instead of keeping its own copy.

**New: `ml_inference.py`** -- loads and caches a trained checkpoint
(by path, so a newly retrained model is picked up automatically),
extracts a storm-centered patch matching the training patch size (not
run against the full scene -- a much larger input than the model has
ever seen would be a real distribution shift, even though the
architecture has no size-dependent layers that would technically
prevent it), runs the model, and blends the corrected patch back with
a smooth taper so the patch boundary isn't a visible seam. Critically:
every failure path (no torch, no checkpoint, a bad forward pass)
returns the backbone completely unchanged rather than raising --
required given this is now unconditionally wired into every generation
call, and the current real state is "no checkpoint exists yet."

Verified thoroughly before wiring it in, in three stages:
1. The actual current state (no torch installed, no checkpoint file) --
   confirmed `apply_ml_correction` returns arrays that are exactly,
   bit-for-bit identical to the input. Zero regression risk to every
   image already generated.
2. The pure-logic pieces (patch bounds near an edge vs. fully in-bounds,
   the blend taper's shape) tested directly against known expected values.
3. The full pipeline with a mocked model producing a known, predictable
   output -- confirmed the correction is de-normalized to the correct
   real-Kelvin magnitude, applied exactly at the storm center, and
   completely absent outside the patch region.

Then wired into `generate_synthetic_mw` right after the backbone is
finalized, with `progress_callback` threaded through from both GUI call
sites so "ML correction applied" (once a checkpoint exists) or any
skip reason will actually surface in the log. Re-ran the full
end-to-end generation test (including with real MW fusion active
simultaneously) after wiring this in -- confirmed identical, correct
output to before, since there's still no checkpoint to apply.

**Where this actually leaves things**: this is real, tested
integration code, not a placeholder -- but it has never processed a
real trained model's output, because no model has been trained yet.
The next real milestone is training producing an actual checkpoint at
`~/.synthetic_mw_tc/ml_checkpoints/mw_correction_best.pt`, at which
point this wiring activates automatically with no further code changes
needed.

## MWSynth 0.68 — patch size increased to 256, and the actual path to start training

**Patch size**: `PATCH_SIZE` (in the shared `ml_constants.py`) increased
from 128 to 256, per direct feedback after reviewing real output where
128px (~256x256km at GOES resolution) was cutting off exactly the outer
spiral banding visible in the GOES imagery. 256px (~512x512km) covers
even large, mature systems (ROCI 450-500km) with margin. Since both
`ml_train.py` and `ml_inference.py` import this constant rather than
keeping their own copies (a previous round's fix specifically to avoid
training/inference drift), this one change propagates correctly to
both automatically -- confirmed directly rather than assumed.

Lowered the training batch size default from 8 to 2 to compensate: 256px
is 4x the pixel count of 128px, and full training (forward + backward +
optimizer state) uses meaningfully more memory than the forward-pass-only
shape check already run on the real RTX 3050. This is an estimate, not a
measurement on that hardware -- documented clearly in the code as
something to watch and adjust on the first real run, not a guarantee.

**New: the actual missing link to start training.** The GUI's download
button only ever cached raw files -- it never paired them with GOES
imagery or exported anything to the training-data folder, which would
have left the download finishing with nothing to train on. Fixed with
`ml_data_mining.mine_local_tcprimed_cache()`, which processes whatever's
already sitting in the local cache directly -- parsing filenames alone
(extended `tcprimed_ingest._parse_overpass_filename` to also pull out
basin/storm/season, confirmed against real filenames) to group files by
storm with zero additional network calls, then pairs each with GOES
imagery and exports training examples exactly like the live pipeline
does. Tested the parsing/grouping logic directly against a realistic
fake cache (two different storms, a deliberately malformed entry, and
the "era5" environmental file that should never count as an overpass) --
confirmed it correctly discovers and groups only the real overpass
files.

**New: `run_ml_pipeline.py`** -- the actual single command to run once
the download finishes: processes the local cache into training data,
prints a summary, and (only if at least one example was actually saved)
starts training. Refuses to start a training run against zero data
rather than doing that silently.

## MWSynth 0.69 — a real, deserved bug, and an actual re-check of every ML file

`run_ml_pipeline.py` crashed with `NameError: name 'os' is not defined`
on the very first real run. Direct cause: `ml_data_mining.py` never had
`import os` at the top of the file, even though the function added a
few rounds ago uses `os.path.isdir`, `os.listdir`, and `os.path.join`
directly. This is exactly the class of bug that `ast.parse` (used for
every "syntax check" claimed in prior rounds) structurally cannot catch
-- `os.path.isdir(...)` is completely valid Python syntax whether or
not `os` was ever imported; the name lookup only fails at the moment
that line actually executes. Confirmed directly: even `compile()` (one
step further than `ast.parse`) still can't catch it, for the same
reason. Fixed the missing import.

Given that this was a real, shipped bug, went back through every
recently-added ML file rather than just patching the one line and
moving on -- specifically by actually calling every function with
mocked dependencies and watching for `NameError`, not just checking
that files parse:
- `ml_data_mining.py`: every function now runs (or fails with an
  expected, non-NameError exception like a blocked network call) --
  confirmed the fix actually works and didn't miss a sibling bug.
- `tcprimed_ingest.py`: every function exercised the same way, all
  clean.
- `training_data_export.py`: pure numpy/os, fully exercised with no
  mocking needed at all -- clean.
- `ml_train.py`: non-torch logic (patch extraction, train/val split)
  re-confirmed correct after the recent patch-size change.
- `ml_model.py`: not re-mocked further here, since it already has
  something strictly better than anything I can fake in this sandbox --
  a real, successful run on the actual RTX 3050 with real torch,
  already reported directly.
- `run_ml_pipeline.py`: ran the actual `main()` function end-to-end
  with a properly complete mock (my first attempt at this had its own
  incomplete-mock gap, caught and fixed before trusting the result) --
  confirmed it correctly detects the "no data yet" case and exits
  cleanly with a clear message, rather than crashing or silently
  starting a training run against nothing.

Also grep-cross-checked every import against every module-prefix usage
across all these files as a first, fast pass before the deeper
call-through testing -- two suspicious-looking flags turned out to be
false positives (substring matches inside words like "timestamp," not
actual module usage) and are noted as such rather than left ambiguous.

## MWSynth 0.70 — WP/IO/SH best-track via IBTrACS (Himawari-9 and full-disk crop still to come)

The "no_best_track: HTTPError" skip (192 files) was `besttrack.py` only
ever knowing about NHC's server, which is explicitly scoped to NHC's
own area of responsibility (AL/EP/CP) -- confirmed directly rather than
assumed. Western Pacific, North Indian Ocean, and Southern Hemisphere
storms are JTWC's responsibility, and JTWC's own site turns out to be a
webpage-based portal rather than a simple, reliably-scrapable directory
listing.

**Fixed via IBTrACS** (NOAA/NCEI's own unified, multi-agency archive,
confirmed at
`.../international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/`),
which already merges NHC's and JTWC's data into one consistent format.
Cross-checked the actual column layout against multiple independent
real-data sources before writing any parsing code -- critically, found
a real example row confirming the `USA_ATCF_ID` column (e.g.
"SH062023") directly matches TC-PRIMED's own basin/storm/year
convention, which sidesteps a real wrinkle: IBTrACS splits "Southern
Hemisphere" into SI (South Indian) and SP (South Pacific) separately,
while JTWC/TC-PRIMED use one combined "SH" -- filtering by
`USA_ATCF_ID` works regardless of which of those two a storm is
actually filed under, without needing to guess or map between them.
Used `USA_WIND`/`USA_RMW`/`USA_ROCI` specifically (not `WMO_WIND` etc.)
since those stay consistently in JTWC's 1-minute-wind convention,
matching what NHC's b-decks (and the rest of this project) already
assume -- `WMO_WIND` can silently switch averaging conventions
depending on which agency is "currently responsible" for a basin.

`fetch_best_track` now routes AL/EP/CP to the existing NHC path
unchanged, and everything else to IBTrACS automatically. The multi-basin
IBTrACS file is cached locally (atomic write, same lesson as an earlier
truncated-file bug elsewhere in this project) rather than re-downloaded
per storm.

Tested thoroughly, learning directly from last round's shipped `import
os` bug -- not just syntax-checked:
- Built a realistic mock IBTrACS CSV matching the real, confirmed
  column structure (including the units row real files have between
  header and data) and verified parsing against it: correct filtering
  by `USA_ATCF_ID`, correct use of `USA_WIND` over `WMO_WIND`, correct
  RMW/ROCI extraction, and a nonexistent storm returning cleanly rather
  than crashing.
- Verified `fetch_best_track`'s routing directly -- confirmed AL/EP/CP
  hit the NHC path and WP/IO/SH hit IBTrACS, not by inspection but by
  intercepting both code paths and counting actual calls.
- Verified the cache freshness logic across all three real states
  (missing, fresh, stale) rather than assuming the age comparison logic
  was correct.
- Ran the real (non-mocked) module's functions end-to-end specifically
  hunting for `NameError` from the new `csv`/`timedelta` imports, the
  same category of bug that shipped last round.

**Explicitly not done in this round**, and worth being direct about
rather than rushing: Himawari-9 ingestion (for actually fetching GOES-
equivalent imagery over WP/IO/SH storms) and the full-disk-crop fallback
(for storms that never had a dedicated mesoscale sector). Confirmed
`s3://noaa-himawari9/` is public/unsigned, same access pattern as GOES,
and that only Full Disk products exist -- no mesoscale-style sectors,
matching exactly what was described. Building the actual ingestion
module correctly needs Himawari's exact S3 key-naming convention and
native file format (likely HSD, different from GOES's NetCDF), which is
real, separate work for the next round rather than something to
compress into this one under time pressure.

## MWSynth 0.71 — real TC-PRIMED file errors, fixed at the likely root cause

Two real errors from actual cached files, both traced to the same
underlying issue: GMI's error was directly informative -- requesting
`TB_36.64V` from group S2 got xarray's own suggestion of `TB_166.0V`,
an S1-only channel per the documented mapping. That means the group
actually opened was S1, not the S2 that was requested. AMSR2's
`OSError: [Errno 22] Invalid argument` on Windows, for a differently-
numbered group, is consistent with the same root cause rather than a
separate problem: passing a nested "passive_microwave/S2"-style path to
xarray's `group=` parameter not reliably opening the right group across
engines/platforms.

**Fixed by removing the hardcoded group numbers entirely** rather than
re-guessing new ones from documentation already shown to be unreliable
in practice. `_discover_tcprimed_channels` now walks the actual file via
h5py's `visititems()` -- reading the frequency directly out of
TC-PRIMED's own variable naming (e.g. "TB_36.64V"), matching a strict
`TB_<freq><pol>` pattern that deliberately excludes higher-order names
like "TB_183.31_3.0V" (an offset-from-line-center channel, not a
window frequency). `read_overpass_as_swath` now reads everything --
discovery AND the actual TB/lat/lon/time data -- directly via h5py's
own path navigation, never constructing a "parent/child" group string
for xarray to interpret. This sidesteps both reported failure modes at
what's most likely their shared root, rather than patching each
symptom separately.

Verified thoroughly using a mock built specifically to replicate h5py's
real, documented API precisely (Group/Dataset distinction, `visititems`
walking full relative paths, `[()]` indexing) -- not a loose
approximation:
- Reconstructed the exact reported bug (S1 has TB_166.0-series
  channels, S2 has the needed TB_36.64/TB_89.0 channels) and confirmed
  discovery now correctly finds S2, ignoring S1's string-similar but
  wrong channels.
- Confirmed AMSR2's different target frequency (36.5 vs GMI's 36.64)
  is discovered correctly.
- Confirmed fill-value masking (-9999.9 -> NaN) still works correctly
  in the rewritten reader.
- Confirmed a genuine "channels not found" case raises a useful,
  diagnostic error showing exactly what groups/variables actually
  exist, rather than a cryptic failure.

**Honest limit**: this is the most likely shared root cause based on
the evidence in both error messages, fixed and tested against a
carefully faithful h5py mock -- but not yet confirmed against a real
TC-PRIMED file on real hardware, including specifically whether it
resolves the Windows-specific OSError. The next real run against your
cached files is what actually confirms this.

## MWSynth 0.72 — the AMSR2 89GHz bug wasn't basin-specific, it was mine

Reported as "seems to be North Indian Ocean specific," but the actual
diagnostic error told a different story: S1-S4 (10.65/18.7/23.8/36.5GHz,
none of which carry an A/B scan prefix) were all found correctly --
only 89GHz was missing. That's not a basin pattern, it's a channel-name
pattern.

**Real cause, and it's a regression from last round, not a NIO-specific
issue**: AMSR2's 89GHz channels carry a documented A/B scan-indicator
prefix (e.g. "TB_A89.0V") -- this was correctly known and hardcoded
several rounds back, then genuinely dropped when 0.71 rewrote channel
discovery from hardcoded groups to a dynamic regex. The regex
(`^TB_(\d+\.?\d*)([HV])$`) required the frequency to start immediately
after "TB_", so "TB_A89.0V" simply never matched at all. This affects
every AMSR2 file with 89GHz data, in any basin -- the apparent "NIO
only" pattern is just "whichever basin's AMSR2 files happened to be
processed first, and possibly other basins' downloaded storms didn't
include any AMSR2 passes at all."

**Fixed**: the regex now accepts an optional leading "A" or "B" before
the frequency, and A-scan is preferred deterministically when both
exist for the same channel (interlaced sub-swaths at essentially the
same locations, so this is a reasonable, simple default) -- explicitly
NOT just "whichever h5py's `visititems()` happens to walk first," which
could vary by file or platform and silently disagree run to run.

Verified thoroughly given this is the second real bug in this same
function in two rounds:
- Confirmed GMI-style names (no A/B prefix) still match correctly --
  no regression for the sensor that was already working.
- Reconstructed the exact reported failure (S1-S4 present, S5/S6 with
  A/B-prefixed 89GHz channels) and confirmed A-scan is now found and
  preferred.
- Specifically tested the walk encountering the B-scan group *before*
  the A-scan group in iteration order, confirming the preference is
  deterministic and doesn't depend on which one h5py happens to visit
  first.
- Re-ran the full `read_overpass_as_swath` pipeline end-to-end with a
  realistic multi-group AMSR2 structure (S1/S4/S5/S6 all present,
  fill-value masking included) to confirm nothing else broke.

## MWSynth 0.73 — single file open per read; the "slowdown" is mostly the fix working

Confirmed directly: SHEM had the same 89GHz failure on 0.71, which
settles that the A/B-prefix bug was never basin-specific -- that was a
hypothesis last round, now verified rather than assumed.

**On 0.72 seeming slower**: the primary explanation is almost certainly
not a performance regression. On 0.71 every AMSR2 file failed fast --
one file open, discovery fails, RuntimeError, skip, with no data read,
no best-track interpolation, no GOES fetch, no algorithm run, and no
export. On 0.72 those same files now succeed, so each one runs the
entire pipeline, including a network fetch of three GOES bands, which
very likely dominates everything else. NIO went from "13 files erroring
instantly" to "13 files each doing a full download-and-process cycle."
That's the fix working, not the code getting slower per unit of work.

**Real inefficiency found and fixed anyway**: each file was being opened
twice -- once inside `_discover_tcprimed_channels`, once again in
`read_overpass_as_swath`. Now opened once, with discovery accepting
either a path or an already-open handle (kept path-accepting so it
stays independently callable for diagnosing a single file by hand).
Verified by counting actual opens in a test: 1, down from 2, with A/B
scan preference and fill-value masking both confirmed unchanged.

**On the NESDIS HDF5 tutorial**: reviewed the offer but didn't fetch it
for this round -- it's a 2015 general introduction to HDF5 concepts
(groups, datasets, attributes, the HDF4→HDF5 transition), and the
current issues aren't from misunderstanding HDF5's data model. The bugs
so far have been a wrong-group assumption (fixed by dynamic discovery)
and a variable-naming convention specific to AMSR2 (fixed by the regex).
If a future problem turns out to involve HDF5 internals where general
background would genuinely help -- chunking/compression affecting read
performance, or an unusual storage layout -- that reference would be
worth pulling in then, rather than reading it speculatively now.

## MWSynth 0.74 — two real bugs the run log exposed, one of them data-poisoning

**1. All AL/EP/CP best tracks were 404ing.** `/atcf/btk/` holds ONLY the
current season -- confirmed against the live listing, which contained
nothing but 2026 storms. Historical tracks live at
`/atcf/archive/<year>/b<basin><nn><year>.dat.gz` (gzipped). That's why
the NATL/EPAC/CPAC phase "took mere seconds" and produced 90
no_best_track errors: every 2023/2024 storm in those basins failed
instantly, silently discarding exactly the basins GOES can actually
see. Now tries btk first, falls back to the gzipped archive.

**2. `no_goes_imagery: 1654` was hiding something worse than slowness.**
`goes_fetch.find_nearest_file` selects purely by TIME -- it returns
whichever mesoscale sector was scanning nearest the requested
timestamp, with no check of where on Earth it was pointed. GOES cannot
see the Indian Ocean or most of the Western Pacific, so for WP/IO/SH
storms the "nearest in time" file is a sector over a different part of
the planet. Files that returned nothing were merely slow; files that
returned *something* were worse -- they were exported as training
examples pairing a real MW target over (say) the Bay of Bengal with
GOES imagery of the Atlantic.

This means the 166 examples from this run are suspect and should be
deleted rather than trained on. All 166 came from WP/IO/SH storms
(AL/EP/CP all failed at best-track), i.e. exactly the basins where this
mismatch occurs.

Fixed with `_goes_covers_storm`, which verifies the storm's position
falls inside the fetched image's actual lat/lon bounds (with a margin,
so a storm on the exact edge doesn't count) before the example is used.
This is the same failure mode as the MW geographic-coverage miss found
much earlier in this project: matching on time without validating
geography. Verified against the real cases -- an Atlantic storm in an
Atlantic sector passes; NIO, WP, and SH storms against that same sector
are all correctly rejected, as is a storm sitting exactly on the
boundary.

**3. Deprecated `torch.cuda.amp.*` calls** updated to `torch.amp.*`.

**Note on CUDA**: the run reported "no CUDA device found," so training
ran on CPU. The RTX 3050 is present, so this is almost certainly a
CPU-only PyTorch build -- `pip install torch` pulls CPU-only wheels on
Windows unless the CUDA index URL is specified. Check
`torch.cuda.is_available()` and reinstall from pytorch.org's selector
if it returns False.

**Still open**: Himawari-9 ingestion. With the coverage check in place,
WP/IO/SH storms will now skip fast and cleanly instead of silently
producing bad data -- but they still produce no training examples until
Himawari support exists. AL/EP/CP should now work properly, which is
where usable data will come from in the meantime.

## MWSynth 0.75 — the 2.5-hour run explained, and actually fixed

Asked why WP/IO/SH took 2.5 hours before rerunning. Traced it instead
of guessing, and found 0.74 would NOT have fixed it.

**Root cause**: `list_available_files()` lists the S3 prefix
`ABI-L1b-RadM/YYYY/JJJ/HH/`, which contains every band and both meso
sectors for a whole hour -- roughly 2000 objects -- then filters
client-side and discards 15/16 of them. It is called once PER BAND, so
fetching bands 13/9/7 for one scene re-listed the same ~2000 objects
three separate times. At ~6 paginated S3 calls per attempt x 1821
attempts, ~2.5 hours is exactly the expected arithmetic.

**0.74's coverage check did not address this** -- it ran AFTER all three
fetches, so out-of-view storms still paid the full network cost before
being rejected. It fixed data quality, not speed. Worth stating plainly
rather than letting the previous round's fix look more complete than it
was.

**Fix 1 -- `goes_fetch.satellite_can_see()`**: pure geometry, zero
network. Computes angular distance from the satellite's sub-satellite
longitude (GOES-19: 75.2W, GOES-18: 137W) and rejects anything beyond
65 deg from nadir (tighter than the geometric ~81 deg limit, since
viewing quality degrades badly toward the limb). Called in the mining
loop BEFORE any GOES fetch, so every WP/IO/SH attempt now costs
microseconds instead of ~5 seconds. Verified against real positions
from the actual run: Atlantic/EPac/CPac pass, NIO/WPac/SHEM all
rejected.

**Fix 2 -- per-hour listing cache** in `list_available_files()`. Bands
13/9/7 for one scene now share a single listing. Measured: 6 S3 LIST
calls -> 2 for the same work, a 3x reduction that applies to every
legitimate Atlantic/EPac scene too, not just the skipped ones.

Expected effect on the rerun: the ~1654 WP/IO/SH attempts collapse from
hours to seconds, and the AL/EP/CP storms (now reachable at all thanks
to 0.74's archive-URL fix) fetch roughly 3x fewer S3 listings each.

## MWSynth 0.76 — storm-level skip before any file is opened

Asked for a runtime estimate before rerunning, which surfaced one more
real cost 0.75 left in place: `read_overpass_as_swath()` fully
decompresses and reads a file's channel arrays BEFORE the per-file
field-of-view check runs. For WP/IO/SH -- basins GOES cannot see at all
-- that meant ~1800 files of pointless disk I/O.

Added a storm-level check immediately after best-track is fetched: if
no position on the storm's entire track is within the satellite's view,
the whole storm is skipped and none of its files are ever opened.
Verified by asserting zero calls to `read_overpass_as_swath` for a WPac
storm.

## MWSynth 0.77 — parallel mining (and a thread-safety bug fixed first)

The `.npz` files aren't downloaded, they're generated locally -- what's
actually consuming a connection is GOES imagery from S3, so that's the
real optimization target.

**Found a genuine hazard before parallelizing**: `download_and_open()`
wrote straight to the final cache path. With concurrent workers, two
threads wanting the same file both see it missing and write the same
path simultaneously, and a third can open a half-written file and get a
corrupt dataset. Now downloads to a thread-unique temp path and
`os.replace()`s it -- atomic, so a reader sees either no file or a
complete one. Same lesson as the PPS NRT truncated-cache bug earlier in
this project. The per-hour listing cache also got a lock.

**Parallelized the per-file mining loop** (`max_workers`, default 6).
Per-file work is network-bound, so threads genuinely help despite the
GIL -- boto3 and the HDF5/numpy reads release it while waiting.
Measured with simulated 100ms-per-fetch latency over 24 files: 7.23s
serial vs 1.21s at 6 workers, a ~6x speedup, with attempted/saved
counts exactly correct at both settings (confirming the locked counters
aren't racing).

6 is deliberately modest rather than very high -- past ~8 the
bottleneck shifts to S3 throttling and the CPU cost of
generate_synthetic_mw, so more workers mostly add contention.
`MAX_WORKERS` at the top of run_ml_pipeline.py is easy to raise.

## MWSynth 0.78 — why only 2025 saved: GOES-19 didn't exist yet

"Only 2025 storms are saving" had a concrete cause, confirmed against
NOAA/NESDIS and the AWS Open Data registry: **GOES-19 became operational
GOES-East on April 7, 2025.** Before that, GOES-East was GOES-16. The
project's BUCKETS dict contained only GOES-18 and GOES-19, and the
mining default was hardcoded to "GOES-19" -- so every 2023/2024
Atlantic and East Pacific storm queried a bucket with no data for those
dates and returned "no imagery." The storms were perfectly visible; we
were asking the wrong satellite.

Confirmed operational eras now encoded:
  GOES-16  East 75.2W   2017 -> 2025-04-07
  GOES-19  East 75.2W   2025-04-07 -> present
  GOES-17  West 137W    2019 -> 2023-01-10
  GOES-18  West 137W    2023-01-10 -> present

Added `noaa-goes16` and `noaa-goes17` buckets plus
`select_satellite(lat, lon, when)`, which picks the satellite that was
actually operational at that date AND can see that position, preferring
the smaller viewing angle. Mining now auto-selects per scene; passing an
explicit `satellite=` still overrides.

Verified against real cases -- 2023/2024 Atlantic now routes to
GOES-16, 2025 Atlantic to GOES-19, pre-2023 Pacific to GOES-17, and
uncovered basins/dates return None cleanly. One test "failure" during
this work turned out to be my expectation being wrong rather than the
code: an East Pacific storm at 117W is 42 deg from GOES-East's sub-point
but only 20 deg from GOES-West's, so selecting GOES-18 there is correct,
not a bug.

Expected effect: 2023 and 2024 should now produce training examples
instead of silently yielding nothing.

## MWSynth 0.79 — the 0.0000 loss was training on nothing

302 examples across three seasons and all four sensors is real progress.
The training run, however, did not train. A loss of exactly 0.0000 from
epoch 1 through 50 is not fast convergence -- the output layer is
zero-initialized, so the model predicts 0 at the start, and a zero loss
means the target it is compared against is also zero everywhere the loss
counts.

`masked_loss()` divides the masked error sum by a denominator clamped to
a minimum of 1. With an all-zero mask that is 0/1 = exactly 0.0, for any
target whatsoever, forever -- and the gradient is zero too, so the
weights never move. Confirmed directly: with a target of -1.05
everywhere, an empty mask yields 0.0 and a full mask yields 1.05.

Reproduced the export -> dataset path end-to-end at the real 256px
patch size with a realistic 500x500 scene: mask sum 160,000 and residual
mean -1.04, i.e. the logic is sound and *should* produce a nonzero loss.
So the problem is in the actual mined data, which cannot be inspected
from here.

**Added `diagnose_training_data.py`** to answer this definitively
against the real files. It reports, per example: grid bounds, whether
the storm falls inside them, what fraction of the whole grid has finite
MW target values, and -- the decisive number -- how many supervised
pixels exist inside the storm-centered patch the trainer actually uses.
It separates the two possible causes: an empty mask (MW coverage exists
in the scene but not at the storm) versus a zero residual (backbone and
target identical, which would be an export bug).

**Made the silent failure loud.** Training now prints a `supervised_px`
count every epoch and raises immediately after epoch 1 if it is zero,
rather than running 50 epochs and writing a checkpoint that encodes
nothing. The existing checkpoint at
`~/.synthetic_mw_tc/ml_checkpoints/mw_correction_best.pt` should be
deleted -- it is a zero-correction model, and ml_inference will happily
load and apply it.

Also worth noting from the mining summary: `goes_does_not_cover_storm`
at 1270 is now by far the largest skip reason. That is the mesoscale
sector limitation -- sectors follow whatever NHC is actively watching,
so most storm-times have no sector on them. Full-disk fallback is the
clear next lever for increasing the dataset size, once the zero-loss
issue is understood.

## MWSynth 0.80 — narrowing the empty-mask cause

The diagnostic returned 10/10 empty masks, 0 zero-residual. That rules
out the export bug (backbone and target are genuinely different arrays)
and confirms the real MW simply never lands at the storm on the GOES
grid.

Two candidate causes remain, and they need different fixes, so
`diagnose_training_data.py` now also reports **average MW coverage over
the whole GOES grid**, which separates them cleanly:
  - 0% across the entire scene -> the swath never lands on the grid at
    all. That is swath geolocation: a longitude convention mismatch
    (0..360 vs -180..180) or lat/lon arrays whose shape does not match
    the Tc array they geolocate.
  - nonzero but absent at the storm -> geolocation is fine and the
    problem is patch placement / storm position.

**Added `diagnose_tcprimed_swath.py`**, which reads a raw cached file
exactly as the pipeline does and prints, per frequency: Tc/lat/lon
shapes (flagging a mismatch outright), lat and lon ranges (flagging any
longitude above 180), finite-value counts, and whether the swath
actually contains the storm position the file's own
`overpass_storm_metadata` reports. A swath that does not cover its own
storm is unambiguous proof of a geolocation error.

This matters especially for AMSR2, where 37GHz and 89GHz live in
different groups (S4 and S5) at different resolutions -- pairing one
group's latitude/longitude with another group's Tc would produce
exactly this symptom.

## MWSynth 0.81 — root cause: TC-PRIMED longitude is 0..360

The swath diagnostic found it outright. TC-PRIMED stores longitude in
0..360; every other component of this project -- GOES grids, NHC and
IBTrACS best-track, the PPS NRT swaths -- uses -180..180.

Real numbers from an AL012023 file: the swath spanned lon
[274.29, 298.96] and reported its own storm at 289.54. The GOES scene
for that storm sat near -70. Those share no longitude at all, so the
regridded MW landed nowhere on the grid -- 0.00% coverage across the
entire scene, an empty supervision mask in every patch, and therefore a
loss of exactly 0.0000 for 50 epochs.

Worth noting what was NOT wrong: the swath data itself was pristine --
100% finite values, physically sensible Tc ranges (123-279 K), lat/lon
shapes correctly matching their Tc arrays including AMSR2's 37GHz
(198,243) vs 89GHz (198,486) split across different groups, and the
swath genuinely containing its own storm. Only the coordinate
convention was wrong.

Fixed with `_normalize_longitude()` applied to every lon array read from
TC-PRIMED. Verified against the exact values from the real diagnostic
(274.29 -> -85.71, 289.54 -> -70.46, 303.69 -> -56.31), with
already-negative input passing through unchanged, and confirmed the
converted swath now overlaps a realistic GOES scene where before it
shared no longitude whatsoever.

End-to-end confirmation through the full export -> dataset path using a
swath built in native 0..360: **101,760 supervised pixels (38.8%)** and
a residual mean of -0.92, versus zero before.

**Re-mine before training.** Every existing .npz was exported with the
broken geolocation and contains an all-NaN target. Delete both the
training data and the useless checkpoint:
  ~/.synthetic_mw_tc/training_data/
  ~/.synthetic_mw_tc/ml_checkpoints/mw_correction_best.pt
The cached TC-PRIMED .nc files are fine and do not need re-downloading --
only the export step was wrong.

## MWSynth 0.82 — why 4 hours to reach SH092024, and a basin filter

**The answer to "how could this take so long":**
`fetch_best_track_ibtracs()` streamed the ENTIRE IBTrACS ALL-basin CSV
through `csv.DictReader` on every single storm lookup. That file covers
every basin from 1840 to present -- roughly 700k rows of ~180 columns --
and DictReader allocates a dict per row. Looking up ~100 WP/IO/SH storms
meant ~100 full passes and tens of millions of throwaway dicts, all so
each storm could be discarded immediately afterwards for being outside
GOES coverage. The storm-level field-of-view skip added in 0.76 ran
AFTER this lookup, so it never prevented the expensive part.

Two fixes:

**1. IBTrACS is now indexed once.** Built with `csv.reader` and fixed
column indices (far cheaper than DictReader), storing only the eight
fields actually used, keyed by USA_ATCF_ID. Per-storm lookup becomes a
dict hit. Measured on a synthetic 12k-row file: 124 ms to build, then
0.037 ms per lookup -- roughly 3300x per lookup, and real IBTrACS is
~60x larger than that test file.

**2. Basin exclusion before any lookup.** `exclude_basins` defaults to
("WP", "IO", "SH") -- basins with no GOES coverage at all. These are now
skipped from the filename alone, with zero best-track lookups and zero
file reads. Verified directly: given SH/IO/WP/AL storms, only AL
triggers a lookup.

Set `EXCLUDE_BASINS = ()` at the top of run_ml_pipeline.py to re-enable
them once Himawari-9 support exists.

Expected effect: the WP/IO/SH phase should drop from hours to
effectively instant, and AL/EP/CP best-track lookups also get the
indexed path for free.

## MWSynth 0.83 — the mining speedup that actually matters

Worth separating two things that were being conflated: AMP,
torch.compile and DataLoader tuning affect TRAINING, but training was
never the slow part. The 50-epoch run completed without complaint. The
hours were all in mining, which is network-bound, and none of those
techniques touch it.

**The real win: download one band, not three, before rejecting.**
Mining fetched bands 13, 9 and 7, and only then checked whether the
scene actually contained the storm. Coverage rejection is the single
most common outcome -- 1270 of 1739 attempts in the real run -- so two
thirds of the traffic on the dominant path was downloaded and
discarded. Now band 13 is fetched alone, coverage is checked, and bands
9 and 7 are fetched only if the scene passes. Verified directly: 10
rejections cost 10 downloads, previously 30.

Rough arithmetic on the last run: ~1270x3 + ~300x4 ~= 5000 downloads
becomes ~1270x1 + ~300x4 ~= 2500. Combined with `max_workers` raised
from 6 to 10 (reasonable on a 200 Mbps line) and the 0.82 basin filter
removing the WP/IO/SH phase entirely, well under an hour is a
reasonable expectation.

**Training-side additions** (real, but secondary):
- `pin_memory` for faster host->GPU copies, `persistent_workers` and
  `prefetch_factor` so workers are not respawned each epoch -- which
  matters more on Windows, where workers are spawned rather than forked.
- `torch.compile` attempted with a silent fallback. It can speed up
  convolutional stacks meaningfully, but Triton support on Windows has
  historically been patchy, so a failed compile prints a note and
  training continues uncompiled rather than failing.
- TF32 matmul/conv and `cudnn.benchmark` enabled on CUDA. The 3050 is
  Ampere, the input size is fixed, and TF32 precision is well within
  tolerance for brightness-temperature regression.
- AMP was already present (autocast + GradScaler) and is unchanged.
- `BATCH_SIZE` raised 2 -> 4 in run_ml_pipeline.py. This is still an
  estimate, not a measurement on the actual 6GB card -- if it hits an
  out-of-memory error, drop it back to 2; if epoch 1 runs comfortably,
  raising it further is the biggest remaining training speedup.

## MWSynth 0.84 — torch.compile fallback actually works now; step selection

**The crash was my bug.** `torch.compile` is LAZY -- it returns a wrapped
model immediately and only compiles during the first forward pass. The
try/except in 0.83 wrapped only the `torch.compile()` call, so it caught
nothing: the real failure (TritonMissing) surfaced later inside the
training loop and killed the run. A fallback that cannot catch the
failure it exists for is not a fallback.

Fixed by forcing a real warmup forward pass with a correctly-shaped
dummy batch INSIDE the guard, so compilation either succeeds there or
falls back to the uncompiled model before training starts. Verified the
uncompiled path is retained rather than lost.

**Default changed to off.** Not worth installing Triton here: Windows
needs a separate version-matched triton-windows package, and the same
run reported "Not enough SMs to use max_autotune_gemm mode" -- a 3050 is
too small for compile's better optimizations. Real upside is small
relative to the setup pain. `COMPILE_MODEL = True` in run_ml_pipeline.py
for anyone who wants to try; it now degrades safely.

**Step selection added**, as requested:
    python run_ml_pipeline.py             # prompts
    python run_ml_pipeline.py --step 1    # mine only
    python run_ml_pipeline.py --step 2    # train only
    python run_ml_pipeline.py --step all  # both
Verified each option runs exactly the intended phases. Step 2 is the
common case from here: 298 examples are already on disk, mining is
network-bound and slow, training is neither -- so iterating on training
settings should never require re-mining.

Mining results this run: 1514 attempted, 298 saved, and the WP/IO/SH
basins now cost 102 instant filename-only skips instead of hours.
`goes_does_not_cover_storm` at 1140 remains the dominant loss and is
the mesoscale-sector limitation; full-disk fallback is the lever for
growing the dataset substantially.

## MWSynth 0.85 — first real trained model

Training worked. 58.4M supervised pixels per epoch (zero, two versions
ago), and the loss actually moved.

In physical units -- the loss is mean absolute error on the residual
divided by TB_STD (40 K), so multiplying back:
  starting point (~0.51):  backbone off by ~20 K MAE vs real MW
  best validation (0.2893): ~11.6 K MAE
a ~44% reduction in the parametric backbone's error, measured on
validation storms the model never trained on (the split is by storm,
not by example, so this is generalisation rather than memorisation).

Two things the curve shows honestly:
- Overfitting begins around epoch 29. Train loss keeps falling
  (0.287 -> 0.253) while validation flattens at ~0.29-0.30 and bounces.
  With only 43 validation examples that bounce is largely noise; epoch
  44's 0.2893 is barely better than epoch 29's 0.2959.
- 255/43 examples is still a small dataset. The result is real but not
  yet robust.

Added in response:
- `ReduceLROnPlateau` (halve LR after 4 stalled epochs) and early
  stopping (`early_stop_patience=10`), so future runs stop wasting
  epochs at the plateau instead of drifting toward the training storms.
- A safeguard in ml_inference: checkpoints saved from a torch.compile()'d
  model carry an `_orig_mod.` prefix on every key and would fail to load
  into the uncompiled inference model. The prefix is now stripped when
  present. Not needed for the current checkpoint (compile was off) but
  it would be a confusing failure later.

**The checkpoint is now live.** ml_inference loads
mw_correction_best.pt automatically, so the next generated image has the
correction applied with no further action.

## MWSynth 0.86 — controls for evaluating the correction honestly

First image with the trained correction live. Mixed read, stated
plainly rather than spun either way:

Plausible: the small warm eye sits exactly where the IR eye is, in both
37 and 89 GHz. A compact intense storm (RMW 33 km) genuinely should
have a larger, more saturated core than the weaker Karina frames from
0.62-0.67, so a bigger signature is not by itself wrong.

Concerning: the core is unusually uniform. Real 37 GHz imagery of a
storm this organised normally shows eyewall structure and banding
within the core rather than a smooth mass. Spatial smoothness is the
known signature of a pixel-wise L1 loss trained on limited data -- it
converges toward the conditional median, which is smooth. 255 examples
is well short of what learning fine structure would take.

Rather than guess, added the controls to measure it:

- `MWSYNTH_ML_STRENGTH` env var (default 1.0) scales the correction.
  Set to 0 to disable it entirely without deleting the checkpoint,
  which makes a clean A/B of the same frame trivial.
- `MWSYNTH_ML_MAX_K` (default 25) clamps per-pixel correction
  magnitude. A larger correction than this is not a refinement of the
  backbone, it is the model asserting more than the training data can
  justify.
- The progress log now reports mean and peak |delta| in Kelvin per
  frame, so the correction's actual magnitude is visible instead of
  inferred from colours.

The mean/peak numbers are the thing to look at first: if peak |delta|
is pinned at the clamp across the core, the model is over-asserting and
strength should come down. If mean |delta| is a few K and peak is
under the clamp, the magnitude is reasonable and the smoothness is a
structure problem to solve with more data, not a magnitude problem.

## MWSynth 0.87 — the A/B verdict, and fixing what the A/B exposed

Two A/B pairs, EP11 (Aug 31, high confidence) and Edouard AL05 at
landfall (Sep 1, age confidence 0.15, NEXRAD deliberately off). The
0.86 controls did their job, and the answer is more interesting than
either "it works" or "it doesn't."

**0.86's diagnosis was wrong, and backwards.** The uniform core blamed
on L1 smoothing is the parametric backbone's own signature, not the
model's. At strength 0 on EP11 the 37 GHz core is a filled disk out to
r≈56 px with 0.33 centre fill; at strength 1.0 it is an annulus with an
open eye, 0.00 centre fill, and *higher* internal gradient (4.46 vs
3.07). The correction adds structure. On EP11 it put the eyewall at a
radius of ~33 km against a reported RMW of 33 km, which is the fine
structure 0.85/0.86 doubted 255 examples could teach.

**On Edouard it does the opposite, and is wrong.** Age confidence 0.15
with no radar puts the backbone at 69% of the output (vs 25% on EP11),
so this is the first frame where the correction actually leads.
Recovering Kelvin by inverting the composite colour tables: it warms
the 37 GHz core by ~18 K in both polarizations, implying a raw
correction of ~26 K against a 25 K clamp, with 41% of V37 and 65% of
H37 pixels inside r<40 px pinned at that clamp. That pushes red_raw
from 277.1 to 294 — out the top of the [260,280] window — and the
37 GHz emission signature goes from 56 red pixels to zero. The 89 GHz
scattering core drops 87%. Edouard genuinely had a core on radar and in
MW from 35-40 kt; the correction deleted it. Best guess is regression
to the intensity prior: vmax 50 kt normalises to -0.75σ and RMW 9 km to
-1.26σ, and on 255 examples a model handed "weak, small" scalars
learns to suppress, overriding imagery that shows a real core.

Note that dialling strength back does not rescue that frame. With only
2.9 K of headroom below the window top and a 17.2 K push, strength would
have to drop to ~0.17 to preserve any red at all. The sign is wrong
there, not just the size.

### Fixes this exposed

**The clamp bounded the wrong space.** `MAX_CORRECTION_K` is per-channel
in raw Tb, but the composites render PCT combinations:
PCT37 = 2.181·V - 1.181·H over a 20 K window, PCT89 = 1.818·V - 0.818·H
over 90 K. A correction moving V and H oppositely is amplified 3.36x at
37 GHz, so under 6 K per channel traverses the entire visible range —
and 0.86's mean/peak-|delta|-in-Kelvin log would have reported that as
comfortably within budget. Added `MWSYNTH_ML_MAX_PCT_K` (default 12),
which bounds the combination by scaling each frequency's V/H pair
together, preserving polarization structure rather than clipping the two
channels into a ratio neither implied.

**Strength was a non-linear dial.** The old order scaled by strength and
then clamped, so a raw 60 K prediction gave 25 K at both strength 1.0
and strength 0.5 — identical output exactly in the core, where the clamp
binds and where the dial matters most. Now clamps first, then scales.
Verified: 0.5 gives exactly half of 1.0.

**Newly mined training examples were being poisoned.** The ML step runs
on the backbone inside `generate_synthetic_mw`, and `diag["backbone_*"]`
was captured after it, so once a checkpoint existed every mined example
carried the current model's prediction — as an input channel and inside
the (real_MW − backbone) target. A v2 trained on that would learn the
residual of an already-corrected backbone while inference still applied
it to an uncorrected one, under-correcting by exactly v1's contribution
and compounding each round. The correction is purely additive, so the
delta is now recorded and subtracted back out. The existing 298 examples
predate the checkpoint and are clean.

**Out-of-bounds patch padding was ~6σ out of distribution.** A storm
near its crop edge got a zero-filled patch margin, normalising to
(0 − 260)/40 = −6.5 — a value the model never saw in training, presented
as a hard-edged block against real data. Now pads with `TB_MEAN`, i.e.
exactly 0.

**Retrained checkpoints were never picked up.** The model cache keyed on
path alone, but retraining overwrites the same
`mw_correction_best.pt`, so a running session served the stale model
indefinitely while the docstring claimed otherwise. Now keyed on
(path, mtime, size).

**RMW fallback disagreed with the backbone.** `synthetic_algorithm`
falls back to a Willoughby RMW estimate when best-track RMW is missing,
but `ml_inference` independently fell back to `RMW_MEAN` (30 nm). On any
such frame the model was told "average storm" while sitting on a
backbone shaped by a different radius. The caller now passes the same
value the backbone used. Not active on Edouard (its 9 km came from best
track; Willoughby would have given ~35 km), but silent whenever it fires.

### Other changes

- ML strength is now a typed GUI field (0.00–2.00) rather than an env
  var read once at import, which made within-session A/B impossible. The
  strength, mean/peak |delta|, PCT-space peak as a percentage of each
  colour window, and a clamp-saturation warning are stamped into the
  figure title — an A/B pair saved to disk previously carried no record
  of which frame was which.
- Credentials reduced to the two services actually used: PPS (the NRT
  feed carrying GMI/WSFM/AMSR3) and Earthdata (archive GMI). JAXA
  G-Portal and NOAA CLASS entries removed — AMSR2-NRT retired after
  Aug 31 2026, SSMIS shut down in September 2026, and NOAA CLASS was
  never automated at all (its fetch raised `NotImplementedError`), so
  that box could never have done anything. Archive GMI is kept
  deliberately: it has no NRT retention window and has historically been
  the path that still worked when NRT did not. The G-Portal AMSR2 code
  in `mw_ingest.py` is left in place, just without a credential entry.
- Sensor rotation dropped SSMIS-NRT and AMSR2, which sat *ahead* of
  AMSR3-NRT, so every generate failed over two dead sensors before
  reaching a live one.
- `calibration_state.get_status_summary()` now warns when an offset
  reaches 90% of `MAX_OFFSET_K`. The 37 GHz offset is currently −36.8 K
  against a 40 K bound: the EMA is close to saturating, at which point
  it stops tracking the real bias and merely reports the bound. A
  standing 37 K offset is not a calibration nudge, it is a sign the
  37 GHz background Tb is structurally wrong.

## MWSynth 0.88 — measuring the correction in the space it actually lives

0.87 established that the correction helps on an intense storm and hurts
on a weak-but-organised one. Everything here is aimed at making that
distinction measurable automatically, rather than by inverting colour
tables off saved PNGs by hand.

### Structural scoring (`mw_structure_metrics.py`, new)

Every quantitative number this project had was a bias/RMSE statistic and
none of them could see what a correction does. The residual from
`mw_compare` is measured on the scalar `tb37`/`tb89` fields, which the
correction never touches. Mean/peak |delta| measures magnitude but not
direction — it rates "sharpened the eyewall to the right radius" and
"erased the core" identically, because both are a few K per pixel.

The new module scores three things on the composite's red-channel PCT
combination (the thing actually rendered, not the standalone PCT
product):

- **eyewall radius vs RMW.** Radius of peak azimuthal-mean redness. An
  `rmw_ratio` near 1.0 means the signature peaks where the wind maximum
  is, which is where a real one belongs.
- **eye contrast and core area.** Ring vs filled disc, and how much area
  the core covers in km² rather than an uncomparable pixel count.
- **PCT excursion and clipping.** How far outside the colour window the
  field runs. Clipping stats are measured within 150 km of the centre —
  computed over the whole grid they read ~0.95 for every frame, because
  the far field is clipped low everywhere, and the signal drowns.

Validated against synthetic ring and disc fields: a ring at 40 km scores
`eyewall 38 km, eye_contrast 0.99`; a filled disc at r=90 km scores
`eyewall 2 km, eye_contrast 0.00`.

The comparison warning needed care. Keying it on core-area collapse alone
flagged a known-good disc→ring improvement as a failure, because opening
an eye legitimately shrinks the core. The discriminator is peak redness:
opening an eye leaves the eyewall as red as the disc was, while pushing
the field out of the window takes the peak down with it. The warning now
requires both to collapse, and separately emits a positive note when the
eyewall moves toward the RMW.

### Single-run A/B

`_weighted_fuse` is a per-pixel linear combination whose weights depend
only on which sources have coverage, never on their values, and the ML
correction enters through the backbone alone. So re-running the fuse with
the backbone replaced by the recorded ML delta — and the other sources
replaced by zeros carrying their original NaN pattern, so the identical
weight renormalization happens — recovers the correction's exact
contribution to the fused output. Verified to 1.1e-13 against a
separately-computed uncorrected fuse.

This matters because a two-run A/B re-draws the noise floor and advances
the calibration EMA between the two frames, so part of any measured
difference was never the correction at all. That ambiguity is why the
first A/B pair was hard to read. A new GUI checkbox swaps the
convective-signal panel for the ML-OFF 89 GHz composite from the same
generate, and the structural comparison for both frequencies is logged
and stamped into the figure title.

### Training changes

**PCT-space loss term.** Nothing in training ever saw the V/H
combination the composites render, so nothing penalised a correction
moving V and H in opposite directions — amplified up to 3.36x at 37 GHz.
Worse, `best_val_loss` selected the saved checkpoint on that same blind
metric. `masked_pct_loss` adds an L1 term on the PCT combination,
supervised only where both channels of a pair have coverage (taking one
alone would invent a polarization difference the data doesn't contain).
Confirmed on constructed cases: a common-mode (+1,+1) error and an
opposite-sign (+1,−1) error score identically under per-channel L1, and
3.36x apart under the PCT term. Weight is `PCT_LOSS_WEIGHT` (default
0.5); set to 0 to reproduce the old objective.

**Intensity-stratified validation split.** The split shuffled storms and
took the first 15%, which at this dataset size could easily produce a
validation set with no weak storm in it — so a model that suppresses
weak-storm signatures would score as healthy. The split now samples
within each intensity band (TD / TS / cat1-2 / major), and validation
loss is reported per band each epoch and recorded in the checkpoint.
Tested on a deliberately lopsided set (12 major, 5 cat1-2, 2 TS, 1 TD):
every populated band with more than one storm appears in validation, and
a single-storm band goes to training rather than being held out into a
regime the model would then never see.

**Scalar conditioning dropout.** `vmax_layer` and `rmw_layer` are
constant across the patch, making them by far the cheapest features for a
fully-convolutional network to key on — cheaper than learning structure
from imagery. On a small dataset that is a trap, and it matches the
Edouard failure exactly. `SCALAR_DROPOUT_P` (default 0.25) blanks both
layers on a fraction of training samples, forcing the imagery to carry
the prediction. Zero is the right blank value: these layers are already
normalized, so 0 means "average storm", the same imputation already used
when best-track RMW is missing.

### Also fixed

`combined_loss` took `pct_weight=PCT_LOSS_WEIGHT` as a default argument,
which binds at def time — so `run_ml_pipeline`'s override would have been
silently ignored. Resolved at call time instead. That is exactly the kind
of knob that appears to work and does nothing.

## MWSynth 0.89 — the new metric was wrong, and the first real frame proved it

Karina 20:59Z, an intense storm at RMW 28 km, run with the 0.88
instrumentation. Two things came out of it.

### The metric had a defect

The title reported `37GHz eyewall 38->50 km` and `89GHz eyewall 50->46
km`. Measuring the rendered 37 GHz panel independently reproduced the
50 km exactly -- but reproducing it also showed why the number is not
trustworthy. The azimuthal-mean redness profile at 37 GHz is:

    r(km)  0    12   24   36   44   48   52   56   60   68   76
    red    0.00 0.09 0.44 0.76 0.99 1.00 1.00 1.00 1.00 0.93 0.81

Four consecutive bins at exactly 1.00. `nanargmax` over a saturated
plateau returns the FIRST bin of it, so "eyewall = 50 km" actually meant
"where clipping starts" -- a function of the colour-table boundary, not
of the storm. Shift the whole field warm or cold and that number moves
while the structure is identical. At 89 GHz it degenerated further: the
profile decreases monotonically from 0.98 at the centre, so argmax
returns the innermost bin.

Cause: `redness()` clips to [0,1] because that is what the rendered image
does, and 0.88 then used that clipped field to LOCATE the maximum.
Locating a maximum requires an unbounded quantity. `radial_pct_profile()`
now finds the eyewall as the minimum of the unclipped red-channel PCT,
with the clipped redness still used for area and contrast where clipping
is the correct behaviour. Verified: a ring at 40 km now reports 38 km
whether or not it is saturated, where before the saturated version
reported the clip boundary instead.

Two guards added alongside it. `saturated_annulus_km` reports how wide
the saturated band is, because a correctly-located eyewall inside a 32 km
saturated annulus is real but invisible in the rendered image. And the
eyewall/RMW comparison is now suppressed entirely when `eye_contrast` is
below 0.05: on a flat filled disc the PCT minimum is arbitrary within the
disc, so reporting a radius there invites reading a trend into noise.

### Correction to the 0.87 record

0.87 claimed the correction "put the eyewall at a radius of ~33 km
against a reported RMW of 33 km." That measurement was the equivalent
radius of the enclosed EYE -- the inner edge of the eyewall -- not the
eyewall itself. On this frame the same two quantities are 30 km (inner
edge) and 50 km (peak). They are not interchangeable, and comparing an
inner-edge measurement against RMW flattered the correction. The claim
should not be relied on; it needs redoing with the fixed metric before
anything is concluded about whether the correction places the eyewall
correctly.

### What the frame shows regardless

`peak |delta| 25.0K` with `35% of patch pinned at clamp` -- exactly the
clamp-saturation condition seen on Edouard, now on an INTENSE storm with
a fresh 70% pass. So clamp saturation is not a weak-storm failure mode;
it is the model's normal operating state, and the 25 K clamp is doing
real shaping work on most frames rather than acting as a rare safety
bound. That is worth knowing independently of the eyewall numbers.

The single-run A/B panel worked as intended: the ML-OFF 89 GHz composite
is visibly a larger, more filled core than the corrected one, from the
same generate, with no run-to-run noise between them.

### 0.89 also covers (versioning, UI, and Tier 2)

**Versioning.** The previous entry was briefly labelled 0.88.1. There are
no patch components in this scheme: versions count up by one, and a fix
to the last version is simply the next number. `APP_VERSION` now carries
that rule in a comment, including what happens past 0.99 — the counter
keeps going (0.100, 0.101, ...) rather than rolling to 1.0, because the
leading 0 means "not finished" and MWSynth should not promote itself
because a counter wrapped. Anything sorting versions must compare the
part after the dot as an integer, since these do not sort as strings.
The window title said 0.86 while the code was three versions further on.

**Scrollable settings.** The controls column had outgrown a non-maximized
window and was CLIPPING the lower group boxes — they were unreachable,
not merely off-screen. The controls now sit in a `QScrollArea`, with the
width constraint moved to the outer container (constraining the inner
panel left no room for the scrollbar, which then overlapped the
controls). The status log is attached outside the scroll area so progress
output stays pinned instead of scrolling away.

**37 GHz baseline (Tier 2).** `bg_tb_37` was 165.0 K and `bg_tb_89` was
250.0 K. Real AMSR-class clear tropical ocean is about 200-205 K at
37 GHz V-pol and 265-270 K at 89 GHz V-pol, and `mw_compare` measures
`freq_37ghz` against real V-pol. 165 K sits between ocean H-pol and V-pol
and matches neither — most likely an H-pol figure used where a V-pol one
was needed.

The evidence is unusually direct. The persisted EMA, which learns
(synthetic − real) and knows nothing about these constants, had converged
to −36.8 K and −16.5 K over 54 observations. Those are the baseline
errors, to within about a Kelvin on both channels. The EMA had
independently rediscovered the bug and was spending its whole 40 K clamp
budget hiding it, within 8% of saturating.

Set to 202.0 / 266.0. `max_depression_37/89` were retuned from 90/140 to
35/105 in the same change, because they had been tuned as offsets from
the wrong baselines — left alone they now implied floors of 112 K and
126 K, which no real observation approaches. These constants drive the
scalar fields only; the V/H composites use separate constants and the
imagery is unchanged.

**Calibration state is now versioned.** Changing the baseline while a
stale offset survives would double-count: a leftover −36.8 K would have
added another 36.8 K on top of the fix, leaving the field ~37 K too WARM
and looking from the outside exactly like a fresh calibration problem,
with the EMA then spending another ~50 passes unlearning it.
`CALIBRATION_BASELINE_ID` is stored in the state file, mismatched state
is discarded on load, and the reset is reported in the status line rather
than happening silently. The flag is deliberately sticky for the process
lifetime — `load_state()` runs several times per generate, so clearing it
on the next call would have wiped the notice before anything displayed
it.

**Mirror augmentation off by default (Tier 2).** `fliplr` reverses a
cyclone's rotational handedness, and SH basins are excluded from mining —
so half of all augmented samples had a handedness that never occurs at
inference, spending capacity on an impossible case and denying the model
spiral handedness as a cue. `ALLOW_MIRROR_AUGMENTATION` gates it; turn it
on if SH storms are ever added, at which point mirroring becomes
legitimate. Rotation is kept.

**Novelty-scaled strength (Tier 2).** Strength now tapers as the storm's
(vmax, RMW) moves away from the training distribution: full inside
1 sigma, down to a 0.25 floor beyond 2.5 sigma. Distance is measured
against statistics recorded from the actual training files at checkpoint
time (`train_scalar_stats`), falling back to the ml_constants
normalization values. Effective strength is reported separately from the
dial value in both the log and the title, so a correction quietly scaled
to 0.25 while the GUI reads 1.00 cannot happen.

Being honest about its limits: this would have applied ~0.77 on Edouard,
where preserving the signature needed roughly 0.17. Edouard sits only
~1.5 sigma out because what made it unusual was weak intensity combined
WITH an organised core, and a distance over two scalars cannot see a
combination. The taper is a conservative mitigation, not a fix for that
failure — scalar dropout and more training data remain the real answers.

## MWSynth 0.90 — the storm centre stopped moving

Lowell (EP12) at 11:04Z, GOES-only. The synthetic core sat visibly east
of the eye in the GOES imagery underneath it.

Cause: `besttrack.interpolate_fix()` ended with

    if not after:
        return before[-1]

Past the last best-track entry it returned that fix unchanged, including
its `valid_time`. Best track lands every 6 h, so an 11:04Z frame was
being built around the 06:00Z centre with five hours of motion thrown
away. For a storm moving 10-12 kt that is about 0.7 degrees of longitude
-- roughly 80 km -- and the centre positions everything: the radial
weight profile, the eyewall ring, the RMW/ROCI envelope, and the ML
correction patch. All of it was placed beside the storm it was
describing.

This is worst exactly where it is least obvious. On a fused frame the
real MW pass anchors the pattern and partly masks the error. On a
GOES-only frame like this one the backbone carries 100% of the output,
so the misplacement is the whole image.

`interpolate_fix` now extrapolates the position along the storm's recent
motion instead of freezing. Motion is estimated over a 12 h arm rather
than the last pair alone, because best-track positions are quantized to
0.1 degree and a single 6 h pair carries that rounding as noise. On a
reconstructed Lowell-like track the 11:04Z centre moves 0.72 degrees
(78 km) west of the 06Z fix, which matches the offset in the image.

Position only. Intensity, RMW and ROCI are carried forward unchanged --
linearly extrapolating a rapidly intensifying storm's vmax produces
values that never occur, and RMW/ROCI are noisy enough between advisories
that a projected trend is mostly noise. Those fields degrade gracefully
when stale; position does not.

Three guards, all tested: gaps beyond 12 h fall back to the old freeze
(a linear projection over half a day cannot represent recurvature, and a
confidently wrong centre is worse than an obviously stale one); implied
speeds above 50 kt are treated as a data error and ignored; a single fix
with no motion estimate is left alone. The symmetric backward case is
handled the same way.

**This also fixes the MW morphing, which was never broken.**
`mw_ingest.morph_swath_to_time` advects a real pass by the displacement
between two `interpolate_fix` calls. With the endpoint frozen, a pass
observed after the last best-track entry computed a zero displacement and
was left unshifted at its original position. The advection logic was
correct all along; the positions it was handed were not. Every consumer
-- generation, morphing, and mining -- routes through this one function,
so they are all corrected together.

A projected centre is now labelled in the figure title
(`centre PROJECTED 5.1h past last best track`) and carried in
`StormFix.extrapolated_hours`. A projected centre is otherwise
indistinguishable from an observed one in the output, which is how this
survived as long as it did.

Also confirmed working on this frame: 0.89's calibration-baseline reset
fired correctly and said so in the title, rather than silently reusing
offsets learned against the old baseline.

## MWSynth 0.91 — the eyewall was in the wrong place, and the background was boiling

Started from the Lowell finding that the structural metrics reported
`eyewall/RMW 2.27`, unchanged by the ML correction. Chasing that one
number turned up four separate bugs in the generation path, two of which
had nothing to do with the eyewall.

### 1. Ring width ignored RMW

`ring_width_km` interpolated on Vmax alone, in absolute kilometres:
`[20,60,100,140,180] -> [80,60,40,25,18]`. Nothing in it knew the radius
it was centred on. Lowell at ~110 kt with a 19 km RMW got a 36 km ring
width — a Gaussian whose sigma is nearly twice its own centre radius is
not a ring, it is a blob. Width is now a FRACTION of RMW with the
intensity dependence kept as a multiplier on that fraction, bounded to
8–60 km.

The error scaled with compactness, which is why it went unnoticed: the
old constants gave peak/RMW near 1.0 for RMW 55–80 km, 1.7 at 28 km, and
2.2 at 19 km. Typical storms looked fine.

### 2. The radial weight saturated and destroyed the ring

`np.clip(0.55*ring + 0.65*envelope, 0, 1)` — the coefficients sum to
1.20, so wherever both terms were strong the sum exceeded 1 and was
flattened. For Lowell that clipped across **36% of the inner 60 km**,
turning the eyewall into a plateau of exactly 1.0. The argmax was then
decided by `eye_suppress` shaving the inner side, pushing the apparent
eyewall to the OUTER edge of the clipped region. The ring term was being
erased precisely in the storms it exists to represent.

Now divided by the coefficient sum instead. This is the same failure
mode as the metric bug in 0.89 — taking a maximum over a saturated
plateau — appearing independently in the physics.

Together the two fixes land the eyewall within a few percent of RMW:

| case | RMW | old peak | new peak |
|---|---|---|---|
| Lowell 19 km / 110 kt | 19 | 41.5 (2.18x) | 19.8 (1.04x) |
| Karina 28 km / 115 kt | 28 | 48.2 (1.72x) | 29.0 (1.04x) |
| Edouard 9 km / 50 kt | 9 | 0.2 (0.03x) | 8.0 (0.89x) |
| typical 55 km / 90 kt | 55 | 59.5 (1.08x) | 51.8 (0.94x) |

End-to-end on a synthetic 19 km-RMW storm the metric now reports
`eyewall 18 km, ratio 0.94`, against 2.27 on the real frames before.

Note the Edouard row. Its filled, ringless core — which 0.87 attributed
to the ML correction — was the backbone all along: a 65 km ring width
around a 9 km RMW, with no eye suppression below 65 kt, peaks at r=0.
The correction was being blamed for a shape it inherited.

Deliberate side effect: peak weight is now ~0.82–0.95 rather than a
saturated 1.0, so response amplitude drops slightly. The old 1.0 was an
artifact of clipping, not a real maximum, and the calibration EMA absorbs
the level change.

### 3. The noise floor was re-drawn every frame

`np.random.default_rng()` with no seed. Harmless in a still; in a loop it
is not. Measured on the real 5-frame Lowell GIF, a calm far-field ocean
corner changed **17–18 RGB levels between consecutive 10-minute frames
while the storm core changed only 6–7**. The most meteorologically
stable part of the scene was the most visually volatile — the background
boiled while the storm sat still.

Now seeded from storm ID and grid shape, so the floor is fixed in grid
space across a loop while still differing between storms and domains.
`zlib.crc32` rather than `hash()`, since Python salts string hashing per
process and `hash()` would have reintroduced flicker between runs.
Verified: repeat generation of the same frame is now bit-identical.

### 4. `_weighted_fuse` emitted 0 K where nothing covered

Where no source was valid it substituted a weight of 1.0 against a sum of
0, producing 0.0. That is finite, so `assert_finite` would not catch it,
and 0 K renders as fully saturated red in both colour tables — a silent,
confident, physically impossible blob. Emits NaN now so the existing
guard fires with a clear message. Should be unreachable; it was the
failure mode that mattered.

### Smaller

The band-2 term added 0.08 on top of a `convective_signal` whose three
coefficients already summed to exactly 1.0, then clipped — flattening
every pixel already above 0.92. Same saturation pattern in miniature,
now renormalized. And `backbone_response_37` and `_89` were computed
twice from identical inputs; consolidated, with a note that the absence
of any frequency-dependent SPATIAL response is a real modelling
simplification worth revisiting.

### Flagged, not changed

Real MW gets FULL confidence out to `MW_FUSION_MAX_DISTANCE_KM * 0.7` =
210 km from the nearest real sample, tapering to zero at 300 km. Given
AMSR/GMI cross-track sample spacing of ~5–10 km, anything beyond roughly
20 km is outside the swath rather than in an inter-scan gap — so a
substantial share of a reported "70% coverage" may be extrapolation
carried at full weight, smeared perpendicular to the swath edge.

Left alone deliberately. These distances were tuned against real seam
artifacts across several versions, and tightening them blind here, with
no real swath available to test against, risks regressing something
already validated visually. Worth measuring first: report the
distance-to-nearest-real-sample distribution alongside the coverage
percentage, so "70% coverage" can be split into real and extrapolated.

## MWSynth 0.92 — measuring the centre instead of inferring it

0.91 fixed the eyewall placement (2.27x RMW -> 1.19x on the next real
Lowell frame) but the 37 GHz emission signature vanished with it: core
area went 6,417 km2 to 159 km2 uncorrected and 0 km2 corrected, with
literally zero red pixels in the panel. Two candidate causes, and no way
to tell them apart: either the 37 GHz constants are wrong, or the centre
the narrowed ring is drawn around is.

Three attempts across two sessions to locate the IR eye by hand kept
locking onto gaps in the cirrus canopy — a warm hole surrounded by cold
cloud looks locally identical to an eye. That guesswork is now a module.

### tc_center_fix.py (new)

Estimates the storm centre from GOES IR and compares it to the centre the
field was actually built around. Three constraints, all necessary, all
learned from the failed attempts:

1. Search only within 200 km of the prior centre. An eye is never 200 km
   from the best-track position; a canopy gap easily is.
2. Require the warm region to be fully ENCLOSED by the cold CDO, not
   merely warm and nearby.
3. Score on thermal contrast, compactness AND distance from the prior,
   rather than taking the largest or warmest candidate.

A fourth constraint emerged in testing: a GOES crop is a parallelogram
inside a rectangular array, so the white off-grid corners are enclosed by
cloud as far as the array is concerned and register as enormous,
infinitely-warm "eyes". The contrast term then evaluated to NaN, and
because NaN comparisons are False such a candidate slipped past the score
test and won by default. Candidates containing non-finite pixels are now
rejected outright.

Validated against the real Lowell 11:52Z IR panel: converges on the same
eye from five priors spanning 120 km in every direction (contrast 41 K,
confidence 0.70-0.86). On synthetic controls it recovers a known eye to
2 km, correctly reports an 81 km offset when handed a wrong prior,
degrades to an explicitly low-confidence CDO centroid when there is no
eye, and returns None when there is no cold cloud.

Deliberately diagnostic only. Nothing is repositioned from it. Automated
centre-fixing from imagery is a real discipline and a
blur-and-find-the-warm-hole heuristic is not a substitute for one — this
exists to make a misplaced centre visible, not to fix it.

The IR panel now carries a cyan "+" at the fix used and a red "x" at the
detected eye with a line between them, and the offset appears in the
title, escalating to FIELD IS BUILT AROUND THE WRONG POINT past 40 km.
It is placed above the structural numbers because it qualifies all of
them: an eyewall radius measured from the wrong centre is precise and
meaningless.

### What the diagnostic already says about the 37 GHz loss

Running the same idealised storm twice, once correctly centred and once
with the fix displaced 75 km:

| | 37 GHz core area | 89 GHz eye contrast |
|---|---|---|
| centre correct | 114 km2 | 1.00 |
| fix 75 km off | 0 km2 | 0.27 |

So centre error does collapse the 37 GHz core, supporting the hypothesis
— a narrow ring drawn around the wrong point lands on warm eye and clear
air instead of the eyewall, and the radial response never reaches the
level the emission term needs.

But it does not fully explain it. Even perfectly centred, that scene
produced only 114 km2 of 37 GHz core. The 37 GHz amplitude is genuinely
low after 0.91, independent of any centre error. Both things are true,
and the constants are NOT being retuned on the strength of an idealised
synthetic scene — the point of building the diagnostic was to stop
guessing. Run a real frame with the overlay: if the eye marker sits on
the fix and 37 GHz is still empty, it is the constants.

### Metric guard

`compare_structure` reported `eyewall/RMW 1.19` on a field whose core
area was exactly 0 km2 — a confident number computed from nothing, since
with no core the PCT minimum is located on a noise floor. The
eye_contrast guard from 0.89 covered the flat-disc case but not the
empty-field case. Radius reporting is now also suppressed below
`MIN_MEANINGFUL_CORE_KM2` (450 km2, about a 12 km-radius disc), with a
distinct message for the two situations.

## MWSynth 0.93 — pre-release audit: dateline, concurrency, tests, packaging

A full pass over the project ahead of an open-source release. Four real
bugs, a regression suite, and a restructure of the documentation.

### The dateline bug

`interpolate_fix` subtracted longitudes naively. A storm crossing 180 has
consecutive best-track fixes at, say, +179.5 and -179.5 — a 1 degree
westward move — and the difference read as -359 degrees. The linear
interpolation then swept **backwards across the entire globe**:

    fix 00Z: lon +179.5      fix 06Z: lon -179.5
    interpolated at 03Z  ->  lon 0.00      (the Atlantic)

Everything positioned from the fix followed it there. The extrapolation
path was affected differently: the implied speed came out around 3,600 kt,
tripped the plausibility guard, and silently fell back to freezing.

This could never fire in the Atlantic or east Pacific, which is why it
survived — it would have fired on the first west Pacific storm to cross
the dateline, which is exactly the case that arrives with Himawari
support. Fixed by taking the shortest arc around the circle: no tropical
cyclone moves 359 degrees in six hours.

The same naive subtraction appeared in every radius and distance
calculation in the project — `_radial_weight`, `tc_center_fix`,
`mw_structure_metrics`, the ML patch centring in both `ml_inference` and
`ml_train`, the satellite-overpass distance in `tle_predict`, and the
morph displacement in `mw_ingest`. All now use a shared wrap. Verified on
a grid spanning 180: maximum distance from a point near the dateline is
now 781 km rather than ~20,000.

### Naive datetimes

`load_local_swath` used `datetime.now()` as the scene-time fallback when a
filename could not be parsed. That is naive and in LOCAL time, while every
other timestamp in the project is tz-aware UTC — comparing them (as
best-track matching and pass-age confidence both do) raises
`can't compare offset-naive and offset-aware datetimes`, and where it did
not raise the value was wrong by the machine's UTC offset. Only reachable
when a filename is malformed, which is precisely when a local file is
least likely to be well-formed. Also replaced a deprecated
`datetime.utcnow()`.

### Calibration state durability

`save_state` wrote directly into the target path, so a crash or a second
writer between truncation and completion left a half-written file.
`load_state` recovers from that by falling back to defaults, which means
the failure is **silent** — calibration learned over dozens of passes just
quietly resets. Now writes to a temp file in the same directory and
`os.replace`s it in, which is atomic on POSIX and Windows.

`update_offset` also does a read-modify-write and is now serialised under
a lock. The frame-loop path already skips calibration updates to avoid
this race, but a lock costs nothing and covers what remains: pressing
Generate twice, or the Reset button firing while a worker is mid-update.
Verified with 200 concurrent updates across 8 threads — none lost, no temp
files left behind.

### Edge-case sweep

The generation path was stress-tested against a storm outside the grid, a
storm on the grid corner, vmax 0 and 200, missing/zero/huge RMW and ROCI,
all-warm and all-cold imagery, IR containing NaNs, uniform imagery with
zero gradient, a 4x4 grid and a single-row grid. All produced finite,
physically-bounded output without raising. Graceful degradation was
confirmed by blocking `torch`, `arm_pyart`, `sgp4`, `skyfield`, `gportal`
and `earthaccess` at import: every module still imports and the affected
feature disables itself.

### tests/test_mwsynth.py (new)

47 regression tests, needing no network, credentials, torch or display.
Nearly every case corresponds to a bug that actually shipped and was found
by staring at an image afterwards: the saturated radial weight, the frozen
storm centre, the boiling noise floor, the dateline sweep, the strength
dial that was not linear, the 0 K fusion output, the off-grid corners
winning as giant eyes, the eyewall radius reported on an empty field.

Each is cheap to check automatically and was expensive to notice by eye,
which is the trade a test suite should make. Real ingest, GUI construction
and model training are deliberately not covered — those need credentials,
a display and a GPU respectively, and pretending to test them here would
be theatre.

### Documentation and packaging

- `README.md` rewritten as a project front page: what it does, how it
  works, install, credentials, usage, layout, tests, and an explicit
  **Known limitations** section covering the hand-tuned calibration, the
  6-hourly best track, the ML correction's observed 37 GHz behaviour, the
  210 km full-confidence MW extrapolation, the missing WP/IO basins, and
  the identical 37/89 GHz spatial response. The previous README opened by
  describing the project as a "Non-ML algorithm" and listing AMSR2/SSMIS
  as working, both of which stopped being true some versions ago.
- The 101-entry development log moved to `CHANGELOG.md` (this file).
- `requirements.txt` split into core and optional groups with the reason
  each optional package exists and what breaks without it.
- `.gitignore` added — notably covering `credentials.json`, which holds
  real service passwords, and the `.pt`/`.npz` model and training
  artifacts.

`LICENSE` is referenced by the README but deliberately not created: the
licence choice belongs to the project owner, not to this audit.

## MWSynth 0.94 — real MW fusion is a toggle again

The checkbox was removed some versions ago on the reasoning that real
data is the point of the tool, not an optional extra. That is right for
normal use and wrong for diagnosis: with fusion always on, there is no
way to force a GOES-only frame when a pass IS available, and therefore no
way to look at the parametric backbone or the ML correction in isolation.
Real MW masks both wherever it has coverage — on a 98%-coverage frame the
37 GHz field is almost entirely real data, so nothing about the backbone
can be judged from it.

All the plumbing was still intact; only the control had been removed and
`auto_calibrate` hardcoded to True. Restored as "Fuse real MW when
available", defaulting to on, wired through both the single-frame and
loop paths.

The GOES-only branch messages were reworded to distinguish a DELIBERATE
GOES-only frame from a failed search. Both produce a frame with no real
MW in it, and reading "no pass found" on a frame where fusion was
switched off would send you hunting a data problem that does not exist.

Also corrected `ml_inference`'s module docstring, which still said the
correction was "ALWAYS APPLIED -- not a toggle". That stopped being true
when the runtime strength control was added in 0.89.

### What the toggle immediately showed about 37 GHz

Forcing GOES-only on a synthetic storm with the centre verified correct
(5 km offset) reproduces the Lowell result: the 37 GHz core barely
registers. Measuring the generated field directly rather than
approximating it:

    r(km)    V37     H37    V-H   PCT37  redness
       15  248.4   178.5   69.9   330.9     0.00
       19  213.8   147.7   66.1   291.8     0.01
       25  196.5   135.4   61.1   268.6     0.57
       30  201.6   139.0   62.7   275.6     0.22

There IS a signature, but it peaks at redness 0.57 and sits right on the
0.5 core threshold — which is why it reads as 159 km2 on one frame and
0 km2 on another. It is marginal, not absent, and resolution-dependent.

The mechanism is the polarization difference. Real 37 GHz ocean V-H is
about 55-65 K and COLLAPSES to 10-20 K in heavy rain, because emission
from liquid water is weakly polarized; that collapse is what drives PCT
down and makes an eyewall red. Here V-H only falls from 77 K to 61 K.
Two compounding causes:

1. `bg_v_37 - bg_h_37` is 80 K, against a real ocean value nearer 55-65 K
   — the field starts over-polarized.
2. The emission term is multiplied by `(1 - scat_pot_vh)`, and
   `vh_scatter_warm_bound = 230 K` drives that factor to about 0.29 at an
   eyewall IR of ~195 K. So the maximum 30 K V-H collapse the constants
   allow is throttled to roughly 6 K exactly where the eyewall is. At
   37 GHz ice scattering is weak and liquid emission dominates in the
   eyewall, so treating cold IR as near-pure scattering is the wrong
   physics for this frequency specifically.

NOT retuned here. The diagnosis is now precise enough to act on, but it
should be confirmed against a real GOES-only frame with the centre marker
overlapping the fix before constants are changed — which is exactly what
this toggle now makes possible.

## MWSynth 0.95 — 37 GHz had the sign inverted, validated against real GMI

A GOES-only Lowell frame at 15:59Z with the IR centre check confirming
the fix to 2 km (confidence 0.98), alongside a real GPM GMI pass 44
minutes earlier. That is the first time this project has had synthetic
output and same-storm truth side by side with the centre independently
verified, and it settled the 37 GHz question immediately.

### The finding

Reading 37H off the GMI colour bar:

| region | real GMI 37H | synthetic 37H |
|---|---|---|
| far-field ocean | 182-193 K | 170 K |
| eyewall / core | **269 K** (p10 257, p90 276) | **135 K** |

Real 37H RISES about 80 K from ocean to eyewall. The synthetic FELL 35 K.
The sign was inverted and the eyewall was wrong by roughly 134 K.

The cause was physical rather than a tuning error. `scat_pot_vh` was a
single field shared by both frequencies. At 89 GHz that is right — ice
scattering dominates deep convection and depresses Tb hard, and the real
89H imagery confirms it. At 37 GHz ice scattering is weak; the eyewall
signature is EMISSION from liquid precipitation, which warms both
polarizations toward the physical temperature and collapses the
polarization difference. Sharing the field made the 37 GHz eyewall a
scattering feature, which it is not.

### The fix

`vh_scatter_warm_bound` is now per-frequency: 230 K at 89 GHz (unchanged)
and 190 K at 37 GHz, so only genuine overshooting tops scatter there.
200 K was tried first and was still too warm — a normal eyewall top near
195 K came out 35% scattering, holding H37 about 70 K low.

The 37 GHz V/H constants were retuned against the measured values:
`bg_h_37` 170 -> 182, `emission_boost_h37` 85 -> 93,
`emission_boost_v37` 55 -> 28, `max_depression_v37` 110 -> 20,
`max_depression_h37` 70 -> 12. 89 GHz is untouched; it does not use
`scat_pot_vh` at all.

On a test scene the eyewall now warms instead of cooling and V-H collapses
from 68 K to 33 K. It does not yet reach the real ~5-10 K, because the
radial response on that scene only reaches about 0.44 — its WV band is
synthetic, so the convective signal is understated. Worth re-checking on a
real frame rather than tuning further against a scene that cannot saturate.

Being explicit about a weakness this exposed: using CLOUD-TOP IR to decide
emission-versus-scattering is physically weak at 37 GHz in the first
place, since the emission comes from liquid water below the freezing
level that cloud-top temperature only loosely constrains. The threshold is
a pragmatic stand-in and the obvious place a radiative-transfer treatment
would replace guesswork.

### A metric that was measuring the wrong thing

The real 37 GHz core is bright WHITE in the NRL composite, not red —
an optically thick, nearly unpolarized emitting layer gives high G and B
with only a small pink ring at the very centre. So "37 GHz core area
0 km2" was never the right complaint: the real product is barely red
either. The synthetic field was wrong by 134 K in a way redness cannot
see.

`pol_diff_far_k`, `pol_diff_core_k` and `pol_collapse_k` added to
`mw_structure_metrics`, reporting how far V-H falls from the far field to
the core. That is the physically meaningful 37 GHz descriptor. At 89 GHz
it stays informative but inverted in meaning — scattering depresses H more
than V, so V-H tends to widen.

### The western artifact

Differencing the ML-ON and ML-OFF 89 GHz panels pixel-for-pixel, the
change is confined to a hard-edged block spanning about 613 km, matching
PATCH_SIZE (256 px) at this latitude's pixel size. The "weird stuff in
the western half" is the ML correction, and outside that block the field
is untouched — so on a storm whose visible field spans over 1000 km the
patch boundary shows up as a change in character partway across the
image. The taper smooths the transition over roughly 75 km but cannot
hide that the inner region has been treated and the outer has not.

Also measured: the synthetic 89 GHz core is about 3.4x the area of the
real one (32,187 km2 against roughly 9,355 km2), and — more telling than
the ratio — the real signature is a compact eyewall ring plus discrete
spiral bands, while the synthetic is one contiguous mass. The envelope
term scaled from ROCI 444 km spreads response far too broadly.

## MWSynth 0.96 — the 37 GHz fix validated, and the checkpoint is now stale

Two 15:59Z Lowell frames, GOES-only and GOES+GMI (98% coverage,
confidence 0.34, so roughly half backbone and half real MW), with the IR
centre check confirming the fix to 2 km in both. Recovering V37/H37 from
the rendered panels and profiling by radius:

| | far field | eyewall (15 km) |
|---|---|---|
| GOES-only  H37 | 182.0 K | **255.5 K** |
| GOES+GMI   H37 | 180.9 K | 262.1 K |
| real GMI   H37 | 182-193 K | 269 K (p90 276) |
| GOES-only  V-H | 67.9 K | 22.2 K |
| GOES+GMI   V-H | 51.5 K | 4.8 K |

**The 0.95 fix worked.** The GOES-only eyewall went from 135 K to
255.5 K, so the 134 K error is now about 14 K, and the far-field level
lands inside the observed range. The eyewall warms rather than cools, as
it should.

What GOES alone still cannot do is collapse the polarization difference:
V-H reaches 22 K at the eyewall against 4.8 K in the fused frame. That
gap is not a tuning problem. IR measures cloud-top temperature and
passive MW measures the integrated hydrometeor column, and the two
decouple precisely at the eyewall — a mature eyewall and a decaying CDO
can share a cloud-top signature and look nothing alike in MW. Added to
the README's limitations as a first-class constraint rather than a
caveat: GOES-only output is an interpolation between real passes, not a
substitute for one.

### The checkpoint is stale by construction

0.95 inverted the 37 GHz eyewall from a scattering depression to an
emission signature, a ~120 K change at the core. The ML model was trained
to predict residuals against the OLD backbone, so every residual it
learned is now measured from a backbone that does not exist. It showed
exactly that way on these frames:

- GOES-only: 37 GHz eyewall 22 -> 26 km, ratio 1.19 -> **1.40** (away
  from RMW)
- GOES+GMI: 37 GHz eyewall 26 -> **2 km**, ratio 1.40 -> **0.11**, core
  area 3213 -> 1823 km2

Collapsing the eyewall radius to the innermost bin on a frame that is
half real MW is the correction actively destroying observed structure.

A model cannot detect this about itself — it emits confident corrections
either way — so the backbone now announces its own identity.
`synthetic_algorithm.VH_PHYSICS_ID` is recorded into every checkpoint at
training time, and `ml_inference` compares it against the live value,
warning in the log and in the figure title when they differ. Existing
checkpoints have no ID and are reported as `pre-0.95 (unversioned)`.

This is the same pattern as the calibration `baseline_id` from 0.89, and
for the same reason: a learned correction that outlives the thing it was
learned against fails silently and confidently.

**Retraining is required before the ML correction is trustworthy again.**
Mining also needs re-running, since the exported backbones carry the old
physics. Until then, strength 0 is the right setting.

## MWSynth 0.97 — land is not ocean (prompted by Li et al. 2025)

Reading Li, Tan & Bai's DeepTCTransfer preprint (essoar 10.22541/
essoar.173655526.67865984/v1) — a diffusion model generating TC PCT from
multi-channel geostationary IR. Same task, same data sources, and it
points at a gap MWSynth had from the start.

### The gap: no surface type at all

Every background brightness temperature in CALIBRATION was an open-ocean
value applied uniformly across the grid. There was no land mask anywhere
in the project.

That is not a small offset. At 37/89 GHz ocean is a poor, strongly
polarizing emitter (V-H ~55-80 K), while land is a near-blackbody in BOTH
polarizations (emissivity ~0.9-0.95), sitting near its physical
temperature with V-H of a few K. A warm, unpolarized surface is exactly
what heavy precipitation looks like over ocean — it is the signature the
emission model exists to produce. So unmasked land was actively imitating
the feature the composites are built to show, over the coast, which is
where a landfalling storm matters most. Edouard was at landfall.

This is also why polarization-corrected temperature exists in the first
place (Spencer et al. 1989; Cecil & Chronis 2018): to suppress surface
emissivity differences so the scattering signal survives a coastline.

`surface_type.py` resolves a land fraction through `global_land_mask`
(small, pure-Python, embedded mask, no runtime network), falling back to
cartopy's Natural Earth geometries, then to all-ocean with a single
explicit warning. The fallback reproduces the previous behaviour rather
than failing a generate, but says so. Backgrounds are blended by land
fraction and smoothed with the field sigma so a coastline is a shoreline
rather than a one-pixel step.

Verified with an injected coastline: ocean side V-H 68 K, land side 6 K,
land H37 282 K. Two regression tests added, including that a missing
backend degrades to ocean rather than raising.

Land constants are deliberately NOT split by land-cover type. Without a
vegetation or soil-moisture input that would be false precision, and the
goal is to stop rendering coastlines as convection, not to retrieve land
emissivity.

### What the paper independently confirms

- **37 GHz is warm rain and shallow convection, not ice scattering.** The
  authors deprioritize PCT37 for exactly this reason. That is the same
  physics 0.95 arrived at from the GMI comparison, reached independently.
- **Pixel-wise losses over-smooth.** Their ensemble mean beats individual
  members on PSNR/SSIM but loses the high-frequency detail that matters
  for localized extremes, scoring worse on LPIPS and classification bias.
  That is the L1-smoothing concern from 0.86-0.87, confirmed.
- **Storm-level splits with intensity stratification.** They require the
  intensity distribution to match across splits and keep each storm in
  one split only — the same design as 0.88, arrived at for the same
  reason.
- **IR does not cleanly determine what is under the cloud top.** They cite
  Hilburn et al. (2021) for this, which is the constraint added to the
  README limitations in 0.96.

### What it says MWSynth should do differently

Recorded here rather than implemented, since each is substantial:

1. **More IR channels.** Their single-channel experiment (band 13 + mask)
   was worst on every metric; adding bands 8-16 produced the largest
   single improvement in the paper. MWSynth uses four bands (13, 9, 7, 2).
   This is the highest-value change available.
2. **Elevation as an input.** Adding it improved their results further,
   and it is the natural companion to the land mask above.
3. **A probabilistic architecture.** Diffusion beat conditional GAN, which
   beat deterministic CNNs; Pix2Pix could only produce fuzzy contours.
   MWSynth's correction model is a deterministic U-Net with L1 — the
   weakest of the three families they tested.
4. **Two orders of magnitude more training data.** 13,213 paired samples
   from 2015-2021 against MWSynth's ~298. Their TC PRIMED route is
   already half-built here (`tcprimed_ingest.py`).
5. **A larger patch.** They use 256 x 256 at 4 km (1024 km); MWSynth uses
   256 at ~2 km (~512 km), which is why the ML patch boundary was visible
   partway across a large storm in 0.95.
6. **Multi-task output helps.** Generating PCT37 alongside PCT89 improved
   PCT89 as well.

Their PCT coefficients (2.15/1.15 at 37 GHz, 1.70/0.70 at 89 GHz, after
Cecil & Chronis) sit close to but not identical with the NRL colour-table
red-channel coefficients MWSynth uses (2.181/1.181 and 1.818/0.818),
which confirms the existing note in `mw_composites.py` that the two are
related but distinct quantities.

## MWSynth 0.98 — wider footprint, richer inputs (Li et al. steps 1, 2, 5)

Scaffolding for the paper-driven work, done first so the harder items
(probabilistic architecture, two orders of magnitude more data) land on a
model that already has the right inputs and receptive field.

### Step 5: wider footprint at coarser sampling

`PATCH_SAMPLE_STRIDE = 2`, so the model still sees 256 x 256 but each
model pixel spans two native GOES pixels and the patch now COVERS 512
native pixels — roughly 1024 km instead of 512. Li et al. use 256 x 256
at 4 km for the same task.

This was already a measured problem, not a theoretical one: differencing
the ML-on and ML-off panels on Lowell showed the correction confined to a
hard-edged block of about 613 km on a storm whose field spanned well over
1000 km, with the boundary visible partway across the image.

Downsampling is a block AVERAGE, not subsampling, in both training and
inference — the footprint exists to give the model context, and dropping
three of every four pixels would discard the texture that context is
meant to summarize. The two implementations mirror each other
deliberately; training on subsampled data while inferring on averaged
data would be a silent input mismatch of exactly the kind that produces
confident nonsense rather than an error. The correction is upsampled back
by pixel repetition plus a light blur so the stride does not read as
blocking, and the blend taper scales with the stride.

### Steps 1 and 2: more IR channels, and surface conditioning

Input channels go from 9 to 18, with the order fixed in
`ml_constants.INPUT_CHANNEL_LAYOUT` and shared by both paths:

    ir, wv, swir                      (bands 13, 9, 7 -- as before)
    ir_band8/10/11/12/14/15/16        (new, EXTRA_IR_BANDS)
    backbone_v37/h37/v89/h89
    vmax, rmw
    land_fraction, elevation          (new)

Li et al.'s largest single ablation gain came from moving beyond one IR
band — their band-13-only experiment was worst on every metric. Adding
elevation improved things further, and it is the natural companion to the
land mask added in 0.97.

**Bands that a frame doesn't have become a neutral plane rather than
being dropped.** The channel COUNT is therefore constant, so a checkpoint
stays loadable whether or not every band was fetched, and older training
exports remain usable instead of silently shifting every channel index by
one. That property is what makes it safe to land this before the GOES
fetch side is extended.

Elevation currently derives a nominal land elevation from the land mask
rather than pretending to real topography — there is no bundled DEM and
downloading one at generation time is not acceptable for a real-time
tool. That captures the part that matters for passive MW (whether the
surface below is land at all) without inventing terrain detail. The
channel and its normalization exist, so dropping in ETOPO later changes
nothing downstream.

Verified with a stub model: input arrives as 18 x 256 x 256, a supplied
extra band normalizes correctly, an absent one is exactly neutral, and
the region actually modified is 512 native pixels wide.

### Still to do

- **Fetch the extra bands.** `goes_fetch` and the GUI still pull only
  bands 2, 7, 9, 13, so the seven new channels are neutral in practice
  until that is extended. Everything downstream is ready for them.
- **`training_data_export`** has the optional-key hook but does not yet
  write the extra bands or surface fields.
- **Steps 3 and 4** (probabilistic architecture, TC PRIMED-scale
  dataset) are the substantial ones and are deliberately untouched here.

Any existing checkpoint is now doubly invalid — wrong input channel
count, wrong footprint, and already flagged stale by `VH_PHYSICS_ID` from
0.96. Retraining is required, and mining should be re-run.

## MWSynth 0.99 — the re-mine prerequisites, and what the published paper changed

Li, Tan & Bai's peer-reviewed version (JGR: MLC, 10.1029/2026JH001257)
supersedes the preprint and carries two things the preprint did not: a
per-channel IR saliency analysis, and a cloud-type breakdown. Both are
directly actionable, and both are landed here BEFORE re-mining rather than
after — anything that changes the backbone invalidates exports, so the
right moment for it is now.

### The ordering, completed

1. **Extra ABI bands are fetched.** `goes_fetch.fetch_extra_ir_bands()`,
   wired into both the single-frame and loop paths. Best-effort per band:
   a failure means that channel is absent, and the model fills it with a
   neutral plane rather than changing input shape. That property is what
   makes fetching these on a latency budget safe.
2. **The exporter writes them.** Plus `land_fraction`, `elevation_m`, and
   `vh_physics_id`. The regridded arrays are routed through diagnostics so
   the file stores precisely what the model was fed — regridding twice
   would risk the two drifting apart.
3. **Re-mine** is now the next step, and will capture all of the above.

### Bands are now ordered by measured importance

The paper's Figure 5 turns "add more bands" into a ranked list, so
`EXTRA_IR_BANDS` is reordered `(11, 10, 15, 16, 14, 8, 12)`:

- **ch11** (8.5 um, cloud-top phase) — their most critical single input
- **ch10** (7.4 um, low-level WV) — dominant predictor
- ch15 secondary (split-window pair with 13); ch09 intermediate
- ch16 and ch12 minimal — stratospheric CO2/O3, physically decoupled from
  the precipitation layer
- ch08 saturates near the tropopause; ch14 is spectrally redundant with 13

`EXTRA_IR_FETCH_LIMIT` (default 3) trims from the END, so a reduced fetch
drops the channels the paper found least informative rather than an
arbitrary subset. Fetching all seven costs seven extra S3 downloads per
frame; fetching the top three gets ch11 and ch10, which is most of the
value.

### Cirrus is transparent at microwave — the response now reflects that

`convective_signal` was
`0.55*scat_pot + 0.30*wv_mask*scat_pot + 0.15*texture`. The 0.55 base
term was **ungated**: a thick cirrus canopy, cold in IR but with wv_mask
near zero, still scored about 0.54. The paper's cloud-type analysis is
explicit that cirrus and warm water are largely transparent at 37/89 GHz,
with their PMW signal dominated by background emission — effectively
indistinguishable from clear sky.

Now `(0.25 + 0.60*wv_mask) * scat_pot + 0.15*texture`:

| wv_mask | regime | old | new |
|---|---|---|---|
| 1.0 | deep convection | 0.81 | 0.81 |
| 0.3 | thick anvil / cirrus | 0.62 | 0.43 |
| 0.0 | pure cirrus canopy | 0.54 | 0.27 |

Deep convection is unchanged; cirrus halves. Coefficients still sum to
exactly 1.0, so nothing clips — the same discipline the `_radial_weight`
bug in 0.91 taught.

This is very likely the mechanism behind a measured discrepancy: against
a real GMI pass the synthetic 89 GHz core came out about 3.4x the observed
area and contiguous, where the real signature was a compact eyewall ring
plus discrete spiral bands. A cold CDO is mostly cirrus canopy, and
treating the whole canopy as convective inflates in exactly that way.

**Honest limitation on the evidence:** this could not be validated on the
synthetic test scene, because that scene's WV band is derived as `ir + 3`,
so `wv_mask` is near-constant and the scene cannot exercise cirrus
discrimination at all. The weighting is verified numerically and by
regression test; the *effect on a real storm* needs a real frame. Worth
checking on the next Lowell-class case whether the 89 GHz core area moves
toward the observed value.

`VH_PHYSICS_ID` bumped to `0.99-emission37-cirrusgate`.

### Mixed-vintage datasets are now refused, not warned about

`check_physics_consistency()` groups a dataset by the backbone physics
each example was generated under and raises if there is more than one.
Fatal rather than advisory on purpose: the target is
(real_MW − backbone), so examples from either side of a physics change
describe different quantities. Training across both does not average two
views of one thing, it fits one model to two things and converges toward
neither — while every loss curve looks entirely healthy. Pass
`strict=False` to override deliberately.

It lives in `training_data_export` rather than `ml_train` because it only
reads npz files and has no business requiring torch; `ml_train`
re-exports it. That also makes it testable in a suite whose whole point
is running without a GPU.

### Fixes found while doing the above

- The first attempt at wiring the band fetch into the GUI corrupted the
  file: the 8-space `if band13 is None...` pattern is a substring of the
  12-space one, so the second replacement matched inside the first.
  Redone with indentation-aware insertion.
- `_extra_ir_grids` was initially defined after its first use, and the
  exporter was initially passed an `extra_ir` that wasn't in scope at the
  call site (the worker's `finished` signal doesn't carry it). Both
  caught before packaging; the regridded arrays now travel via
  diagnostics, which is the better design regardless.

52 tests passing.

## MWSynth 0.100 — diffusion training path, sized for the actual machine

The diffusion model and its inference wiring already existed; what was
missing is that **`ml_train` had no diffusion path at all** — the model
could be sampled but never trained. That is now closed, along with two
bugs found doing it.

### Bug: training built a 9-channel model

    model = MWCorrectionUNet(in_channels=9, ...)

went stale the moment 0.98 took the input stack to 18. `ml_model`'s
default was updated but this explicit argument overrode it, so training
would have constructed a 9-channel network and then been handed
18-channel input. It would have failed loudly on the first batch rather
than silently, but only after a full mine — the worst time to find it.
Both architectures now take the width from `MODEL_IN_CHANNELS`.

### Bug: the validation loss would have been meaningless

Flow matching regresses a VELOCITY, not a per-pixel prediction. Routing
that through `combined_loss` would have applied the PCT-space term
(0.88) to a velocity field, where the 2.181/-1.181 combination has no
physical meaning whatsoever. The number would have printed happily and
measured nothing. Training, validation and the per-band attribution all
now branch on `CORRECTION_ARCH`.

### Sizing for a laptop RTX 3050

Li et al. trained on four RTX 4090s: 96 GB against 4, a 24x gap. Their
settings are not a target, and shrinking their pixel-space DiT until it
fits would give up the global attention that made it worth choosing.

| | before | now |
|---|---|---|
| sample steps | 24 | 12 |
| ensemble members | 8 | 4 |
| forward passes per frame | 192 | **48** |

Roughly 2.0M conv parameters at `base_channels=48`, with the activation
peak at 256x256 batch 2 fp16 well inside 4 GB. Members are sampled
sequentially on purpose: batching N members multiplies activation memory
by N for little wall-clock gain, since the GPU is already saturated at
batch 1 at this resolution. Drop `DIFFUSION_BASE_CHANNELS` to 32 if it
OOMs — that costs capacity, not correctness.

Honest caveat on the uncertainty field: 4 members gives a noisy spread,
and the sample standard deviation is biased low at small N even with the
ddof correction. Read it as "where is the model unsure", not as a
calibrated error bar.

### Algorithm review: does the parametric backbone still earn its place?

Asked directly, and the answer is **keep it** — the diffusion change makes
it more important, not less.

What the measurements say. Against a real GMI pass 44 minutes from a
GOES-only frame with the centre independently verified to 2 km, the
backbone puts far-field 37H at 182.0 K against 182-193 K observed, and
the eyewall at 255.5 K against 269 K. After 0.91 the eyewall lands at
1.02-1.04x RMW across the intense/compact range. That is a prior worth
keeping.

Why it matters more under diffusion, not less:

  * **Data.** The papers train on 18,165 paired samples. This project
    will have a few hundred. A model that generates the field from
    scratch must learn the entire IR-to-PMW mapping from that; a model
    that generates a residual only has to learn what IR genuinely
    under-determines — which is exactly the quantity Li et al. identify
    as irreducible. The prior carries the physics the data cannot.
  * **Failure floor.** An undertrained from-noise generator produces
    plausible invention. An undertrained residual generator produces a
    small residual, and the output falls back to the backbone — wrong in
    ways that are known, bounded and documented. For a tool whose output
    can be mistaken for an observation, that is the right direction to
    fail in.
  * **Guardrails survive.** The Kelvin clamp, the PCT-space bound and the
    strength dial all operate on a correction. None of them mean anything
    if the model IS the field.
  * **Real MW.** The papers are IR-only by construction. MWSynth fuses
    real passes, and the backbone the residual is measured against
    already contains them, so the correction is learned in the presence
    of real data rather than in place of it.

That is the substantive departure from both papers, and it is closest to
Mardani et al. (2025) residual corrective diffusion, which both cite and
neither applies here.

Real weaknesses that remain, none of which argue for a rewrite:

  1. The 37 and 89 GHz spatial responses are identical by construction —
     all frequency dependence lives in the calibration constants. Real
     sensors differ spatially.
  2. Constants are hand-tuned first guesses, not a radiative-transfer
     model or a fit.
  3. Using cloud-top IR to split emission from scattering is physically
     weak at 37 GHz, where the signal comes from liquid below the
     freezing level. The 0.95 threshold is a stand-in.
  4. The 89 GHz core measured 3.4x the observed area; the 0.99 cirrus
     gate is the candidate fix and is still unvalidated on a real frame.
  5. Real MW is carried at full confidence 210 km from the nearest real
     sample.

Items 1 and 3 are where a radiative-transfer treatment would replace
guesswork, and are the strongest remaining argument for outside
contribution.

## MWSynth 0.101 — the training loss was not measuring what it looked like

First real diffusion training run: 264 train / 34 val, one physics vintage
(the 0.99 provenance guard worked), converged 1.12 -> 0.0754, early-stopped
at epoch 43 with best at 33. Everything behaved. The problem is what the
number means.

### The flow-matching loss is not interpretable on its own

For z_t = t*x1 + (1-t)*x0 with x1 the residual and x0 ~ N(0,1), the best
achievable loss has a closed form:

    E_t[ s^2 / (t^2 s^2 + (1-t)^2) ]

where s is the residual SCALE. It depends on almost nothing else. So:

| residual scale | in Kelvin | Bayes-optimal loss |
|---|---|---|
| 0.05 | 2 K | 0.079 |
| 0.10 | 4 K | 0.157 |
| 0.25 | 10 K | 0.393 |
| 1.00 | 40 K | 1.571 |

The observed best of **0.0754 is essentially the optimum for a residual
scale of about 2 K**, and is simply unreachable if the true residuals are
larger. Watching that curve fall confirms the optimizer works. It does not
confirm the correction is worth applying, and it cannot distinguish a
model that learned the correction from one that learned the correction is
small. Both are real outcomes and they look identical here.

(An earlier version of this analysis claimed a degenerate predictor could
drive the loss to zero regardless. That was wrong -- it blew up at the
t -> 1 singularity. The closed form above is the correct statement.)

### sampled_skill(): the number to actually judge on

Draw a sample and compare its error against the error of predicting zero
everywhere:

    skill = 1 - rmse_model / rmse_zero

    skill > 0    the correction beats leaving the backbone alone
    skill ~ 0    it learned the residual is small -- true, and useless
    skill < 0    applying it makes the frame worse

Reported in Kelvin, masked to pixels with real MW behind them, run every
5 epochs and at the end. Verified against constructed cases: a perfect
model scores +1.000, predicting zero scores exactly 0.000, half-right
scores +0.500, and pure noise of the right amplitude scores -0.415.

**Re-run training to get this number before trusting the checkpoint.** If
skill comes out near zero, the honest reading is that the backbone after
0.95/0.99 is already close enough that 264 examples cannot improve on it
-- which would be a real result, not a failure, and an argument for
leaving ML strength at 0 until there is more data.

### Also fixed

The log printed `(L1=... PCT=...)` with both values identical to
val_loss on every line, because the diffusion branch sets `l1 = pct =
loss` -- `combined_loss` has no meaning against a velocity field, since
the PCT combination is defined on brightness temperatures. Printing them
implied a decomposition that does not exist. Diffusion runs now print
`(flow-matching MSE)` instead.

### Reading the rest of that run

- **val below train throughout** is expected, not overfitting-in-reverse:
  DropPath (0.1) and scalar dropout (0.25) are active in training and off
  in eval.
- **per-band numbers are too small to read.** With n = 6/15/8/5 the
  band means swing (0-34kt went 0.101 -> 0.341 -> 0.155 on consecutive
  epochs). That is sampling noise, not weak-storm degradation. Worth
  keeping for when the dataset is larger; not worth interpreting now.
- **training distribution vmax 60+/-30 kt, RMW 43+/-38 nm.** The RMW
  spread is nearly as large as its mean, so the novelty taper is now
  working off a genuinely wide reference -- and a compact storm like
  Lowell (about 10 nm) sits near -0.9 sigma rather than being exotic.

## MWSynth 0.102 — the skill metric worked, and immediately found two bugs

The 0.101 skill check did its job on the first run. It showed the model
going from actively harmful to modestly useful:

    epoch  5   rmse 25.57K vs zero 15.85K   skill -0.613  (WORSE than nothing)
    epoch 10   rmse 15.65K                  skill +0.012
    epoch 25   rmse 14.41K                  skill +0.091
    epoch 40   rmse 13.65K                  skill +0.139  (best)

So the correction is real but small: about a 14% reduction in residual
RMSE. Without this number the run looked like a clean success at
val_loss 0.0721, and epoch 5 -- where the model was ADDING 10 K of error
-- would have been invisible.

### Bug 1: the checkpoint was selected on the wrong metric

Correlating the two across the run gives **r = -0.41** between val_loss
and skill, where a perfect selection metric would be -1.0. That is weak
enough to matter, and it did:

- best val_loss: **epoch 36** (0.0721) -- this is what got saved
- best skill: **epoch 40** (+0.139, rmse 13.65 K), val_loss 0.0944

The saved checkpoint was chosen by a number I had already established
(0.101) does not measure usefulness. Selection and early stopping now run
on skill for the diffusion architecture, and skill is computed every
epoch rather than every fifth, because it is now the selection metric and
sampling every epoch costs seconds where selecting wrongly costs the run.

### Bug 2: the skill metric penalised the model for being generative

`sampled_skill` drew **one** member. RMSE is minimised by the conditional
MEAN, and a single draw from a conditional distribution has expected
squared error of bias^2 + 2*sigma^2 against the ensemble mean's
bias^2 + sigma^2/N. Scoring one sample counts the model's spread as if it
were error -- which is the exact asymmetry Li et al. describe from the
other side, where their ensemble mean wins on pixel metrics while
individual members win on perceptual ones. It is in this module's
docstring, and I scored with one member anyway.

Now scores the mean of 4 members, and reports the spread alongside.
Expected effect if bias is small: RMSE falls by a factor of ~0.79, so the
observed 13.65 K should land near 10.8 K and skill near **+0.32** rather
than +0.139. That is a prediction, not a result -- worth checking against
the next run.

Sampling steps in the check also went 8 -> 16. Euler error on the learned
(curved) marginal velocity field is a plausible second contributor to the
gap between what the loss implies and what sampling recovers, and steps
are cheap here.

### The gap worth understanding

val_loss 0.072 implies the residual is well predicted GIVEN the
conditioning, yet sampling recovered only 14% of it. Part of that is
bug 2 above. If skill does not improve substantially on the next run,
the remaining explanation is integration error, and the test is simply
whether skill rises with sampling steps -- if it does, the sampler is the
limiter, not the model.

Note also that the zero-correction RMSE, 15.85 K, is the first direct
measurement of the residual scale the backbone leaves behind. It is
consistent with the ~14 K eyewall gap measured against real GMI in 0.96,
which is a reassuring cross-check from a completely different direction.

## MWSynth 0.103 — skill nearly doubled; the remaining error is bias, not sampling

The ensemble-mean fix from 0.102 worked. Best skill went **+0.139 ->
+0.256** (RMSE 13.65 K -> 11.78 K against a 15.85 K zero-correction
baseline), which is a real correction: about a quarter of the residual
the backbone leaves behind.

The 0.102 prediction was +0.32. It came in at +0.256, and the shortfall
is informative rather than disappointing.

### Decomposing the error

With single-draw and 4-member RMSE both measured, bias and sampling noise
separate exactly:

    single draw   MSE = bias^2 + sigma^2       (13.65 K)
    mean of N=4   MSE = bias^2 + sigma^2 / 4   (11.78 K)

    => bias  11.09 K
    => sigma  7.96 K

The derived sigma of 7.96 K sits close to the independently reported
ensemble spread of 6.85 K, which is a good cross-check that the model is
the right one.

The 0.102 prediction assumed bias was negligible. It is not -- **bias is
70% of the residual**, and no amount of sampling touches it:

| members | RMSE | skill |
|---|---|---|
| 1 | 13.65 K | +0.139 |
| 4 | 11.78 K | +0.256 |
| 8 | 11.44 K | +0.278 |
| 16 | 11.26 K | +0.289 |
| infinite | 11.09 K | **+0.301** |

So 4 members already captures most of what ensembling can give. Going to
16 buys +0.03 for four times the sampling cost, which on a 3050 is not a
trade worth making. **The lever on the remaining error is more training
data, not more members or more steps.** `sampled_skill` now reports this
decomposition and the ceiling every epoch, so the point at which more
sampling stops helping is visible rather than inferred.

### The uncertainty field was overconfident, and is now calibrated

Reported spread 6.85 K against an actual error of 11.78 K: the ensemble
**understates its own error by about 1.7x**.

That is not surprising -- a model trained on 264 examples cannot
represent uncertainty arising from everything it has never seen -- but it
matters because that spread is displayed as a confidence field, and an
uncertainty estimate that is confidently too small is worse than none at
all. `ENSEMBLE_SPREAD_CALIBRATION = 1.7` is applied in `ml_inference`
before the field leaves the function, and `sampled_skill` reports the
measured overconfidence every epoch so the constant can be re-derived
whenever the model or dataset changes. It is empirical, not derived, and
labelled as such.

Also switched the spread to `ddof=1`, which was already the case in the
training metric but not at inference -- at 4 members the biased estimator
is low by roughly another 8% on top of the 1.7x.

### Where this leaves the correction

Honest summary: the correction removes about a quarter of the residual,
its uncertainty estimate is now roughly calibrated, and it is bounded by
bias that only more data will move. That is a defensible thing to ship at
a non-zero strength -- unlike the pre-0.96 checkpoint, which the metrics
of the time could not distinguish from harmful.

Worth noting what made the difference: every conclusion in this entry
comes from `sampled_skill`, which did not exist two versions ago. The
same run under the old logging looked like a clean success at val_loss
0.0721 and would have shipped a checkpoint selected on a metric
correlating -0.41 with usefulness.

## MWSynth 0.104 — read TC PRIMED in place on S3, never landing it on disk

The binding constraint on dataset size was laptop storage: 2023-2025 WHEM
GOES already cost ~50 GB of a 500 GB machine, and TC PRIMED at the scale
that would help is far larger than the disk it would sit on.

It does not need to sit there. netCDF4 is HDF5, HDF5 is a random-access
format, and the exporter needs only a handful of variables per file.
Given a seekable file object, the HDF5 library reads only the bytes it
wants -- so the download step becomes ranged GETs and nothing is written
locally.

### s3_range_reader.py (new)

A seekable, caching file-like object over an S3 object using ranged GETs,
handed straight to `h5py.File`. **No new dependencies**: boto3 is already
core and these buckets are already opened unsigned, where s3fs/fsspec
would add two. `tcprimed_ingest.open_overpass_streaming()` is the entry
point; the download path is kept for files read repeatedly.

**The block cache is not optional.** HDF5 issues many small reads walking
metadata -- superblock, B-trees, heap, attributes -- and one GET per read
would mean hundreds of requests per file and be slower than downloading.
Reads are served from aligned blocks.

Block size was measured, not guessed. Simulating an HDF5-like pattern
against a 30 MB file (scattered small metadata reads, then contiguous
chunk reads):

| block | requests | fetched | % of file |
|---|---|---|---|
| 256 KB | 29 | 7.6 MB | 24% |
| 512 KB | 17 | 8.9 MB | 28% |
| 1 MB | 14 | 14.7 MB | 47% |
| 4 MB | 6 | 23.1 MB | 73% |

512 KB is the knee: roughly half the transfer of a 1 MB block for three
more requests, where 256 KB buys another 4 points for nearly double the
requests. S3 request latency is tens of milliseconds, so requests are not
free either. `stats()` reports bytes fetched against object size, so the
saving is measured per file rather than assumed.

Five regression tests cover random access against the source bytes, all
three seek modes, short reads at EOF, the fetched fraction, and the
BufferedReader path h5py actually uses. A subtle seek bug here would
corrupt training data silently rather than failing, which is why the
correctness tests are more thorough than the efficiency one.

### The constraint moved, so it is now addressed too

Removing the source-file problem immediately made the EXPORT the binding
constraint:

| format | per example | 10,000 examples (compressed) |
|---|---|---|
| float32 @ 512 footprint | 23.1 MB | ~115 GB |
| uint16 @ 512 footprint | 11.5 MB | ~58 GB |
| uint16 @ 256 (model res) | 2.9 MB | **~14 GB** |

`pack_tb` / `unpack_tb` store brightness temperatures as scaled uint16 at
0.01 K resolution, spanning 0-655 K with NaN preserved via a reserved
sentinel. Measured round-trip error is 0.005 K, against sensor noise of
0.5-1 K, and it beats float16 in this range (float16's relative precision
gives ~0.15 K at 300 K). Halves the file for no meaningful loss.

Storing at model resolution rather than the native footprint is the
larger win -- another 4x -- and is NOT done here, because it forecloses
changing `PATCH_SAMPLE_STRIDE` later and would make re-centring on a
corrected storm fix impossible without re-mining. That trade is worth
making deliberately once the patch geometry is settled, not by default
while it is still moving.

### What is still required before this can mine

`read_overpass_as_swath` takes a local path and must accept an open h5py
handle; the variable-group discovery it already does will work unchanged
against the streamed handle. That, plus routing `fetch_storm_swaths`
through the streaming opener, is the remaining work. The reader itself is
tested and the ingest entry point exists.

Honest scope note: the range reader is verified against a mock S3, not
against the real bucket -- there is no network in the environment this
was written in. The HDF5 access pattern it was tuned against is a
simulation of one, so the first real mining run should be watched for the
fetched fraction it reports, and the block size revisited if it comes out
far from ~25%.

## MWSynth 0.105 — streaming ingest completed and wired

Finishes 0.104. `read_overpass_as_swath` now accepts an already-open h5py
handle, so the identical parsing logic runs against a file streamed from
S3 or one on disk with no duplicated code to drift apart.
`read_overpass_streaming()` reads one overpass straight off the bucket,
and `fetch_storm_swaths(stream=True)` is now the default path.

Streaming by default is the right choice here specifically because each
file is read exactly once during mining -- there is nothing for a local
copy to amortise. `stream=False` keeps the download path for anything
read repeatedly.

### Three bugs found finishing this

**Wrong key name.** `fetch_storm_swaths` passed `f.get("size")` where the
listing produces `size_bytes`. It would have returned None on every file,
forcing a HEAD request per overpass -- one extra round trip per file
across a whole mining run, and silent, since the code would still work.

**Naive datetime, again.** `datetime.utcfromtimestamp()` is deprecated
from Python 3.12 and returns a NAIVE datetime, while every other
timestamp in the project is tz-aware UTC. Comparing them raises "can't
compare offset-naive and offset-aware datetimes". Same bug as
`mw_ingest` at 0.93, still sitting in this module.

**Validation after the optional import.** `import h5py` came before the
instrument check, so a bad instrument name reported itself as a missing
dependency, and the argument could not be validated at all on a machine
without h5py. Cheap checks first.

### Handle ownership

`read_overpass_as_swath` closes only what it opened. A streamed handle
belongs to the caller, which also owns the underlying S3 reader it needs
for transfer stats. Tested on the ERROR path specifically, since that is
where a leaked close is easiest to miss and hardest to notice.

### Test coverage

61 tests. The chain is verified end to end with a stub h5py that
exercises the file object the way HDF5 does -- seek to the front, read
the signature, seek to the end from `SEEK_END`, read the tail -- and
confirms only the blocks touched were fetched. Plus handle ownership on
the error path, and rejecting an unknown instrument without touching I/O.

**What is still unverified:** h5py cannot be installed in the environment
this was written in, so a genuine HDF5 parse over the range reader has
never been executed, and the block size was tuned against a SIMULATED
access pattern. The plumbing, the byte-exactness of the reader, and the
call chain are all tested. The HDF5 library's real access pattern is not.

On the first real run, watch the fetched fraction the streaming path
prints per file. Around 25% means the tuning transferred. Far above that
means HDF5 is scattering reads more widely than simulated and BLOCK_SIZE
in `s3_range_reader` should come down; far below with many requests means
it can go up.

## MWSynth 0.106 — the pipeline now actually streams

0.104/0.105 built the streaming reader and wired it into
`mine_local_tcprimed_cache(stream=True)`. But `run_ml_pipeline.run_mining()`
never passed that flag, so step 1 still defaulted to `stream=False` and
read the LOCAL cache. Running the pipeline would have re-processed the
same files and produced roughly the same 298 examples, with the streaming
work having no effect at all -- and nothing in the output would have said
so.

`STREAM_FROM_S3` (default True), `SEASONS` and `BASINS` are now
pipeline-level settings, passed through. The step-1 header states which
mode is active, and says explicitly that the local-cache mode will not
grow the dataset.

### Pre-flight estimate

`estimate_mining()` runs before any work and reports storm count,
overpass count, total size on S3, and expected transfer. Listing S3 costs
no file bodies, so this takes seconds and turns "how long will this take"
into a number.

The expected transfer is given as a RANGE (20-40% of total) rather than a
point estimate, because the ~25% figure comes from a simulated HDF5
access pattern and has never been measured against the real library.

It also states plainly that the GOES fetch -- 4 base bands plus up to 3
extra per surviving overpass -- is usually the slower half. The streaming
work removed a storage ceiling, not a time one, and it would be
misleading to imply otherwise.

`SEASONS` defaults to a single season (2022) rather than a decade.
A first run should establish the per-storm cost before committing to a
long one, and seasons before roughly 2018 will mostly skip on missing
GOES-R coverage regardless.

## MWSynth 0.107 — step 1 is configurable from the command line

Mining settings were module constants that had to be edited before each
run. They are now CLI arguments:

    python run_ml_pipeline.py --step 1 --start 2018 --end 2025 --stream true
    python run_ml_pipeline.py --step 1 --start 2022 --agency NHC
    python run_ml_pipeline.py --estimate --start 2018 --end 2025

`--start`/`--end` give an inclusive season range, `--stream` toggles S3
streaming, `--basins` takes an explicit list, and `--estimate` reports
what a run would process and exits.

Overrides are applied to the module-level constants in place rather than
threaded through as parameters, so the constants remain the single source
of truth and running with no arguments behaves exactly as before.

### --agency, and an interaction that would have failed quietly

`NHC` maps to AL/EP/CP, `JTWC` to WP/IO/SH.

`EXCLUDE_BASINS` defaults to `("WP", "IO", "SH")`, so `--agency JTWC`
would have selected exactly the basins the exclusion then removed --
producing zero storms while looking like it worked. An explicit basin
choice now clears the matching entries from the exclusion, with a
regression test asserting no selected basin remains excluded.

The JTWC basins are also outside GOES coverage, and every training
example needs GOES-R ABI infrared, so selecting them today yields almost
nothing but `no_goes` skips. The flag warns and proceeds rather than
refusing: the mapping is correct and will be useful the moment
Himawari/Meteosat ingest exists, and silently mining nothing for an hour
is the worse failure.

Six regression tests over the argument handling: year-range expansion,
the single-year default, agency mapping, the exclusion interaction,
explicit basins overriding agency, and the stream toggle. Invalid ranges
(`--end` before `--start`, `--end` without `--start`) exit with a clear
argparse error rather than silently mining nothing.

67 tests.

## MWSynth 0.108 — fix wrong keyword names in the S3 listing calls

`--estimate` crashed immediately:

    TypeError: list_available_storms() got an unexpected keyword argument 'season'

The real signatures are `list_available_storms(season_start, season_end,
basins, ...)` and `list_storm_overpass_files(basin, storm_num, season,
...)`, returning storms keyed `storm_num`. Both call sites used `season=`
and `cyclone_number=`.

The identical error was in `mine_local_tcprimed_cache`'s streaming block,
so mining would have failed the same way -- and only AFTER the estimate
had succeeded, which is the worse ordering.

### Why the tests missed it

The 0.107 CLI tests replaced `estimate_mining` with a stub in order to
assert on the parsed arguments. That made them tests of argparse rather
than of the thing argparse calls, and the one function with a bug in it
was the one being stubbed out. Mocking the unit under test is not a
subtle mistake and it is worth naming plainly.

Two tests added that would have caught it:

- `estimate_mining()` is now driven for real against a fake
  `tcprimed_ingest` whose functions carry the TRUE signatures, so a wrong
  keyword raises in the suite rather than on the user's machine.
- The mining streaming block is checked against
  `inspect.signature` of the real ingest functions, asserting that names
  the API does not accept do not appear in its source.

The second is a weaker style of test -- it inspects source text rather
than behaviour -- but that block needs live S3 to execute, and a
signature check is worth considerably more than no coverage.

69 tests.

## MWSynth 0.109 — the estimate was counting instruments we cannot read

The 0.108 estimate reported 2022 as 4,245 overpasses and 23.1 GB, and
2018 as 6,294 and 32.2 GB. Those work out to roughly 120-150 overpasses
per storm, which is far too many for two sensors and was the tell.

`list_storm_overpass_files(instrument_filter=None)` returns the whole GPM
constellation -- SSMIS, ATMS, MHS, AMSU-B and more -- and neither the
estimate nor the mining streaming block passed a filter.
`read_overpass_as_swath` accepts GMI and AMSR2 only, by design.

Two consequences, one cosmetic and one not:

- the estimate over-reported by several times, which is exactly the
  number being used to decide whether a run is worth starting;
- **mining would have spent a ranged GET on every unreadable file**,
  then thrown it away. Wasted requests and wasted time on files that
  could never become training examples.

`tcprimed_ingest.SUPPORTED_INSTRUMENTS` is now the single declaration of
what can be read, and both call sites filter on it. The estimate reports
the usable count broken down by instrument, and separately reports what
was present but skipped, so the difference is visible rather than
silently absorbed:

    usable overpasses: 792  (AMSR2 360, GMI 432)
    other instruments present but not read: 3456 (SSMIS 1440, MHS 1116, ATMS 900)

Re-run `--estimate` for real numbers. Based on the constellation mix the
readable share is likely a fifth or less of what 0.108 reported, so 2022
is probably closer to 800 overpasses and a few GB than 4,245 and 23 GB.
That is a guess about the mix, not a measurement -- the corrected
estimate will say.

Two tests added: the estimate is driven for real against a fake ingest
returning a mixed-instrument listing and must count only the readable
ones, and the mining block is checked to reference SUPPORTED_INSTRUMENTS.

71 tests.

## MWSynth 0.110 — the estimate now sizes the GOES side too

With instrument filtering fixed, real numbers came back sane: 2022 is 613
usable overpasses (AMSR2 362, GMI 251) across 36 storms and 8.7 GB on S3;
2022-2025 is 2,127 overpasses, 126 storms, 29.7 GB. About 17 usable
overpasses per storm, which is what two sensors over a storm lifetime
should look like.

But the estimate was still only sizing half the job. Each surviving
overpass needs a GOES fetch of 4 base bands plus up to 3 extra, and at
~12 MB per ABI mesoscale sector that is the larger number by an order of
magnitude:

    2022-2025, if 40% of overpasses yield an example:
        ~870 examples, ~73 GB of GOES traffic
    2022-2025, if 70%:
        ~1500 examples, ~128 GB of GOES traffic

Against 6-12 GB of TC-PRIMED streaming. Saying "the GOES side dominates"
in a NOTE was not useful when it is the thing deciding the runtime, so
the estimate now prints it, with a transfer-bound wall-clock figure at
two stated bandwidths.

Every number there is an ASSUMPTION and labelled as one -- the survival
rate especially, which varies with basin and season and can only be
pinned down by running a season and reading the skip reasons. A range is
honest; a single figure would not be.

This also reframes what the streaming work bought. It removed a STORAGE
ceiling, not a time one. The dataset can now grow past what fits on the
disk, but a decade of seasons is still hundreds of GB of GOES transfer,
and that is the constraint to plan around.

## MWSynth 0.111 — the time model was wrong for a fast link

2018-2025 estimates at 301 storms, 4,894 usable overpasses, 68.6 GB on
S3, and 164-288 GB of GOES traffic. On a 1.05 Gbps connection that is
about 20-40 minutes of line time -- at which point the transfer-bound
model 0.110 introduced simply stops being the right one.

Two changes:

**`--mbps`** so the estimate reports the actual link rather than two
guessed defaults. `--estimate --start 2018 --end 2025 --mbps 1050`.

**A CPU floor.** Every example costs scattered-point regridding of the MW
swath, several gaussian filters, the composites, the structure metrics
and the IR centre check, none of which bandwidth removes. At a rough
4 s/example serial that is ~2-4 h for 2018-2025, against ~0.4-0.6 h of
transfer at 1 Gbps -- so on this machine the CPU is the binding
constraint by roughly an order of magnitude, the opposite of the 50 Mbps
case the previous estimate was written around.

Above 500 Mbps the estimate now says so explicitly, and notes that
raising MAX_WORKERS helps only until the cores saturate.

`SECONDS_PER_EXAMPLE_CPU = 4.0` is a GUESS, labelled as one. It exists so
a fast link cannot make the estimate promise a runtime the CPU will not
deliver, which is the failure mode that matters -- an estimate that is
optimistic about the wrong resource is worse than no estimate. Correct it
once a real season has been timed: divide the observed wall clock by the
example count and the effective parallelism.

## MWSynth 0.112 — sensor PSF, fittable calibration, lightning

Three changes aimed at the 11 K bias and at structural realism, plus
retiring NEXRAD.

### Sensor antenna pattern (mw_psf.py)

The backbone rendered V/H at ~2 km GOES spacing with no antenna pattern,
so its 37 GHz field carried detail no 37 GHz radiometer can resolve. Real
IFOVs are frequency-dependent -- GMI is 8.6 x 14.0 km at 37 GHz but
4.4 x 7.2 km at 89, a factor of four in area -- and this also finally
gives the two frequencies different SPATIAL responses, closing a known
limitation where all frequency dependence lived in the constants.

Verified against published figures: a point source blurred at 37 GHz
recovers 8.4 x 13.5 km against a published 8.6 x 14.0.

**Applied to the backbone only.** Real MW arrives carrying its own
sensor's PSF; blurring the fused field would convolve genuine
observations twice.

**Applied AFTER texture and the noise floor**, which corrects a first
attempt that applied it before. Real radiometer noise is per-footprint --
adjacent grid cells inside one footprint see the same sample -- so
grid-scale texture left unblurred is finer than the instrument can
produce. Measured: applying the PSF before texture gave a 37/89 sharpness
ratio of 1.00, because the frequency-independent grid-scale texture
simply overwrote the blur. After the move it is 1.56.

Ellipse orientation is a deliberate simplification: real footprints are
along-scan/along-track and rotate across a swath, and this applies them
axis-aligned. Getting the magnitude right matters far more than the
angle, but a scan-geometry treatment would be a genuine improvement.

### Fittable calibration constants (calibrate_constants.py)

The constants are the largest known error source: 11.09 K of bias out of
a 15.85 K residual, so about 70% of what the correction model is asked to
absorb is systematic offset that least squares removes far better than a
few-hundred-example network.

Fits without re-mining. The exports do not store the intermediate
response and scattering fields, but for 37 GHz the backbone is linear in
two shared weights across V and H, so (A, B) are recoverable per pixel
from a 2x2 solve given the constants that produced the file. 89 GHz has
no emission term, so one channel suffices.

Recovers known constants EXACTLY on synthetic data -- ten constants, all
within 0.5 K, RMSE to zero.

Caveat stated in the module: stored backbones also carry the baseline
shift, texture, noise floor and now the PSF, all applied after the
formula, so recovered weights absorb some of that and the fit is a large
improvement rather than a clean inversion. Storing response/scat_pot at
mining time would make it exact.

Deliberately does NOT apply the fit. These constants define the backbone
every stored residual is measured against, so changing them invalidates
the training set -- that must be a decision, with a VH_PHYSICS_ID bump
and a re-mine, not a side effect of running a script.

### GLM lightning (glm_lightning.py)

The one genuinely new physical observable available. Flashes require
graupel and supercooled water colliding in strong updrafts, which is
close to a direct observation of what 89 GHz scattering measures -- and
it is information IR does not carry, since a cirrus canopy and an active
core can share a cloud-top temperature and differ completely in flash
rate. Neither paper uses it, and it is on the same platform already being
fetched.

Input channel 19, appended at the END so every existing channel keeps its
index. Absent GLM gives all zeros, which is also what a lightning-free
scene gives -- deliberately, so the input distribution does not shift
with product availability.

Honest about what it cannot do: lightning is sparse and biased toward
vigorous convection, and mature eyewalls are often electrically quiet. It
is a strong positive indicator and a weak negative one, so it goes in as
a model INPUT where the network can learn that asymmetry, and is NOT
wired into the parametric convective signal where it would act as a hard
vote.

**Still required:** `goes_fetch.get_glm_flashes()` does not exist yet.
The module detects that and returns zeros with a log line rather than
failing, so everything is wired and testable ahead of the reader.

### NEXRAD retired

Removed from the default dependencies (commented in requirements, so
turning it back on is one line). It only ever covered storms near the US
coast, is a display layer better served by any radar app, and produced
frames that looked different in character from every other frame.
`radar_ingest.py` is retained and still works.

If radar is ever wanted as a physical CONSTRAINT, GPM DPR is the better
source: global, storm-centred, vertically resolved, and already inside
the TC PRIMED files being streamed.

79 tests.

## MWSynth 0.113 — parallax: the IR was in the wrong place

Reviewing the parametric algorithm for accuracy turned up something it
did not handle at all. GOES views a TC from the equator at a slant, so a
cloud top 12-16 km up is SEEN displaced away from the subsatellite point
by h*tan(zenith). The microwave target does not share that displacement:
PMW senses the column, from a completely different orbit geometry.

Measured for real cases in this project:

| case | zenith | h=12 km | h=16 km |
|---|---|---|---|
| Lowell EP12 / GOES-18 | 20.8 | 4.6 km | 6.1 km |
| Karina EP11 / GOES-18 | 20.6 | 4.5 km | 6.0 km |
| Edouard AL05 / GOES-16 | 34.1 | 8.1 km | 10.8 km |
| Atlantic far east / G16 | 39.8 | 10.0 km | 13.3 km |

Against Lowell's 19 km RMW that is about a third of the radius of maximum
wind. Every training pair has carried this offset, so the correction
model has been quietly asked to learn a spatially-varying coordinate
transform on top of the physics -- a poor use of a few hundred examples,
and one that shows up as bias and blur rather than as anything visibly
wrong.

`parallax.py` corrects the convective signal before radial weighting, so
a single resample carries every IR-derived input. Per-pixel by height
rather than one shift per scene: a warm eye is low and barely moves while
the cold tops around it move several km, and that differential is exactly
the structure at stake.

### An angle-definition trap

`goes_fetch._view_angle_deg()` returns the Earth-CENTRAL angle, not the
satellite zenith angle -- 20.82 vs 24.40 degrees for Lowell. The central
angle is a reasonable proxy for a coverage cutoff and goes_fetch's
thresholds are calibrated to it, so that function is left alone. But
parallax needs the angle from the local vertical, and using the central
angle would have under-corrected by 15-20%. Both are now computed
independently with a test asserting zenith > central.

Verified end to end: with uniform cloud height the measured shift is
7.32 km against a predicted 7.35 km, in the correct direction. An earlier
sharp-edged test read 4.7 km against 7.3 -- that was the test case, not
the code: a hard-edged disc resamples against unshifted background at its
rim and drags the centroid measurement back.

### What is deliberately NOT modelled

Cloud-top height comes from brightness temperature against a fixed
tropical lapse rate, capped at 17 km. Real retrievals use CO2-slicing or
split-window channels and handle semi-transparent cirrus, where the
radiating level sits below the physical top. A 2 km height error is about
1 km of position error at 25 degrees zenith -- inside a grid cell -- so
the crude estimate is adequate for the geometry even though it would not
be adequate for cloud physics.

### Still outstanding in the algorithm

`backbone_response_37` and `backbone_response_89` remain literally the
same array. The 0.112 PSF gave the two frequencies different
RESOLUTION, but their underlying response field is still identical, and
real 37 and 89 GHz respond to different hydrometeors at different heights
-- liquid below the freezing level versus ice above it. That is the next
substantive physics gap, and a larger change than parallax: it needs a
defensible per-frequency response curve, not just a constant.

84 tests.

## MWSynth 0.114 — frequency-dependent response, and wind roughening

### The two frequencies finally differ physically

`backbone_response_37` and `backbone_response_89` were literally the same
array. The 0.112 PSF gave them different RESOLUTION, but the underlying
hydrometeor response was identical, which is wrong in a way that matters:
89 GHz responds to ICE above the freezing level, where scattering is
threshold-like and therefore concentrated on the deepest convection,
while 37 GHz responds to LIQUID below it -- a deep, diffuse layer that
extends well into stratiform rain.

Implemented in `mw_surface.frequency_response` as a per-frequency gamma
(1.35 at 89, 0.70 at 37) plus extra smoothing at 37 GHz. A shape
correction, not radiative transfer, but it moves both fields the way the
physics requires and it is testable: 89 GHz response is now measurably
more concentrated than 37 GHz on the same scene.

**A bug caught doing this.** An earlier in-lined attempt at the same
split referenced four CALIBRATION keys that were never added
(`response_gamma_37`, `envelope_scale_37`, ...). It would have raised
KeyError on the first generate -- and since mining catches per-example
exceptions as skips, a mining run would have produced ZERO examples while
reporting skip reasons rather than failing. Replaced with the tested
module rather than patched, so there is one implementation with coverage
instead of two.

### Wind roughening of the ocean background

A tropical cyclone is a wind field, and wind raises ocean emissivity
sharply at H-pol -- roughly 0.75 K per m/s at 37H against 0.25 at 37V,
because H-pol starts much further from unity. The backgrounds were
uniform constants, so the storm's own wind contributed nothing and the
background was flat where it should rise toward the core.

`mw_surface` adds it from a modified-Rankine profile built on vmax and
RMW, referenced to a 7 m/s ambient, capped for foam saturation, and
suppressed over land. Far-field 37H now reads 184 K against the measured
real GMI 182-193, rising to 219 K in the eyewall before any precipitation
emission.

### A correction to my own reasoning

The first version of this lowered `bg_h_37` from 182 to 168, on the
theory that wind roughening explained the gap between the measured
182-193 K and a calm-ocean 150-155 K. **That was wrong.** A 0.75 K per
m/s slope needs ~40 m/s to bridge 30 K, and the sampled region had maybe
15. The rest is ATMOSPHERIC -- water vapour and cloud liquid over a humid
tropical ocean add roughly 20-25 K at 37H, and this project has no
atmospheric term at all.

So the constant stays at 182 and is now documented as what it actually
is: the ambient tropical background INCLUDING the mean atmospheric
contribution, with wind adding only the storm-relative excess above it.
An explicit atmospheric term is the next improvement, and would let this
be decomposed properly instead of bundled.

89 tests.

## MWSynth 0.115 — the mining run that saved nothing

A 2018-2025 mine on 0.113 attempted 4,867 overpasses and saved **zero**,
every single one failing with TypeError.

Reproduced exactly against the shipped 0.113:

    goes_fetch.select_satellite(15.0, -95.0, <tz-aware datetime>)
    TypeError: can't compare offset-naive and offset-aware datetimes

**This was my fault, introduced by a fix.** 0.105 changed TC PRIMED's
`scene_time` from `datetime.utcfromtimestamp()` (naive, deprecated) to
`datetime.fromtimestamp(tz=utc)` (aware). Correct in isolation, and
correct against this project's convention. But `goes_fetch` was still
naive throughout -- the satellite-era table at module level, and every
timestamp parsed out of a GOES filename -- so the first comparison
between the two raised, and the satellite selector is the first thing
every mined overpass touches.

Making one side of a boundary correct is not a fix. The 0.105 entry even
noted this was "the same bug as mw_ingest at 0.93, still present here",
without checking what consumed the value.

`goes_fetch` is now tz-aware throughout: era bounds carry
`tzinfo=timezone.utc`, parsed filenames return aware times, and `_as_utc()`
coerces at the public entry points so a naive input from anywhere is
normalized rather than fatal. Four tests pin it, including one asserting
the era table is aware.

### The logging failure that hid it

The run reported:

    Skip reasons:
      exception: TypeError: 4867

Which is almost useless. The skip label carried the exception TYPE but
not its MESSAGE, so a completely diagnostic error string -- "can't
compare offset-naive and offset-aware datetimes" -- was thrown away, and
a run that failed identically 4,867 times looked like an ordinary skip
tally. Skip labels now include the message (truncated to 80 chars).

That is the more important fix of the two. The datetime bug was one line;
the reason it consumed a whole mining run is that the failure was
invisible until the summary, and uninformative even then.

### Re-mine required

Nothing was saved, and the earlier 298 examples are also gone from the
export directory. Re-run step 1 on this build. Since 0.112-0.114 changed
the backbone physics (PSF, per-frequency response, wind roughening),
`VH_PHYSICS_ID` should be bumped and this is the right single re-mine to
carry all of it.

93 tests.

## MWSynth 0.116 — one datetime convention, enforced

The 0.115 fix did not work: 613 attempted, 0 saved, same TypeError. The
improved logging did its job though -- the message came through this time
and pointed straight at the next boundary.

### The real defect was the absence of a convention

An audit of every datetime construction in the project found a genuine
mixture:

- `besttrack` parsed naive with `strptime` (three sites)
- `mw_ingest` **deliberately stripped** tzinfo in three places
- `radar_ingest` was naive
- `tcprimed_ingest` was naive until 0.105, aware after
- `goes_fetch` was naive until 0.115
- the GUI built an aware time and immediately stripped it back to naive

Every module was internally consistent. The combination was not, and
patching one boundary at a time actively made it worse: fixing TC PRIMED
in 0.105 moved the clash to goes_fetch, and fixing goes_fetch in 0.115
moved it to best track. Each fix was locally correct and the bug
survived, because the defect was never in any one module.

`timeutil.py` now states the convention -- **every datetime crossing a
module boundary is tz-aware UTC** -- and provides `as_utc()`, which is
idempotent and treats naive values as UTC rather than rejecting them.
Everything here genuinely is UTC (satellite filenames, synoptic
best-track times, TC PRIMED epochs), so a naive value is under-specified
rather than wrong, and raising on it would turn a harmless ambiguity into
an outage.

Applied at the source in `besttrack`, `mw_ingest`, `radar_ingest`,
`tcprimed_ingest` and the GUI; `goes_fetch` keeps its entry-point
coercion from 0.115. The three deliberate strips in `mw_ingest` existed
only to match the old naive convention -- there was no external-library
reason -- so converting them is safe.

Aware rather than naive because `utcnow()` and `utcfromtimestamp()` are
deprecated from Python 3.12, so the naive convention is on a path to
being unsupported.

### Tests that check the convention, not the sites

Six new tests, deliberately written against the RULE rather than any
individual call, since site-by-site fixes are what failed twice:

- `interpolate_fix` accepts all four aware/naive combinations and always
  returns aware
- `select_satellite` and `is_daytime` agree for aware and naive input
- `as_utc` is idempotent and upgrades naive
- **no module outside timeutil may contain `replace(tzinfo=None)`** --
  that string is exactly how the naive convention crept into three
  separate modules
- **no module may use `datetime.utcnow()` or `utcfromtimestamp()`**

The last two scan the source tree, so a future regression fails in the
suite rather than after 613 wasted overpasses.

Verified end to end: a full generate now succeeds with aware bands and
aware fixes, naive bands and aware fixes, and aware bands with naive
fixes, producing identical output in all three.

99 tests.

## MWSynth 0.117 — make a broken run fail in seconds, not hours

Two mining runs were lost not because of one bug each, but because
nothing in the pipeline was built to notice a run that could not succeed.
4,867 attempted / 0 saved, then 613 / 0 saved -- both grinding to
completion, both silent until the summary. These changes target that,
rather than any individual defect.

### Preflight

`run_ml_pipeline` now runs one synthetic example end to end before
touching the network, plus an explicit time-convention check. Verified
against both real failures:

- missing `CALIBRATION` key (the 0.114 bug) -> **caught**
- naive/aware datetime clash (0.105-0.116) -> **caught**

The second needed a dedicated check. A synthetic generate passes happily
while mining fails on every overpass, because generation never crosses
the boundary that broke: a TC PRIMED scene time compared against
best-track fixes and the satellite-era table. So preflight exercises that
comparison directly, with all four aware/naive combinations.

Both now fail in under a second with the actual error and a traceback.

### Fail-fast

If the first 150 attempts save nothing AND one reason accounts for 95% of
them AND that reason is an ERROR (not a legitimate skip), the run stops
and says why. Legitimate skips are expected to dominate early -- "no GOES
coverage" for 200 storm-times in a row is normal, one exception repeated
200 times is not -- so only `exception:`, `no_best_track:` and
`tcprimed_fetch_failed:` prefixes can trigger it, and a single save
disables it permanently.

### Live progress

A line every 50 attempts with the running save count and the most common
skip reason. Previously nothing printed until the end, so a broken run
and a slow one looked identical for hours.

### Resumability

The export filename is fully determined by storm, scene time and sensor,
so an existing file is the same example. A multi-hour mine interrupted by
a dropped connection or a closed lid previously started over and
re-downloaded every GOES band already processed; now it skips what is
done and costs only what is left.

Only skips files written under the CURRENT `VH_PHYSICS_ID`. A stale
vintage is redone rather than reused -- reusing it would silently mix
vintages and the 0.99 guard would refuse to train on the result.

### A guard with an explicit opt-out

The `replace(tzinfo=None)` source scan from 0.116 flagged preflight, which
constructs a naive datetime deliberately in order to test for it. Rather
than loosen the rule, lines may now carry a `naive-ok` marker. The marker
has to be deliberate, so an accidental strip still fails the suite.

102 tests.

## MWSynth 0.118 — script-by-script audit: per-channel normalization

A pass over every module looking for bugs, failure modes, and things
costing accuracy. Most classes came back clean: no mutable default
arguments, no unguarded divisions (the best-track interpolation guards
equal timestamps, ring/envelope widths are clipped, pixel counts are
checked), and nm/km conversions are disciplined -- `_nm` is always
converted at the boundary and functions take `_km` explicitly.

Three findings.

### 1. One normalization for every channel (accuracy)

`TB_MEAN = 260, TB_STD = 40` was applied to all brightness-temperature
inputs. The channels do not share a range:

| channel | normalized span, old |
|---|---|
| IR band 13 | -1.75 .. 1.00 |
| 37V | -0.50 .. 0.62 |
| WV bands 8/9/10 | -1.50 .. 0.00 |

The water-vapour channels sat **entirely on one side of zero** and 37V
varied over barely one sigma, so the network spent capacity undoing a
constant per-channel offset before it could learn anything. Not a
crash-class bug -- a quiet tax on every example. `ml_constants`'s own
docstring had flagged the constants as unconfirmed and worth revisiting
"once there's enough real training data".

`CHANNEL_NORM` now gives each channel a physically-motivated centre and
spread; every channel spans roughly +/-1.5 sigma. These are still
DEFAULTS, not measured statistics: computing true per-channel statistics
from the mined dataset and persisting them in the checkpoint -- as
`train_scalar_stats` already does for vmax/RMW -- is the proper fix and
the obvious follow-up.

**One implementation, shared.** `ml_train` and `ml_inference` each
carried their own normalization arithmetic. Identical, but only by
coincidence, and a train/inference normalization mismatch throws no error
at all -- it quietly produces wrong corrections. Both now delegate to
`ml_constants.normalize_channel`, with a test asserting neither module
normalizes inline.

While wiring this, `ml_train._normalize_tb` turned out to be a
MODULE-level function taking one argument while the new call sites passed
two -- caught by the suite, would have been a TypeError on the first
training batch.

### 2. A silently dropped input band (reliability)

`synthetic_algorithm` swallowed regrid failures for the supplementary IR
bands with a bare `pass`. A band that failed to regrid became a neutral
plane with no message, so the model quietly lost an input it was trained
with -- degradation that surfaces later as unexplained accuracy loss
rather than as an error. Now logged.

### 3. Silent handlers reviewed

Twelve broad `except: pass` sites audited. Ten carry a justifying comment
and are correct (optional dependencies, non-critical UI, offline
fallbacks). The two that did not are fixed above and in `ml_train`, where
the TF32/cudnn tuning fallback is now explained rather than bare.

107 tests.

## MWSynth 0.119 — mock-generation validation and physical-consistency tests

Generated frames across the intensity range and checked them against the
real GMI benchmarks measured from the Lowell pass, then locked the
findings in as tests.

### What the mock sweep confirms

Far-field 37H comes out **183-186 K across every intensity**, against the
measured 182-193 K. The wind-roughening and background work from 0.114 is
behaving on synthetic scenes, and the value is now stable rather than
depending on which radius happened to be sampled.

89 GHz is measurably sharper than 37 GHz on every frame (gradient ratio
1.27), which before 0.112 was 1.00 by construction.

### What it does not confirm

The 89 GHz core area still comes out 3.5-4.9x the real 9,355 km2. The
0.99 cirrus gate remains unvalidated, because these synthetic scenes
derive the WV band as `ir + 3` and so cannot exercise cirrus
discrimination at all -- the same limitation noted when that gate was
written. It needs a real frame.

Eyewall 37H reaches only 204-232 K against a real ~269 K, versus 255.5 K
measured on the real Lowell frame. The synthetic scenes understate the
convective response, so this is a limit of the test rather than a new
finding.

### An investigation that came back negative

The sweep showed eyewall/RMW drifting 0.77 (35 kt) to 1.17 (140 kt) -- a
0.40 spread in a ratio that should be flat, biased worst for the intense
compact storms where placement matters most.

Ruled out, in order:

- **Sensor PSF.** Identical drift with it disabled.
- **Grid resolution.** Identical from 3 to 27 pixels across the RMW.
- **The radial weight itself.** Peaks at 0.90-1.03 x RMW throughout.
- **Eye suppression.** The leading hypothesis, since `eye_frac` scales
  with intensity and the kernel removes 13.5% of the response at
  0.7*RMW. Narrowing it from 0.35 to 0.22*RMW changed the spread by
  **nothing at all** (0.40 before, 0.40 after).

So the hypothesis was wrong and the constant is **left alone**. The
remaining candidate is the interaction between the IR-driven convective
signal and the radial prior -- but the cloud-shield geometry in these
scenes is an assumption, so this may be measuring the test rather than
the algorithm. Recorded in the code with everything that was excluded, to
be re-checked against real frames across intensity once the mine exists.
Changing a tuned constant on synthetic evidence would have been worse
than leaving it.

### Physical-consistency tests

Six sensitivity checks in the spirit of Li et al.'s validation, which
perturbs the input and confirms the output responds as the physics
requires. All held when written:

1. deeper convection -> colder 89V (monotonic)
2. **a convection-free scene produces no core signature** -- the model
   must not manufacture an eyewall from nothing
3. stronger storm -> warmer 37H at 150-250 km (wind roughening)
4. larger RMW -> larger eyewall radius (monotonic)
5. 89 GHz always sharper than 37 GHz
6. output stays physical (finite, 50-350 K) across 35-140 kt

These need no data and catch the failure mode unit tests miss entirely: a
physics change that runs clean, produces plausible output, and is wrong.

113 tests.

## MWSynth 0.120 — fix UnboundLocalError, and a static guard for the class

Mining aborted after 169 attempts with

    UnboundLocalError: cannot access local variable 'target_time'

**My bug, from the 0.115 datetime work.** A scripted edit inserted
`target_time = _as_utc(target_time)` after a function's docstring using a
position heuristic, and it landed in `download_and_open(ref, local_dir)`
-- which has no `target_time` parameter at all. Guaranteed to raise the
moment it ran. The line was meant for `find_nearest_file`, which never
got it; both are now correct.

### Why nothing caught it

`download_and_open` only runs against the network. The preflight cannot
reach it, and no unit test calls it, so the bad edit was invisible to
every check in the project. It took a real mining run to surface.

### The guard

An AST check over the whole source tree for `x = f(x)` where `x` is
neither a parameter nor assigned earlier in the function -- a guaranteed
UnboundLocalError. Verified two ways: it detects the exact shipped bug
when reconstructed, and does not flag legitimate reassignment of a
parameter, which is the same line in a function that actually takes the
argument.

This is the right shape of test for this project. Static checks cover
functions that need S3 to execute, which is a large share of the mining
path and exactly where scripted edits have gone wrong twice now.

### The fail-fast worked

169 attempts instead of 4,867, with the error text in the abort message
rather than a bare exception count. The two mechanisms added in 0.115 and
0.117 -- message in the skip label, and stopping once one error dominates
-- did what they were built for on their first real outing.

114 tests.

## MWSynth 0.121 — the measurement overturned the block-size tuning

First real streaming run produced the number 0.104 asked to be watched,
and it came back very different from the prediction.

### 93%, not 25%

Across 13 real TC PRIMED overpasses: **190 MB of 204 MB fetched, 93%, in
386 requests** -- about 30 requests per 16 MB file.

The 512 KB block size was chosen against a SIMULATED HDF5 access pattern
that assumed the library touches a small share of a file. It does not.
The variables this project reads -- two full coordinate grids and four
channels at swath resolution -- span most of a 13-20 MB object.

So ranging saved about 7% of transfer and cost ~30 round trips per file.
At 50 ms latency that is 1.4 s of pure latency per file against 0.05 s for
a single GET: roughly **14 minutes wasted over one season, and over an
hour across 2018-2025**, for no benefit at all.

Objects under 64 MB are now fetched whole in one request; larger ones keep
the ranged path, where partial reads may genuinely pay. Verified
byte-exact on both paths, and the threshold keeps peak memory sane -- one
object per worker, ten workers, well inside 16 GB.

This is what the fetched-fraction reporting was for. The tuning was
defensible when written and wrong in fact, and only a real run could say
so.

### The estimate was wrong too

`--estimate` predicted "20-40% fetched" from the same simulation. It now
reports ~95% and says the figure is measured.

### A mock that hid the fallback

The reader's test double implemented `get_object(Bucket, Key, Range)` with
Range REQUIRED, while the real boto3 API makes it optional. The new
whole-object call therefore raised inside the mock and silently exercised
the ranged fallback -- the test failed for the right reason, but would
have passed for the wrong one had the assertion been weaker. Mocks that
are stricter than the API they stand in for quietly test the wrong path.

### Survival rate, first real data point

AL012022: 6 examples saved from 13 overpasses, **46%**. That sits in the
pessimistic half of the 40-70% band used for estimating, which puts
2018-2025 at roughly 2,250 examples -- about 7.5x the old 298.

115 tests.

## MWSynth 0.122 — 799 examples, skill +0.324, and the 83% we are leaving on the table

First full 2018-2025 run: 4,867 attempted, **799 saved**, and training
reached **skill +0.324** against +0.256 on 298 examples.

### What more data did, and did not, do

| | 298 examples | 799 examples |
|---|---|---|
| skill | +0.256 | **+0.324** |
| bias | 11.09 K | 11.51 K |
| ceiling (infinite members) | +0.301 | +0.369 |
| overconfidence | 1.7x | 1.9-2.0x |

The gain came almost entirely from **reduced spread, not reduced bias** --
bias is essentially unchanged. That is the clearest signal yet that the
remaining error is systematic backbone offset rather than anything more
data will fix, and it is exactly what `calibrate_constants.py` exists for.
Worth running now that there is a real dataset behind it.

`ENSEMBLE_SPREAD_CALIBRATION` re-measured 1.7 -> **1.95**. More data made
the ensemble tighter without making it proportionally more accurate, so it
became MORE overconfident, not less -- which is what a model that has seen
more of the distribution but still cannot represent what it has never seen
should do.

### The big finding: we only look at mesoscale sectors

Skip reasons were dominated by coverage:

    goes_does_not_cover_storm                  3754
    no_goes_satellite_for_this_time_and_place   269
                                               ----
                                               4023  = 83% of all attempts

`goes_fetch.list_available_files` lists **RadM only** -- the two
steerable ~1000 x 1000 km mesoscale sectors, which operators point at
whatever is operationally interesting. Most storm-times simply have M1
and M2 aimed elsewhere.

CONUS (RadC) and full disk (RadF) cover vastly more. Recovering even half
of those attempts at the current 16% yield would give roughly **2,700
examples**; recovering most of them, around **3,800** -- a 3.4x to 4.8x
increase, dwarfing anything else on the list.

The obvious objection is transfer: a full-disk band is hundreds of MB
against ~10 MB for a mesoscale file. But the mechanism for that already
exists here. `s3_range_reader` reads HDF5 in place over ranged GETs, and
its ranged path is retained specifically for objects above 64 MB, where
partial reads genuinely pay -- which is precisely the full-disk case. A
storm-centred crop is a small fraction of a full disk, so this is the one
place the original 0.104 design assumption actually holds.

Not implemented here: it needs fixed-grid geolocation to map lat/lon onto
full-disk scan angles and read only the covering chunks, which is a real
piece of work rather than a patch. Recorded as the highest-value item
remaining.

### Two Windows bugs, one found by the other's guard

`PermissionError: [WinError 5] Access is denied: '/tmp/goes_cache\\...'`
appeared once in the run. `download_and_open` defaulted `local_dir` to the
POSIX literal `/tmp/goes_cache`, which Windows cannot create. Rare only
because it needs the download path rather than the cached one; on a clean
machine it fails every time.

Both that and a second call site now use `tempfile.gettempdir()`. The
source-scan guard added alongside then immediately found a **third**
instance -- `mw_ingest.CACHE_DIR = "/tmp/mw_cache"` -- which had never
been hit because that path runs less often.

That is the pattern these scans keep proving: a bug fixed in one module is
usually present in two others.

117 tests.

## MWSynth 0.123 — fixed-grid geolocation: full disk becomes reachable

The 83% of storm-times lost to "no covering sector" was never a data
problem. Mining only ever listed RadM -- the two steerable mesoscale
sectors, which operators point at whatever is operationally interesting.
Full disk covers the whole hemisphere every 10 minutes and has been there
the entire time.

### goes_fixed_grid.py

ABI products are on a fixed grid in SCAN ANGLE, not lat/lon: a
geostationary perspective projection where each pixel is a fixed (x, y)
angle pair. The forward and inverse transforms are implemented directly
from the GOES-R PUG Vol.5 section 4.2.8, rather than pulling in a
projection library for forty lines of trigonometry.

Both directions are needed -- forward to locate the storm in the array,
inverse to attach real lat/lon to the pixels that come back.

**Validated:**

- sub-satellite point maps to exactly (0, 0)
- round trip lat/lon -> scan -> lat/lon is exact to **3.3e-13 degrees**,
  which is sub-millimetre, against a 2 km pixel
- beyond-the-limb points are rejected rather than returning the
  plausible-looking nonsense the raw geometry produces there

### Why the crop makes this affordable

A full-disk band 13 is 5424 x 5424, 59 MB uncompressed, against ~10 MB
for a mesoscale file. A 512 km storm-centred crop is **0.64% of the disk
-- 0.4 MB**, roughly a 150x reduction.

That is exactly the case `s3_range_reader`'s ranged path was kept for.
0.121 found ranging bought nothing on small TC-PRIMED files and switched
those to whole-object GETs, keeping ranging only above 64 MB. Full disk
is that case, and here the original design assumption genuinely holds.

The crop over-covers slightly toward the limb, because the angular
half-width is computed at nadir where one radian subtends the least
ground. That is the safe direction: a larger crop costs a few hundred KB,
a small one loses the storm.

### Product generalisation

`list_available_files` takes `product` ("RadM", "RadC", "RadF") and the
filename parser now handles all three. CONUS and full disk carry no
sector digit, so `sector` comes back "C"/"F" rather than "M1"/"M2" --
correct rather than missing, since they have no sub-sector to choose. A
first attempt produced the sector string "MNone" for those, caught by the
tests.

Eight new tests, including two for failure modes that produce wrong
answers rather than errors: the y axis runs north-to-south, and reading
it ascending yields an EMPTY slice rather than raising; and off-disk
storms must return None rather than an out-of-range crop.

### What remains before this is live

The geolocation and listing are done and tested. Still to wire: reading
the cropped `Rad` sub-array through the range reader, applying
scale/offset and the Planck conversion, and adding the RadF fallback to
the mining path when RadM misses. That is integration work against real
S3 objects -- deliberately not written blind here, since the last three
failures all came from scripted edits to code that could not be executed
in this environment.

125 tests.

## MWSynth 0.124 — full-disk integration wired and exercised offline

Completes 0.123. The geolocation was validated; this is the read path,
the fallback, and the plumbing that makes both reach mining.

### The read path

`goes_fixed_grid.read_cropped_radiance()` takes an already-OPEN h5py-like
handle rather than a path -- the same choice `read_overpass_as_swath`
made, and for the same reason: identical logic runs against a file
streamed from S3 and against an offline fake, with no duplicated parsing
to drift apart.

Exercised against a fake that models the real ABI format -- scaled int16
coordinates with a centred `add_offset`, scaled radiance with a fill
sentinel. Results on a synthetic disk with a storm at 25N 60W:

- crop is **0.64% of the array**
- descaling exact to **half a quantum** (0.005 against a 0.01 step)
- crop centre lands at 24.995, -59.964 for a request of 25.0, -60.0
- the coldest brightness-temperature pixel falls on the storm centre
- an off-disk storm returns None rather than an out-of-range crop

**The fake mattered as much as the test.** A first version scaled x/y
into 0..65534 and stored them as int16, which overflows at 32767. The
reader correctly returned None and the FIXTURE was wrong -- the same
lesson as the boto3 mock that made `Range` mandatory and quietly
exercised the fallback path. A mock that does not model the format
faithfully tests the wrong thing.

### Fallback order, and why

`get_band_image_any_sector()` tries mesoscale FIRST, then a full-disk
crop. Mesoscale is 1-minute cadence, already proven, and a small
whole-file read; full disk is 10-minute and needs a ranged crop.
Preferring full disk would be slower and coarser for the majority of
cases mesoscale already covers.

### A gap found while wiring it

Bands 9, 7 and 2 were still on the mesoscale-only path after band 13 had
been switched. That would not raise -- it would build frames from
**mismatched geometry**, with band 13 on a full-disk crop and the rest on
a mesoscale grid. The supplementary IR bands had the same problem.

All of them now take the storm centre and follow the same fallback, with
a test that scans `ml_data_mining` for any remaining mesoscale-only call.

### Expected effect

83% of attempts were lost to "no covering sector". At the observed 16%
yield, recovering most of them puts 2018-2025 near **3,000-3,800
examples** against the current 799.

Honest scope note: every piece here is tested offline against fakes, and
none of it has met a real full-disk object. The projection is exact and
the read path is verified against a faithful format model, but the first
real run should be watched -- particularly the reported crop fraction and
fetched bytes, which will say immediately whether ranged reads are
behaving on objects this size.

134 tests.

## MWSynth 0.125 — the full-disk crop was working and being thrown away

`full-disk fallback failed (NameError: name 'format_bytes' is not
defined)`. `format_bytes` lives in tcprimed_ingest and was never imported
into goes_fetch.

Worse than a plain crash. The call sat BEFORE the return, so every
full-disk crop was located, fetched, descaled and converted -- and then
discarded by a NameError **in a progress message**. The log said the
fallback failed; the work was already done.

Two fixes: the byte count is formatted inline, and the logging block now
runs AFTER the BandImage exists and inside its own try. Diagnostics must
never be able to fail the operation they describe.

### A static check for the class, which found a second one

An AST scan for calls to names defined in no scope -- module globals,
imports, builtins, parameters, locals. Only CALLED names are checked,
which keeps false positives near zero while covering the case that bites.

It flagged five sites. Four were false positives from my first version,
which walked functions flat and so did not model closures: nested helpers
legitimately using a `progress_callback` from their enclosing function.
Teaching it the scope chain cleared those.

The fifth was **real**: `mw_ingest.fetch_ssmis_swath_nrt()` is top-level,
does not take `progress_callback`, and calls it in its coverage-debug
branch. A latent NameError that has never fired only because SSMIS left
the sensor rotation in 0.87 -- it would have raised the moment that
branch ran.

That is the argument for fixing the scope chain rather than loosening the
check: a real bug was hiding among four false positives, and relaxing the
rule would have discarded it along with them.

Verified both ways: the checker detects the exact shipped bug when
reconstructed, and does not flag a legitimate closure.

### Also confirmed in this run

Whole-object fetching from 0.121 is live -- every TC PRIMED file now
reads in **1 request** instead of ~30.

135 tests.

## MWSynth 0.126 — two optimizations were cancelling each other out

The full-disk fallback is working: AL032021 saved **14 of 14** overpasses
against the 46% seen before it existed. But one line gave away a problem:

    full-disk crop band 13: 0.64% of the array, 30.1 MB fetched (1 request)

0.64% of the array, and the whole 30 MB came down anyway.

### Size cannot decide this; intent can

0.121 made objects under 64 MB fetch whole in one request. That is right
for TC PRIMED, where HDF5 touches ~93% of a 13-20 MB file and ranging
cost 30 round trips for nothing.

0.124 then added full-disk crops -- which read under 1% of a ~30 MB
object. Also under the threshold, so the whole file was fetched and the
crop saved nothing but decode time. Both decisions were individually
correct and together they contradicted.

Size alone cannot resolve it: what matters is how much of the object the
caller intends to read, and only the caller knows that. `prefer_ranged`
is now an explicit flag on the reader, set by the full-disk path.

Measured on a 30 MB object reading a 0.6% slice:

| | mode | fetched |
|---|---|---|
| default | whole | 31.5 MB |
| prefer_ranged | ranged | **0.5 MB** |

At four bands per frame and ~3,000 frames that is roughly **360 GB down
to 18 GB**.

### The log was under-counting by 4x

Only band 13 was given a `progress_callback` by the mining path. Bands 9,
7 and 2 took the identical full-disk route **silently**, so one visible
line per frame represented about four times the transfer it appeared to.
The message now says so explicitly rather than adding three more lines
per frame to a run that already produces thousands.

That is the second time in three versions that a diagnostic has been the
thing at fault -- once failing an operation it described, once
understating it. Both are worth more attention than they get: a log that
is quietly wrong is harder to catch than one that is absent.

138 tests.

## MWSynth 0.127 — audit of every optimization: are they on, and do they fight?

Prompted by three separate cases of optimizations interfering. A pass
over all of them, asking two questions: is it actually running, and does
it contradict another?

### Dead: the uint16 packing was never called

`pack_tb` / `unpack_tb` were written in 0.104 and **never invoked**. The
exporter kept writing float32, so every .npz has been twice its intended
size since. An optimization that silently does nothing is
indistinguishable from one that works.

Now wired, and backward compatible by construction: the loader keys on
DTYPE, not on a format flag, so float32 files already mined stay readable
and a dataset may contain both -- which it will, since this lands
mid-mine. Treating a packed array as float would train on values around
25,000 K and look like a physics failure rather than a format one.

### Conflicting: band 2 dragging 941 MB through the full-disk path

Band 2 is 0.5 km, so full disk is 21696 x 21696 -- **941 MB** against
59 MB for a 2 km band. And `synthetic_algorithm` regrids it straight down
onto band 13's grid, so every one of those extra pixels is discarded
before use. The least valuable band, by far the most expensive to reach.

The full-disk fallback now returns None for band 2. Frames are built
without the visible channel, which is already the normal case at night.

### Silent: the band 2 coarsen fallback

`ds.coarsen(...)` was wrapped in a bare `except: pass`. If it fails
against a real file the code silently runs at full 0.5 km resolution --
correct, slower, and invisible. Now warns. This is the same failure shape
as the dead packing, and worth catching by habit rather than by accident.

### Verified working

Resumability skips on a matching physics vintage; the model cache is
keyed on (path, mtime, size) and hits; `PATCH_SAMPLE_STRIDE` agrees
between training and inference; the instrument filter is applied in both
the estimate and the mining loop.

### A guard that passed vacuously

Wiring the loader introduced an UnboundLocalError -- `_load_tb` called at
line 205, defined at 221 -- and **all 138 tests passed**. The 0.120 guard
only caught `x = f(x)`, not use-before-definition generally.

Generalising it took four attempts, each failing differently, which is
worth recording:

1. walked every statement subtree: dozens of false positives, because in
   `for x in items:` the target is bound by the statement itself
2. restricted to straight-line statements: still flagged closure
   variables, since it seeded only from module names and parameters
3. added the scope chain and comprehension targets: went green, but
   **returned clean on the reconstructed bug** -- the module-name
   collection walked the whole tree, so nested function names looked
   module-global
4. collected module names from `tree.body` only: caught the bug, then
   flagged two annotated module globals (`AnnAssign`, not `Assign`)

A test that goes green is not evidence until it has been shown to fail on
the thing it exists to catch. Attempt 3 would have shipped as a working
guard while detecting nothing at all.

139 tests.

## MWSynth 0.128 — the training backbone was pre-aligned to the target

Looking specifically at the GOES-only path, which is what the tool exists
for, and found the most consequential bug in a while.

### What was happening

`baseline_shift_*` measures the far-field level of THIS FRAME'S real MW,
compares it to the assumed background constant, and adds the difference
to the backbone, clamped to +/-20 K. It runs only when a real pass
exists.

Every training example has a real pass. No GOES-only frame does.

So the stored backbone -- the model's input -- had been nudged toward the
target by up to 20 K, while at GOES-only inference it had not. Two
distinct problems from one line:

1. **Target leakage.** The input was partially pre-aligned to the answer,
   so the model learned a SMALLER correction than the GOES-only case
   needs.
2. **Train/inference mismatch**, in precisely the mode that matters most.

The clamp is 20 K. The measured bias the correction is supposed to remove
is 11.5 K. Same order of magnitude, which is consistent with more data
improving spread while leaving bias almost untouched (0.122): the model
was never being asked to learn the offset.

### The fix

The shift is now subtracted when recording the ML-free backbone, exactly
as `ml_delta` already was, and the measured values are recorded in
diagnostics so the effect is auditable rather than implicit.

Verified with a synthetic pass offset 15 K from the assumed background:

    measured shift: v37 +15.0, h37 +15.0, v89 -5.0, h89 +8.0 K
    stored backbone far-field 37H   185.9 K
    GOES-only frame, same storm     186.0 K

Those now agree to 0.1 K. Before, they differed by the full shift.

The residual target grows accordingly, which is the point -- the model
should be learning that systematic offset rather than being handed it.

### A bug introduced while fixing it

`baseline_shift_*` was assigned only inside `if real_swath is not None`,
so reading it unconditionally made every GOES-only frame raise
UnboundLocalError. Defaulted to zero. Caught immediately by running a
GOES-only generate, which is the one thing the change was about.

### Note for the dataset

Examples mined before this carry the pre-aligned backbone. They are not
corrupt -- the targets are real observations -- but they teach a smaller
correction than GOES-only needs. Since the physics vintage is unchanged,
resumability will NOT re-do them. Worth a clean re-mine before the next
training run if the GOES-only case is the priority, or at minimum
comparing skill trained on pre- and post-0.128 examples.

142 tests.

## MWSynth 0.129 — the extra IR channels were never in the training data

Pushing GOES-only toward its ceiling, and the largest finding is that the
work already done to get there had never reached the data.

### Seven dead channels

`ml_data_mining` called `generate_synthetic_mw(...)` **without
`extra_ir`**. The supplementary IR bands were fetched on the GUI path
only, so the code looked wired while every training example carried seven
NEUTRAL channels.

That is the multi-channel IR work from 0.98/0.99 -- and Li et al. found
that exact ablation to be their single largest gain, their band-13-only
experiment being worst on every metric. The model has been trained with
those inputs present in shape and empty in content, which costs capacity
and teaches it to ignore them.

Now fetched and passed at every mining call site, with a test that parses
each `generate_synthetic_mw(...)` call and asserts `extra_ir` and
`flash_density` appear.

### GLM lightning is live

`goes_fetch.get_glm_flashes()` existed only as a hole: `glm_lightning`
degrades to zeros when it is missing, so its absence was invisible at
runtime -- the channel simply stayed flat.

Implemented against GLM L2 LCFA, read whole (granules are tiny) through
the streaming reader, with scale_factor/add_offset applied. Skipping that
descaling would place every flash outside the grid and silently return an
empty field, the same failure shape as the ABI coordinate scaling.

Why it matters for GOES-only, measured on the Norbert frame:

    IR enhanced cold cloud      16.0% of frame
    89 GHz deep scattering       9.2%
    COLDEST IR tops              0.9%

The answer sits between two thresholds an order of magnitude apart in
area. No fixed IR threshold lands on it, which is why a learned model is
needed -- and lightning is the one available observable that sees the
mixed-phase column IR cannot.

### VH_PHYSICS_ID bumped -- 0.128 should have done it

0.128 changed the stored backbone from (parametric + per-frame baseline
shift) to pure parametric, so the residual target changed MEANING, and
the identifier was not bumped. That is exactly the failure the guard
exists to prevent: `ml_train` would have mixed vintages silently, and
`calibrate_constants` -- which recovers intermediate weights FROM the
stored backbone -- would have fitted across the boundary and baked the
leak 0.128 removed straight back into the constants.

Now `0.129-psf-freqresp-wind-nobaselineleak`, and
`calibrate_constants` refuses a mixed-vintage dataset outright rather
than producing plausible, wrong constants.

### Two process failures worth recording

**The indentation-substring trap, again.** A 12-space pattern is a
substring of the same line at 16 spaces, so a scripted replace matched
inside its own earlier replacement and corrupted the file. This is the
second time; the fix both times was line-based, indentation-aware
editing.

**Test pollution.** The GLM test did `del goes_fetch.get_glm_flashes` in
its teardown -- harmless when no such function existed, destructive once
0.129 added one, silently breaking every test that ran afterwards. Caught
by the new test asserting the reader exists. Save-and-restore now.

### Re-mine required

Examples mined before this have seven neutral IR channels, no lightning,
and a baseline-shift-leaked backbone. The physics vintage now differs, so
the guard will refuse to mix them -- which is the correct outcome and
means a clean re-mine before the next training run.

145 tests.

## MWSynth 0.130 — steps 3 and 4

    python run_ml_pipeline.py --step 3                    # fit, show the diff
    python run_ml_pipeline.py --step 3 --apply-fit        # fit and write it
    python run_ml_pipeline.py --step 4 --start 2018 --end 2025 --apply-fit

**Step 3** fits the V/H constants against the mined dataset. This is the
direct attack on the bias term, which barely moved between 298 and 799
examples (11.09 -> 11.51 K) while spread fell -- the signature of
systematic constant error rather than anything more data fixes.

**Step 4** runs the sequence: mine -> fit -> re-mine -> train.

Two mines is not redundancy. The fit needs data generated under the
CURRENT backbone, and applying it MOVES the backbone, so the first
dataset is then measured against something that no longer exists.
Chaining without the re-mine would hand `ml_train` a dataset the vintage
guard is correct to refuse.

### The decision point stays explicit

Without `--apply-fit`, step 3 and step 4 both print the diff and stop,
changing nothing. This is deliberate: `calibrate_constants` has refused
to apply its own fit since it was written, because these constants define
the backbone every stored residual is measured against.

Automating that away would be the wrong kind of convenience -- the fit is
a least-squares result from one dataset, and it is worth looking at before
it becomes the physics.

### The writer

`apply_fitted_constants()` rewrites only numeric literals matched on their
quoted key, refuses anything it cannot match unambiguously (reported
rather than guessed), writes a timestamped .bak first, and bumps
`VH_PHYSICS_ID` -- which is what stops the old dataset from being mixed
with the new one, and is exactly what 0.128 forgot to do.

Four tests pin it: only the named constants change, untouched ones are
byte-identical, the file still parses, the id is bumped, a backup exists,
and an empty fit is a no-op.

149 tests.

## MWSynth 0.131 — latency, not bandwidth

A 2018-2025 run reached AL06 in 2h30m, against ~5h30m for the whole thing
on 0.121. Extrapolating, that is roughly 15 h -- about 3x slower.

### Most of that is productive

The full-disk fallback (0.124) means ~83% more storm-times now actually
PROCESS instead of skipping instantly on "no covering sector". Those
frames were previously free because nothing happened. Frames per hour is
the number that matters, not storms per hour, and the dataset should be
3-5x larger for it.

### But two paths were needlessly serial

On a 250 Mbps link these are round-trip bound, not byte bound -- the
files involved are well under a megabyte each.

**GLM granules.** A +/-5 min window is ~30 granules of 20 s each, read one
at a time: ~1.5 s of pure latency per frame. Now read through a 12-thread
pool, about 0.12 s. Threads are right here because the work is entirely
network-bound, so the GIL costs nothing.

**Extra IR bands.** Three independent list+read operations, serial. Now a
4-thread pool, bounded deliberately so it cannot compete with the mining
pool's own workers for connections.

GLM hour listings now share the existing `_LISTING_CACHE`, which the ABI
path has used since earlier; consecutive overpasses of one storm often
fall in the same hour, making those free.

Checked before changing: the per-band ABI listing cache was already
present and working, so the 14-LIST-per-frame figure was already down to
one or two. Not everything that looks like waste is.

Verified the parallel paths return exactly what the serial ones did,
including that a single failing band does not drop the batch -- the
classic way a naive parallel map loses data silently.

149 tests.

## MWSynth 0.132 — the missing atmospheric term, and where the ceiling is

### Where the ceiling actually is

The error decomposition says the whole thing:

    zero-correction RMSE 18.24 K
    bias 11.51 K        per-draw spread 8.83 K

**Bias is 76% of the error budget**, and it did not move between 298 and
799 examples (11.09 -> 11.51 K) while spread fell. More data cannot learn
something the input never varies with, so by elimination that is wrong
constants or missing physics.

What removing it would be worth, at the current 4-member ensemble:

| bias | RMSE | skill |
|---|---|---|
| 11.5 K (now) | 12.33 K | +0.324 |
| 7 K | 8.28 K | +0.546 |
| 5 K | 6.67 K | +0.634 |
| 3 K | 5.34 K | +0.707 |

Everything else is small by comparison. Going from 4 ensemble members to
infinite is worth +0.045. The realistic GOES-only ceiling is set by how
much of that 11.5 K is systematic rather than irreducible.

### The missing term

`bg_h_37 = 182 K` is documented as the ambient tropical background
"including the mean atmospheric contribution". Clear calm ocean at 37H is
nearer 150-155 K; the remaining 25-30 K is water vapour and cloud liquid
emitting, plus its reflection off a poorly-emitting sea surface.

That is real, spatially-varying physics folded into a constant -- so it
could not vary with the moisture field, which varies a great deal. Flagged
in 0.114 and not built until now.

Li et al. (2026) point straight at it: their saliency analysis makes IR
channel 10 (low-level water vapour) a dominant predictor, and finds
channels 13/15 contributing "baseline constraints on cloud optical
thickness and total precipitable water" through differential water vapour
absorption. The information is in channels this project already fetches.

`mw_surface.atmospheric_anomaly_k()` drives the background from band 9
(mid-level WV, always fetched). Parameterized as an ANOMALY about the
tropical mean, not an absolute addition -- the constants already carry the
mean and adding it again would double-count. Same discipline as the wind
term being referenced to an ambient wind rather than to calm.

Measured on a full frame, far-field 37H:

    dry column   (band 9 = 252 K)   175.0 K
    mean         (band 9 = 238 K)   186.0 K
    moist column (band 9 = 224 K)   197.0 K

A +/-11 K swing -- the same magnitude as the unexplained bias, which is
what makes this the plausible candidate rather than a speculative one.

Band 10 or the 13-15 split window would be better TPW proxies. Both are
OPTIONAL bands here and band 9 is guaranteed, so band 9 is the honest
choice; the swing constants are first estimates of the right sign and
order, and are exactly what `calibrate_constants` should refine once a
dataset exists with this term active.

### Sequence

`VH_PHYSICS_ID` -> `0.132-atmospheric-anomaly`. This changes the backbone,
so it belongs in the same re-mine as everything else pending. The order
that gets the most out of it:

    --step 4 --start 2018 --end 2025          # mine, fit, review the diff
    --step 4 --start 2018 --end 2025 --apply-fit

The fit now has a moisture-dependent term to work with rather than a
constant standing in for one, so the constants it returns should mean
more than they would have last version.

158 tests.

## MWSynth 0.133 — backbone audit: three structural gaps

Reading the backbone itself rather than its constants. What it already
gets right is most of it: the emission/scattering split per frequency
(0.95), per-frequency response shaping and sensor PSF, parallax, wind
roughening, surface type, and now the atmospheric anomaly. Three
structural gaps remained, all of them one-signed and therefore
bias-shaped.

### 1. The mapping was LINEAR in response

    Tb = bg + E*resp - D*resp

Real radiative transfer saturates: emission and scattering both approach
a limit as the layer becomes optically thick. The difference is largest
exactly where most pixels live:

    response   linear   saturating   ratio
      0.10      0.10       0.27       2.7
      0.25      0.25       0.56       2.2
      0.50      0.50       0.82       1.6
      1.00      1.00       1.00       1.0

A ramp calibrated to be right at full response is too WEAK everywhere
below it -- a one-signed error over the bulk of every frame.

`mw_surface.saturate()` normalizes so sat(0)=0 and sat(1)=1 exactly, which
keeps the existing constants meaningful at the endpoints and leaves
`calibrate_constants` working unchanged: it solves for the PRODUCTS
resp*(1-scat) and resp*scat, so substituting sat(resp) does not touch that
algebra.

`SATURATION_K = 3.0` stands in for the unknown mapping from this
project's normalized response to actual optical depth. A first estimate of
the right shape, not a derived value.

### 2. 89 GHz had no emission term at all

    v89 = bg - D * resp

Pure scattering, monotonically decreasing -- the same error 37 GHz had
before 0.95. Over ocean, cloud and rain liquid emit at 89 GHz and raise Tb
slightly before ice takes over. The real response is a hook, not a ramp.

**The magnitude was caught by checking against reality, twice.** A first
attempt used boosts of 12/22 K, which put V-pol at 280 + 12 + 8
(atmospheric) = 300 K, at the sea-surface physical temperature an emitting
layer cannot exceed. And the real Norbert 89H frame shows the ocean
background near 280 K with EVERY convective feature colder than it -- so
at 89 GHz over ocean the background already sits near saturation and there
is very little room to rise. Reduced to 5/12 K, giving a few-K hook rather
than a prominent bump.

### 3. No freezing-level dependence

Ice scattering depends on frozen mass above the freezing level, which
drops from ~4.9 km in the deep tropics to ~3.6 km near 35 degrees. The
same cloud-top temperature therefore implies less ice aloft poleward, so a
latitude-blind model over-depresses poleward storms. Measured effect on
eyewall 89H: 191 K at 12N against 214 K at 32N.

### A test that had encoded the bug

`test_deeper_convection_gives_colder_89` demanded monotonic cooling. That
was only true because 89 GHz had no emission term -- the test was
asserting the ramp. It now asserts the real shape: deepest convection much
colder than background and colder than moderate, a small warm bump at
shallow depth permitted, and the bump bounded below 5 K so the emission
term cannot grow implausible. Plus a direct check that neither 89 GHz
emission peak can reach the physical temperature.

Worth naming: this is the second time a test has encoded an artifact of
the model rather than a property of the world. Passing tests are evidence
about the code, not about the physics.

### Still open, deliberately

- `SATURATION_K` and the 89 emission boosts are first estimates. They are
  the right sign and order; `calibrate_constants` should refine them.
- Beam filling is handled only implicitly through the sensor PSF.
- Band 10 or the 13-15 split window would be better TPW proxies for the
  atmospheric term than band 9, but both are optional bands.

`VH_PHYSICS_ID` -> `0.133-saturation-89emission-freezinglevel`.

165 tests.

## MWSynth 0.134 — a physics question the data should answer, not me

Continuing the backbone audit against the papers. The remaining
paper-supported gap is a real one, and the evidence for acting on it is
not.

### The gap

Li et al. (2026) describe an interaction the backbone does not model. 37
and 89 GHz are computed INDEPENDENTLY here from the same response field,
so nothing lets ice aloft screen the liquid emission beneath it. Their
cloud-type analysis is explicit that under opaque ice the 37 GHz emission
from lower-level liquid is "only partially attenuated by the ice layer
above".

### Why it was not implemented

Checking it against the real Norbert frames came back ambiguous, and the
ambiguity is instructive. Binning 37H by 89H ice class:

    89H class            n        37H emitting
    deep (<180 K)      6,751          0.03
    strong (180-212)  31,193          0.73
    moderate (212-228) 38,018         0.76
    light (228-254)   94,960          0.37

That looks like dramatic screening at the deepest ice. It may equally be
the opposite: the deep-ice class is tiny, and the 37H "emitting" colour
mask can EXCLUDE the very hottest pixels, which render dark red to black
in the NRL table. A low emitting-fraction could mean the liquid signal is
screened, or that it is hotter than the mask catches.

A nonlinear colour table read through RGB thresholds cannot resolve this,
and an earlier pass of the same analysis was contaminated by the grey IR
strip along the frame edge -- 296,254 pixels of it -- which had to be
excluded before the numbers meant anything at all.

Adding an attenuation term on that evidence would have been guessing with
extra steps. It is also the sort of term that would look plausible in
output while being wrong, since ice and liquid genuinely co-vary.

### What was built instead

`calibrate_constants.ice_coupling_profile()` measures the relationship
from the MINED DATA. Every .npz holds the real 37 and 89 GHz brightness
temperatures on the same grid, so this is directly measurable rather than
inferred from a picture: median 37H binned by 89H, with quartiles.

Falling 37H toward the coldest 89H bins is screening. Flat or rising says
ice and liquid simply co-vary and no attenuation term is warranted.

Verified it distinguishes both cases on constructed datasets -- including
the failure that matters, which is reporting screening where there is only
co-variation, since that would motivate a term the data does not support.

Run it after the next mine. If it shows screening, the term is worth
adding and its magnitude comes out of the same profile. If it does not,
that is a settled question rather than an open one.

168 tests.

## MWSynth 0.135 — saturation softened after checking it against reality

0.133 introduced the optical-depth saturation curve with
`SATURATION_K = 3.0`. Checked against the one metric with a real
reference, that was too aggressive.

The 89 GHz core area was ALREADY 23% larger than the observed Lowell
value on the old linear ramp (11,480 against ~9,355 km2). Saturation
inflates it further:

    linear    11,480 km2     +0%
    k=0.75    13,538 km2    +18%
    k=1.50    15,054 km2    +31%
    k=3.00    17,545 km2    +53%

The physical argument is unchanged and still sound: emission and
scattering saturate, and a linear ramp is too weak below full response.
The problem is that every calibration constant was tuned against the
LINEAR form, so changing the shape without refitting over-strengthens the
whole field -- and it does so in the direction of a discrepancy already
known to exist.

Reduced to 0.75: correct shape, limited magnitude change, until
`calibrate_constants` can fit the constants to the new curve. Raise it
once that has run and core area can be checked against real data instead
of a synthetic scene.

### A test that pinned a tunable

`test_mid_range_is_lifted` required a 1.2x lift at mid-range -- a number
calibrated to k=3.0. Lowering the constant failed a test that was
asserting a tuning choice rather than a physical fact.

Replaced with the actual property: the curve must lie above the diagonal
and be concave. How far above is exactly what the fit should decide, and
a test has no business fixing it.

Third time a test has encoded an artifact instead of a fact. The pattern
is consistent enough to name: when a test breaks because a constant
moved, check whether the test was asserting physics or asserting the
constant.

168 tests.

## MWSynth 0.136 — the identifier now derives itself, and bands fetch in parallel

### The hole 0.135 opened

0.135 changed SATURATION_K from 3.0 to 0.75 -- a genuine backbone change
-- and did not bump VH_PHYSICS_ID. Two datasets with different physics
would have carried the same label. Resumability would have skipped the
older files as current, and check_physics_consistency would have seen one
vintage and approved. The guard defeated by exactly the thing it guards
against, for the second time (0.128 was the first).

A hand-maintained string cannot survive this. `VH_PHYSICS_ID` is now
DERIVED: a hash over every numeric CALIBRATION entry plus the
physics-affecting constants in `mw_surface` (SATURATION_K,
RESPONSE_GAMMA, WIND_SENSITIVITY, ATMOS_SWING_K, FREEZING_LEVEL_KM and
the rest). Any tuning change produces a new id automatically, so mixing
becomes impossible without anyone having to remember.

`calibrate_constants.apply_fitted_constants` no longer rewrites the
identifier either -- writing new constants changes it by construction.

Two tests pin the property: editing a CALIBRATION value changes the id,
and so does editing a mw_surface constant, which is the case 0.135 missed.

### Parallel band fetches

Bands 13, 9 and 7 were fetched one after another, and each may be a
full-disk crop -- a LIST plus several ranged GETs on a ~30 MB object.
At ~36 s per saved example against a ~4 s CPU floor, the mining loop is
bound by serial round trips, not computation.

`goes_fetch.fetch_bands_parallel()` fetches them concurrently, bounded at
4 workers so a single frame cannot monopolise connections the mining
pool's own workers need. Same fix already applied to the supplementary IR
bands (0.131) and GLM granules (0.131).

### A test that asserted an implementation detail

`test_bumps_the_physics_id` checked that the new id appears as literal
text in the rewritten file. That was true when the id was a string
constant and is meaningless now that it is computed. Replaced with the
property that actually matters: changing a constant changes the id.

Fourth time a test has encoded an artifact rather than a fact. The tell
each time was a test breaking on a change that was clearly correct.

169 tests.

## MWSynth 0.137 — a scan for things that look wired and do nothing

An AST scan for every public function and constant defined and
referenced nowhere -- including by tests. Thirteen names. Most are
harmless leftovers (legacy IBTrACS parsing, retired NEXRAD station
tables). Three were real.

### 1. The one direct check of synthetic against real was discarded

`mw_compare.compare_both_frequencies` runs on every fused frame, and the
GUI consumed only `bias_k` from it. The RMSE and pixel counts against
both real V-pol and real PCT -- the project's single most direct
quantitative check of synthetic output against observation -- were
computed and thrown away. `format_stats_summary` existed to render
exactly that and was never called.

Now emitted to the progress log and stored in diagnostics.

### 2. No way to tell whether the land mask was active

`surface_type.backend_name()` has existed since 0.97 and was never
called. The all-ocean fallback warns once per process, which is easy to
lose in a run producing thousands of lines, and a working backend said
nothing at all.

That distinction matters more than most: over land, ocean emissivity at
37/89 GHz mimics heavy precipitation, so an unnoticed fallback renders
coastlines as convection. Each frame now reports the backend and the land
fraction, or says plainly that no mask is installed.

### 3. A tuning knob that did nothing

`ml_diffusion.DEFAULT_ENSEMBLE` duplicated
`ml_constants.ENSEMBLE_MEMBERS`, which is what `ml_inference` actually
reads. Changing it altered nothing, silently. Now an alias of the live
constant, so there is one knob.

### A correction to my own finding

I first reported `mw_compare` as entirely dead. That was wrong -- a
`head -6` truncation in my own grep hid the GUI call site. The module is
used; only `format_stats_summary` was not. Worth recording because the
mistake is the same shape as the bugs being hunted: a tool that appears
to show everything while quietly showing part.

### A standing guard

`test_no_new_unreferenced_public_names` fails when a public function is
defined and referenced nowhere, with an explicit allow-list for
deliberate leftovers and entry points. Anything new that lands unwired
now shows up as a test failure rather than as months of silence -- which
is how seven dead IR channels, dead uint16 packing, and a checkpoint
selected on the wrong metric all survived.

173 tests.

## MWSynth 0.138 — M-PERC-style EWRC diagnosis, and three retired settings

### ewrc.py

Kossin et al. (2023, Wea. Forecasting 38, 1405) build M-PERC from ARCHER
ring-score radial profiles reduced by PCA over 1787 profiles, feeding an
18-predictor logistic regression whose coefficients are not published in
usable form. **This does not reproduce that model and is not called
M-PERC.** It implements the mechanism M-PERC rests on and reports a
confidence, not a probability:

- ring score at each radius measures how well the convective signature
  fits a circle -- strong AND azimuthally uniform, so a single intense
  cell scores low
- sampled every 6 km to 200 km from TC center, as in the paper
- the profile is searched for a SECONDARY MAXIMUM separated from the
  primary by a real moat, not a shoulder on one peak
- KS09's criterion applied as a gate: an outer ring must close at least
  75% of a circle
- the Vmax-only baseline is reported alongside, following the paper's
  practice of displaying the full model beside a reduced intensity-only
  one, so it is visible whether the cloud presentation says more than
  intensity already did

**Real microwave only**, as specified, and for the reason the paper
gives: IR does not see through the cirrus canopy covering a TC, which is
why M-PERC uses microwave at all. Running this on the synthetic field
would be circular -- the backbone derives its response from IR, so any
ring it showed is one IR already implied.

Sanabia et al. (2015) track the same cycle in WV-minus-IR profiles and
find inner-eyewall decay detectable EARLIER there than in IR. That is a
genuine geostationary-only signal and a natural extension, but it
diagnoses a different stage and was deliberately left out.

### Two bugs found by testing it rather than trusting it

**Scene percentiles broke the normalization.** Both ends of the
convective-strength scale came from percentiles of the scene. A real
eyewall covers well under 1% of a 200 km-radius frame, so the cold
percentile landed in the BACKGROUND, span collapsed to ~1 K, and every
depressed pixel saturated. The profile read flat-topped from 12 to 48 km
on a textbook single ring at 30 km -- no peak, so no primary eyewall and
no secondary maximum. The cold reference is now fixed at 100 K below the
scene background, the observed 89 GHz range over ocean.

**Fixed azimuth bins failed at small radius.** A 28 km ring has a 176 km
circumference, ~50 pixels, so a third of 36 bins were empty and per-bin
scatter exploded. Uniformity read near zero and a textbook inner eyewall
scored BELOW a partial outer arc. Bins now scale with circumference.

**And one silent failure caught by the project's own pattern.** The first
wiring wrote `diagnostics["ewrc"]` at a point where no such dict exists
-- it is built later, at the result construction. The write raised
NameError inside a try/except that swallowed it and printed
"EWRC diagnosis unavailable". Assigned to a local and merged at the
construction site instead.

### Retired settings

- **NEXRAD checkbox removed entirely.** Retired in 0.112 with arm_pyart
  commented out of requirements; the checkbox offered a toggle for a path
  that could no longer run.
- **"Save paired GOES+real-MW training example" removed.** A 0.65-era
  holdover from before `ml_data_mining` existed. Training data now comes
  from a pipeline that is reproducible, vintage-tagged and resumable; a
  one-off checkbox export was none of those.
- **"Taper strength on unusual storms" now defaults OFF.** It suppresses
  the correction exactly where the backbone is least trustworthy -- rare,
  extreme or oddly-structured storms, which are the ones worth looking
  at. Still available as an opt-in.

184 tests, and `test_no_new_unreferenced_public_names` confirms the new
module is wired rather than sitting inert.

## MWSynth 0.139 — WVIR staging: ERC awareness between overpasses

`ewrc.py` (0.138) diagnoses a secondary eyewall from a real 89 GHz pass,
which is the right instrument -- but passes are hours apart, and Kossin
et al. note the temporal gaps can be large, particularly in the tropics.
Sanabia et al. (2015) mapped an entire ERC in Typhoon Sinlaku from
WV-minus-IR radial profiles at geostationary cadence.

So microwave anchors, and `wvir.py` tracks progression between
overpasses. GOES-only, every frame.

### Why WVIR works where IR does not

The broad weighting functions of IR and WV each make it hard to isolate
deep convective cores from surrounding cloud. Their DIFFERENCE does not:
WV minus IR exploits the temperature inversion at the tropopause, and
positive values indicate convection that has PENETRATED it. Cirrus canopy
and anvil sit below the tropopause and produce no positive difference.

That is the same discrimination this project's cirrus gate (0.99)
approximates with a tuned water-vapour blend, reached from a different
direction and with a physical basis instead.

### The six stages

Implemented as Sanabia et al. define them, with the amplitude cue they
report: single-eyewall and transition maxima run 3.0-3.5 K, dropping to
1.5-2.0 K once concentric eyewalls form. Stages defined by CHANGE --
outer erosion, inner decay, contraction -- require history, so without a
previous frame they are not asserted rather than guessed. The authors are
explicit that the stages are subjective, so every result carries a
confidence and the evidence behind it, never a bare label.

Verified end to end on the Sinlaku progression: 1 -> 2 -> 3 -> 4 -> 5 ->
6, including the return to 5 when the ring holds radius and back to 6
when it resumes contracting.

### Two filtering bugs, both found by testing rather than assuming

**A mean smooth crushed narrow rings.** At 6 km radial sampling a 12 km
ring is about two samples; averaging over three dropped it below the
deep-convection threshold entirely. Switched to a median, which
suppresses single-bin noise equally well and preserves peak height.

**Then the median did the same thing for a different reason.** A decaying
inner eyewall near 30 km can occupy ONE bin, and suppressing single-bin
features is exactly what a median is for. Peaks are now found on the RAW
profile, with smoothing retained only for the moat, where single-bin
noise would otherwise manufacture separation between two halves of one
ring.

Both bugs collapsed stage 4 into stage 5 -- destroying precisely the
distinction the paper reports WVIR detects earlier than IR, which is the
main operational reason to have built this.

### A test fixture that was wrong twice

The first fixture offset WVIR by -1.0 K, putting its stated peaks below
the detection threshold, so stages 3-5 "failed". The second set a
decaying inner ring at 1.0 K, which is already below the threshold and
therefore already stage 5 -- conflating the two stages in the input
rather than in the code. Both times the module was right.

190 tests.

## MWSynth 0.140 — the fitter was reading packed data as Kelvin

Found while checking what `--step 4` does after mining finishes, since
that path had never executed.

### The bug

`calibrate_constants.fit_from_examples` read the backbone and target
arrays with a bare `np.asarray`. 0.127 wired in scaled-uint16 storage,
so a 182 K background came back as **18200**, and the least-squares
fitted every constant against values a hundred times too large.

No error. No warning. Just nonsense constants at the end of a 23-hour
mine -- which `--apply-fit` would then have written straight into
CALIBRATION and bumped the physics vintage for.

`ml_train` and `ice_coupling_profile` both key on dtype. This was the one
reader that did not, and it is the one whose output gets written back
into the physics.

Now keyed on dtype like the others, so a dataset holding both float32
(pre-0.127) and uint16 files reads correctly -- which any dataset
spanning that change will.

### Verified by recovery, not by inspection

The test builds a packed dataset FROM the current constants and asserts
the fit returns them unchanged. Reading packed data as float would return
values two orders of magnitude off, so this fails loudly on exactly the
bug it exists to catch. It recovers bg_h_37, emission_boost_h37 and
max_depression_v89 to within 0.5 K.

A second, standing check requires any module reading `backbone_*` or
`target_*` arrays to mention uint16 at all.

### Note on --step 4

Without `--apply-fit`, step 4 mines, fits, prints the diff and STOPS --
nothing is changed and no training runs. With `--apply-fit` it applies
the constants and immediately **re-mines from scratch**, because the
physics vintage has changed and resumability correctly refuses the old
files.

192 tests.

## MWSynth 0.141 — SATURATION_K restored, and three real bugs in the fitter

"Something about constants not finding 3 things" turned out to be worth
chasing. The fitter had three separate defects, all of which would have
corrupted a fit run against a real dataset.

### 1. A bound that excluded the truth

`bg_v_37` is 250 K. Its fitting bound was (170, 230). The bound went
stale when the 0.95 emission work retuned the backgrounds, and nothing
checked -- so the fitter CLAMPED the 37V background 20 K below its actual
value on every run.

Visible once tested against data generated from the exact constants: the
fit made 37V *worse*, RMSE 0.00 K -> 20.00 K, precisely the clip
distance. Only the upper bound was stale, so only it moved; a first fix
raised the lower bound as well and promptly clipped a legitimate 195 K
test value, the same mistake in the opposite direction.

A standing test now asserts every CALIBRATION value lies inside its own
bound.

### 2. The 89 GHz model no longer matched the generator

The fit inverted `v89 = bg - D * B89` -- pure depression, one weight from
one channel. That matched the backbone until 0.133 gave 89 GHz an
emission term. After that the generator was `bg + E*A - D*B`, the
recovered weight silently absorbed the emission, and both 89 GHz
constants were fitted against a model the data did not come from.

Now solved as two channels and two unknowns, exactly as 37 GHz always
was, with a three-column design. All six 89 GHz constants are recovered
exactly, including the two that previously could not be fitted at all.

### 3. Silence about what is not fitted

Fourteen CALIBRATION entries do not enter the linear backbone model --
land backgrounds, IR bounds, scattering thresholds. That is by design,
but the fit said nothing about them, so their absence looked like
failure. They are now listed explicitly as not fitted, by design.

### SATURATION_K back to 3.0

0.135 softened it to 0.75 because the constants were tuned against the
LINEAR form and a stronger curve over-drove the field. That was
explicitly interim, holding only until the constants were refitted to the
curve. Keeping 3.0 means a dataset mined at that value stays usable
rather than needing a 23-hour re-mine.

### A fifth test encoding the old model

`TestCalibrationFit` generated its 89 GHz data as pure depression, so it
asserted the pre-0.133 backbone. With the fitter now solving for
emission, a depression-only generator is data the model cannot have
produced. Updated.

That is five now. The tell is always the same: a test breaking on a
change that is clearly correct.

196 tests.

## MWSynth 0.142 — cutting the mining time, and cross-module mismatches

### Four dead input channels

The input layout spanned all seven EXTRA_IR_BANDS while
EXTRA_IR_FETCH_LIMIT is three, so **four channels were permanently
neutral planes** -- model capacity spent on inputs that never carry
anything, and a model taught they are worthless.

The layout is now DERIVED from the fetch limit, so the two cannot drift
apart again. MODEL_IN_CHANNELS drops 19 -> 15. Nothing is lost: the bands
are ordered by Li et al.'s saliency, the first three are the informative
ones, and the four removed are the ones that analysis found contribute
minimally. Raising the limit widens the model automatically.

This is the same lesson as the derived physics ID: two constants
describing one thing will eventually disagree.

### Making the run shorter

Overpasses were already processed six at a time, so the obvious
parallelism was taken. Two new levers:

- `--max-per-storm N` caps examples per storm, spread EVENLY across each
  storm's lifetime rather than taking the first N, so intensification and
  decay both stay represented. Overpasses of one storm are highly
  correlated -- fifteen frames of one hurricane add far less than fifteen
  frames of fifteen different ones, at the same cost. `--max-per-storm 6`
  should cut a 2018-2025 run to roughly a third.
- `--workers N` exposes the concurrency (default 10).

### And instrumentation, because the rest is guesswork

At roughly 100 s of work per attempt against a ~4 s CPU floor, the time
is going somewhere, and every previous performance assumption in this
project was wrong until measured -- the block size tuned on a simulated
access pattern (real answer 93%, not 25%), and the paths that turned out
to be serial.

Each run now reports seconds per saved example split across mw_read,
goes, extra_glm, generate and export. The next run will say where the
time actually goes instead of leaving it to be reasoned about.

### Cross-module checks added

Four standing tests for things that are not bugs in any single file but
disagree between files: every input slot is actually filled, patch
geometry is self-consistent, every layout channel has a normalization
entry, and every fitted constant exists in CALIBRATION with a bound.

200 tests.

## MWSynth 0.143 — one pool per frame

### Bands were still being fetched one at a time

0.136 parallelised 13/9/7 and 0.131 parallelised the supplementary bands,
but they ran as two pools back to back -- and bands 2, and 9 and 7 on the
streaming path, were still fetched INDIVIDUALLY. A frame paid up to five
separate round trips for work that is entirely independent.

`goes_fetch.fetch_frame_bands()` fetches base and supplementary bands in
ONE pool of six. Measured against a 0.25 s stand-in round trip: 1.75 s
serial to 0.50 s, a 3.5x reduction in the GOES phase.

**Band 13 deliberately stays first and alone** on the streaming path. It
gates the coverage check, coverage rejection is the most common outcome,
and `goes_fetch` matches on TIME ONLY -- so that check cannot be skipped,
only moved. Fetching the whole frame up front would undo an optimization
that already exists. A test pins the ordering.

### A break I introduced, and two guards that missed it

The first edit replaced the streaming path's `extra_ir` assignment with a
comment while line 700 still used it -- a NameError on every attempt in
the hot path. Both the unbound-local and undefined-name guards passed:
the use sits inside nested blocks, which the linear check skips, and
`extra_ir` is bound in other functions, which the scope walk credited.

Caught by reading the diff rather than by the suite. The guards cover
straight-line code in one scope; they do not catch a name used deep
inside a nested block whose binding was removed elsewhere.

### A third test pinning an implementation detail

`test_mining_fetches_the_extra_ir_bands` asserted the string
`fetch_extra_ir_bands` appeared in the mining source. Merging the two
pools -- a pure speed change -- failed it. It now checks that the bands
ARRIVE and are passed to generate, which is the property that matters.

### Estimated 2018-2025 runtime

| configuration | hours |
|---|---|
| 0.134, as measured | 23.0 |
| + band batching | 16.6 |
| + `--workers 14` | 13.2 |
| + `--max-per-storm 8` | 8.2 |
| + `--max-per-storm 6` | 6.6 |
| + `--max-per-storm 4` | 4.8 |

The batching figure is calculated from round-trip counts, not measured
end to end, and the per-phase instrumentation added in 0.142 will confirm
or correct it on the next run. Every previous performance estimate here
was wrong until measured.

203 tests.

## MWSynth 0.144 — the 15-channel change would have crashed training

Checked before an overnight run, and it was needed.

0.142 narrowed the input layout to the three supplementary IR bands
actually fetched, taking MODEL_IN_CHANNELS from 19 to 15. But both
`ml_train` and `ml_inference` built their input stacks by enumerating
**EXTRA_IR_BANDS** -- all seven -- so each would have stacked 19 planes
into a model declared with 15.

That is a crash on the first training batch, which is the good outcome
for this class. The bad outcome is what it would have cost: discovering
it after a 6.6-hour mine.

`MODEL_EXTRA_IR_BANDS` is now derived from INPUT_CHANNEL_LAYOUT itself,
so the stack and the declared width cannot disagree. Verified end to end
by handing inference all seven bands and confirming the model receives
exactly 15 channels.

This is the third time the same shape of bug has appeared: two constants
describing one thing, drifting apart. The physics ID, the layout versus
the fetch limit, and now the stack versus the layout. Each was fixed by
deriving one from the other rather than maintaining both.

205 tests.

## MWSynth 0.145 — the outward tilt, and CRPS

Two things the papers give that were not being used.

### 1. Eyewall convection tilts outward, and this resolves 0.119

Best-track RMW is a SURFACE WIND radius. What this project synthesises is
a cloud-top and hydrometeor signature, and the two are not at the same
radius.

Sanabia et al. (2015) quantify it from SFMR and IR along aircraft
transects through Sinlaku: correlations between surface wind and IR
brightness temperature are most negative when the winds lag the IR by
**10 km**, with the IR located radially OUTWARD -- agreeing with Sanabia
et al. (2014) on the outward tilt of eyewall convective clouds.

**This closes an open question.** 0.119 measured eyewall/RMW drifting
from 0.77 on a broad storm to 1.17 on a compact one, ruled out the sensor
PSF, grid resolution, the radial weight and eye suppression, tested a
change to the eye kernel that made no difference, and recorded the drift
as unexplained rather than tuning a constant on a guess.

A FIXED outward offset produces exactly that shape: 10 km is 45% of a
22 km RMW and 9% of a 111 km one. The trend was physical the whole time,
and the constant it was hunting never needed changing. Leaving it alone
was the right call.

The ring is now centred at RMW plus the tilt, capped at 0.35*RMW so a
single typhoon's measurement is not extrapolated into displacing a
compact eyewall most of its own radius.

A test asserting "eyewall is near RMW" now asserts the measured property
instead: outward, by a bounded amount. That is the sixth test found
encoding an assumption rather than a fact.

### 2. CRPS

Li et al. (2026) score their synthesis with CRPS throughout. It is a
PROPER scoring rule: it rewards sharpness only when justified.

This project selects on ensemble-mean RMSE skill, which cannot see
calibration at all -- a model could collapse to a single draw and score
identically. The ensemble here is measurably 1.95x overconfident, so the
selection metric was blind to a known defect.

`crps_ensemble()` is reported every epoch alongside skill. Deliberately
NOT the selection metric yet: changing what a run optimises deserves a
measured comparison, not a swap on the strength of an argument.

**The first test of it was wrong**, and instructively so. It centred
members ON the truth, so a collapsed ensemble scored a perfect zero and
the calibration behaviour -- the entire reason for adding CRPS -- was
never exercised. Rewritten so the truth is a DRAW from the predictive
distribution: CRPS is then minimised exactly at the true spread, which is
the property worth having.

### Still not implemented, and why

- **Performance diagrams** (POD/FAR/CSI across BT thresholds): a
  presentation of results, not a model improvement.
- **Cloud-type stratified validation**: needs the NJIAS Himawari
  cloud-type product, which has no GOES equivalent here.
- **The 37 GHz ice-screening term**: the mechanism is real but the
  evidence was ambiguous, and `ice_coupling_profile` (0.134) will settle
  it from the mined data rather than from a guess.

210 tests.

## MWSynth 0.146 — sector selector is M1/M2 only

"Auto (either)" passed `sector=None`, and `find_nearest_file` matches on
**time only**. So it returned whichever mesoscale box scanned closest to
the requested moment, regardless of where that box was pointed -- and M1
and M2 are independently steerable, routinely parked over different
systems. The option could quietly hand back a perfectly valid scene of
somewhere else.

Removed. The selector is M1 or M2.

### A gap that removal made worth closing

With the sector chosen explicitly rather than automatically, a wrong pick
is easier to make -- and the GUI never checked whether the fetched scene
actually contained the storm. The mining path has done so since 0.98;
the GUI path simply rendered whatever came back.

A valid, recent file from the wrong sector produces a convincing frame of
the wrong storm, with no error anywhere. The GUI now warns, naming the
sector and the storm position, and suggests the other one.

210 tests.

## MWSynth 0.147 — the uncertainty map, and diagnostics that never reached the image

Looking at what the image output does with everything now computed. Two
things were being produced and thrown away.

### 1. The per-pixel uncertainty field was discarded

`apply_ml_correction` computes ensemble spread PER PIXEL, calibrated by
the measured 1.95x overconfidence -- and then reduced it to a mean and a
max. So the one thing it was good for, saying WHERE the model is
guessing, was thrown away, leaving two scalars that say only how much on
average.

The field now takes the same path as the correction it describes:
upsampled by the same stride, smoothed the same way, blended onto the
same grid. It renders as a panel in place of the convective-signal
diagnostic when available.

For a synthetic product this is the most honest panel there is. A frame
of plausible-looking microwave imagery generated from infrared invites
more confidence than it has earned; a map marking where the ensemble
disagrees is what makes it safe to read.

Two details that matter:

- Outside the patch the value is **NaN, not zero**. Zero renders as
  CERTAIN in exactly the region the model never looked at.
- The taper edge is NaN too, so the boundary where the correction fades
  out does not read as confident.

### 2. EWRC and WVIR never reached the figure

Both were wired to the progress log only, so on a saved or exported image
they vanished entirely. Now captioned along the bottom: WVIR stage with
its confidence, and EWRC confidence with the secondary ring radius, or a
plain statement that no secondary eyewall was found.

### A bug caught on the way

The uncertainty panel called `np.isfinite`, and `main_window.py` does not
import numpy at module level at all -- it would have raised the moment a
diffusion checkpoint produced a spread field, which is to say on the
first real frame after training. Imported locally.

217 tests.

## MWSynth 0.148 — end-to-end wiring audit

Traced every handoff mechanically rather than by eye: generate -> export
-> train -> checkpoint -> inference -> image.

### What holds

- **export -> train**: nothing the trainer reads is missing from the
  file. Verified by generating a frame, exporting it, and comparing the
  stored keys against every `data[...]` the trainer touches.
- **generate -> export**: the file carries the current physics vintage,
  and packed targets unpack to physical Kelvin.
- **generate -> consumers**: 30 of 37 diagnostics are consumed somewhere.
  The seven that are not -- `fusion_weights`, `radial_weight`,
  `scattering_potential`, `texture`, `wv_mask`, `rmw_km`,
  `baseline_shift_k` -- are inspectable payload on the result object,
  not broken wiring.

### What did not hold

**The checkpoint contract was written and never read.** Training saves
`in_channels` and `patch_sample_stride`; inference read neither, building
the model from whatever the current constants happened to say.

- A channel mismatch surfaced as a torch shape error deep inside
  `load_state_dict` -- loud, but unreadable.
- A STRIDE mismatch was worse. The shapes still match, so the checkpoint
  loads cleanly and the correction is applied over the wrong ground area,
  silently.

Both are now verified before the weights load, with a message naming the
mismatch. This is live: the layout narrowed 19 -> 15 in 0.142, so any
older checkpoint now gets a clear explanation instead of a stack trace.

**`base_channels` fell back to 32** while training uses 48 -- left over
from an earlier architecture. A checkpoint missing the key would have
built the wrong width and failed on the state dict.

### A note on method

Two of my own intermediate findings in this audit were wrong: I reported
`ir`/`wv`/`swir`/`mask` as missing from the export when I had simply
guessed the key names (they are `ir_band13`, `wv_band9`, `swir_band7`),
and I flagged `vis_band2` as dead before checking that it feeds a
daytime sharpening term in the backbone. Both were caught by looking
rather than by assuming, which is the same discipline the code audit
needs.

221 tests.

## MWSynth 0.149 — learning-curve subsets

### What the three added skills actually contain

- **claude-scientific-skills** is a stub: 29 lines, no content, a pointer
  to a GitHub repository.
- **machine-learning** (536 lines) and **ml-fundamentals** (206) cover
  classical tabular supervised learning -- sklearn Pipelines and
  ColumnTransformer, XGBoost with SHAP, SMOTE and class weights,
  RandomizedSearchCV and Optuna, plus a 16-week study path. Their
  `validate.py` validates the skill's own config file, not a model.

None of that transfers directly. MWSynth is a conditional flow-matching
diffusion model over 2D brightness-temperature fields in PyTorch; there
is no tabular design matrix, no class imbalance, and no DMatrix.

**One concept did transfer**, from their leakage table and the
GroupKFold pattern: splitting by ENTITY rather than by row. It is
already implemented here -- `make_train_val_split` splits by storm ID and
stratifies by peak intensity, which goes further than the skill
suggests. Worth the check; nothing to change.

### What was worth taking

Their debug checklist includes learning curves, and this project has
repeatedly reasoned about "does more data help?" without ever drawing
one. Bias sat near 11.5 K across 298 and then 799 examples while spread
fell, which suggested a systematic floor -- but that is an inference from
two points.

`learning_curve_storm_subsets()` builds nested training subsets drawn BY
STORM against a FIXED validation set. Two properties are the whole point,
and both are tested: the subsets nest, so the curve compares more of one
dataset rather than different datasets; and the validation set never
moves, because a curve measured against a shifting target says nothing.

It also prices `--max-per-storm`. That flag shrinks the dataset to cut
mining time, and if the curve is already flat at half the data, the cap
costs nothing and future runs can be shorter still.

225 tests.

## MWSynth 0.150 — offset versus structure

A detail from the 0.134 run: epoch 1 started POSITIVE. On the 0.122 run
it was -0.201, worse than doing nothing, and only turned positive by
epoch 3.

### Why that changed

0.128 removed the baseline-shift leak. Before it, `baseline_shift_*` was
added to the stored backbone, so the systematic offset was HANDED to the
model and the residual left to learn was the hard structural part alone
-- a slow start, with early corrections that actively hurt.

After it, the residual CONTAINS that offset. A near-constant shift is the
easiest thing a network can learn, so epoch 1 picks it up and gains skill
immediately. The fix behaving as designed, and the first real evidence
it did anything.

### Why that is also a warning

Skill that comes mostly from a flat shift is the CALIBRATION CONSTANTS'
job, done badly and expensively by a diffusion model. If skill jumps at
epoch 1 and then flattens, the network is absorbing a bias that step 3
would remove more cleanly and permanently.

That is testable rather than arguable, so each epoch now reports the
skill reachable with a spatially FLAT correction alone, and its share of
the total:

    of that skill, 74% is reachable with a FLAT offset alone
    (skill +0.240) -- the rest is structure the constants cannot supply

A share near 100% that does not fall across epochs says the constants are
wrong and the fit should reclaim most of it. A falling share says the
model is learning real structure, which is what it is for.

### A fixture wrong in the familiar way

The first test of this used ZERO-MEAN truth, where a constant shift
cannot help by construction -- so it reported a 0% flat share for every
input and exercised nothing at all. Rewritten with a genuine systematic
offset, which is the case that exists after 0.128.

229 tests.

## MWSynth 0.151 — the run worked; three things it exposed

**1,644 examples from 1,717 attempts -- a 96% yield**, against 16% before
the full-disk fallback. `goes_does_not_cover_storm` fell from 3,754 to
**10**. Twice the data of the previous best, in a third of the time.

### Step 3 had never been able to run

    no usable examples (need backbone_*, target_* and mask)

`calibrate_constants` required a `mask` key. The exporter has never
written one -- `ml_train` derives its mask from `np.isfinite(target)`,
which is the only definition that exists. So every file failed the key
check and a 1,644-example dataset reported nothing usable.

The fitter now derives the mask the same way, honouring an explicit
`mask` if some future exporter writes one. Verified by exporting real
frames and fitting against them: 6 files, 120,000 pixels, 12 constants.

The 0.148 end-to-end audit checked the TRAINER's keys against the
exporter and never checked the FITTER's. A gap in my own audit, found by
running the thing.

### Lightning was asking the wrong satellite

`extra_glm` measured **0.00 s** -- a fetch that never happened.

The streaming path selects its satellite per frame into `sat`, but passed
`satellite` to the lightning lookup: the enclosing function's parameter,
defaulting to GOES-19. Every frame asked GOES-19 for lightning regardless
of which satellite it actually used -- wrong for every East Pacific
storm, and wrong for all of 2018-2024, when GOES-19 was not operational.

Both names are real, so the undefined-name guard could not see it. A test
now asserts that within one function, GLM is asked about the same
satellite variable the bands were fetched with.

**The lightning channel is therefore still neutral in all 1,644
examples**, exactly as it was before 0.129. Twice now this input has been
wired and not connected.

### Where the time goes, measured

    mw_read    7.37 s
    goes      59.56 s     <- 63%
    generate  27.00 s     <- 29%
    export     0.50 s
    total     94.42 s

The guesswork is over: GOES fetching dominates, and `generate` is a
bigger share than expected for pure CPU. Those are the two targets, and
both are now measurable rather than argued about.

230 tests.

## MWSynth 0.152 — what the training run says, and two bugs it exposed

Training on 1,644 examples: best skill **+0.253** at epoch 4, against
+0.324 on 799. That looks like a regression and is not the interesting
part.

### The diagnostic added in 0.150 gave a verdict

    of that skill, 125% is reachable with a FLAT offset alone
    of that skill, 166% is reachable with a FLAT offset alone
    of that skill, 104% is reachable with a FLAT offset alone

Hovering at or ABOVE 100%, every epoch. Above 100% means a spatially
constant shift scores BETTER than the model's full output -- the
structure it adds is actively harmful.

So the network is doing nothing but correcting a bias, which is the
CALIBRATION CONSTANTS' job. That is exactly the question 0.150 was built
to answer, and the answer is unambiguous.

### The backbone got worse, and that is consistent

Zero-correction RMSE rose 18.24 K -> **24.08 K**. The physics added since
0.112 -- atmospheric anomaly, saturation at k=3, 89 GHz emission, outward
tilt, freezing level -- all landed with constants still tuned for the
configuration before them. Bias rose 11.5 K -> 17-22 K to match.

This is precisely the situation 0.135 anticipated when it softened
SATURATION_K as an interim, and 0.141 restored 3.0 on the grounds that
the fit would reconcile it. The fit has not run yet. Every number in this
run is consistent with that one missing step.

### Two bugs the run exposed

**CRPS was reported in normalized units.** It printed 0.26-0.40 "K"
beside a bias of 17-22 K -- a proper scoring rule reading forty times
better than the error it scores. `sampled_skill` works in normalized
units and multiplies by TB_STD only when reporting rmse; CRPS was handed
the same arrays and reported them raw. Real values were 10-16 K.

**ENSEMBLE_SPREAD_CALIBRATION was not imported.** It was dropped from
ml_inference's import list by a later edit to that list while still being
USED in the diffusion branch, with no try block around it. A NameError on
the first diffusion inference -- invisible only because no diffusion
frame has been generated since 0.121. A test now asserts that every
ml_constants name used in ml_inference or ml_train is actually imported.

### Overconfidence is now measured, not assumed

It climbed **1.43x -> 3.85x within this single run** as the ensemble
collapsed. A hardcoded 1.95 cannot describe both ends of that, let alone
two different runs. Training records the measured value in the
checkpoint and inference prefers it, falling back to the constant only
for older checkpoints.

231 tests.

## MWSynth 0.153 — the crop was fetching 25x what it needed

GLM is confirmed working: real flash counts and non-zero densities in the
log, where the phase previously measured 0.00 s.

The same log gave a number worth acting on:

    full-disk crop band 13: 0.64% of the array, 9.4 MB fetched
    (18 request(s), per band; 3-4 bands per frame)

A 0.64% crop of a 5424 x 5424 int16 band is about **0.38 MB** of pixels.
Fetching 9.4 MB for it is 25x over-fetch, and 18 x 512 KB accounts for it
exactly: each block drags in a large neighbourhood nobody asked for.

512 KB was tuned in 0.104 for TC PRIMED whole-file reads, where it is
right. It was never re-measured after full-disk crops arrived in 0.124 --
a completely different access pattern, scattered chunks rather than a
sequential sweep.

### Measured, not guessed

Simulating that pattern against the reader:

    block   requests   fetched   latency   total
     32 KB        43   1.41 MB    2.15 s   2.31 s
    128 KB        25   3.28 MB    1.25 s   1.61 s
    256 KB        20   5.24 MB    1.00 s   1.58 s
    512 KB        17   8.91 MB    0.85 s   1.84 s

Smaller blocks cut bytes and add round trips, so total time has a broad
minimum between 128 and 256 KB. `RANGED_BLOCK_SIZE = 128 KB` sits in it:
within noise of the best time while moving **2.7x fewer bytes**, which
matters when fourteen workers share one link.

Applies only when `prefer_ranged` is set -- a caller asking to range is
reading a slice. The whole-object path from 0.121 is untouched and still
takes one request.

### Two mistakes worth recording

The edit to add this matched `prefer_ranged: bool = False):` in TWO
places -- the reader's `__init__` and `open_s3_hdf5` -- and broke the
second. That is the third indentation- or substring-collision from a
scripted edit; the fix each time is line-precise targeting.

Then the new attribute was named `self._block`, which is already the
method that FETCHES a block: an int shadowing a callable, caught
immediately by a TypeError. Renamed, and a test now asserts `_block`
stays callable.

231 tests.

## MWSynth 0.154 — 128 KB was wrong; 1 MB is right

0.153 cut the ranged block size 512 KB -> 128 KB to fix what looked like
25x over-fetch. The next run measured it:

    512 KB:  18 request(s),  9.4 MB
    128 KB:  70 request(s),  9.1 MB

**The bytes did not drop.** Only the request count changed, inversely
with block size -- which can only happen if the access is contiguous.

### Why the original analysis was wrong

The crop is 0.64% of the array BY AREA, and I read that as 0.38 MB of
data being fetched as 9.4 MB. But the array is row-major and 5424 wide,
so a 434-column crop reads 434 separate rows **10848 bytes apart**. The
pixels total 0.38 MB; the SPAN containing them is 4.7 MB.

So ~5 MB is a floor set by the geometry. No block size beats it, and
smaller blocks cannot help -- they only divide the same span into more
round trips. My simulation modelled 18 scattered 48 KB chunks, which is
not the access pattern at all, and every conclusion from it was wrong.

### Re-derived against the real pattern

434 row-reads of 868 bytes, strided:

    block   requests   fetched   @50ms   @120ms
    128 KB        37   4.85 MB   2.39 s   4.98 s
    512 KB        10   5.24 MB   1.08 s   1.78 s
   1024 KB         5   5.24 MB   0.83 s   1.18 s
   2048 KB         3   6.29 MB   0.85 s   1.06 s

1 MB reaches the byte floor in five requests and is fastest at both
latencies. So the change was worth making -- in the opposite direction
from the one I took. Verified: 5 requests, 5.24 MB, byte-exact.

### Two tests that encoded the wrong premise

They asserted ranged blocks must be SMALLER than the streaming default,
and that a crop fetches far fewer bytes. Both followed from the bad
simulation rather than from anything measured. Replaced with the property
that actually holds: reach the geometric byte floor in few round trips,
without a block so large it over-reads past the span.

That is the seventh test in this project found asserting an assumption
rather than a fact -- and the first one I wrote, shipped, and had to
retract within a single version.

234 tests.
