# MWSynth — Synthetic Tropical Cyclone Microwave Imagery

Estimates what 37 GHz and 89 GHz passive-microwave imagery of a tropical
cyclone would look like at an arbitrary time, by combining geostationary
infrared imagery, best-track intensity, and whatever real microwave and
radar data happens to be available.

Passive microwave is the best way to see a tropical cyclone's inner core,
but polar-orbiting sensors only pass over a given storm a few times a
day. Between passes, forecasters have infrared and not much else. MWSynth
fills those gaps with a physically-motivated estimate, fusing real
observations wherever they exist and falling back to a parametric model
where they don't.

**MWSynth output is synthetic. It is not an observation.** Where real
microwave data is fused in it is weighted and labelled as such, and the
figure title always records what went into the frame. Nothing here should
be treated as a substitute for a real overpass, and the structural
metrics it reports describe the synthetic field, not the storm.

---

## What it does

Four panels per frame:

- **Input GOES IR** (band 13), with the storm centre used for generation
  marked, plus an independent IR-derived eye position when one is found.
- **Synthetic 37 GHz** in the NRL colour table — the emission signature
  from liquid precipitation, which shows the low-level eyewall.
- **Synthetic 89 GHz** in the NRL colour table — ice scattering, which
  shows deep convection and rainband structure.
- **A diagnostic panel**, either the convective-signal field or the same
  89 GHz composite with the ML correction disabled, for A/B comparison.

It also does multi-frame loops with GIF export, automatic calibration
against real passes, and optional NEXRAD fusion near the US coast.

## How it works

```
GOES ABI (bands 2/7/9/13)  --+
NHC best track (position,    +--> convective signal --+
  intensity, RMW, ROCI)    --+    + radial weighting  |
                                                      +--> weighted fusion --> 37/89 GHz V/H
Real MW pass (GMI/AMSR3/WSFM) --> morphed to frame ---+         ^
NEXRAD Level II (optional)  ----> regridded ----------+         |
                                                    ML correction (optional)
```

1. **Convective signal.** IR brightness temperature, the IR-WV difference
   (which discriminates deep convection from thin cirrus), and texture
   are combined into a 0-1 field.
2. **Radial weighting.** A parametric vortex — an eyewall ring at the
   RMW, a decaying envelope scaled by ROCI, and eye suppression — imposes
   TC structure the IR alone cannot supply.
3. **Fusion.** Real MW and radar are blended per-pixel with the
   parametric backbone, weighted by coverage and by pass age. A fresh
   pass dominates; a six-hour-old one contributes little. Real passes are
   MIMIC-TC-style morphed to the frame time along the storm's motion.
4. **Radiative mapping.** The fused response is mapped to V- and
   H-polarised brightness temperatures per frequency, then rendered
   through the NRL colour tables.
5. **ML correction (optional).** A U-Net trained on paired GOES/real-MW
   scenes predicts a residual against the parametric backbone, applied
   within a storm-centred patch. Inactive if no checkpoint is present.

## Install

Python 3.10+.

```bash
git clone <repo-url> && cd synthetic_mw_tc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Only the core dependencies are required. Everything under "Optional" in
`requirements.txt` degrades gracefully: without `arm_pyart` radar fusion
is skipped, without `sgp4`/`skyfield` the sensor rotation is tried in
default order, and without `torch` the ML correction becomes a no-op and
generation proceeds on the parametric backbone alone.

### Credentials

Real MW ingest needs free accounts, entered in the **MW Data
Credentials** tab and stored in `credentials.json` next to the source
(git-ignored):

| Service | Used for | Register |
|---|---|---|
| NASA PPS | Near-real-time GMI / WSFM / AMSR3 | https://registration.pps.eosdis.nasa.gov/registration/ |
| NASA Earthdata | GES DISC archive GMI (fallback, no retention limit) | https://urs.earthdata.nasa.gov/users/new |

These are separate accounts with separate registrations. Without them,
MWSynth runs GOES-only.

## Usage

Pick a satellite, sector, storm and time, then **Generate synthetic MW**.

**Fuse real MW when available** (on by default) searches for a pass that
actually observed the storm's core and falls back to GOES-only if nothing
is found. Untick it to force a GOES-only frame even when a pass exists —
that is a diagnostic mode, and the only way to see the parametric
backbone and the ML correction in isolation, since real MW masks both
wherever it has coverage.

Enable NEXRAD fusion for storms within ~200 mi of a US radar.

Frame loops render N frames at a chosen interval and export as GIF.

**ML correction strength** (0.00-2.00) scales the learned correction. Set
it to 0 to compare against the raw parametric backbone; the ML-OFF
comparison panel does the same A/B from a single run, which avoids the
run-to-run noise of generating the same frame twice.

### Mining training data

```bash
python src/run_ml_pipeline.py --estimate --start 2018 --end 2025   # size it first
python src/run_ml_pipeline.py --step 1 --start 2018 --end 2025     # mine
python src/run_ml_pipeline.py --step 2                             # train
```

TC PRIMED overpasses are read **in place on S3** via ranged GETs, so no
source files are written to disk — storage stops being the limit on
dataset size. `--estimate` lists what would be processed without fetching
any file bodies, which takes seconds and is the right way to size a run.

`--agency NHC` is AL/EP/CP; `JTWC` is WP/IO/SH. The JTWC basins sit
outside GOES coverage and will almost entirely skip until
Himawari/Meteosat ingest exists; the flag warns and proceeds rather than
silently mining nothing.

## Project layout

```
src/
  main.py                    entry point
  gui/main_window.py         PyQt6 interface, rendering, frame loops
  gui/credentials_tab.py     credential entry and cache management
  synthetic_algorithm.py     the core model: signal, weighting, fusion, mapping
  mw_composites.py           NRL 37/89 GHz colour tables (PCT formulas)
  mw_structure_metrics.py    structural scoring (eyewall radius, core area)
  tc_center_fix.py           independent IR centre estimate, for verification
  goes_fetch.py              GOES ABI from NOAA public S3
  besttrack.py               ATCF b-deck / IBTrACS parsing and interpolation
  mw_ingest.py               real MW retrieval, QC, morphing
  radar_ingest.py            NEXRAD Level II
  calibration_state.py       persisted bias offsets
  ml_*.py                    correction model: data mining, training, inference
tests/
  test_mwsynth.py            regression suite (no network/GPU/display needed)
```

## Tests

```bash
python -m pytest tests/ -v      # or: python tests/test_mwsynth.py
```

47 tests, requiring no network, credentials, torch or display. They are
regression tests rather than a specification — nearly every case
corresponds to a bug that actually shipped and was found by looking at an
image afterwards. See the file's docstring for what is deliberately not
covered.

## Known limitations

Worth understanding before trusting output:

- **Infrared and microwave measure different things, and GOES-only
  frames are limited by that, not by tuning.** IR sees cloud-top
  temperature; passive microwave sees the vertically integrated
  hydrometeor column. They decouple exactly where it matters — a mature
  eyewall and a decaying central overcast can share a cloud-top
  signature while looking nothing alike in MW. Measured against a real
  GMI pass 44 minutes from a GOES-only frame, the backbone gets the
  far-field level right (H37 182 K, against 182–193 K observed) and the
  eyewall level close (255 K against 269 K), but under-collapses the
  polarization difference at the eyewall (V−H 22 K, against 5 K in the
  fused frame) and spreads the 89 GHz core over roughly 3.4× the real
  area as one contiguous mass where reality is a compact ring plus
  discrete spiral bands. Fusing a real pass fixes all three. Treat
  GOES-only output as an interpolation between real passes, which is what
  it is, not as a substitute for one.
- **The parametric backbone is calibrated by hand, but now fittable.**
  The constants in `CALIBRATION` began as physically-motivated guesses.
  `calibrate_constants.py` fits them by least squares against real MW
  targets in the mined dataset, which is the direct attack on the ~11 K
  systematic bias; run it once a season or two has been mined.
  Using cloud-top IR to split emission from scattering is the weakest
  link, particularly at 37 GHz where the signal comes from liquid water
  below the freezing level.
- **Best track is 6-hourly.** Between fixes the centre is interpolated;
  past the last fix it is extrapolated along recent motion and labelled
  `centre PROJECTED Nh`. The IR centre check reports the disagreement,
  and everything in the frame is positioned from that centre.
- **The ML correction is trained on a few hundred examples, and the
  current checkpoint is stale.** 0.95 changed the 37 GHz backbone physics,
  so any checkpoint trained before it learned residuals against a
  backbone that no longer exists; MWSynth detects this and warns, but the
  correct fix is retraining. Even setting that aside, it adds genuine
  fine structure at 89 GHz while having been observed suppressing real
  signatures at 37 GHz. Treat a non-zero strength as an experiment, not a
  default, and use the ML-OFF panel.
- **Real MW is extrapolated up to 210 km from the nearest real sample at
  full confidence.** A reported coverage percentage therefore includes
  some extrapolation, and structure can smear perpendicular to a swath
  edge.
- **Land is masked only if `global_land_mask` is installed.** Without it
  the grid is treated as ocean, and land in view gets ocean emissivity —
  which at 37/89 GHz resembles heavy precipitation. MWSynth warns when
  this fallback is in force.
- **No Himawari/Meteosat**, so the west Pacific and Indian Ocean are not
  supported. The code is dateline-safe in anticipation, but untested
  there.
- **Radar fusion is off by default.** NEXRAD only ever covered storms
  near the US coast and is a display layer better served by any radar
  app; `radar_ingest.py` still works if enabled. GPM DPR, already inside
  the TC PRIMED files being streamed, is the better route if radar is
  wanted as a constraint.

## Contributing

Issues and pull requests welcome. Some things that would help most:

- Himawari/Meteosat ingest, to open the WP and IO basins (this would also
  roughly double the available training data).
- A radiative-transfer-based replacement for the hand-tuned calibration.
- Validation against real overpasses at scale — the structural metrics
  exist but have only been applied to a handful of frames.
- Splitting reported coverage into real versus extrapolated.

If you are fixing a bug, a regression test in `tests/` is worth more than
the fix itself. Most of what has gone wrong here was invisible in the
output until someone looked hard at one specific image.

`CHANGELOG.md` is a full chronological development log. It is long, and
deliberately records *why* decisions were made and how bugs were
diagnosed rather than only what changed — useful if you are wondering why
something is the way it is.

## Related work

Li, Tan & Bai (2025), *Generative Deep Learning Reconstructs Tropical
Cyclone Microwave Data from Geostationary Infrared Radiometers*
([preprint](https://doi.org/10.22541/essoar.173655526.67865984/v1)), take
the same IR-to-microwave problem with a conditional diffusion model over
13,213 TC PRIMED samples. Worth reading before extending the ML side
here: their ablations quantify how much multi-channel IR and elevation
matter as inputs, and their ensemble-mean-versus-member comparison is a
clean demonstration of why pixel-wise losses over-smooth. See the 0.97
changelog entry for a point-by-point mapping onto this project.

The peer-reviewed successor — Li, Tan & Bai (2026), *Physically Consistent
Synthesis of Tropical Cyclone Microwave Brightness Temperatures From
Geostationary Infrared Observations*, JGR: Machine Learning and
Computation ([doi](https://doi.org/10.1029/2026JH001257)) — adds a
per-channel IR saliency analysis that MWSynth now uses directly to order
which ABI bands are worth fetching, and a cloud-type breakdown showing
that cirrus is largely transparent at 37/89 GHz. Its closing caveat is
worth quoting as the honest bound on this whole class of tool: there is
an identifiability limit in the IR-to-microwave mapping, and real
observations remain irreplaceable.

Also directly adjacent: Haynes et al. (2024) on synthetic 89/37 GHz from
operational geostationary satellites, and Wimmers & Velden (2007) on
MIMIC-TC, whose morphing approach MWSynth's pass advection follows.

## Data sources and credits

- GOES-18/19 ABI via the NOAA public S3 buckets (`noaa-goes18`,
  `noaa-goes19`), unsigned access.
- NHC ATCF best-track b-decks; IBTrACS for historical storms.
- GPM GMI, AMSR3 and WSFM via NASA PPS and GES DISC.
- NEXRAD Level II via the public NOAA S3 archive.
- Colour tables ported from NRL Monterey's GeoIPS 37 GHz and 89 GHz
  product definitions, including the PCT coefficients.

## Licence

See `LICENSE`.
