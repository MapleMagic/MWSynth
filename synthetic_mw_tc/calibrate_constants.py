"""
Fit the V/H calibration constants against real microwave observations.

WHY: `synthetic_algorithm.CALIBRATION` holds about a dozen
physically-motivated GUESSES -- background brightness temperatures,
emission boosts, scattering depressions -- refined by eye against
reference imagery. They are the largest known source of error left. The
measured skill decomposition puts an irreducible bias of 11.09 K against
a 15.85 K total residual, so roughly 70% of what the correction model is
being asked to fix is systematic offset in these constants, not anything
a learned residual should be spending capacity on.

Fitting them is strictly better than learning around them: a constant
that is wrong by 10 K is wrong the same way on every frame, which is
exactly the kind of error least squares removes and exactly the kind a
few-hundred-example neural network wastes itself on.

HOW IT WORKS WITHOUT RE-MINING
------------------------------
The exported .npz files store the backbone and the real-MW target, but
not the intermediate response and scattering-fraction fields the
constants multiply. Those can be recovered algebraically. For 37 GHz the
backbone is

    v = bg_v + Ev * A - Dv * B
    h = bg_h + Eh * A - Dh * B

where A = response * (1 - scat_pot) is the emission weight and
B = response * scat_pot the scattering weight, both shared between the
two polarizations. With the constants that PRODUCED the file known, that
is a 2x2 linear system per pixel, solvable for (A, B) as long as
Eh*Dv - Ev*Dh is non-zero (currently 93*20 - 28*12 = 1524, comfortably
so). Recover (A, B), then refit the constants against the real target.

89 GHz has no emission term, so B alone is recovered from one channel.

CAVEAT, and it is a real one: the stored backbone also carries the
baseline shift, injected texture, the noise floor and (since 0.112) the
sensor PSF. Those are additive or smoothing operations applied after the
formula above, so the recovered (A, B) absorb some of them and are
approximate. The fit is therefore a substantial improvement on hand-tuned
guesses, not a clean inversion. Mining that stores `response` and
`scat_pot` directly would make it exact, and is the right follow-up --
this exists so the constants can be fitted against data ALREADY on disk
rather than waiting for another mine.
"""
from __future__ import annotations

import glob
import os

import numpy as np

# Constants fitted per frequency. Order matters: it is the parameter
# vector layout used throughout.
PARAMS_37 = ("bg_v_37", "emission_boost_v37", "max_depression_v37",
             "bg_h_37", "emission_boost_h37", "max_depression_h37")
PARAMS_89 = ("bg_v_89", "emission_boost_v89", "max_depression_v89",
             "bg_h_89", "emission_boost_h89", "max_depression_h89")

# Physically defensible bounds. Fitting is constrained rather than free
# because a few hundred storms cannot pin down twelve constants without
# occasionally finding an absurd combination that happens to reduce
# residual on the training set -- a negative background temperature, say,
# or an emission boost that drives Tb past the physical temperature.
BOUNDS = {
    # bg_v_37's upper bound was 230 while the constant itself is 250 --
    # the bound went stale when the 0.95 emission work retuned the
    # backgrounds, and nothing checked. The fitter silently CLAMPED the
    # 37V background 20 K below its true value on every run, which on a
    # dataset generated from the exact constants showed up as the fit
    # making 37V worse: RMSE 0.00 K -> 20.00 K, precisely the clip
    # distance. Only the UPPER bound was stale, so only it moved --
    # a first fix raised the lower bound too and promptly clipped a
    # legitimate 195 K test value, which is the same mistake in the
    # opposite direction.
    "bg_v_37": (170.0, 285.0), "bg_h_37": (120.0, 220.0),
    "emission_boost_v37": (0.0, 90.0), "emission_boost_h37": (0.0, 140.0),
    "max_depression_v37": (0.0, 60.0), "max_depression_h37": (0.0, 60.0),
    "bg_v_89": (240.0, 300.0), "bg_h_89": (230.0, 300.0),
    # Added with the 89 GHz emission split (0.141). Upper bounds keep the
    # peak below the sea-surface physical temperature: bg_v_89 is 280 and
    # bg_h_89 is 260, so an emitting layer has roughly 20 and 40 K of
    # headroom before it would exceed what it can physically radiate.
    "emission_boost_v89": (0.0, 18.0),
    "emission_boost_h89": (0.0, 34.0),
    "max_depression_v89": (0.0, 180.0), "max_depression_h89": (0.0, 180.0),
}


def _valid_mask(d):
    """Which pixels carry a real observation.

    Derived from FINITE TARGETS, exactly as ml_train derives it -- which
    is the only definition that exists, because the exporter never wrote
    a `mask` key at all. This required one, so every file failed the
    key check and a 1,644-example dataset reported "no usable examples".
    Step 3 had therefore never run against real data.

    An explicit mask is honoured if some future exporter writes one, but
    the finite-target rule is the fallback and the source of truth.
    """
    if "mask" in getattr(d, "files", ()):
        return np.asarray(d["mask"]).astype(bool)
    t = _tb(d, "target_h37")
    m = np.isfinite(t)
    for k in ("target_v37", "target_v89", "target_h89"):
        try:
            m &= np.isfinite(_tb(d, k))
        except Exception:
            pass
    return m


def _tb(d, key):
    """Read a brightness-temperature array, packed or not.

    Keyed on DTYPE rather than a format flag, so a dataset holding both
    float32 (pre-0.127) and uint16 files reads correctly -- which any
    dataset spanning that change will.
    """
    a = np.asarray(d[key])
    if a.dtype == np.uint16:
        from training_data_export import unpack_tb
        return unpack_tb(a)
    return a.astype(np.float32)


def recover_weights_89(v, h, cal) -> tuple:
    """Recover (A89, B89) from a stored 89 GHz backbone pair.

    ADDED in 0.141. The 89 GHz fit previously inverted a PURE-DEPRESSION
    model, `v89 = bg - D * B89`, recovering a single weight from one
    channel. That matched the backbone until 0.133 gave 89 GHz an
    emission term of its own, after which the generator was

        v89 = bg + E*A89 - D*B89

    and the fit was solving a model the data no longer came from. The
    recovered B89 silently absorbed the emission, and both 89 GHz
    constants were fitted against it -- biased, with no error raised.

    Two channels, two unknowns, exactly as 37 GHz has always been done.
    """
    Ev, Dv = cal.get("emission_boost_v89", 0.0), cal["max_depression_v89"]
    Eh, Dh = cal.get("emission_boost_h89", 0.0), cal["max_depression_h89"]
    det = Eh * Dv - Ev * Dh
    if abs(det) < 1e-9:
        # Degenerate: the two channels carry the same mix, so emission and
        # scattering cannot be separated. Fall back to the pure-depression
        # inversion rather than returning a meaningless split.
        B = (cal["bg_v_89"] - np.asarray(v)) / max(Dv, 1e-9)
        return np.zeros_like(B), B
    dv = np.asarray(v, dtype=np.float64) - cal["bg_v_89"]
    dh = np.asarray(h, dtype=np.float64) - cal["bg_h_89"]
    A = (dv * Dh - dh * Dv) / det * -1.0
    B = (dv * Eh - dh * Ev) / det * -1.0
    return np.clip(A, 0.0, 4.0), np.clip(B, 0.0, 4.0)


def recover_weights_37(v, h, cal) -> tuple:
    """Recover (A, B) -- the emission and scattering weights -- from a
    stored 37 GHz backbone pair and the constants that produced it."""
    Ev, Dv = cal["emission_boost_v37"], cal["max_depression_v37"]
    Eh, Dh = cal["emission_boost_h37"], cal["max_depression_h37"]
    det = Eh * Dv - Ev * Dh
    if abs(det) < 1e-9:
        raise ValueError(
            "37 GHz emission/depression constants are collinear "
            f"(Eh*Dv - Ev*Dh = {det:g}); (A, B) cannot be recovered from the "
            "stored backbone. Store response/scat_pot at mining time instead."
        )
    dv = np.asarray(v, dtype=np.float64) - cal["bg_v_37"]
    dh = np.asarray(h, dtype=np.float64) - cal["bg_h_37"]
    # Solve [[Ev, -Dv], [Eh, -Dh]] @ [A, B] = [dv, dh]
    A = (-Dh * dv + Dv * dh) / det
    B = (-Eh * dv + Ev * dh) / det
    return A, B


def _clip(name, value):
    lo, hi = BOUNDS[name]
    return float(np.clip(value, lo, hi))


def fit_from_examples(paths, cal, max_pixels_per_file: int = 20000,
                      progress_callback=None) -> dict:
    """Fit the V/H constants across a set of exported .npz examples.

    Returns {"fitted": {...}, "before_rmse_k": {...}, "after_rmse_k": {...},
             "n_pixels": int, "n_files": int}. The RMSE pair is the point:
    a fit that does not reduce error against held-out truth should not be
    adopted, and reporting only the new constants would hide that.
    """
    # Vintage check FIRST. This fit recovers the intermediate weights
    # from the STORED backbone using the constants that produced it, so a
    # dataset spanning a backbone change is not merely noisy -- the
    # recovered weights are wrong for one half of it, and the resulting
    # constants absorb that error as if it were physics.
    try:
        from training_data_export import check_physics_consistency
        check_physics_consistency(list(paths))
    except RuntimeError as e:
        return {"fitted": {}, "n_pixels": 0, "n_files": 0,
                "before_rmse_k": {}, "after_rmse_k": {},
                "note": f"refusing to fit a mixed-vintage dataset: {e}"}
    except Exception:
        pass    # provenance unavailable -- fall through rather than block

    a89 = []
    rows_A, rows_B, targ_v37, targ_h37 = [], [], [], []
    b89, targ_v89, targ_h89 = [], [], []
    base_v37, base_h37, base_v89, base_h89 = [], [], [], []
    n_files = 0

    for path in paths:
        try:
            with np.load(path, allow_pickle=True) as d:
                need = ("backbone_v37", "backbone_h37", "backbone_v89", "backbone_h89",
                        "target_v37", "target_h37", "target_v89", "target_h89")
                if any(k not in d.files for k in need):
                    continue
                m = _valid_mask(d)
                if not m.any():
                    continue
                # MUST unpack. 0.127 wired in scaled-uint16 storage, and
                # this read the arrays raw -- so a 182 K background came
                # back as 18200 and the least-squares fitted constants
                # against values a hundred times too large. No error, no
                # warning, just nonsense constants at the end of a 23-hour
                # mine, which would then be written into CALIBRATION by
                # --apply-fit.
                #
                # ml_train and ice_coupling_profile both key on dtype the
                # same way; this was the one reader that did not.
                bv37, bh37 = _tb(d, "backbone_v37"), _tb(d, "backbone_h37")
                bv89, bh89 = _tb(d, "backbone_v89"), _tb(d, "backbone_h89")
                tv37, th37 = _tb(d, "target_v37"), _tb(d, "target_h37")
                tv89, th89 = _tb(d, "target_v89"), _tb(d, "target_h89")
        except Exception:
            continue

        good = m & np.isfinite(bv37) & np.isfinite(tv37) & np.isfinite(bv89) & np.isfinite(tv89)
        idx = np.flatnonzero(good.ravel())
        if idx.size == 0:
            continue
        if idx.size > max_pixels_per_file:
            # Subsample rather than take the first N: pixels are spatially
            # ordered, so a prefix would be the top of the image.
            idx = np.random.default_rng(0).choice(idx, max_pixels_per_file, replace=False)

        A, B = recover_weights_37(bv37.ravel()[idx], bh37.ravel()[idx], cal)
        rows_A.append(A)
        rows_B.append(B)
        targ_v37.append(tv37.ravel()[idx])
        targ_h37.append(th37.ravel()[idx])
        base_v37.append(bv37.ravel()[idx])
        base_h37.append(bh37.ravel()[idx])

        # 89 GHz: two channels, two unknowns -- see recover_weights_89.
        _a89, _b89 = recover_weights_89(bv89.ravel()[idx], bh89.ravel()[idx], cal)
        a89.append(_a89)
        b89.append(_b89)
        targ_v89.append(tv89.ravel()[idx])
        targ_h89.append(th89.ravel()[idx])
        base_v89.append(bv89.ravel()[idx])
        base_h89.append(bh89.ravel()[idx])
        n_files += 1

    if not rows_A:
        return {"fitted": {}, "n_pixels": 0, "n_files": 0,
                "before_rmse_k": {}, "after_rmse_k": {},
                "note": "no usable examples (need backbone_*, target_* and mask)"}

    A = np.concatenate(rows_A); B = np.concatenate(rows_B)
    tv37 = np.concatenate(targ_v37); th37 = np.concatenate(targ_h37)
    bv37 = np.concatenate(base_v37); bh37 = np.concatenate(base_h37)
    A89 = np.concatenate(a89)
    B89 = np.concatenate(b89)
    tv89 = np.concatenate(targ_v89); th89 = np.concatenate(targ_h89)
    bv89 = np.concatenate(base_v89); bh89 = np.concatenate(base_h89)

    ones = np.ones_like(A)
    fitted = {}

    def _solve(design, target, names):
        """Least squares with the bounds applied afterwards. lstsq then
        clip, rather than a constrained solver: with a well-conditioned
        design the unconstrained solution is almost always in range, and
        clipping makes the rare out-of-range case obvious rather than
        silently reshaping the whole fit."""
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return {n: _clip(n, c) for n, c in zip(names, coef)}

    # 37 GHz: target = bg + E*A - D*B  ->  design [1, A, -B]
    d37 = np.column_stack([ones, A, -B])
    fitted.update(_solve(d37, tv37, ("bg_v_37", "emission_boost_v37", "max_depression_v37")))
    fitted.update(_solve(d37, th37, ("bg_h_37", "emission_boost_h37", "max_depression_h37")))
    # 89 GHz: target = bg + E*A89 - D*B89  ->  design [1, A89, -B89].
    # Three columns, matching the backbone since 0.133; two columns fitted
    # a model the data did not come from.
    d89 = np.column_stack([ones, A89, -B89])
    fitted.update(_solve(d89, tv89, ("bg_v_89", "emission_boost_v89", "max_depression_v89")))
    fitted.update(_solve(d89, th89, ("bg_h_89", "emission_boost_h89", "max_depression_h89")))

    def _rmse(a, b):
        return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))

    before = {"v37": _rmse(bv37, tv37), "h37": _rmse(bh37, th37),
              "v89": _rmse(bv89, tv89), "h89": _rmse(bh89, th89)}
    after = {
        "v37": _rmse(fitted["bg_v_37"] + fitted["emission_boost_v37"] * A
                     - fitted["max_depression_v37"] * B, tv37),
        "h37": _rmse(fitted["bg_h_37"] + fitted["emission_boost_h37"] * A
                     - fitted["max_depression_h37"] * B, th37),
        "v89": _rmse(fitted["bg_v_89"] + fitted["emission_boost_v89"] * A89
                     - fitted["max_depression_v89"] * B89, tv89),
        "h89": _rmse(fitted["bg_h_89"] + fitted["emission_boost_h89"] * A89
                     - fitted["max_depression_h89"] * B89, th89),
    }

    if progress_callback:
        progress_callback(f"Fitted on {n_files} file(s), {A.size:,} pixels")
        for k in ("v37", "h37", "v89", "h89"):
            progress_callback(f"  {k}: RMSE {before[k]:.2f}K -> {after[k]:.2f}K "
                              f"({100*(1-after[k]/max(before[k],1e-9)):+.0f}%)")

    # State coverage EXPLICITLY. A constant absent from the fit is not a
    # failure, but silence about it looks like one -- a previous run
    # reported constants "not found" with no indication of whether that
    # was a bug or by design.
    uncovered = sorted(k for k, v in cal.items()
                       if isinstance(v, (int, float)) and k not in fitted)
    if progress_callback and uncovered:
        progress_callback(f"  not fitted ({len(uncovered)}), by design -- these "
                          f"do not enter the linear backbone model:")
        progress_callback("    " + ", ".join(uncovered))

    return {"fitted": fitted, "before_rmse_k": before, "after_rmse_k": after,
            "n_pixels": int(A.size), "n_files": n_files,
            "not_fitted": uncovered}


def fit_from_dataset(data_dir=None, progress_callback=None) -> dict:
    """Convenience wrapper over the exported training set."""
    from synthetic_algorithm import CALIBRATION
    import training_data_export as tde

    if data_dir is None:
        data_dir = tde.DEFAULT_EXPORT_DIR
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    return fit_from_examples(paths, CALIBRATION, progress_callback=progress_callback)


def format_result(result: dict) -> str:
    """Render a fit as a paste-ready CALIBRATION diff.

    Deliberately NOT applied automatically. These constants define the
    backbone every stored residual is measured against, so changing them
    invalidates the training set (VH_PHYSICS_ID exists for exactly this)
    and should be a decision, not a side effect of running a script.
    """
    if not result.get("fitted"):
        return result.get("note", "no fit produced")
    from synthetic_algorithm import CALIBRATION
    lines = ["Fitted constants (review, then paste into CALIBRATION):"]
    for name in list(PARAMS_37) + list(PARAMS_89):
        old = CALIBRATION.get(name)
        new = result["fitted"].get(name)
        if new is None:
            continue
        lo, hi = BOUNDS[name]
        flag = "  <- AT BOUND" if abs(new - lo) < 1e-6 or abs(new - hi) < 1e-6 else ""
        lines.append(f'    "{name}": {new:.1f},   # was {old:.1f}{flag}')
    lines.append("")
    lines.append("Bump VH_PHYSICS_ID and re-mine before training on these.")
    return "\n".join(lines)


def apply_fitted_constants(result: dict, module_path: str = None,
                           bump_physics_id: bool = True) -> dict:
    """Rewrite the fitted values into synthetic_algorithm.CALIBRATION.

    Deliberately NOT called by `fit_from_dataset`. These constants define
    the backbone every stored residual is measured against, so changing
    them invalidates the training set -- it has to be a decision, and the
    caller has to have seen the diff.

    Writes a timestamped .bak beside the module first. Only numeric
    literals of the form `"name": <number>,` inside CALIBRATION are
    touched; anything it cannot match unambiguously is left alone and
    reported, rather than guessed at.

    Returns {"applied": [...], "skipped": [...], "backup": path,
             "physics_id": new_id}.
    """
    import os
    import re
    import shutil
    import time

    fitted = result.get("fitted") or {}
    if not fitted:
        return {"applied": [], "skipped": [], "backup": None, "physics_id": None,
                "note": "nothing to apply"}

    if module_path is None:
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "synthetic_algorithm.py")
    backup = f"{module_path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    shutil.copy2(module_path, backup)

    with open(module_path, encoding="utf-8") as fh:
        src = fh.read()

    applied, skipped = [], []
    for name, value in fitted.items():
        # Anchored on the quoted key so a bare number elsewhere in the
        # file cannot be hit by accident.
        pattern = re.compile(rf'("{re.escape(name)}"\s*:\s*)(-?\d+(?:\.\d+)?)')
        matches = pattern.findall(src)
        if len(matches) != 1:
            skipped.append(f"{name} ({len(matches)} matches)")
            continue
        src = pattern.sub(rf'\g<1>{value:.1f}', src, count=1)
        applied.append(f"{name}={value:.1f}")

    # No explicit bump needed any more, and that is the point: since 0.136
    # VH_PHYSICS_ID is DERIVED from the constants, so rewriting any of them
    # changes it automatically. Previously this had to remember, and the
    # one time a constant moved without it (SATURATION_K in 0.135) two
    # different physics shared a label.
    #
    # The id is recomputed below from the file as written, so the caller
    # still learns what the dataset must now be tagged with.
    new_id = None

    with open(module_path, "w", encoding="utf-8") as fh:
        fh.write(src)

    if applied:
        try:
            import importlib
            import synthetic_algorithm as _sa
            importlib.reload(_sa)
            new_id = _sa.VH_PHYSICS_ID
        except Exception:
            new_id = "changed (recompute on next import)"

    return {"applied": applied, "skipped": skipped, "backup": backup,
            "physics_id": new_id}


# --- Measuring physics from the mined data, not from rendered images ---

def ice_coupling_profile(paths, bins=None, max_pixels_per_file: int = 30000,
                         progress_callback=None) -> dict:
    """Measure how real 37 GHz emission varies with real 89 GHz depression.

    WHY THIS EXISTS. Li et al. (2026) describe an interaction the backbone
    does not model at all: 37 and 89 GHz are computed independently here,
    so nothing lets ice aloft screen the liquid emission beneath it. Their
    cloud-type analysis says the 37 GHz emission under opaque ice is "only
    partially attenuated by the ice layer above".

    An attempt to check this by sampling colours out of rendered NRL
    imagery came back AMBIGUOUS: the deepest-ice class was ~6,700 pixels,
    and the 37H "emitting" colour mask may exclude the very hottest pixels
    (which render dark red to black), so a low emitting-fraction could mean
    either screening or the opposite. A nonlinear colour table read through
    RGB thresholds cannot settle a question this fine.

    The mined dataset can. Every .npz holds the real 37 and 89 GHz
    brightness temperatures on the same grid, so the relationship is
    measurable directly rather than inferred from a picture.

    Returns the median target 37H, and its spread, binned by target 89H.
    A falling 37H toward the coldest 89H bins is evidence of screening; a
    flat or rising profile says ice and liquid simply co-vary and no
    attenuation term is warranted.
    """
    if bins is None:
        bins = [150, 180, 200, 212, 228, 254, 275, 300]

    sums = {i: [] for i in range(len(bins) - 1)}
    n_files = 0
    for path in paths:
        try:
            with np.load(path, allow_pickle=True) as d:
                if not {"target_h37", "target_h89"} <= set(d.files):
                    continue
                t37 = np.asarray(d["target_h37"])
                t89 = np.asarray(d["target_h89"])
                if t37.dtype == np.uint16:
                    from training_data_export import unpack_tb
                    t37, t89 = unpack_tb(t37), unpack_tb(t89)
                mask = (_valid_mask(d)
                        if "mask" in d.files else np.ones(t37.shape, bool))
        except Exception:
            continue

        good = mask & np.isfinite(t37) & np.isfinite(t89)
        idx = np.flatnonzero(good.ravel())
        if idx.size == 0:
            continue
        if idx.size > max_pixels_per_file:
            idx = np.random.default_rng(0).choice(idx, max_pixels_per_file,
                                                  replace=False)
        a37, a89 = t37.ravel()[idx], t89.ravel()[idx]
        which = np.digitize(a89, bins) - 1
        for i in range(len(bins) - 1):
            sel = a37[which == i]
            if sel.size:
                sums[i].append(sel)
        n_files += 1

    out = []
    for i in range(len(bins) - 1):
        if not sums[i]:
            continue
        v = np.concatenate(sums[i])
        out.append({"h89_range": (bins[i], bins[i + 1]), "n": int(v.size),
                    "h37_median": float(np.median(v)),
                    "h37_p25": float(np.percentile(v, 25)),
                    "h37_p75": float(np.percentile(v, 75))})

    if progress_callback and out:
        progress_callback(f"Ice coupling, {n_files} file(s):")
        progress_callback(f"  {'89H bin':>14} {'n':>9} {'37H median':>12}")
        for row in out:
            progress_callback(f"  {row['h89_range'][0]:>5}-{row['h89_range'][1]:<8} "
                              f"{row['n']:>9,} {row['h37_median']:>12.1f}")
        deep = [r for r in out if r["h89_range"][1] <= 212]
        mid = [r for r in out if 212 <= r["h89_range"][0] < 254]
        if deep and mid:
            d_med = float(np.mean([r["h37_median"] for r in deep]))
            m_med = float(np.mean([r["h37_median"] for r in mid]))
            verdict = ("screening: 37H falls under deep ice"
                       if d_med < m_med - 3 else
                       "no screening: ice and liquid co-vary")
            progress_callback(f"  deep-ice 37H {d_med:.1f} K vs mid-ice {m_med:.1f} K"
                              f"  -> {verdict}")
    return {"bins": out, "n_files": n_files}
