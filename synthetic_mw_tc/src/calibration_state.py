"""
Persistent auto-calibration state for the synthetic MW algorithm.

Design choice, and why: rather than having the auto-ingested real MW data
directly perturb synthetic_algorithm.CALIBRATION's dozen interacting
physical constants (background Tb, emission boost, depression magnitude,
radial-weight shape, ...) from a single scene's comparison, this applies
a much simpler, safer correction -- a persisted per-frequency ADDITIVE
BIAS OFFSET, updated via an exponential moving average (EMA) each time a
real MW pass is successfully ingested and compared. Reasoning:

  - A single scene's bias/RMSE isn't enough signal to responsibly refit
    a dozen coupled constants without risking overfitting to whatever
    that one storm happened to look like (or actively destabilizing the
    physically-motivated shape of the field for unrelated storms).
  - An additive offset is easy to reason about, easy to bound/sanity-
    check, and directly addresses the most common form of "this runs a
    bit warm/cold overall" miscalibration without touching the model's
    internal structure (radial profile, RMW/ROCI response, etc.).
  - EMA update (rather than overwriting with the latest bias) means the
    offset reflects a running consensus across many storms/passes over
    time, not just whatever the most recent one happened to show --
    genuinely "learns" with repeated use rather than resetting each run.

State persists at ~/.synthetic_mw_tc/calibration_state.json, loaded once
per generate_synthetic_mw() call and applied as a flat Kelvin shift to
the final tb37/tb89 output. A conservative EMA alpha and a hard clamp on
the offset magnitude keep a single anomalous comparison (e.g. a bad
regrid, a mismatched storm) from swinging the correction too far in one
update.
"""
from __future__ import annotations

import json
import tempfile
import threading
import os

STATE_PATH = os.path.expanduser("~/.synthetic_mw_tc/calibration_state.json")

# Identifier for the CALIBRATION baseline the persisted offsets were
# learned against. Bump this whenever synthetic_algorithm's bg_tb_37 /
# bg_tb_89 (or anything else that shifts the mean level of tb37/tb89)
# changes.
#
# WHY THIS EXISTS: the offset is an EMA of (synthetic - real). If the
# baseline that produced "synthetic" changes but a stale offset survives,
# the correction double-counts -- after 0.89 raised bg_tb_37 by 37 K, a
# leftover -36.8 K offset would have gone on adding another 36.8 K on top
# of the fix, leaving the field ~37 K too WARM and looking, from the
# outside, exactly like a fresh calibration problem. Worse, the EMA would
# then have spent another 50-odd passes slowly unlearning it. Detecting
# the mismatch and starting clean is the only safe behaviour, and it
# cannot be left to the user remembering to press "Reset persisted
# calibration".
CALIBRATION_BASELINE_ID = "0.89-bg37_202-bg89_266"

# EMA update rate: how much a single new comparison shifts the persisted
# offset. Low (not overreacting to one scene); not zero (still adapts
# over repeated use).
EMA_ALPHA = 0.15

# Hard bound on the offset magnitude -- if the running bias estimate ever
# wants to go beyond this, something's more likely wrong with a specific
# comparison (bad regrid, wrong storm matched, real data QC issue) than
# with the underlying model needing a correction this large.
MAX_OFFSET_K = 40.0

DEFAULT_STATE = {
    "offset_37": 0.0,
    "offset_89": 0.0,
    "n_updates_37": 0,
    "n_updates_89": 0,
    "baseline_id": CALIBRATION_BASELINE_ID,
}

# Set when load_state() discards offsets learned against a superseded
# baseline, so the GUI can say so instead of silently showing zeros.
LAST_LOAD_WAS_RESET = {"reset": False, "old_id": None}


def load_state() -> dict:
    """Load persisted offsets, discarding any learned against a different
    CALIBRATION baseline (see CALIBRATION_BASELINE_ID)."""
    if not os.path.exists(STATE_PATH):
        return dict(DEFAULT_STATE)
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        merged = dict(DEFAULT_STATE)
        merged.update({k: v for k, v in state.items() if k in DEFAULT_STATE})

        # A state file written before baseline_id existed has no way to
        # say which baseline it came from, so it is treated as stale --
        # which it is, since the only pre-existing files were written
        # against the pre-0.89 baselines.
        stored_id = state.get("baseline_id")
        if stored_id != CALIBRATION_BASELINE_ID:
            LAST_LOAD_WAS_RESET["reset"] = True
            LAST_LOAD_WAS_RESET["old_id"] = stored_id or "pre-0.89 (unversioned)"
            fresh = dict(DEFAULT_STATE)
            save_state(fresh)
            return fresh
        # NOTE: deliberately does NOT clear the flag. load_state() is
        # called several times per generate (and again by
        # get_status_summary), so the very next call after a reset would
        # see the freshly-written matching file and wipe the notice
        # before anything displayed it -- the user would be silently
        # reset with no indication. The flag is sticky for the process
        # lifetime instead.
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_STATE)


# Serializes the read-modify-write in update_offset(). The frame-loop
# path deliberately skips calibration updates for exactly this reason,
# but a lock costs nothing and covers the cases that remain: a user
# pressing Generate twice, or the Reset button firing while a single-frame
# worker is mid-update.
_STATE_LOCK = threading.Lock()


def save_state(state: dict) -> None:
    """Write atomically: serialize to a temp file in the same directory,
    then os.replace() it into place.

    Writing directly into STATE_PATH means a crash, power loss, or a
    second writer between truncation and completion leaves a half-written
    file. load_state() recovers from that by falling back to defaults, so
    the failure is silent -- the calibration learned over dozens of passes
    just quietly resets. os.replace is atomic on POSIX and Windows, so a
    reader sees either the old state or the new one, never a partial."""
    directory = os.path.dirname(STATE_PATH)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".calibration_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_PATH)
    except Exception:
        # Never leave the temp file behind on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_offset(freq: int, bias_k: float) -> dict:
    """Update the persisted offset for `freq` (37 or 89) using a new
    observed bias (synthetic - real, from mw_compare.py's convention),
    via EMA, clamp, and persist. Returns the updated state dict.

    bias_k > 0 means the synthetic field ran warmer than real -- the
    correction should SUBTRACT that bias going forward, so the stored
    offset is the negative of the EMA-updated bias (added directly to
    future synthetic output: corrected = raw - offset... see
    apply_offset() below for the exact sign convention actually used).
    """
    # Read-modify-write held under a lock. Without it two concurrent
    # updates both read the same starting offset, both apply their own
    # EMA step, and whichever writes last silently discards the other --
    # so an observation is lost with no indication it happened.
    with _STATE_LOCK:
        state = load_state()
        key = f"offset_{freq}"
        n_key = f"n_updates_{freq}"

        current = state.get(key, 0.0)
        updated = (1 - EMA_ALPHA) * current + EMA_ALPHA * bias_k
        updated = max(-MAX_OFFSET_K, min(MAX_OFFSET_K, updated))

        state[key] = updated
        state[n_key] = state.get(n_key, 0) + 1
        save_state(state)
        return state


def apply_offset(tb, freq: int):
    """Apply the current persisted correction to a synthetic Tb field
    (any numpy array). Offset is stored as (synthetic - real) bias, so
    subtracting it moves the synthetic field toward the real-data mean
    observed so far."""
    state = load_state()
    offset = state.get(f"offset_{freq}", 0.0)
    return tb - offset


def get_status_summary() -> str:
    """Human-readable state, including an explicit warning when an offset
    is approaching MAX_OFFSET_K. That threshold matters: the EMA silently
    saturates there, so an offset sitting at the bound stops tracking the
    real bias and starts merely reporting the bound. A persistent ~37 K
    offset at 37 GHz is not a small residual calibration nudge -- it is a
    sign the parametric background Tb for that frequency is structurally
    wrong, and no amount of further EMA updating will surface that,
    because the number just stops moving."""
    state = load_state()
    parts = []
    for freq in (37, 89):
        offset = state[f"offset_{freq}"]
        n = state[f"n_updates_{freq}"]
        flag = ""
        if abs(offset) >= 0.9 * MAX_OFFSET_K:
            flag = f"  [WARNING: at/near the {MAX_OFFSET_K:g}K bound -- EMA is saturating, investigate the {freq} GHz baseline]"
        parts.append(f"{freq} GHz: offset={offset:+.1f}K over {n} update(s){flag}")
    summary = "; ".join(parts)
    if LAST_LOAD_WAS_RESET.get("reset"):
        summary += (f"  [offsets were RESET: previous values were learned against "
                    f"baseline '{LAST_LOAD_WAS_RESET['old_id']}', which 0.89 replaced. "
                    f"Reusing them would have double-counted the fix.]")
    return summary
