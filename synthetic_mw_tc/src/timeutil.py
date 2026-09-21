"""
One datetime convention for the whole project: tz-aware UTC.

WHY THIS EXISTS. The project accumulated a mixture. Best track parsed
naive times with `strptime`, TC PRIMED returned naive until 0.105 and
aware after, `mw_ingest` deliberately STRIPPED tzinfo in three places to
force naive, `goes_fetch` was naive until 0.115, and `radar_ingest` is
naive. Every module was internally consistent and the combination was
not.

That is not a theoretical problem. It cost two full mining runs: 4,867
overpasses attempted and 0 saved, then 613 and 0 saved, both with

    TypeError: can't compare offset-naive and offset-aware datetimes

and the first of those was invisible until the summary because the skip
label recorded the exception type without its message.

Patching one boundary at a time made it worse rather than better: fixing
TC PRIMED in 0.105 moved the clash to goes_fetch, and fixing goes_fetch
in 0.115 moved it to best track. Every fix was locally correct and the
bug survived, because the actual defect was the absence of a convention.

THE CONVENTION: **every datetime crossing a module boundary is tz-aware
UTC.** Naive values are treated as UTC and upgraded, never rejected --
everything in this project genuinely is UTC (satellite filenames,
best-track synoptic times, TC PRIMED epochs), so a naive value is
under-specified rather than wrong, and raising on it would convert a
harmless ambiguity into an outage.

Aware UTC rather than naive UTC because `utcnow()` and
`utcfromtimestamp()` are deprecated from Python 3.12, so the naive
convention is on a path to being actively unsupported.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def as_utc(t: Optional[datetime]) -> Optional[datetime]:
    """Coerce a datetime to tz-aware UTC. None passes through.

    A naive input is assumed to be UTC and labelled as such; an aware
    input in another zone is converted. Idempotent, so it is safe to
    apply at every boundary without tracking whether it was already done.
    """
    if t is None:
        return None
    if getattr(t, "tzinfo", None) is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def as_naive_utc(t: Optional[datetime]) -> Optional[datetime]:
    """Strip to naive UTC, for the few places that must hand a datetime to
    an external library that rejects aware values.

    Only for that purpose. Anything internal should use as_utc().
    """
    t = as_utc(t)
    return None if t is None else t.replace(tzinfo=None)


def utcnow() -> datetime:
    """Aware-UTC replacement for the deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


def same_instant(a: Optional[datetime], b: Optional[datetime]) -> bool:
    """Compare two datetimes regardless of awareness, without raising."""
    if a is None or b is None:
        return a is b
    return as_utc(a) == as_utc(b)
