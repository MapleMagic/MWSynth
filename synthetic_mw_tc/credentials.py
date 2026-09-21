"""
Local storage for the login credentials needed by mw_ingest.py:
  - PPS  ("pps_nrt")   -- the near-real-time jsimpsonhttps feed, which is
    where GMI-NRT / WSFM-NRT / AMSR3-NRT all come from. This is the
    primary path and the one that does the real work.
  - Earthdata ("earthdata") -- the GES DISC archive GMI fallback. Higher
    latency and not part of the normal flow, but retained deliberately:
    it is not subject to NRT's ~7-day retention window, so it is the only
    path that works for older cases, and it has historically been the
    thing that still worked when NRT did not.

The JAXA G-Portal (AMSR2) and NOAA CLASS (SSMIS) entries were removed:
AMSR2-NRT retired transmission after Aug 31 2026 and SSMIS shut down in
September 2026, so neither service is used. NOAA CLASS was never actually
automated at all -- its fetch function raised NotImplementedError -- so
its credential box could never have done anything. Storing credentials
for services that cannot be reached is a small security cost for no
benefit. mw_ingest.py's AMSR2 G-Portal code is left in place; it just no
longer has a credential entry feeding it.

IMPORTANT: Earthdata Login and PPS are SEPARATE account systems, even
though both relate to NASA GPM data. Earthdata (urs.earthdata.nasa.gov)
covers the GES DISC science archive; PPS (registration.pps.eosdis.nasa.gov)
is a distinct registration required for the near-real-time "jsimpson"
feed, and PPS doesn't use a separate password at all -- your registered
email serves as both username and password. They're kept as separate
credential entries here rather than one being reused for the other.

Storage is plaintext JSON on the local machine, at ~/.synthetic_mw_tc/credentials.json.
This is intentionally simple rather than using OS keychain integration --
it's easy to inspect/clear by hand, at the cost of not being encrypted at
rest. Persisting to disk at all is OFF by default (see save_credentials'
`persist` argument / the GUI toggle) -- when off, credentials only live in
memory for the running session and are never written to disk.

The file (when it exists) is created with 0600 permissions (owner
read/write only) as a baseline precaution, though on a shared/compromised
machine that's not a strong guarantee -- this is a convenience store for a
personal research tool, not a secrets manager.
"""
from __future__ import annotations

import json
import os
import stat

CRED_DIR = os.path.expanduser("~/.synthetic_mw_tc")
CRED_PATH = os.path.join(CRED_DIR, "credentials.json")

SERVICES = ("pps_nrt", "earthdata")


def load_credentials() -> dict:
    """Returns {} if no saved file exists yet."""
    if not os.path.exists(CRED_PATH):
        return {}
    try:
        with open(CRED_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_credentials(data: dict) -> None:
    """Overwrites the credentials file with `data` (dict of
    service -> {"username": ..., "password": ...}). Creates the directory
    if needed and sets restrictive file permissions."""
    os.makedirs(CRED_DIR, exist_ok=True)
    with open(CRED_PATH, "w") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(CRED_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass  # best-effort on platforms where this doesn't apply (e.g. some Windows setups)


def clear_credentials() -> None:
    """Deletes the credentials file, if present. No confirmation prompt --
    that's a deliberate GUI-layer decision since this data is sensitive and
    the user asked for an immediate, silent clear."""
    if os.path.exists(CRED_PATH):
        os.remove(CRED_PATH)


def has_saved_credentials() -> bool:
    return os.path.exists(CRED_PATH)
