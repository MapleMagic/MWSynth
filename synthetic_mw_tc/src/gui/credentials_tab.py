"""
Credentials tab: username/password entry for the two real MW data
services actually in use -- NASA PPS (the near-real-time feed carrying
GMI-NRT / WSFM-NRT / AMSR3-NRT) and NASA Earthdata (the GES DISC archive
GMI fallback). See credentials.py for the storage behavior/rationale and
for why the JAXA G-Portal and NOAA CLASS entries were removed.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QMessageBox,
    QSpinBox,
)
from PyQt6.QtCore import QThread, pyqtSignal

import os
import json
from datetime import datetime, timezone

import credentials
import mw_ingest
import tcprimed_ingest


NRT_CACHE_PREFS_PATH = os.path.expanduser("~/.synthetic_mw_tc/nrt_cache_prefs.json")


class TCPrimedEstimateWorker(QThread):
    """Runs TC-PRIMED storm discovery + size estimation in the
    background -- both are pure S3 LIST operations (no actual file
    downloads), but a multi-year, multi-basin range can still mean many
    individual calls and take real time, so this shouldn't block the
    GUI thread."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, dict)  # (storms, estimate_result)
    failed = pyqtSignal(str)

    def __init__(self, start_year: int, end_year: int):
        super().__init__()
        self.start_year = start_year
        self.end_year = end_year

    def run(self):
        try:
            storms = tcprimed_ingest.list_available_storms(
                self.start_year, self.end_year, progress_callback=self.progress.emit,
            )
            estimate = tcprimed_ingest.estimate_download_size(storms, progress_callback=self.progress.emit)
            self.finished.emit(storms, estimate)
        except Exception as e:
            self.failed.emit(str(e))


class TCPrimedDownloadWorker(QThread):
    """Runs the actual TC-PRIMED bulk download in the background, given
    an already-discovered storms list (from TCPrimedEstimateWorker) --
    kept as a separate step/class from the estimate so the user has
    already seen and confirmed the size before any real download
    traffic starts."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, storms: list):
        super().__init__()
        self.storms = storms

    def run(self):
        try:
            result = tcprimed_ingest.download_storms(self.storms, progress_callback=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


# Ordered primary-first: PPS is the path that actually carries the
# current sensor rotation; Earthdata is the archive fallback.
SERVICE_LABELS = {
    "pps_nrt": "NASA PPS (near-real-time GMI / WSFM / AMSR3 — separate account from Earthdata)",
    "earthdata": "NASA Earthdata (GES DISC archive GMI — fallback, no 7-day retention limit)",
}

# PPS doesn't use a separate password -- your registered email IS both the
# username and password (see mw_ingest.py / credentials.py docstrings).
# Give it a single "email" field in the UI instead of two identical boxes.
SINGLE_FIELD_SERVICES = {"pps_nrt": "Registered PPS email (used as both username & password):"}


class CredentialsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._fields: dict[str, tuple[QLineEdit, QLineEdit]] = {}

        layout = QVBoxLayout(self)

        note = QLabel(
            "Used by the real-microwave-data ingestion module (mw_ingest.py). "
            "Stored as plaintext JSON on this machine at "
            "~/.synthetic_mw_tc/credentials.json -- only if you enable saving "
            "below. Otherwise these stay in memory for this session only. "
            "Note: Earthdata and PPS are separate NASA account systems, even "
            "though both relate to GPM data -- registering with one does not "
            "register you with the other."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        for service, label in SERVICE_LABELS.items():
            box = QGroupBox(label)
            form = QFormLayout(box)
            if service in SINGLE_FIELD_SERVICES:
                email_edit = QLineEdit()
                form.addRow(SINGLE_FIELD_SERVICES[service], email_edit)
                self._fields[service] = (email_edit, email_edit)  # same widget for both
            else:
                user_edit = QLineEdit()
                pass_edit = QLineEdit()
                pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
                form.addRow("Username:", user_edit)
                form.addRow("Password:", pass_edit)
                self._fields[service] = (user_edit, pass_edit)
            layout.addWidget(box)

        self.persist_checkbox = QCheckBox("Save credentials to disk (plaintext JSON) — off by default")
        self.persist_checkbox.setChecked(credentials.has_saved_credentials())
        layout.addWidget(self.persist_checkbox)

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.on_save)
        self.clear_btn = QPushButton("Clear saved credentials")
        self.clear_btn.clicked.connect(self.on_clear)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        nrt_box = QGroupBox("NRT download cache")
        nrt_layout = QVBoxLayout(nrt_box)
        nrt_note = QLabel(
            "Real MW passes (GMI/SSMIS/WSFM/AMSR3) are downloaded to "
            "~/.synthetic_mw_tc/NRT/ and kept for reuse. SSMIS/AMSR2/AMSR3/WSFM "
            "publish long swaths (often 90+ minutes of orbit per file), so this "
            "can add up in storage over time -- this clears it manually; nothing "
            "is deleted automatically."
        )
        nrt_note.setWordWrap(True)
        nrt_layout.addWidget(nrt_note)
        self.clear_nrt_btn = QPushButton("Clear NRT download cache...")
        self.clear_nrt_btn.clicked.connect(self.on_clear_nrt_cache)
        nrt_layout.addWidget(self.clear_nrt_btn)
        self.nrt_status_label = QLabel("")
        nrt_layout.addWidget(self.nrt_status_label)
        layout.addWidget(nrt_box)

        tcprimed_box = QGroupBox("TC-PRIMED training data (for ML correction model)")
        tcprimed_layout = QVBoxLayout(tcprimed_box)
        tcprimed_note = QLabel(
            "Downloads pre-validated GMI/AMSR2 real-MW passes from TC-PRIMED's "
            "public S3 bucket (no credentials needed) into the training-data "
            "folder, for training the ML correction model -- see ml_train.py. "
            "This is a genuinely large public dataset; nothing downloads until "
            "you explicitly start it, and you'll see a real size estimate first."
        )
        tcprimed_note.setWordWrap(True)
        tcprimed_layout.addWidget(tcprimed_note)

        year_row = QHBoxLayout()
        year_row.addWidget(QLabel("Download storms from year:"))
        self.tcprimed_year_spin = QSpinBox()
        self.tcprimed_year_spin.setRange(1987, datetime.now(timezone.utc).year)
        self.tcprimed_year_spin.setValue(max(1987, datetime.now(timezone.utc).year - 3))
        self.tcprimed_year_spin.setToolTip(
            "TC-PRIMED covers 1987-present, but GMI only exists from 2014 and "
            "AMSR2 from 2012 -- picking an earlier year won't find older data for "
            "either, just spends extra time checking seasons that can't have any."
        )
        year_row.addWidget(self.tcprimed_year_spin)
        year_row.addWidget(QLabel("onward"))
        year_row.addStretch(1)
        tcprimed_layout.addLayout(year_row)

        button_row = QHBoxLayout()
        self.tcprimed_estimate_btn = QPushButton("Estimate download size...")
        self.tcprimed_estimate_btn.clicked.connect(self.on_estimate_tcprimed)
        button_row.addWidget(self.tcprimed_estimate_btn)
        self.tcprimed_download_btn = QPushButton("Start download...")
        self.tcprimed_download_btn.setEnabled(False)
        self.tcprimed_download_btn.setToolTip("Run 'Estimate download size' first -- the confirmation needs a real number to show you.")
        self.tcprimed_download_btn.clicked.connect(self.on_download_tcprimed)
        button_row.addWidget(self.tcprimed_download_btn)
        tcprimed_layout.addLayout(button_row)

        self.tcprimed_status_label = QLabel("")
        self.tcprimed_status_label.setWordWrap(True)
        tcprimed_layout.addWidget(self.tcprimed_status_label)
        layout.addWidget(tcprimed_box)

        layout.addStretch(1)

        self._load_existing()
        self._tcprimed_storms = None  # populated by a successful estimate; required before download can start
        self._tcprimed_estimate_worker = None
        self._tcprimed_download_worker = None

    def _load_existing(self):
        data = credentials.load_credentials()
        for service, (user_edit, pass_edit) in self._fields.items():
            entry = data.get(service, {})
            user_edit.setText(entry.get("username", ""))
            pass_edit.setText(entry.get("password", ""))

    def get_credentials(self) -> dict:
        """In-memory read, for mw_ingest.py to call regardless of whether
        persistence to disk is enabled."""
        return {
            service: {"username": u.text(), "password": p.text()}
            for service, (u, p) in self._fields.items()
        }

    def on_save(self):
        if self.persist_checkbox.isChecked():
            credentials.save_credentials(self.get_credentials())
            self.status_label.setText(f"Saved to {credentials.CRED_PATH}")
        else:
            self.status_label.setText("Kept in memory for this session only (not written to disk).")

    def on_clear(self):
        # No confirmation dialog, per design: this is sensitive account
        # info and the clear action should be immediate.
        credentials.clear_credentials()
        for user_edit, pass_edit in self._fields.values():
            user_edit.clear()
            pass_edit.clear()
        self.status_label.setText("Cleared.")

    def _load_skip_confirm_pref(self) -> bool:
        try:
            with open(NRT_CACHE_PREFS_PATH) as f:
                return json.load(f).get("skip_clear_confirm", False)
        except Exception:
            return False

    def _save_skip_confirm_pref(self, skip: bool):
        try:
            os.makedirs(os.path.dirname(NRT_CACHE_PREFS_PATH), exist_ok=True)
            with open(NRT_CACHE_PREFS_PATH, "w") as f:
                json.dump({"skip_clear_confirm": skip}, f)
        except Exception:
            pass  # non-critical -- worst case the confirmation just reappears next time

    def on_clear_nrt_cache(self):
        skip_confirm = self._load_skip_confirm_pref()

        if not skip_confirm:
            box = QMessageBox(self)
            box.setWindowTitle("Clear NRT download cache?")
            box.setText(
                "This permanently deletes all downloaded NRT swath files "
                "(GMI/SSMIS/WSFM/AMSR3) from ~/.synthetic_mw_tc/NRT/. "
                "They'll be re-downloaded automatically next time they're needed. "
                "This cannot be undone."
            )
            box.setIcon(QMessageBox.Icon.Warning)
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            dont_show_again = QCheckBox("Don't ask me again")
            box.setCheckBox(dont_show_again)

            result = box.exec()
            if dont_show_again.isChecked():
                self._save_skip_confirm_pref(True)
            if result != QMessageBox.StandardButton.Yes:
                self.nrt_status_label.setText("Cancelled -- nothing deleted.")
                return

        try:
            files_deleted, bytes_freed = mw_ingest.clear_nrt_cache()
            mb_freed = bytes_freed / (1024 * 1024)
            self.nrt_status_label.setText(f"Deleted {files_deleted} file(s), freed {mb_freed:.1f} MB.")
        except Exception as e:
            self.nrt_status_label.setText(f"Error clearing NRT cache: {e}")

    def on_estimate_tcprimed(self):
        start_year = self.tcprimed_year_spin.value()
        end_year = datetime.now(timezone.utc).year
        self.tcprimed_estimate_btn.setEnabled(False)
        self.tcprimed_download_btn.setEnabled(False)
        self.tcprimed_status_label.setText(f"Checking TC-PRIMED for {start_year}-{end_year}... this can take a little while for a wide year range.")

        self._tcprimed_estimate_worker = TCPrimedEstimateWorker(start_year, end_year)
        self._tcprimed_estimate_worker.progress.connect(self.tcprimed_status_label.setText)
        self._tcprimed_estimate_worker.finished.connect(self._on_tcprimed_estimate_finished)
        self._tcprimed_estimate_worker.failed.connect(self._on_tcprimed_estimate_failed)
        self._tcprimed_estimate_worker.start()

    def _on_tcprimed_estimate_finished(self, storms, estimate):
        self.tcprimed_estimate_btn.setEnabled(True)
        self._tcprimed_storms = storms
        size_str = tcprimed_ingest.format_bytes(estimate["total_bytes"])
        self.tcprimed_status_label.setText(
            f"Found {estimate['n_files']} GMI/AMSR2 file(s) across {estimate['n_storms_with_data']} "
            f"storm(s) ({len(storms)} storms checked total) -- estimated size: {size_str}. "
            "Click 'Start download' to actually download this."
        )
        self.tcprimed_download_btn.setEnabled(estimate["n_files"] > 0)

    def _on_tcprimed_estimate_failed(self, error_msg):
        self.tcprimed_estimate_btn.setEnabled(True)
        self.tcprimed_status_label.setText(f"Estimate failed: {error_msg}")

    def on_download_tcprimed(self):
        if not self._tcprimed_storms:
            return  # shouldn't happen -- button is disabled until an estimate exists

        # Re-run just the size lookup isn't necessary -- reuse the numbers
        # already shown from the estimate step, which is exactly the
        # number that will be downloaded (same storms list, same
        # instruments) if nothing on the bucket has changed since.
        box = QMessageBox(self)
        box.setWindowTitle("Download TC-PRIMED training data?")
        box.setText(
            f"This downloads GMI/AMSR2 overpass files for {len(self._tcprimed_storms)} storm(s) "
            "from TC-PRIMED's public S3 bucket into ~/.synthetic_mw_tc/tcprimed_cache/. "
            "See the estimate above for the expected size. This can take a long time "
            "depending on your connection and how wide a year range you picked. "
            "You can close this dialog and re-check the estimate with a narrower year "
            "range first if you'd rather not commit to this yet."
        )
        box.setIcon(QMessageBox.Icon.Warning)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        result = box.exec()
        if result != QMessageBox.StandardButton.Yes:
            self.tcprimed_status_label.setText("Cancelled -- nothing downloaded.")
            return

        self.tcprimed_download_btn.setEnabled(False)
        self.tcprimed_estimate_btn.setEnabled(False)
        self.tcprimed_status_label.setText("Starting download...")

        self._tcprimed_download_worker = TCPrimedDownloadWorker(self._tcprimed_storms)
        self._tcprimed_download_worker.progress.connect(self.tcprimed_status_label.setText)
        self._tcprimed_download_worker.finished.connect(self._on_tcprimed_download_finished)
        self._tcprimed_download_worker.failed.connect(self._on_tcprimed_download_failed)
        self._tcprimed_download_worker.start()

    def _on_tcprimed_download_finished(self, result):
        self.tcprimed_estimate_btn.setEnabled(True)
        self.tcprimed_download_btn.setEnabled(True)
        size_str = tcprimed_ingest.format_bytes(result["total_bytes"])
        self.tcprimed_status_label.setText(
            f"Done. Downloaded {result['n_files']} file(s) ({size_str}) from "
            f"{result['n_storms_with_data']} storm(s)."
        )

    def _on_tcprimed_download_failed(self, error_msg):
        self.tcprimed_estimate_btn.setEnabled(True)
        self.tcprimed_download_btn.setEnabled(True)
        self.tcprimed_status_label.setText(f"Download failed: {error_msg}")
