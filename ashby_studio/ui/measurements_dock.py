"""Functional Measurements panel (Phase 1).

Live per-ROI geometry + candidate-mineralization summary with explicit
calibration, filters, a tissue-mask denominator and CSV export.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..domain.annotations import LABELS
from ..measurements.engine import (
    Scale,
    measure_rois,
    summarize_candidates,
)
from ..measurements.io import export_results_csv

REVIEW_STATES = ["manual", "suggested", "edited", "accepted", "rejected"]


class MeasurementDock(QWidget):
    tissueMaskRequested = Signal()
    refreshRequested = Signal()

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self._win = win
        self._tissue_used = True
        lay = QVBoxLayout(self)

        cal = QGroupBox("Calibration & denominator")
        cf = QFormLayout(cal)
        self._cal_edit = QLineEdit()
        self._cal_edit.setPlaceholderText("µm/pixel (leave empty = uncalibrated)")
        self._cal_edit.setToolTip("e.g. 0.25 means 0.25 µm per pixel. Empty means pixel units.")
        cf.addRow("µm/pixel", self._cal_edit)
        self._tissue_chk = QCheckBox("Restrict candidate area to tissue mask")
        self._tissue_chk.setChecked(True)
        cf.addRow("", self._tissue_chk)
        self._tissue_btn = QPushButton("Create auto tissue mask (Otsu)")
        self._tissue_btn.clicked.connect(self.tissueMaskRequested)
        cf.addRow("", self._tissue_btn)
        lay.addWidget(cal)

        filt = QGroupBox("Filters (results + summary)")
        ff = QFormLayout(filt)
        self._class_combo = QComboBox()
        self._class_combo.addItem("All classes", None)
        for label in sorted(LABELS):
            self._class_combo.addItem(label, label)
        ff.addRow("Class", self._class_combo)
        self._origin_combo = QComboBox()
        self._origin_combo.addItem("All origins", None)
        self._origin_combo.addItem("Human", "human")
        self._origin_combo.addItem("AI (suggested)", "ai")
        ff.addRow("Origin", self._origin_combo)
        self._review_combo = QComboBox()
        self._review_combo.addItem("All states", None)
        for st in REVIEW_STATES:
            self._review_combo.addItem(st, st)
        ff.addRow("Review", self._review_combo)
        lay.addWidget(filt)

        self._summary = QLabel("No measurements.")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        self._table = QTableWidget(0, 10)
        self._table.setHorizontalHeaderLabels(
            ["ROI id", "Name", "Class", "Origin", "State",
             "Area px²", "Area µm²", "Area mm²", "Perimeter px", "Perimeter µm"]
        )
        lay.addWidget(self._table, 1)

        row = QHBoxLayout()
        ref = QPushButton("Refresh")
        ref.clicked.connect(self.refresh)
        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.clicked.connect(self._export_csv)
        row.addWidget(ref)
        row.addWidget(self._export_btn)
        lay.addLayout(row)

        self._cal_edit.editingFinished.connect(self.refresh)
        self._tissue_chk.toggled.connect(lambda _: self.refresh())
        self._class_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self._origin_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self._review_combo.currentIndexChanged.connect(lambda _: self.refresh())

    # ---- state --------------------------------------------------------------
    def set_tissue_mask(self, mask):
        self._tissue_mask = mask
        self.refresh()

    def _scale(self) -> Scale:
        try:
            val = float(self._cal_edit.text().strip())
            return Scale(um_per_px=val)
        except ValueError:
            return Scale(um_per_px=None)

    def _filters(self) -> dict:
        return {
            "classes": self._class_combo.currentData(),
            "origins": self._origin_combo.currentData(),
            "review_states": self._review_combo.currentData(),
        }

    def _current_cell(self, text, *, editable=False):
        it = QTableWidgetItem(str(text))
        if not editable:
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        return it

    # ---- compute & display ---------------------------------------------------
    def refresh(self):
        svc = self._win.service
        asset = self._win.asset
        self._table.setRowCount(0)
        if svc is None or not svc.annotations:
            self._summary.setText("No annotations yet.")
            return
        filt = self._filters()
        anns = svc.annotations
        if filt["classes"]:
            anns = filter_annotations(anns, classes=[filt["classes"]])
        if filt["origins"]:
            anns = [a for a in anns if a.origin == filt["origins"]]
        if filt["review_states"]:
            anns = [a for a in anns if a.review_state == filt["review_states"]]

        scale = self._scale()
        rows = measure_rois(anns, scale)
        self._table.setRowCount(len(rows))
        for ri, r in enumerate(rows):
            vals = [r.roi_id, r.name, r.label, r.origin, r.review_state,
                    f"{r.area_px2:.3g}", f"{r.area_um2:.3g}" if r.area_um2 is not None else "—",
                    f"{r.area_mm2:.3g}" if r.area_mm2 is not None else "—",
                    f"{r.perimeter_px:.3g}", f"{r.perimeter_um:.3g}" if r.perimeter_um is not None else "—"]
            for ci, v in enumerate(vals):
                self._table.setItem(ri, ci, self._current_cell(v))

        # Summary candidates over candidate classes, honoring the same filters.
        if asset is not None:
            tissue = getattr(self, "_tissue_mask", None)
            tissue = tissue if (self._tissue_chk.isChecked() and tissue is not None) else None
            summary_anns = self._filtered_for_summary(svc, filt)
            summary = summarize_candidates(
                summary_anns,
                asset.data.shape,
                scale=scale,
                tissue_mask=tissue,
                filters=filt,
            )
            summary.image_sha256 = asset.sha256
            self._summary_summary = summary
            cal = f"calibrated {scale.um_per_px:.4g} µm/px" if scale.calibrated else "uncalibrated (pixels)"
            self._summary.setText(
                f"Candidate area {summary.candidate_area_px2:.3g} px²  /  denominator "
                f"{summary.denominator_desc} = {summary.denominator_area_px2:.3g} px²"
                f"  →  {summary.candidate_fraction_pct:.3f} %\n"
                f"Distinct objects: {summary.n_objects}  | overlaps deduped: {summary.overlaps_deduped}  | {cal}\n"
                f"image sha256 {asset.sha256[:12]}"
            )
        else:
            self._summary.setText("Measurements ready; open an image.")

    def _filtered_for_summary(self, svc, filt) -> list:
        anns = svc.annotations
        if filt["classes"]:
            anns = [a for a in anns if a.label == filt["classes"]]
        if filt["origins"]:
            anns = [a for a in anns if a.origin == filt["origins"]]
        if filt["review_states"]:
            anns = [a for a in anns if a.review_state == filt["review_states"]]
        return anns

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _export_csv(self):
        if self._win.asset is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements CSV", "", "CSV (*.csv)")
        if not path:
            return
        summary = getattr(self, "_summary_summary", None)
        export_results_csv(
            path,
            self._current_rows(),
            summary,
            case_id=self._win.service.case_id,
        )
        self._win.statusBar().showMessage(f"Exported measurements to {path}", 6000)

    def _current_rows(self):
        from ..measurements.engine import measure_rois
        scale = self._scale()
        filt = self._filters()
        anns = self._filtered_for_summary(self._win.service, filt)
        return measure_rois(anns, scale)