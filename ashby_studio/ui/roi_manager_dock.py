"""Searchable ROI Manager (Phase 2).

Sortable table of annotations with text search, class/origin/review filters,
multi-select, selection centering, and audited bulk assign-review/delete via the
AnnotationService (never direct mutation).
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal, QItemSelectionModel
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QGridLayout, QHeaderView, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QLayout,
)

from ..domain.annotations import DEFAULT_LABEL, STATE_ACCEPTED, STATE_REJECTED
from ..measurements.engine import polygon_area_px2

REVIEW_STATES = ["manual", "suggested", "edited", "accepted", "rejected"]


class RoiManagerDock(QWidget):
    changed = Signal()
    booleanRequested = Signal(str)  # union | intersection | subtract

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self._win = win
        self.setObjectName("ashby_roi_manager")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(9)
        lay.setSizeConstraint(QLayout.SetMinimumSize)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search ROI names, IDs, notes…")
        self._search.textChanged.connect(self.refresh)
        lay.addWidget(self._search)

        title = QLabel("Filters")
        title.setStyleSheet("font-weight: 600; margin-top: 3px;")
        lay.addWidget(title)
        ff = QGridLayout()
        ff.setContentsMargins(0, 0, 0, 0)
        ff.setHorizontalSpacing(8)
        ff.setVerticalSpacing(6)
        ff.setColumnStretch(1, 1)
        self._class_combo = QComboBox()
        self._class_combo.addItem("All classes", None)
        self._origin_combo = QComboBox()
        self._origin_combo.addItem("All origins", None)
        self._origin_combo.addItem("Human", "human")
        self._origin_combo.addItem("AI proposal", "ai")
        self._review_combo = QComboBox()
        self._review_combo.addItem("All states", None)
        for st in REVIEW_STATES:
            self._review_combo.addItem(st, st)
        for row, (label, widget) in enumerate((("Class", self._class_combo),
                                               ("Origin", self._origin_combo),
                                               ("Review", self._review_combo))):
            ff.addWidget(QLabel(label), row, 0)
            ff.addWidget(widget, row, 1)
        lay.addLayout(ff)
        self._class_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self._origin_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self._review_combo.currentIndexChanged.connect(lambda _: self.refresh())

        self._count = QLabel("0 ROIs")
        lay.addWidget(self._count)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["id", "name", "class", "origin", "state", "area px²", "model run"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.setMinimumHeight(170)
        self._table.setMaximumHeight(390)
        self._table.setAlternatingRowColors(True)
        # Keep all columns in the data model, but show the ones relevant to a
        # narrow dock. Hidden details remain available in the row tooltip.
        for col in (0, 3, 5, 6):
            self._table.setColumnHidden(col, True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, 1)

        title = QLabel("Selected ROI actions")
        title.setStyleSheet("font-weight: 600; margin-top: 6px;")
        lay.addWidget(title)
        a = QHBoxLayout()
        a.setSpacing(7)
        self._bulk_review = QComboBox()
        self._bulk_review.addItem("Choose review state", None)
        for st in REVIEW_STATES:
            self._bulk_review.addItem(st, st)
        b1 = QPushButton("Apply")
        b1.setToolTip("Apply review to selected ROI(s)")
        b1.clicked.connect(self._bulk_apply_review)
        a.addWidget(self._bulk_review, 1)
        a.addWidget(b1)
        lay.addLayout(a)

        a2 = QHBoxLayout()
        a2.setSpacing(7)
        self._class_assign = QComboBox()
        self._class_assign.addItem("Choose class", None)
        b2 = QPushButton("Set class")
        b2.clicked.connect(self._bulk_assign_class)
        a2.addWidget(self._class_assign, 1)
        a2.addWidget(b2)
        lay.addLayout(a2)

        a3 = QHBoxLayout()
        a3.setSpacing(7)
        b3 = QPushButton("Center on ROI")
        b3.clicked.connect(self._center_selected)
        b4 = QPushButton("Delete selected")
        b4.setToolTip("Audited deletion of selected ROIs")
        b4.clicked.connect(self._bulk_delete)
        a3.addWidget(b3)
        a3.addWidget(b4)
        lay.addLayout(a3)

        title = QLabel("Boolean operations")
        title.setStyleSheet("font-weight: 600; margin-top: 7px;")
        lay.addWidget(title)
        bo = QHBoxLayout()
        bo.setSpacing(7)
        self._bool_op = QComboBox()
        self._bool_op.addItem("Union", "union")
        self._bool_op.addItem("Intersection", "intersection")
        self._bool_op.addItem("Subtract (first − others)", "subtract")
        apply_bo = QPushButton("Apply")
        apply_bo.clicked.connect(lambda: self.booleanRequested.emit(self._bool_op.currentData()))
        bo.addWidget(self._bool_op, 1)
        bo.addWidget(apply_bo)
        lay.addLayout(bo)
        lay.addStretch(1)
        self._sync_class_choices()

    def _sync_class_choices(self):
        """Offer pathologist-created and loaded case classes, not global presets."""
        current = self._class_combo.currentData()
        assigned = self._class_assign.currentData()
        labels = {DEFAULT_LABEL}
        labels.update(getattr(self._win, "_custom_classes", {}).keys())
        labels.update(a.label for a in getattr(self._win.service, "annotations", ()))
        labels = sorted(labels)
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("All classes", None)
        for label in labels:
            self._class_combo.addItem("Unclassified" if label == DEFAULT_LABEL else label.replace("_", " "), label)
        idx = self._class_combo.findData(current)
        self._class_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._class_combo.blockSignals(False)
        self._class_assign.blockSignals(True)
        self._class_assign.clear()
        self._class_assign.addItem("Choose class", None)
        for label in labels:
            self._class_assign.addItem("Unclassified" if label == DEFAULT_LABEL else label.replace("_", " "), label)
        idx = self._class_assign.findData(assigned)
        self._class_assign.setCurrentIndex(idx if idx >= 0 else 0)
        self._class_assign.blockSignals(False)

    # ---- selection ------------------------------------------------------------
    def _selected_ids(self):
        ids = []
        for idx in sorted({i.row() for i in self._table.selectedIndexes()}):
            item = self._table.item(idx, 0)
            if item is not None and item.text():
                ids.append(item.text())
        return ids

    def _visible_annotations(self):
        svc = self._win.service
        anns = list(svc.annotations or [])
        if self._class_combo.currentData():
            anns = [a for a in anns if a.label == self._class_combo.currentData()]
        if self._origin_combo.currentData():
            anns = [a for a in anns if a.origin == self._origin_combo.currentData()]
        if self._review_combo.currentData():
            anns = [a for a in anns if a.review_state == self._review_combo.currentData()]
        q = self._search.text().strip().lower()
        if q:
            anns = [
                a for a in anns
                if q in a.roi_id.lower() or q in (a.name or "").lower()
                   or q in a.label.lower() or q in (a.notes or "").lower()
            ]
        return anns

    # ---- display --------------------------------------------------------------
    def refresh(self):
        svc = self._win.service
        if svc is None:
            return
        self._sync_class_choices()
        anns = self._visible_annotations()
        selected = set(self._selected_ids())
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(anns))
        for ri, a in enumerate(anns):
            row = [
                a.roi_id, ("AI · " if a.origin == "ai" else "") + (a.name or a.roi_id),
                a.label, a.origin, a.review_state,
                f"{polygon_area_px2(a.points):.3g}", a.model_run_id or "",
            ]
            for ci, val in enumerate(row):
                it = QTableWidgetItem(str(val))
                it.setToolTip(
                    f"ID: {a.roi_id}\nName: {a.name or a.roi_id}\n"
                    f"Class: {a.label}\nOrigin: {a.origin}\nState: {a.review_state}\n"
                    f"Area: {polygon_area_px2(a.points):.3g} px²\nModel run: {a.model_run_id or '—'}"
                )
                self._table.setItem(ri, ci, it)
        self._table.setSortingEnabled(True)
        # Preserve selected rows when resorting or rebuilding data.
        for row in range(self._table.rowCount()):
            id_item = self._table.item(row, 0)
            if id_item is not None and id_item.text() in selected:
                self._table.selectionModel().select(
                    self._table.model().index(row, 1),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
        total = len(svc.annotations or [])
        self._count.setText(f"{len(anns)} shown of {total} ROIs")

    # ---- bulk actions (audited via service) ------------------------------------
    def _bulk_apply_review(self):
        st = self._bulk_review.currentData()
        ids = self._selected_ids()
        if not st or not ids:
            return
        for rid in ids:
            try:
                self._win.service.review(rid, st)
            except KeyError:
                continue
        self._after_mutation(f"Set review={st} on {len(ids)} ROI(s).")

    def _bulk_assign_class(self):
        lab = self._class_assign.currentData()
        ids = self._selected_ids()
        if not lab or not ids:
            return
        for rid in ids:
            try:
                self._win.service.update_polygon(rid, label=lab)
            except KeyError:
                continue
        self._after_mutation(f"Assigned class {lab} to {len(ids)} ROI(s).")

    def _bulk_delete(self):
        ids = self._selected_ids()
        if not ids:
            return
        for rid in ids:
            try:
                self._win.service.delete(rid)
            except KeyError:
                continue
        self._after_mutation(f"Deleted {len(ids)} ROI(s) (audited).")

    def _center_selected(self):
        ids = self._selected_ids()
        if len(ids) == 1:
            a = self._win.service.find(ids[0])
            if a is not None and a.points:
                xs = [p[0] for p in a.points]
                ys = [p[1] for p in a.points]
                self._win.viewer.camera.center = ((min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2)
                self._win.viewer.camera.zoom = 4

    def _after_mutation(self, msg):
        self._win._rebuild_all()
        self.changed.emit()
        self._win.statusBar().showMessage(msg, 6000)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()