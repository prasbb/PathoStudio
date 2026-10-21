"""Qt dock widgets for Virchow's Suite (M0).

Layout per ARCHITECTURE: left = layers/modalities + model selector; right = ROI
detail + reviewer actions; bottom = history + measurements (M1 placeholder).
Panels not yet backed by real functionality are disabled with a reason to avoid
claiming unimplemented behavior.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..domain.annotations import LABELS, DEFAULT_LABEL
from ..models.registry import MODEL_NONE


class ModalityDock(QWidget):
    modalityChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Modality (reference image)"))
        self.combo = QComboBox()
        self.combo.addItem("H&E", "he")
        for key, name in (("polarized_light", "Polarized Light"), ("ihc", "IHC"), ("other", "Other")):
            self.combo.addItem(name, key)
            idx = self.combo.count() - 1
            self.combo.setItemData(idx, "Local image/registration not implemented (M2)", Qt.ToolTipRole)
        self.combo.setEnabled(False)
        self.combo.setToolTip("Only H&E is available in M0; other modalities need registration (M2).")
        self.combo.currentIndexChanged.connect(self._changed)
        lay.addWidget(self.combo)
        self._hint = QLabel("Other modalities require an image and verified registration (M2).")
        self._hint.setWordWrap(True)
        lay.addWidget(self._hint)
        lay.addStretch(1)

    def _changed(self, idx: int):
        self.modalityChanged.emit(self.combo.itemData(idx))

    def current(self) -> str:
        return self.combo.currentData()


class ModelDock(QWidget):
    modelChanged = Signal(str)
    runMockClicked = Signal()

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("AI model"))
        self.combo = QComboBox()
        self._entries: dict[int, str] = {}

        for i, meta in enumerate(registry.all_entries()):
            self.combo.addItem(meta.ui_name, meta.id)
            self._entries[i] = meta.reason
            disabled = meta.status == "disabled"
            self.combo.setItemData(i, meta.reason, Qt.ToolTipRole)
            self.combo.model().item(i).setEnabled(not disabled)

        self.combo.currentIndexChanged.connect(self._changed)
        layout.addWidget(self.combo)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self.run_mock = QPushButton("Run mock AI overlay (SIMULATED)")
        self.run_mock.clicked.connect(self.runMockClicked)
        layout.addWidget(self.run_mock)
        from .feature_controls import FeatureControls
        self.feature_controls = FeatureControls(self)
        layout.addWidget(self.feature_controls)
        layout.addStretch(1)
        self._update_status()

    def _changed(self, idx: int):
        self._update_status()
        self.modelChanged.emit(self.combo.currentData())

    def _update_status(self):
        idx = self.combo.currentIndex()
        reason = self._entries.get(idx, "")
        self._status.setText(reason or "Manual annotation only; no AI.")
        self.run_mock.setVisible(self.combo.currentData() == "mock")

    def current(self) -> str:
        return self.combo.currentData()

    def select_none(self):
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == MODEL_NONE:
                self.combo.setCurrentIndex(i)
                break


class AnnotationClassDock(QWidget):
    labelChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Annotation class (color)"))
        self.combo = QComboBox()
        self.combo.addItem("Unclassified", DEFAULT_LABEL)
        self.combo.setCurrentIndex(0)
        self.combo.currentIndexChanged.connect(lambda _: self.labelChanged.emit(self.combo.currentData()))
        lay.addWidget(self.combo)
        self._note = QLabel(
            "H&E can only flag *candidate* mineralized foci.\n"
            "Chemical identity (e.g. calcium oxalate) stays unconfirmed without corroborating evidence."
        )
        self._note.setWordWrap(True)
        lay.addWidget(self._note)
        lay.addStretch(1)

    def current(self) -> str:
        return self.combo.currentData()


class RoiDetailDock(QWidget):
    acceptClicked = Signal()
    rejectClicked = Signal()
    deleteClicked = Signal()
    commitEditClicked = Signal()
    saveNotesClicked = Signal()
    saveNameClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_roi_id: str | None = None
        self._notes_dirty = False
        form = QFormLayout(self)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)

        self.id_label = QLabel("—")
        self.bounds_label = QLabel("—")
        self.origin_label = QLabel("—")
        self.state_label = QLabel("—")
        self.model_label = QLabel("—")
        for value_label in (self.id_label, self.bounds_label, self.origin_label,
                            self.state_label, self.model_label):
            value_label.setWordWrap(True)
            value_label.setMinimumWidth(0)
            value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(200)
        self.save_name_btn = QPushButton("Save ROI name")
        self.save_name_btn.clicked.connect(self.saveNameClicked)
        self.name_edit.returnPressed.connect(self.saveNameClicked)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Pathologist notes…")
        self.notes_edit.setMinimumHeight(75)
        self.notes_edit.setMaximumHeight(140)
        self.notes_edit.textChanged.connect(self._notes_changed)
        self.save_notes_btn = QPushButton("Save notes")
        self.save_notes_btn.clicked.connect(self.saveNotesClicked)
        self.save_notes_btn.setEnabled(False)
        self.notes_status = QLabel("Saved in current case")
        self.notes_status.setWordWrap(True)

        form.addRow("ROI id", self.id_label)
        form.addRow("Bounds", self.bounds_label)
        form.addRow("Origin", self.origin_label)
        form.addRow("Review state", self.state_label)
        form.addRow("Model run", self.model_label)
        form.addRow("Display name", self.name_edit)
        form.addRow(self.save_name_btn)
        form.addRow("Notes", self.notes_edit)
        form.addRow(self.save_notes_btn)
        form.addRow(self.notes_status)

        self.commit_btn = QPushButton("Commit geometry edit")
        self.accept_btn = QPushButton("Accept AI suggestion")
        self.reject_btn = QPushButton("Reject AI suggestion")
        self.delete_btn = QPushButton("Delete ROI")
        self.review_hint = QLabel("Review decisions apply to AI suggestions. Rejected ROIs are retained.")
        self.review_hint.setWordWrap(True)
        row = QWidget()
        hl = QVBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.commit_btn)
        hl.addWidget(self.accept_btn)
        hl.addWidget(self.reject_btn)
        hl.addWidget(self.review_hint)
        hl.addWidget(self.delete_btn)
        form.addRow(row)

        self.commit_btn.clicked.connect(self.commitEditClicked)
        self.accept_btn.clicked.connect(self.acceptClicked)
        self.reject_btn.clicked.connect(self.rejectClicked)
        self.delete_btn.clicked.connect(self.deleteClicked)
        self.setEnabled(False)

    @property
    def notes_dirty(self) -> bool:
        return self._notes_dirty

    def _notes_changed(self):
        if self.current_roi_id is None:
            return
        self._notes_dirty = True
        self.save_notes_btn.setEnabled(True)
        self.notes_status.setText("Unsaved notes — save or switch ROI to commit")

    def mark_notes_saved(self):
        self._notes_dirty = False
        self.save_notes_btn.setEnabled(False)
        self.notes_status.setText("Saved in current case — use Save case as… for disk")

    def show_for(self, ann, actor_id: str | None):
        self.current_roi_id = ann.roi_id if ann is not None else None
        if ann is None:
            self.setEnabled(False)
            for label in (self.id_label, self.bounds_label, self.origin_label,
                          self.state_label, self.model_label):
                label.setText("—")
            self.name_edit.setText("")
            self.notes_edit.blockSignals(True)
            self.notes_edit.setPlainText("")
            self.notes_edit.blockSignals(False)
            self.mark_notes_saved()
            return

        self.setEnabled(True)
        self.id_label.setText(ann.roi_id)
        xs = [p[0] for p in ann.points] or [0]
        ys = [p[1] for p in ann.points] or [0]
        self.bounds_label.setText(f"x[{min(xs):.0f},{max(xs):.0f}] y[{min(ys):.0f},{max(ys):.0f}]")
        self.origin_label.setText("Human (pathologist)" if ann.origin == "human" else "AI proposal")
        self.state_label.setText(ann.review_state.upper())
        self.model_label.setText(ann.model_run_id or "—")
        self.name_edit.setText(ann.name or ann.roi_id)
        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(ann.notes)
        self.notes_edit.blockSignals(False)
        self.mark_notes_saved()
        is_ai = ann.origin == "ai"
        self.accept_btn.setEnabled(is_ai and ann.review_state != "accepted")
        self.reject_btn.setEnabled(is_ai and ann.review_state != "rejected")
        self.review_hint.setText(
            "AI suggestion — choose Accept or Reject. Decision will be audited."
            if is_ai else "Human annotation — AI Accept/Reject not applicable."
        )

    def notes_text(self) -> str:
        return self.notes_edit.toPlainText()


class HistoryDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Action history (audit journal)"))
        self.list = QListWidget()
        lay.addWidget(self.list)

    def set_events(self, events) -> None:
        self.list.clear()
        for ev in reversed(events):
            self.list.addItem(f"{ev.timestamp[:19]}  {ev.actor_type}:{ev.actor_id}  {ev.action}  {ev.target_id}")


class MeasurementsDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Measurements"))
        self.label = QLabel("Available in M1 (denominator-explicit percentage/count).")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)
