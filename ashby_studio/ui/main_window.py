"""Napari/PySide6 main window for Virchow's Suite M0.

Source of truth for annotations is the :class:`AnnotationService`; napari layers
are rebuilt from it (small M0 counts) so shape indices never go stale and AI
vs human provenance stays explicit on separate layers.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import tempfile
import zipfile

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QToolBar,
    QToolButton, QMenu, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLineEdit, QPushButton, QComboBox, QInputDialog,
    QColorDialog, QFrame, QScrollArea, QSizePolicy,
)

from ..annotations.service import AnnotationService
from ..domain.annotations import (
    LABELS,
    DEFAULT_LABEL,
    STATE_ACCEPTED,
    STATE_REJECTED,
)
from ..domain.images import ImageAsset, load_image
from ..models.registry import (
    MODEL_MOCK,
    MODEL_NONE,
    MODEL_CONCH,
    MODEL_MUSK,
    ModelRegistry,
    build_default_registry,
    MOCK_LABEL_PREFIX,
)
from ..storage.repository import CaseRepository, new_case_id
from .docks import (
    AnnotationClassDock,
    HistoryDock,
    ModalityDock,
    ModelDock,
    RoiDetailDock,
)
from .text_to_roi_dock import TextToRoiDock
from .layer_modes import labels_mode
from .measurements_dock import MeasurementDock
from .roi_manager_dock import RoiManagerDock
from .image_tools_dock import ImageToolsDock


def _rgba(color: tuple) -> list[float]:
    """Normalize RGB/RGBA colors for napari."""
    values = [float(c) for c in color]

    if len(values) not in (3, 4):
        raise ValueError("Expected RGB or RGBA color")

    rgb = values[:3]

    # Normalize all RGB channels consistently.
    if any(c > 1.0 for c in rgb):
        rgb = [c / 255.0 for c in rgb]

    alpha = values[3] if len(values) == 4 else 1.0

    if alpha > 1.0:
        alpha /= 255.0

    return [*rgb, alpha]

class MainWindow(QMainWindow):
    def __init__(
        self,
        napari_viewer,
        *,
        registry: ModelRegistry | None = None,
        service: AnnotationService | None = None,
        repository: CaseRepository | None = None,
    ):
        super().__init__()
        self.setWindowTitle("PathoSuite")
        self.viewer = napari_viewer
        self.setCentralWidget(self.viewer.window._qt_viewer)  # type: ignore[attr-defined]

        self.registry = registry if registry is not None else build_default_registry()
        self.service = service or AnnotationService(case_id=new_case_id())
        self.repository = repository  # set when a case dir is chosen

        self.asset: ImageAsset | None = None
        self._rebuilding = False
        self._pending_human_sync = False
        self._drawing_mode = "select"
        # Existing labels are kept solely to render older case files. A new
        # case offers no pathology labels until the pathologist creates them.
        self._legacy_class_colors = dict(LABELS)
        self._builtin_classes = {DEFAULT_LABEL: LABELS[DEFAULT_LABEL]}
        self._custom_classes: dict[str, list[float]] = {}
        self._case_filename: Path | None = None
        self._extracted_case: tempfile.TemporaryDirectory | None = None

        self._image_layer = None
        self._human_layer = self.viewer.add_shapes(name="human_rois · pathologist")
        self._ai_layer = self.viewer.add_shapes(name="ai_rois · SIMULATED suggestions")
        self._overlay = None

        # panels
        self.modality_dock = ModalityDock(self)
        self.model_dock = ModelDock(self.registry, self)
        self.class_dock = AnnotationClassDock(self)
        self.roi_dock = RoiDetailDock(self)
        self.history_dock = HistoryDock(self)
        self.measure_dock = MeasurementDock(self)
        self._roi_manager_dock = RoiManagerDock(self)
        self._image_tools_dock = ImageToolsDock(self)
        self._tissue_mask = None
        self._tissue_layer = None
        self._threshold_layer = None
        self._ttr_dock = TextToRoiDock(self)
        self._ttr_worker = None
        self._ttr_search_token = 0
        self._ttr_saved_keys: set = set()
        from .feature_controller import FeatureController
        self._feature_controller = FeatureController(self)

        # ImageJ/Fiji-inspired instrument layout: one canvas, compact toolbar,
        # an optional ROI Manager, and other tools opened on demand.
        self.setObjectName("ashby_main_window")
        self._dock_widgets: dict[str, QDockWidget] = {}
        self.setDockOptions(QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)
        self._build_toolbar()
        self._build_workspace()
        self._build_menus()
        self._apply_flat_style()
        # Avoid exceeding the useful desktop area on smaller Mac displays.
        self.resize(1200, 790)
        QTimer.singleShot(0, self._size_default_docks)

        self.statusBar().showMessage("Research-only prototype — not for diagnostic use.")
        self._connect()
        self._show_sample_image(force=True)

    _RIGHT_PANEL_KEYS = (
        "roi_manager", "roi", "model", "text_to_roi", "image_tools", "modality"
    )

    def _build_workspace(self):
        """One right-hand tool panel at a time; a large uninterrupted slide.

        No nested/tabified side docks: crowded QDockWidget tab bars and competing
        minimum sizes otherwise cause overlapping controls on laptop screens.
        """
        self._add_panel("roi_manager", "ROI Manager", self._roi_manager_dock,
                        Qt.RightDockWidgetArea)
        self._add_panel("roi", "ROI Properties", self.roi_dock,
                        Qt.RightDockWidgetArea, scroll=True)
        self._add_panel("model", "Local AI Model", self.model_dock,
                        Qt.RightDockWidgetArea, scroll=True)
        self._add_panel("text_to_roi", "Find ROIs from Text", self._ttr_dock,
                        Qt.RightDockWidgetArea, scroll=True)
        self._add_panel("image_tools", "Image Tools", self._image_tools_dock,
                        Qt.RightDockWidgetArea, scroll=True)
        self._add_panel("modality", "Image Modality", self.modality_dock,
                        Qt.RightDockWidgetArea, scroll=True)

        results = self._add_panel("measurements", "Results — Measurements",
                                  self.measure_dock, Qt.BottomDockWidgetArea)
        history = self._add_panel("history", "Audit History",
                                  self.history_dock, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(results, history)
        self._refresh_classes()
        for key, dock in self._dock_widgets.items():
            if key != "roi_manager":
                dock.hide()
        self._show_panel("roi_manager")

    def _add_panel(self, key: str, title: str, content: QWidget, area,
                   *, scroll: bool = False):
        dock = QDockWidget(title, self)
        dock.setObjectName("ashby_" + key + "_dock")
        dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        # Use the platform UI font, not whatever font the embedded napari viewer
        # happens to inherit. Apply only to tool panels, never to the slide.
        font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
        font.setPointSize(11)
        dock.setFont(font)
        content.setFont(font)
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        if scroll:
            viewport = QScrollArea(dock)
            viewport.setObjectName("ashby_" + key + "_scroll")
            viewport.setWidgetResizable(True)
            viewport.setFrameShape(QFrame.NoFrame)
            viewport.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            viewport.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            viewport.setWidget(content)
            dock.setWidget(viewport)
        else:
            dock.setWidget(content)
        self.addDockWidget(area, dock)
        self._dock_widgets[key] = dock
        return dock

    def _hide_right_panel(self):
        for key in self._RIGHT_PANEL_KEYS:
            self._dock_widgets[key].hide()

    def _size_default_docks(self):
        right = next((self._dock_widgets[k] for k in self._RIGHT_PANEL_KEYS
                      if self._dock_widgets[k].isVisible()), None)
        if right is not None and not right.isFloating():
            # Give labels and edit fields adequate width while retaining canvas.
            self.resizeDocks([right], [370], Qt.Horizontal)

    def _show_panel(self, key: str):
        dock = self._dock_widgets.get(key)
        if dock is None:
            return
        # Side tools are alternatives, not a stack of competing windows.
        if key in self._RIGHT_PANEL_KEYS:
            for other_key in self._RIGHT_PANEL_KEYS:
                if other_key != key:
                    self._dock_widgets[other_key].hide()
            if dock.isFloating():
                dock.setFloating(False)
        dock.show()
        dock.raise_()
        if key in self._RIGHT_PANEL_KEYS:
            QTimer.singleShot(0, self._size_default_docks)

    def _apply_flat_style(self):
        """Keep native Qt controls but enforce square corners, no cards."""
        self.setStyleSheet("""
            QMainWindow#ashby_main_window QDockWidget,
            QMainWindow#ashby_main_window QToolBar,
            QMainWindow#ashby_main_window QToolButton,
            QMainWindow#ashby_main_window QPushButton,
            QMainWindow#ashby_main_window QLineEdit,
            QMainWindow#ashby_main_window QComboBox,
            QMainWindow#ashby_main_window QPlainTextEdit,
            QMainWindow#ashby_main_window QTextEdit,
            QMainWindow#ashby_main_window QTabWidget::pane,
            QMainWindow#ashby_main_window QTabBar::tab,
            QMainWindow#ashby_main_window QGroupBox,
            QMainWindow#ashby_main_window QMenu,
            QMainWindow#ashby_main_window QMenuBar,
            QMainWindow#ashby_main_window QFrame,
            QMainWindow#ashby_main_window QTableView,
            QMainWindow#ashby_main_window QListView,
            QMainWindow#ashby_main_window QSlider {
                border-radius: 0px;
            }
            QMainWindow#ashby_main_window QToolBar {
                spacing: 3px;
                padding: 2px;
            }
            QMainWindow#ashby_main_window QPushButton,
            QMainWindow#ashby_main_window QToolButton,
            QMainWindow#ashby_main_window QComboBox,
            QMainWindow#ashby_main_window QLineEdit,
            QMainWindow#ashby_main_window QPlainTextEdit {
                border: 1px solid palette(mid);
                border-radius: 0px;
                padding: 3px;
            }
            QMainWindow#ashby_main_window QToolBar QToolButton:hover,
            QMainWindow#ashby_main_window QPushButton:hover {
                background: palette(light);
            }
            QMainWindow#ashby_main_window QTabBar::tab {
                border-radius: 0px;
                padding: 4px 6px;
            }
            QMainWindow#ashby_main_window QDockWidget::title {
                padding: 3px 5px;
                text-align: left;
            }
            QMainWindow#ashby_main_window QSlider::groove:horizontal {
                height: 5px;
                background: palette(mid);
                border: 0px;
            }
            QMainWindow#ashby_main_window QSlider::sub-page:horizontal {
                background: #7d858c;
            }
            QMainWindow#ashby_main_window QSlider::handle:horizontal {
                background: palette(button);
                border: 1px solid palette(mid);
                border-radius: 0px;
                width: 12px;
                margin: -5px 0px;
            }
        """)

    def _build_menus(self):
        """Fiji/ImageJ-style command taxonomy with no fake menu actions."""
        bar = self.menuBar()
        bar.setNativeMenuBar(False)  # Menu stays discoverable in the application.
        bar.clear()

        file_menu = bar.addMenu("&File")
        file_menu.addAction("Open Image…", self.open_image)
        file_menu.addAction("Open Case…", self.reopen_case)
        file_menu.addAction("Save Case As…", self.save_case)
        file_menu.addSeparator()
        file_menu.addAction("Measurements / Export CSV…", lambda: self._show_panel("measurements"))

        edit_menu = bar.addMenu("&Edit")
        edit_menu.addAction("Commit ROI Geometry Edit", self._commit_edits)
        edit_menu.addAction("Delete Selected ROI", self._delete_selected)

        image_menu = bar.addMenu("&Image")
        image_menu.addAction("Brightness / Contrast / Gamma…", lambda: self._show_panel("image_tools"))
        image_menu.addAction("H&E Color Deconvolution…", lambda: self._show_panel("image_tools"))
        image_menu.addAction("Image Modality…", lambda: self._show_panel("modality"))
        image_menu.addAction("Scale Calibration…", lambda: self._show_panel("measurements"))

        process_menu = bar.addMenu("&Process")
        process_menu.addAction("Threshold and Tissue Masks…", lambda: self._show_panel("image_tools"))
        process_menu.addAction("Threshold to ROI…", lambda: self._show_panel("image_tools"))

        analyze_menu = bar.addMenu("&Analyze")
        analyze_menu.addAction("Measurements / Results…", lambda: self._show_panel("measurements"))
        analyze_menu.addAction("Point Counter…", lambda: self._show_panel("image_tools"))
        analyze_menu.addAction("Audit History…", lambda: self._show_panel("history"))

        roi_menu = bar.addMenu("&ROI")
        roi_menu.addAction("ROI Manager…", lambda: self._show_panel("roi_manager"))
        roi_menu.addAction("ROI Properties / Notes…", lambda: self._show_panel("roi"))
        roi_menu.addSeparator()
        roi_menu.addAction("Navigate", self._set_navigate_mode)
        roi_menu.addAction("Polygon", lambda: self.set_annotation_mode("polygon"))
        roi_menu.addAction("Freehand", lambda: self.set_annotation_mode("freehand"))
        roi_menu.addAction("Select / Edit", lambda: self.set_annotation_mode("select"))
        roi_menu.addSeparator()
        roi_menu.addAction("Boolean Operations…", lambda: self._show_panel("roi_manager"))
        roi_menu.addAction("Add Annotation Class…", self.add_custom_class)

        ai_menu = bar.addMenu("&AI")
        ai_menu.addAction("Select Local Model…", lambda: self._show_panel("model"))
        ai_menu.addAction("Manage Local Models…", self._manage_local_models)
        ai_menu.addAction("Describe Selected ROI…", self.describe_selected_roi)
        ai_menu.addAction("Find ROIs from Text…", self._open_text_to_roi)

        window_menu = bar.addMenu("&Window")
        self._window_menu = window_menu
        for key in ("roi_manager", "roi", "measurements", "image_tools",
                    "text_to_roi", "model", "history", "modality"):
            title = self._dock_widgets[key].windowTitle()
            window_menu.addAction(title, lambda checked=False, k=key: self._show_panel(k))
        window_menu.addSeparator()
        window_menu.addAction("Hide Right Panel", self._hide_right_panel)
        window_menu.addAction("Reset to Default Layout", self.restore_sidebars)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction("About Ashby Studio", lambda: QMessageBox.information(
            self, "About Virchow's Suite",
            "Virchow's Suite — research only, not for clinical diagnosis.\n"
            "All image processing and inference use local resources.",
        ))

    def _manage_local_models(self):
        """Configure individually licensed local checkpoints without bundling them."""
        if self._ttr_worker is not None and self._ttr_worker.isRunning():
            QMessageBox.information(
                self, "Model search active",
                "Wait for the current Text→ROI search to finish before changing models."
            )
            return
        from .local_models_dialog import LocalModelsDialog
        dialog = LocalModelsDialog(self)
        dialog.exec_()
        if not dialog.changed:
            return
        self._invalidate_ttr()
        self._refresh_model_registry()

    def _refresh_model_registry(self):
        """Recreate the existing model selector against refreshed availability."""
        old_selection = self.model_dock.current()
        self.registry = build_default_registry()
        new_widget = ModelDock(self.registry, self)
        new_widget.modelChanged.connect(self._on_model_changed)
        new_widget.runMockClicked.connect(self.run_mock_ai)
        container = self._dock_widgets["model"].widget()
        if isinstance(container, QScrollArea):
            container.setWidget(new_widget)
        else:
            self._dock_widgets["model"].setWidget(new_widget)
        self.model_dock = new_widget
        for index in range(new_widget.combo.count()):
            if new_widget.combo.itemData(index) == old_selection and \
                    self.registry.adapter(old_selection) is not None:
                new_widget.combo.setCurrentIndex(index)
                break
        self._on_model_changed(new_widget.current())
        self.statusBar().showMessage("Local model configuration updated.", 6000)

    def _refresh_classes(self):
        if not hasattr(self, "_class_select"):
            return
        old = self._class_select.currentData()
        self._class_select.blockSignals(True)
        self._class_select.clear()
        self._class_select.addItem("Unclassified", DEFAULT_LABEL)
        existing_case_labels = {
            a.label for a in self.service.annotations if a.label != DEFAULT_LABEL
        }
        # New cases show only user-created classes. Existing cases can still
        # display/reuse their historical labels without losing data.
        options = sorted(set(self._custom_classes) | existing_case_labels)
        for name in options:
            self._class_select.addItem(name.replace("_", " "), name)
        idx = self._class_select.findData(old)
        self._class_select.setCurrentIndex(idx if idx >= 0 else 0)
        self._class_select.blockSignals(False)

    def current_class(self) -> str:
        return self._class_select.currentData() or DEFAULT_LABEL

    def add_custom_class(self):
        name, ok = QInputDialog.getText(self, "New annotation class", "Class name:")
        if not ok:
            return
        key = re.sub(r"\s+", "_", name.strip().lower())
        if not key or not re.fullmatch(r"[a-z0-9_-]{1,60}", key):
            QMessageBox.warning(self, "Invalid class", "Use letters, numbers, spaces, _ or - (maximum 60 characters).")
            return
        if key == DEFAULT_LABEL:
            QMessageBox.information(self, "Reserved name", "Unclassified is the default state; choose a distinct class name.")
            return
        if key in self._custom_classes:
            self._class_select.setCurrentIndex(self._class_select.findData(key))
            return
        color = QColorDialog.getColor(parent=self, title="Choose annotation color")
        if not color.isValid():
            return
        rgba = list(color.getRgbF())
        LABELS[key] = tuple(rgba)
        self._custom_classes[key] = rgba
        self._refresh_classes()
        self._class_select.setCurrentIndex(self._class_select.findData(key))
        self.statusBar().showMessage(
            f"Class '{key}' added; it will be stored in the case archive on save.",
            7000,
        )

    def _build_toolbar(self):
        """Small ImageJ-style command row; analysis tools stay in menus."""
        toolbar = QToolBar("Tools", self)
        toolbar.setObjectName("ashby_tools_toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        toolbar.addAction("Open", self.open_image).setToolTip("Open H&E image")
        toolbar.addAction("Save", self.save_case).setToolTip("Save case archive")
        toolbar.addSeparator()
        toolbar.addAction("Pan", self._set_navigate_mode).setToolTip("Pan / zoom")
        toolbar.addAction("Select", lambda: self.set_annotation_mode("select")).setToolTip("Select or edit ROI")
        toolbar.addAction("Polygon", lambda: self.set_annotation_mode("polygon")).setToolTip("Draw polygon ROI")
        toolbar.addAction("Freehand", lambda: self.set_annotation_mode("freehand")).setToolTip("Draw freehand ROI")
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Class: "))
        self._class_select = QComboBox(self)
        self._class_select.setObjectName("ashby_annotation_class_selector")
        self._class_select.setMinimumWidth(135)
        self._class_select.setMaximumWidth(190)
        toolbar.addWidget(self._class_select)
        toolbar.addAction("+ Class", self.add_custom_class).setToolTip("Create annotation class")
        toolbar.addSeparator()
        toolbar.addAction("ROI Manager", lambda: self._show_panel("roi_manager"))
        toolbar.addAction("Properties", lambda: self._show_panel("roi"))
        toolbar.addAction("Measure", lambda: self._show_panel("measurements"))
        toolbar.addAction("AI Search", self._open_text_to_roi)

        self.mode = QLabel("Navigate")
        self.mode.setObjectName("ashby_active_tool")
        self.mode.setMinimumWidth(92)
        self._image_status = QLabel("No image")
        self.statusBar().addPermanentWidget(self._image_status)
        self.statusBar().addPermanentWidget(self.mode)

    def restore_sidebars(self):
        """Restore the default one-panel Fiji workspace."""
        for key, dock in self._dock_widgets.items():
            dock.hide()
            if dock.isFloating():
                dock.setFloating(False)
        self._show_panel("roi_manager")

    # ---- setup/restore ------------------------------------------------------
    def _connect(self):
        self._human_layer.events.data.connect(self._on_human_data)
        for layer in (self._human_layer, self._ai_layer):
            layer.selected_data.events.items_changed.connect(self._on_selection_change)
            layer.mode = "select"

        # Default to navigation/selection; the toolbar explicitly activates drawing.
        self.human_layer_mode(False)
        self.model_dock.modelChanged.connect(self._on_model_changed)
        self.model_dock.runMockClicked.connect(self.run_mock_ai)
        self._class_select.currentIndexChanged.connect(
            lambda _: self._on_current_label_changed(self.current_class())
        )
        self.roi_dock.acceptClicked.connect(lambda: self._review(STATE_ACCEPTED))
        self.roi_dock.rejectClicked.connect(lambda: self._review(STATE_REJECTED))
        self.roi_dock.deleteClicked.connect(self._delete_selected)
        self.roi_dock.commitEditClicked.connect(self._commit_edits)
        self.roi_dock.saveNotesClicked.connect(self.save_selected_roi_notes)
        self.roi_dock.saveNameClicked.connect(self.rename_selected_roi)

        self._ttr_dock.searchRequested.connect(self._ttr_run_search)
        self._ttr_dock.thresholdChanged.connect(lambda _: self._ttr_apply_threshold())
        self._ttr_dock.clearRequested.connect(self._ttr_clear)
        self._ttr_dock.saveRequested.connect(self._ttr_save_candidates)

        self.measure_dock.tissueMaskRequested.connect(self._create_tissue_mask)
        self._image_tools_dock.thresholdMaskRequested.connect(self._update_threshold_mask)
        self._image_tools_dock.maskCleared.connect(self._clear_threshold_mask)
        self._image_tools_dock.hedShowRequested.connect(self._show_hed)
        self._image_tools_dock.maskEditRequested.connect(self._edit_tissue_mode)
        self._image_tools_dock.maskCommitRequested.connect(self._commit_tissue_edit)
        self._image_tools_dock.convertMaskRequested.connect(self._convert_threshold_to_rois)
        self._image_tools_dock.pointsNewRequested.connect(self._points_new)
        self._image_tools_dock.pointsClearRequested.connect(self._points_clear)
        self._image_tools_dock.pointsCommitRequested.connect(self._points_commit)
        self._roi_manager_dock.booleanRequested.connect(self._roi_boolean)

    def set_annotation_mode(self, mode: str):
        """Switch the manual Shapes layer between polygon, lasso, selection and navigation."""
        if mode == "navigate":
            self._set_navigate_mode()
            return
        napari_modes = {
            "polygon": "add_polygon",
            "freehand": "add_polygon_lasso",
            "select": "select",
        }
        if mode not in napari_modes:
            raise ValueError(f"Unknown annotation mode: {mode}")
        self._human_layer.visible = True
        self.viewer.layers.selection.active = self._human_layer
        self._human_layer.mode = napari_modes[mode]
        self._ensure_layer_order()
        self._drawing_mode = mode
        self.mode.setText({
            "polygon": "Polygon drawing",
            "freehand": "Freehand drawing",
            "select": "Select / edit",
        }[mode])
        tips = {
            "polygon": "Click vertices, then double-click to complete the ROI.",
            "freehand": "Click, move the mouse to trace, then click again to finish. Tablet: drag stylus.",
            "select": "Select an ROI, edit vertices, then use Commit geometry edit.",
        }
        self.statusBar().showMessage(tips[mode], 10000)

    def _set_navigate_mode(self):
        """Restore pan/zoom navigation across all editable layers.

        Switch every mode-capable layer to napari's ``pan_zoom`` and activate
        the read-only image layer so the mouse wheel/drag controls the camera
        instead of being captured by a drawing or selection tool.
        """
        for layer in list(self.viewer.layers):
            if getattr(layer, "mode", None) is None:
                continue
            try:
                layer.mode = "pan_zoom"
            except ValueError:
                continue
        if getattr(self, "_human_layer", None) is not None:
            self._human_layer.visible = True
        target = self._image_layer or (self.viewer.layers[0] if self.viewer.layers else None)
        if target is not None:
            self.viewer.layers.selection.active = target
        self._drawing_mode = "navigate"
        self.mode.setText("Pan / zoom")
        self.statusBar().showMessage(
            "Navigation: drag to pan, scroll to zoom. Choose Annotate ▸ Polygon/Freehand to draw.",
            8000,
        )

    def human_layer_mode(self, enabled: bool):
        """Compatibility with existing callers. Disabling returns to navigation."""
        self.set_annotation_mode("polygon" if enabled else "navigate")

    def rename_selected_roi(self):
        """Request an audited rename; never mutate Annotation.name directly."""
        selected = self._selected_annotations()
        if len(selected) != 1:
            QMessageBox.information(self, "Select ROI", "Select exactly one ROI to rename.")
            return
        name = self.roi_dock.name_edit.text().strip()
        if not name or len(name) > 200:
            QMessageBox.warning(self, "Invalid name", "Enter an ROI name of 1–200 characters.")
            return
        rename = getattr(self.service, "rename_annotation", None)
        if not callable(rename):
            QMessageBox.warning(
                self, "Rename unavailable",
                "Your AnnotationService needs an audited rename_annotation(roi_id, name) "
                "method. The ROI has NOT been changed. Please add that service API "
                "before enabling persistent naming.",
            )
            return
        try:
            if not self._save_pending_roi_notes():
                return
            rename(selected[0].roi_id, name)
            self._rebuild_all()
            self.statusBar().showMessage("ROI name saved; save case to persist on disk.", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Rename failed", str(exc))

    def _save_pending_roi_notes(self) -> bool:
        """Commit the previous ROI's text before selection or layers change."""
        roi_id = self.roi_dock.current_roi_id
        if not roi_id or not self.roi_dock.notes_dirty:
            return True
        try:
            self.service.update_notes(roi_id, self.roi_dock.notes_text())
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Notes not saved", str(exc))
            return False
        self.roi_dock.mark_notes_saved()
        self.history_dock.set_events(self.service.events)
        return True

    def save_selected_roi_notes(self):
        if self._save_pending_roi_notes():
            self.statusBar().showMessage(
                "ROI notes saved in current case; use Save case as… to persist on disk.",
                6000,
            )

    # ---- image ----------------------------------------------------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open H&E image", "", "Images (*.png *.tif *.tiff)"
        )
        if not path:
            return
        try:
            asset = load_image(Path(path), modality=self.modality_dock.current())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open image", str(exc))
            return
        self._set_asset(asset)

    def open_path(self, path: Path) -> ImageAsset | None:
        """Public image-loading API used by the ``--image`` CLI argument."""
        try:
            asset = load_image(path, modality=self.modality_dock.current())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open image", str(exc))
            return None
        self._set_asset(asset)
        return asset

    def _set_asset(self, asset: ImageAsset, *, reset_annotations: bool = True):
        """Attach an image, optionally preserving annotations restored from a case."""
        if not self._save_pending_roi_notes():
            return
        self._feature_controller.invalidate(clear_runs=True)
        self.model_dock.feature_controls.set_runs([])
        self.asset = asset
        if self._image_layer is None:
            self._image_layer = self.viewer.add_image(
                asset.data, name="H&E (read-only)", rgb=True,
                metadata={"asset_id": asset.asset_id},
            )
        else:
            self._image_layer.data = asset.data
            self._image_layer.name = "H&E (read-only)"
            self._image_layer.metadata = {"asset_id": asset.asset_id}
        if reset_annotations:
            self._invalidate_ttr()
            self._hed = None
            self._clear_mask_layers()
            self._reset_classes()
            self._reset_state()
        else:
            self._rebuild_all()
        self.viewer.reset_view()
        self.human_layer_mode(False)
        self._image_status.setText(f"{asset.width} × {asset.height} px")
        QTimer.singleShot(0, self._ensure_layer_order)
        QTimer.singleShot(0, lambda: self._image_tools_dock.refresh_histogram())

    def _show_sample_image(self, force: bool = False):
        from ..imaging.generator import make_synthetic_he_image

        if self.asset is not None and not force:
            return
        self._feature_controller.invalidate(clear_runs=True)
        data, _ = make_synthetic_he_image()
        self.asset = ImageAsset(asset_id="demo-synthetic", path=Path("SYNTHETIC DEMO"), modality="he", data=data)
        if self._image_layer is None:
            self._image_layer = self.viewer.add_image(data, name="Synthetic H&E (demo)", rgb=True, metadata={"asset_id": "demo-synthetic"})
        else:
            self._image_layer.data = data
            self._image_layer.name = "Synthetic H&E (demo)"
        self._invalidate_ttr()
        self._reset_classes()
        self._reset_state()
        self.viewer.reset_view()
        self.human_layer_mode(False)
        self._image_status.setText(f"{self.asset.width} × {self.asset.height} px · demo")
        QTimer.singleShot(0, self._ensure_layer_order)
        QTimer.singleShot(0, lambda: self._image_tools_dock.refresh_histogram())

    def _reset_classes(self):
        for key in list(self._custom_classes):
            if key in self._legacy_class_colors:
                LABELS[key] = self._legacy_class_colors[key]
            else:
                LABELS.pop(key, None)
        self._custom_classes.clear()
        self._refresh_classes()

    def _reset_state(self):
        # When switching images, start a new case instead of silently retaining
        # history from the previous image. Preserve an injected, empty service on
        # first startup so tests and callers retain their service reference.
        if self.service.annotations or self.service.events:
            self.service = AnnotationService(case_id=new_case_id())
        self.repository = None
        self._case_filename = None
        self._refresh_classes()
        self._rebuild_all()

    # ---- shapes sync ---------------------------------------------------------
    def _rebuild_all(self):
        if not self._save_pending_roi_notes():
            return
        # Napari clears Shapes.selected_data when layer.data is replaced.
        # Remember stable ROI IDs, then reselect after the rebuild.
        selected_ids = [
            (layer, {
                self._row_property(layer.properties, idx, "roi_id")
                for idx in list(layer.selected_data)
            })
            for layer in (self._human_layer, self._ai_layer)
        ]
        self._rebuilding = True
        try:
            self._rebuild_layer(self._human_layer, self.service.all_human())
            self._rebuild_layer(self._ai_layer, self.service.all_ai())
            for layer, ids in selected_ids:
                props = getattr(layer, "properties", {})
                selected = {
                    idx for idx in range(len(layer.data))
                    if self._row_property(props, idx, "roi_id") in ids
                }
                layer.selected_data = selected
            self._rebuild_overlay()
            self._ensure_layer_order()
            self.history_dock.set_events(self.service.events)
            self._update_selection_detail()
            self.measure_dock.refresh()
            self._roi_manager_dock.refresh()
        finally:
            self._rebuilding = False

    def _rebuild_layer(self, layer, annotations):
        if not annotations:
            layer.data = []
            layer.properties = {}
            return
        # Preserve true polygon shape types when rebuilding from the domain model.
        # napari also accepts (vertices, shape_type) records in its data setter.
        polys = [(np.asarray(a.as_napari_shape(), dtype=float), "polygon") for a in annotations]
        layer.data = polys
        layer.properties = {
            "roi_id": [a.roi_id for a in annotations],
            "origin": [a.origin for a in annotations],
            "review_state": [a.review_state for a in annotations],
            "annotation_name": [
                f"{'SIMULATED' if a.model_run_id and a.model_run_id.startswith('mock-') else 'AI' if a.origin == 'ai' else 'H'}: "
                f"{a.name or a.label}"
                for a in annotations
            ],
            "label": [a.label for a in annotations],
        }
        face_colors, edge_colors = [], []
        for ann in annotations:
            rgba = _rgba(LABELS.get(ann.label, LABELS[DEFAULT_LABEL]))
            if ann.origin == "ai":
                if ann.review_state == STATE_REJECTED:
                    rgba[3] = 0.06  # rejected candidates are visible but muted
                    edge = [0.58, 0.58, 0.58, 0.7]
                elif ann.review_state == STATE_ACCEPTED:
                    rgba[3] = 0.32
                    edge = [0.12, 0.90, 0.45, 1.0]
                else:
                    rgba[3] = 0.20
                    edge = [1.0, 0.70, 0.10, 1.0]
            else:
                edge = [1.0, 1.0, 1.0, 1.0]
            face_colors.append(rgba)
            edge_colors.append(edge)
        layer.face_color = np.asarray(face_colors, dtype=float)
        layer.edge_color = np.asarray(edge_colors, dtype=float)
        layer.text = "annotation_name"
        layer.edge_width = 2

    def _rebuild_overlay(self):
        if self.asset is None:
            return
        labels = np.zeros((self.asset.height, self.asset.width), dtype=np.int32)
        # Always blank in M0 unless a mock run created AI label masks; suggestions
        # are shown as the ai_rois shapes layer with SIMULATED naming.
        if self._overlay is None:
            self._overlay = self.viewer.add_labels(labels, name="mock AI overlay (SIMULATED)", visible=False)
        else:
            self._overlay.data = labels

    def _ensure_layer_order(self):
        """Keep histology beneath AI heatmaps and all editable ROI shapes.

        napari paints layers in ascending stack order (last is drawn on top).
        Moving each existing layer to the end in bottom-to-top order also works
        when M2.2 layers have not been created yet.
        """
        for layer in (
            self._image_layer,                          # H&E: bottom
            getattr(self, "_hed_e_layer", None),         # deconvolved E
            getattr(self, "_hed_h_layer", None),         # deconvolved H
            self._overlay,                              # optional demo overlay
            getattr(self, "_tissue_layer", None),        # tissue mask
            getattr(self, "_threshold_layer", None),     # threshold preview
            getattr(self, "_ttr_heat_layer", None),    # CONCH similarity heatmap
            self._feature_controller.heat_layer,
            getattr(self, "_ttr_props_layer", None),   # transient AI boxes
            self._feature_controller.box_layer,
            self._ai_layer,                             # persisted AI ROIs
            self._human_layer,                          # manual ROIs
            getattr(self, "_points_layer", None),       # point counter
        ):
            if layer is None or layer not in self.viewer.layers:
                continue
            current = self.viewer.layers.index(layer)
            last = len(self.viewer.layers) - 1
            if current != last:
                self.viewer.layers.move(current, last)

    # ---- interaction hooks ----------------------------------------------------
    def _on_human_data(self, event):
        """Queue sync; never assign Shapes.data inside napari's data callback.

        napari emits ADDED while its polygon finish handler is still executing.
        Synchronously rebuilding layer.data at that moment re-enters its
        `_finish_drawing` implementation and can abort the Qt/VisPy process.
        """
        if self._rebuilding or self.asset is None:
            return
        action = getattr(event, "action", None)
        action_name = str(getattr(action, "value", action)).lower()
        # Geometry changes remain pending until an explicit Commit geometry edit.
        # A direct layer deletion will be undone, never persisted silently.
        if action_name not in {"added", "removed"}:
            return
        if self._pending_human_sync:
            return
        self._pending_human_sync = True
        QTimer.singleShot(0, self._sync_human_shapes_after_event)

    def _sync_human_shapes_after_event(self):
        """Reconcile completed GUI shapes after napari finishes mouse handling."""
        self._pending_human_sync = False
        if self._rebuilding or self.asset is None:
            return

        layer = self._human_layer
        # The very first vertex may add a temporary shape; wait for napari's
        # final ADDED event rather than recording unfinished geometry.
        if getattr(layer, "_is_creating", False):
            return

        existing = {a.roi_id for a in self.service.all_human()}
        if len(layer.data) < len(existing):
            # Direct Shapes deletion is not an auditable AnnotationService action.
            self._rebuild_all()
            self.statusBar().showMessage(
                "To delete an ROI, select it and use Delete selected ROI."
            )
            return
        if len(layer.data) == len(existing):
            # Same count: geometry edits require an explicit commit.
            return

        props = getattr(layer, "properties", None)
        seen: set[str] = set()
        for idx, vertices in enumerate(layer.data):
            roi_id = self._row_property(props, idx, "roi_id")
            if roi_id in existing and roi_id not in seen:
                seen.add(roi_id)
                continue
            points = self._shape_to_points(np.asarray(vertices))
            if len(points) < 3:
                continue
            ann = self.service.create_polygon(
                points,
                label=self.current_class(),
                image_asset_id=self.asset.asset_id,
                modality=self.modality_dock.current(),
                origin="human",
            )
            # Audited name via the service; never mutate Annotation fields directly.
            self.service.rename_annotation(ann.roi_id, f"ROI {ann.roi_id[:6]}")
            # Safe now: the previous mouse event has completed.
            self._rebuild_all()
            self._set_navigate_mode()
            self.statusBar().showMessage(
                "Polygon saved to annotation service; save case to persist to disk.",
                5000,
            )
            return

    @staticmethod
    def _row_property(props, idx: int, key: str):
        if props is None or not hasattr(props, "get") or key not in getattr(props, "keys", lambda: ())():
            return None
        val = props[key].iloc[idx] if hasattr(props[key], "iloc") else props[key][idx]
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        return str(val)

    @staticmethod
    def _shape_to_points(data):
        pts = data.tolist()
        return [[float(p[1]), float(p[0])] for p in pts]  # row,col -> col,row

    def _on_selection_change(self, event):
        if not self._rebuilding:
            self._update_selection_detail()

    def _selected_annotations(self) -> list:
        out = []
        for layer in (self._human_layer, self._ai_layer):
            props = getattr(layer, "properties", None)
            for idx in sorted(getattr(layer, "selected_data", set()) or ()):
                rid = self._row_property(props, idx, "roi_id")
                if rid:
                    a = self.service.find(rid)
                    if a:
                        out.append(a)
        return out

    def _update_selection_detail(self):
        # Notes belong to the old selection until explicitly committed.
        if not self._save_pending_roi_notes():
            return
        sel = self._selected_annotations()
        self.roi_dock.show_for(sel[0] if len(sel) == 1 else None, self.service.actor_id)

    # ---- actions --------------------------------------------------------------
    def _review(self, state: str):
        sel = self._selected_annotations()
        if len(sel) != 1:
            QMessageBox.information(self, "Review AI ROI", "Select exactly one AI proposal to review.")
            return
        if sel[0].origin != "ai":
            QMessageBox.information(self, "Human annotation", "Accept/Reject applies to AI suggestions only.")
            return
        if not self._save_pending_roi_notes():
            return
        try:
            self.service.review(sel[0].roi_id, state)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Review failed", str(exc))
            return
        self._rebuild_all()
        self.statusBar().showMessage(
            f"AI ROI {sel[0].roi_id} marked {state}. Save case to persist on disk.", 7000
        )

    def _delete_selected(self):
        if not self._save_pending_roi_notes():
            return
        sel = self._selected_annotations()
        if not sel:
            return
        for a in sel:
            try:
                self.service.delete(a.roi_id)
            except KeyError:
                pass
        self._rebuild_all()

    def _commit_edits(self):
        # Read current geometry back from the layers into the service (new revision).
        for layer in (self._human_layer, self._ai_layer):
            props = getattr(layer, "properties", None)
            for idx in range(len(layer.data)):
                rid = self._row_property(props, idx, "roi_id")
                if not rid:
                    continue
                a = self.service.find(rid)
                if a is None:
                    continue
                points = self._shape_to_points(layer.data[idx])
                if points and points != a.points:
                    try:
                        self.service.update_polygon(rid, points, label=a.label)
                    except KeyError:
                        pass
        self._rebuild_all()

    def _on_model_changed(self, model_id: str):
        self._feature_controller.model_changed(model_id)
        if model_id == MODEL_MOCK:
            self.model_dock.run_mock.setVisible(True)
        else:
            self.model_dock.run_mock.setVisible(False)

    def _on_current_label_changed(self, label: str):
        if not label:
            return
        sel = self._selected_annotations()
        if not sel:
            return
        for a in sel:
            self.service.update_polygon(a.roi_id, label=label)
        self._rebuild_all()

    def run_mock_ai(self):
        if self.asset is None:
            QMessageBox.information(self, "No image", "Open or show an image first.")
            return
        if self.model_dock.current() != MODEL_MOCK:
            QMessageBox.information(self, "Model", "Set model to 'Mock (demo)' to run the SIMULATED overlay.")
            return
        adapter = self.registry.adapter(MODEL_MOCK)
        if adapter is None:
            return
        # Remove previous mock proposals through the audited service API.
        for ann in list(self.service.annotations):
            if ann.origin == "ai" and ann.model_run_id and ann.model_run_id.startswith("mock-"):
                self.service.delete(ann.roi_id)
        for prop in adapter.suggest(self.asset.width, self.asset.height):
            ann = self.service.create_polygon(
                prop.points,
                label=DEFAULT_LABEL,
                name="SIMULATED " + prop.roi_id,
                notes=f"{MOCK_LABEL_PREFIX}: deterministic demo overlay, not a pathology prediction.",
                image_asset_id=self.asset.asset_id,
                modality=self.modality_dock.current(),
                origin="ai",
                ai_meta={"model_run_id": prop.model_run_id, "revision": 1},
            )
        self._rebuild_all()
        self.history_dock.set_events(self.service.events)
        self.statusBar().showMessage("Mock AI overlay rendered — marked SIMULATED.", 4000)

    def describe_selected_roi(self):
        """Rank controlled image descriptions using the selected local encoder."""
        if self.asset is None:
            QMessageBox.warning(self, "No image", "Open or show an image first.")
            return
        model_id = self.model_dock.current()
        if model_id not in (MODEL_CONCH, MODEL_MUSK):
            QMessageBox.information(self, "Model", "Select a configured local CONCH or MUSK model.")
            return
        adapter = self.registry.adapter(model_id)
        if adapter is None:
            QMessageBox.warning(self, "Model unavailable", "Configure your local model in AI → Manage Local Models.")
            return
        sel = self._selected_annotations()
        if len(sel) != 1:
            QMessageBox.information(self, "Select ROI", "Select exactly one ROI to describe.")
            return
        ann = sel[0]
        from ..models.roi_extraction import extract_roi_crop
        from ..models.descriptions import DEFAULT_CANDIDATES
        from ..models.embedding_cache import DiskEmbeddingCache

        cache = DiskEmbeddingCache(self._score_cache_dir())
        try:
            crop = extract_roi_crop(self.asset.data, ann)
            results, evidence = adapter.rank_descriptions(crop, DEFAULT_CANDIDATES)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "CONCH failed", str(exc))
            return
        self._show_score_results(results, evidence)

    def _score_cache_dir(self):
        cache = Path(os.environ.get("ASHBY_EMBED_CACHE", "")) if os.environ.get("ASHBY_EMBED_CACHE") else Path.home() / "ashby_studio_embeddings"
        return cache

    def _show_score_results(self, results, evidence):
        from qtpy.QtWidgets import QListWidget, QListView

        if not hasattr(self, "_score_dock"):
            self._score_dock = QDockWidget("ROI-to-text similarity (uncalibrated)", self)
            self._score_dock.setObjectName("score_dock")
            self._score_list = QListWidget()
            self._score_list.setSelectionMode(QListView.NoSelection)
            self._score_dock.setWidget(self._score_list)
            self.addDockWidget(Qt.BottomDockWidgetArea, self._score_dock)
            self.tabifyDockWidget(self._dock_widgets["measurements"], self._score_dock)
            self._dock_widgets["roi_text_results"] = self._score_dock
            self._window_menu.insertAction(
                self._window_menu.actions()[-2], self._score_dock.toggleViewAction()
            )
        self._score_list.clear()
        header = (f"{evidence['model_id']} {evidence['model_version']} | {evidence['device']} | "
                  f"weight {evidence['weight_sha256'][:8]} | crop {evidence['crop_sha256'][:8]} | "
                  f"{evidence['score_type']} (uncalibrated)")
        self._score_list.addItem(header)
        for did, text, score in results:
            self._score_list.addItem(f"{score:+.3f}  [{did}]  {text}")
        self._score_dock.show()
        self._score_dock.raise_()
        self.statusBar().showMessage(header, 8000)

    # ---- masks (Phase 1/2) -----------------------------------------------------
    def _create_tissue_mask(self):
        if self.asset is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        from ..measurements.engine import generate_tissue_mask

        mask = generate_tissue_mask(self.asset.data)
        self._tissue_mask = mask
        layer_data = mask.astype(np.int32) * 1  # 0=background, 1=tissue
        if self._tissue_layer is None:
            self._tissue_layer = self.viewer.add_labels(
                layer_data, name="Tissue mask (auto)", visible=True,
                opacity=0.4,
            )
        else:
            self._tissue_layer.data = layer_data
            self._tissue_layer.visible = True
        self.measure_dock.set_tissue_mask(mask)
        self._ensure_layer_order()
        self.statusBar().showMessage("Tissue mask created (Otsu); edit colors/props or correct via ROI tools.", 6000)

    def _update_threshold_mask(self, value: float):
        if self.asset is None:
            return
        gray = np.asarray(self.asset.data, dtype=np.float64).mean(axis=2)
        mask = gray < value
        data = mask.astype(np.int32)
        if self._threshold_layer is None:
            self._threshold_layer = self.viewer.add_labels(
                data, name="Threshold mask (preview)", visible=True, opacity=0.35,
            )
        else:
            self._threshold_layer.data = data
            self._threshold_layer.visible = True
        self._ensure_layer_order()

    def _clear_threshold_mask(self):
        if self._threshold_layer is not None:
            try:
                self.viewer.layers.remove(self._threshold_layer)
            except ValueError:
                pass
            self._threshold_layer = None

    def _show_hed(self, show_h: bool, show_e: bool, opacity: float):
        if self.asset is None:
            return
        from ..image_processing.deconvolve import hed_channels

        if getattr(self, "_hed", None) is None:
            self._hed = hed_channels(self.asset.data)
        h = self._hed[..., 0]
        e = self._hed[..., 1]
        lo, hi = (0.0, float(max(self._hed.max(), 1e-9)))
        if getattr(self, "_hed_h_layer", None) is None:
            self._hed_h_layer = self.viewer.add_image(
                h, name="Hematoxylin (deconvolved)", colormap="blue", opacity=opacity, blending="translucent")
            self._hed_e_layer = self.viewer.add_image(
                e, name="Eosin (deconvolved)", colormap="green", opacity=opacity, blending="translucent")
        self._hed_h_layer.contrast_limits = (lo, hi)
        self._hed_e_layer.contrast_limits = (lo, hi)
        self._hed_h_layer.visible = bool(show_h)
        self._hed_e_layer.visible = bool(show_e)
        self._hed_h_layer.opacity = opacity
        self._hed_e_layer.opacity = opacity
        self._ensure_layer_order()

    def _edit_tissue_mode(self, on: bool):
        if self._tissue_layer is None:
            if self.asset is None:
                return
            self._tissue_mask = np.ones((self.asset.height, self.asset.width), dtype=bool)
            self._tissue_layer = self.viewer.add_labels(
                self._tissue_mask.astype(np.int32), name="Tissue mask (auto)", opacity=0.4)
        self._tissue_layer.mode = labels_mode("paint" if on else "pan_zoom")
        self._tissue_layer.selected_label = 1
        if not on:
            self._set_navigate_mode()

    def _commit_tissue_edit(self):
        if self._tissue_layer is None:
            self._tissue_dock_set(None)
            return
        self._tissue_mask = np.asarray(self._tissue_layer.data, dtype=bool)
        self._tissue_layer.mode = labels_mode("pan_zoom")
        self._tissue_dock_set(self._tissue_mask)
        self._set_navigate_mode()
        self._ensure_layer_order()
        self.statusBar().showMessage("Edited tissue mask committed to measurements.", 5000)

    def _tissue_dock_set(self, mask):
        if getattr(self, "measure_dock", None) is not None:
            self.measure_dock.set_tissue_mask(mask)

    def _convert_threshold_to_rois(self, min_area: float, origin: str):
        if self.asset is None:
            return
        layer = self._threshold_layer
        mask = np.asarray(layer.data, dtype=bool) if layer is not None else None
        if mask is None:
            QMessageBox.information(self, "Threshold", "Show a threshold mask preview first (Image Tools).")
            return
        from ..image_processing.binary import mask_to_polygons

        polys = mask_to_polygons(mask, min_area=int(min_area))
        if not polys:
            self.statusBar().showMessage("No mask regions above the min-area threshold.", 5000)
            return
        created = 0
        for pts in polys:
            ann = self.service.create_polygon(
                pts,
                label=self.current_class(),
                name=f"mask→{'AI' if origin == 'ai' else 'ROI'} {created}",
                notes="Generated from threshold mask; research only.",
                image_asset_id=self.asset.asset_id,
                modality=self.modality_dock.current(),
                origin=origin,
                ai_meta={"model_run_id": f"mask-{self._mask_run_id()}", "revision": 1} if origin == "ai" else None,
            )
            created += 1
        self._rebuild_all()
        self.history_dock.set_events(self.service.events)
        self.statusBar().showMessage(f"Created {created} ROI(s) from threshold mask (audited).", 6000)

    def _mask_run_id(self):
        import uuid
        return f"{uuid.uuid4().hex[:8]}"

    # ---- point counter (audited) ----------------------------------------------
    def _points_new(self):
        if getattr(self, "_points_layer", None) is None:
            self._points_layer = self.viewer.add_points(name="point counter (manual count)", size=6, symbol="ring")
        else:
            self._points_layer.data = []
        self._image_tools_dock.set_point_count(0)

    def _points_clear(self):
        if getattr(self, "_points_layer", None) is not None:
            self._points_layer.data = []
        self._image_tools_dock.set_point_count(0)

    def _points_commit(self):
        layer = getattr(self, "_points_layer", None)
        if layer is None:
            self._points_new()
            return
        pts = [[float(p[1]), float(p[0])] for p in np.atleast_2d(np.asarray(layer.data, dtype=float))]
        ev = self.service.record_point_count(
            count=len(pts), points=pts, notes="Manual mineralization-foci count.",
            modality=self.modality_dock.current(),
        )
        self.history_dock.set_events(self.service.events)
        self._image_tools_dock.set_point_count(len(pts))
        self.statusBar().showMessage(f"Recorded {len(pts)} point(s) (audited {ev.event_id}).", 6000)

    # ---- ROI booleans -----------------------------------------------------------
    def _roi_boolean(self, op: str):
        if self.asset is None:
            return
        ids = self._roi_manager_dock._selected_ids()
        if len(ids) < (2 if op != "subtract" else 2):
            self.statusBar().showMessage(f"{op} needs at least 2 selected ROIs.", 5000)
            return
        anns = [self.service.find(r) for r in ids]
        anns = [a for a in anns if a is not None and len(a.points) >= 3]
        if len(anns) < 2:
            return
        from ..image_processing.binary import rois_boolean

        polys = rois_boolean(
            [a.points for a in anns], op, self.asset.width, self.asset.height
        )
        for pts in polys:
            self.service.create_polygon(
                pts,
                label=self.current_class(),
                name=f"{op} result",
                notes=f"Boolean {op} of ROIs: {[a.roi_id for a in anns]}",
                image_asset_id=self.asset.asset_id,
                modality=self.modality_dock.current(),
                origin="human",
            )
        self._rebuild_all()
        self.history_dock.set_events(self.service.events)
        self.statusBar().showMessage(f"Created {len(polys)} ROI(s) via {op}.", 6000)

    # ---- Text -> ROI (M2.2) --------------------------------------------------
    def _open_text_to_roi(self):
        self._show_panel("text_to_roi")

    def _ttr_run_search(self, query: str):
        if not query:
            self._ttr_dock.set_error("Enter a query first.")
            return
        if self.asset is None:
            self._ttr_dock.set_error("Open or show an image first.")
            return
        model_id = self.model_dock.current()
        if model_id not in (MODEL_CONCH, MODEL_MUSK):
            self._ttr_dock.set_error("Select a configured local CONCH or MUSK model.")
            return
        encoder = self.registry.adapter(model_id)
        if encoder is None:
            self._ttr_dock.set_error("Configure your local model in AI → Manage Local Models.")
            return
        if self._ttr_worker is not None and self._ttr_worker.isRunning():
            self._ttr_dock.set_error("A search is already running.")
            return
        tile_size, stride = self._ttr_dock.tile_params()
        from ..models.embedding_cache import DiskEmbeddingCache

        cache = DiskEmbeddingCache(self._score_cache_dir())
        self._ttr_search_token += 1
        token = self._ttr_search_token
        image_sha = self.asset.sha256
        self._ttr_state = {
            "query": query,
            "model_id": model_id,
            "evidence": None,
            "model_run_id": None,
            "proposals": [],
            "tiles": [],
            "tile_size": tile_size,
            "stride": stride,
            "image_asset_id": self.asset.asset_id,
            "image_sha": image_sha,
        }
        self._ttr_dock.set_running(True)
        from .text_to_roi_dock import TextToRoiWorker
        from ..models.text_to_roi import BATCH_DEFAULT

        worker = TextToRoiWorker(
            encoder, cache, self.asset.data, image_sha, query, tile_size, stride,
            batch_size=BATCH_DEFAULT,
            tissue_mask=getattr(self, "_tissue_mask", None),
        )
        worker.token = token
        worker.progress.connect(self._ttr_on_progress)
        worker.finished_search.connect(self._ttr_on_result)
        worker.failed.connect(self._ttr_on_error)
        worker.cancelled.connect(self._ttr_on_cancelled)
        worker.finished.connect(self._ttr_on_done)
        self._ttr_worker = worker
        worker.start()

    def _ttr_on_progress(self, done: int, total: int):
        # Progress arrives on the GUI thread via a queued Qt signal.
        if getattr(self._ttr_worker, "token", -1) != self._ttr_search_token:
            return
        self._ttr_dock.set_running(True, detail=f"{done}/{total} tiles")

    def _ttr_on_cancelled(self, msg: str):
        if getattr(self._ttr_worker, "token", -1) != self._ttr_search_token:
            return
        self._ttr_dock.set_running(False)
        self._ttr_dock.set_error(msg)
        self.statusBar().showMessage(f"Text→ROI {msg}", 6000)

    def _ttr_on_result(self, proposals, heatmap, tiles, evidence):
        # Ignore stale results: a newer search, image, case, or clear invalidated this token.
        worker = self._ttr_worker
        token = getattr(worker, "token", -1)
        if token != self._ttr_search_token:
            return
        if self.asset is None or getattr(worker, "_image_sha", None) != self.asset.sha256:
            return
        state = getattr(self, "_ttr_state", {})
        state.setdefault("model_run_id", self._ttr_dock.new_run_id())
        state["evidence"] = evidence
        state["proposals"] = proposals
        state["tiles"] = tiles
        self._ttr_heatmap = heatmap
        self._update_text_roi_layers()
        self._ttr_fill_results()
        self.statusBar().showMessage(
            f"Text→ROI: {tiles and len(tiles) or 0} tiles, {proposals and len(proposals) or 0} candidates (research only).",
            8000,
        )

    def _ttr_on_error(self, msg: str):
        if getattr(self._ttr_worker, "token", -1) != self._ttr_search_token:
            return
        self._ttr_dock.set_error(msg)
        self.statusBar().showMessage(f"Text→ROI failed: {msg}", 8000)

    def _ttr_on_done(self):
        self._ttr_dock.set_running(False)
        runner = self._ttr_worker
        if runner is not None and not runner.isRunning():
            self._ttr_worker = None

    def _update_text_roi_layers(self):
        heat = getattr(self, "_ttr_heatmap", None)
        if heat is None:
            return
        if getattr(self, "_ttr_heat_layer", None) is None:
            self._ttr_heat_layer = self.viewer.add_image(
                heat, name="Text→ROI similarity (AI research only)",
                colormap="magma", opacity=0.6, blending="translucent",
                metadata={"label": "AI similarity candidates — research only"},
            )
        else:
            self._ttr_heat_layer.data = heat
        if getattr(self, "_ttr_props_layer", None) is None:
            self._ttr_props_layer = self.viewer.add_shapes(
                name="Text→ROI candidates (AI)" ,
                visible=self._ttr_dock.threshold() <= 1.0,
            )
        self._ttr_apply_threshold()
        self._ensure_layer_order()

    def _ttr_apply_threshold(self):
        layer = getattr(self, "_ttr_props_layer", None)
        if layer is None:
            return
        thr = self._ttr_dock.threshold()
        state = getattr(self, "_ttr_state", {})
        boxes = [p for p in state.get("proposals", []) if p["score"] >= thr]
        if not boxes:
            layer.data = []
            layer.properties = {}
            return
        polys = [[[b["box"][1], b["box"][0]], [b["box"][1], b["box"][2]],
                  [b["box"][3], b["box"][2]], [b["box"][3], b["box"][0]]] for b in boxes]
        layer.data = polys
        layer.properties = {
            "score": [b["score"] for b in boxes],
            "origin": ["ai"] * len(boxes),
            "annotation_name": [f"T→R:{b['score']:.3f}" for b in boxes],
        }
        layer.face_color = [0.9, 0.9, 0.0, 0.15]
        layer.edge_color = "#d4af37"
        layer.edge_width = 1.5
        layer.text = "annotation_name"
        self._ttr_fill_results(boxes)

    def _ttr_fill_results(self, boxes=None):
        state = getattr(self, "_ttr_state", {})
        if boxes is None:
            thr = self._ttr_dock.threshold()
            boxes = [p for p in state.get("proposals", []) if p["score"] >= thr]
        data = boxes or []
        self._ttr_dock.results.clear()
        for b in data:
            self._ttr_dock.results.addItem(f"{b['score']:+.3f}  box {b['box']}")

    def _clear_mask_layers(self):
        for attr in ("_hed_h_layer", "_hed_e_layer", "_tissue_layer", "_threshold_layer"):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except ValueError:
                    pass
                setattr(self, attr, None)
        self._tissue_mask = None

    def _invalidate_ttr(self):
        """Drop stale Text→ROI state/layers and invalidate/cancel any running search."""
        worker = self._ttr_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
        self._ttr_search_token += 1
        for attr in ("_ttr_heat_layer", "_ttr_props_layer"):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except ValueError:
                    pass
                setattr(self, attr, None)
        self._ttr_heatmap = None
        self._ttr_state = {}
        self._ttr_saved_keys = set()
        self._ttr_dock.results.clear()
        self._ttr_dock.set_error("")
        self._ttr_dock.set_running(False)

    def _ttr_clear(self):
        self._invalidate_ttr()

    def _ttr_save_candidates(self):
        if self._ttr_worker is not None and self._ttr_worker.isRunning():
            self._ttr_dock.set_error("Wait for the search to finish before saving.")
            return
        if self.asset is None:
            return
        state = getattr(self, "_ttr_state", {})
        # Never save results computed for a different image or model run.
        if state.get("image_sha") != self.asset.sha256:
            self._ttr_dock.set_error("Results are for a different image; run the search again.")
            return
        thr = self._ttr_dock.threshold()
        boxes = [p for p in state.get("proposals", []) if p["score"] >= thr]
        if not boxes:
            self._ttr_dock.set_error("No candidates above the threshold to save.")
            return
        evidence = state.get("evidence") or {}
        run_id = state.get("model_run_id") or self._ttr_dock.new_run_id()
        state["model_run_id"] = run_id
        query = state.get("query", "")
        width, height = (self.asset.width, self.asset.height)
        saved = 0
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b["box"]]
            x0 = max(0, min(x0, width - 1)); y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width)); y1 = max(y0 + 1, min(y1, height))
            key = (run_id, x0, y0, x1, y1)
            if key in self._ttr_saved_keys:
                continue
            self._ttr_saved_keys.add(key)
            rect = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            notes = (
                f"AI similarity candidate — research only. query={query!r} "
                f"score={b['score']:.4f} box={[x0, y0, x1, y1]} model={evidence.get('model_id')} "
                f"version={evidence.get('model_version')} weight={str(evidence.get('weight_sha256'))[:8]} "
                f"device={evidence.get('device')} image_sha={state.get('image_sha')} "
                f"tile_size={state.get('tile_size')} stride={state.get('stride')} "
                f"score_type={evidence.get('score_type')} calibration={evidence.get('calibration')}"
            )
            self.service.create_polygon(
                rect,
                label=DEFAULT_LABEL,
                name=f"T→R {b['score']:.3f}",
                notes=notes,
                image_asset_id=self.asset.asset_id,
                modality=self.modality_dock.current(),
                origin="ai",
                ai_meta={"model_run_id": run_id, "revision": 1},
            )
            saved += 1
        self._rebuild_all()
        self.history_dock.set_events(self.service.events)
        self.statusBar().showMessage(
            f"Saved {saved} new AI proposal(s) (deduped; research only).", 8000,
        )

    # ---- persistence -----------------------------------------------------------
    def _write_case_directory(self, target_dir: Path):
        """Write the existing repository format plus custom class metadata."""
        repo = CaseRepository(target_dir)
        repo.save(
            case_id=self.service.case_id,
            annotations=self.service.annotations,
            events=self.service.events,
            asset=self.asset,
            embed_image=True,
        )
        (target_dir / "custom_classes.json").write_text(
            json.dumps(self._custom_classes, indent=2), encoding="utf-8"
        )
        self._feature_controller.save(target_dir)

    def save_case(self):
        """Save a named, portable .ashbycase archive (ZIP of repository files)."""
        if not self._save_pending_roi_notes():
            return
        if self.asset is None:
            QMessageBox.warning(self, "Nothing to save", "Open an image first.")
            return
        default_name = str(self._case_filename or (Path.home() / "AshbyCase.ashbycase"))
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save case as", default_name, "Ashby case (*.ashbycase)"
        )
        if not filename:
            return
        output = Path(filename)
        if output.suffix.lower() != ".ashbycase":
            output = output.with_suffix(".ashbycase")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="ashby-export-") as staging:
                base = Path(staging)
                case_dir = base / "case"
                case_dir.mkdir()
                self._write_case_directory(case_dir)
                archive_tmp = base / "case.ashbycase"
                with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for source in sorted(case_dir.rglob("*")):
                        if source.is_file():
                            archive.write(source, source.relative_to(case_dir).as_posix())
                # Atomic replacement: don't destroy an earlier working case on error.
                with tempfile.NamedTemporaryFile(
                    dir=output.parent, prefix=".ashby-", suffix=".tmp", delete=False
                ) as temp_file:
                    swap = Path(temp_file.name)
                try:
                    with open(archive_tmp, "rb") as src, swap.open("wb") as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    os.replace(swap, output)
                finally:
                    swap.unlink(missing_ok=True)
            self._case_filename = output
            self.statusBar().showMessage(f"Saved case: {output.name}", 7000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    @staticmethod
    def _safe_extract_case(archive_path: Path, target: Path):
        """Unpack an archive without accepting traversal paths or symlinks."""
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.infolist()) > 20000:
                raise ValueError("Too many files in case archive")
            total = 0
            for info in archive.infolist():
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Unsafe path in case archive")
                # Reject Unix symlinks.
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Symlinks are not accepted in case archives")
                total += info.file_size
                if total > 5 * 1024**3:
                    raise ValueError("Case archive exceeds 5 GiB unpacked limit")
            archive.extractall(target)

    def reopen_case(self):
        if not self._save_pending_roi_notes():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open case", str(self._case_filename.parent if self._case_filename else Path.home()),
            "Ashby case (*.ashbycase);;Legacy annotations (annotations.json)"
        )
        if not filename:
            return
        archive_path = Path(filename)
        temporary = None
        try:
            if archive_path.suffix.lower() == ".ashbycase":
                temporary = tempfile.TemporaryDirectory(prefix="ashby-open-")
                case_dir = Path(temporary.name)
                self._safe_extract_case(archive_path, case_dir)
            elif archive_path.name == "annotations.json":
                case_dir = archive_path.parent
            else:
                raise ValueError("Choose a .ashbycase archive or annotations.json")

            # Register class definitions *before* decoding annotated records:
            # a domain deserializer may validate label membership against LABELS.
            custom_path = case_dir / "custom_classes.json"
            custom_defs = json.loads(custom_path.read_text(encoding="utf-8")) if custom_path.is_file() else {}
            if not isinstance(custom_defs, dict):
                raise ValueError("Invalid custom_classes.json")
            for key, rgba in custom_defs.items():
                if (not isinstance(key, str) or not re.fullmatch(r"[a-z0-9_-]{1,60}", key)
                    or key in self._builtin_classes
                    or not isinstance(rgba, list) or len(rgba) != 4
                    or not all(type(v) in (float, int) and 0 <= v <= 1 for v in rgba)):
                    raise ValueError(f"Invalid custom class: {key!r}")
            old_labels = dict(LABELS)
            LABELS.update({key: tuple(rgba) for key, rgba in custom_defs.items()})
            try:
                repo = CaseRepository(case_dir)
                case_id, annotations, events, asset = repo.load(
                    resolve_image=repo.case_dir / "images"
                )
                from ..storage.feature_runs import load_runs, validate_runs_for_asset
                feature_runs = load_runs(case_dir, case_id)
                validate_runs_for_asset(feature_runs, asset)
            finally:
                # Restore the current case if loading fails; apply classes only
                # after a complete and valid case is available.
                LABELS.clear()
                LABELS.update(old_labels)
        except Exception as exc:
            if temporary is not None:
                temporary.cleanup()
            QMessageBox.critical(self, "Open case failed", str(exc))
            return

        # Successful load: retain extracted images for the lifetime of this window.
        previous = self._extracted_case
        self._extracted_case = temporary
        self._reset_classes()
        for key, rgba in custom_defs.items():
            LABELS[key] = tuple(rgba)
            self._custom_classes[key] = rgba
        self._refresh_classes()
        # Clear stale ROI-editor identity before replacing the case's records.
        self.roi_dock.show_for(None, self.service.actor_id)
        self.service.case_id = case_id
        self.service.annotations = annotations
        self.service.events = events
        self.service._last_event_hash = events[-1].event_sha256 if events else None
        self._refresh_classes()
        self.repository = repo
        self._case_filename = archive_path if temporary is not None else None
        self._invalidate_ttr()
        self._feature_controller.invalidate(clear_runs=True)
        if asset is not None:
            self._set_asset(asset, reset_annotations=False)
        else:
            self.asset = None
            self._image_status.setText("No image")
            if self._image_layer is not None:
                self.viewer.layers.remove(self._image_layer)
                self._image_layer = None
            if self._overlay is not None:
                self.viewer.layers.remove(self._overlay)
                self._overlay = None
            self._rebuild_all()
            QMessageBox.warning(self, "Image missing", "Case annotations loaded without a source image.")
        self._feature_controller.restore(feature_runs)
        if previous is not None:
            previous.cleanup()
        self.statusBar().showMessage(f"Opened case: {archive_path.name}", 7000)
