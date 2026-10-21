"""Non-destructive image inspection tools (Phase 2).

Brightness/contrast/gamma adjust napari layer display only (source pixels
untouched). An intensity histogram is drawn in-widget, and an Otsu-based
threshold slider previews a mask overlay.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget, QLayout, QSizePolicy,
)

HIST_BINS = 64


class _Histogram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = np.zeros(HIST_BINS, dtype=int)
        self.setMinimumHeight(80)

    def set_counts(self, counts: np.ndarray):
        self._counts = np.asarray(counts, dtype=int)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w = max(1, self.width())
        h = self.height()
        p.fillRect(self.rect(), Qt.black)
        total = int(self._counts.sum())
        if total <= 0 or w <= 0:
            return
        mx = int(self._counts.max()) or 1
        bw = w / HIST_BINS
        for i in range(HIST_BINS):
            bh = int((self._counts[i] / mx) * (h - 2))
            p.fillRect(QRectF(i * bw, h - 1 - bh, max(1.0, bw - 1.0), bh), QColor(200, 200, 200))


class ImageToolsDock(QWidget):
    thresholdMaskRequested = Signal(float)  # threshold value 0..255
    maskCleared = Signal()

    hedShowRequested = Signal(bool, bool, float)   # show_h, show_e, opacity
    maskEditRequested = Signal(bool)               # enter/leave paint mode
    maskCommitRequested = Signal()                 # read edited layer into tissue mask
    convertMaskRequested = Signal(float, str)      # min_area_px2, origin
    pointsNewRequested = Signal()
    pointsClearRequested = Signal()
    pointsCommitRequested = Signal()

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self._win = win
        self.setObjectName("ashby_image_tools")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(9)
        # Keep full-size form rows; the enclosing MainWindow QScrollArea scrolls.
        lay.setSizeConstraint(QLayout.SetMinimumSize)

        def section(title):
            heading = QLabel(title)
            heading.setProperty("sectionHeading", True)
            heading.setStyleSheet("font-weight: 600; margin-top: 7px; margin-bottom: 2px;")
            lay.addWidget(heading)

        def slider_row(text, slider, fmt):
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(7)
            name = QLabel(text)
            name.setMinimumWidth(86)
            line.addWidget(name)
            line.addWidget(slider, 1)
            value = QLabel(fmt(slider.value()))
            value.setMinimumWidth(42)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            line.addWidget(value)
            slider.valueChanged.connect(lambda v: value.setText(fmt(v)))
            lay.addLayout(line)

        section("Display adjustments")
        self._contrast = QSlider(Qt.Horizontal)
        self._contrast.setRange(50, 300)
        self._contrast.setValue(100)
        slider_row("Contrast", self._contrast, lambda v: f"{v}%")
        self._brightness = QSlider(Qt.Horizontal)
        self._brightness.setRange(-50, 50)
        self._brightness.setValue(0)
        slider_row("Brightness", self._brightness, lambda v: str(v))
        self._gamma = QSlider(Qt.Horizontal)
        self._gamma.setRange(50, 200)
        self._gamma.setValue(100)
        slider_row("Gamma", self._gamma, lambda v: f"{v / 100:.2f}")

        section("Intensity histogram")
        self._hist = _Histogram(self)
        self._hist.setFixedHeight(92)
        lay.addWidget(self._hist)

        section("Threshold mask preview")
        self._threshold = QSlider(Qt.Horizontal)
        self._threshold.setRange(0, 255)
        self._threshold.setValue(128)
        slider_row("Threshold", self._threshold, lambda v: str(v))
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Otsu starting value"))
        row.addStretch(1)
        btn = QPushButton("Clear preview")
        btn.clicked.connect(self.maskCleared)
        row.addWidget(btn)
        lay.addLayout(row)

        section("H&E color deconvolution")
        chk_row = QHBoxLayout()
        chk_row.setSpacing(12)
        self._show_h = QCheckBox("Hematoxylin")
        self._show_h.setChecked(True)
        self._show_e = QCheckBox("Eosin")
        self._show_e.setChecked(True)
        chk_row.addWidget(self._show_h)
        chk_row.addWidget(self._show_e)
        chk_row.addStretch(1)
        lay.addLayout(chk_row)
        self._ded_opacity = QSlider(Qt.Horizontal)
        self._ded_opacity.setRange(10, 100)
        self._ded_opacity.setValue(100)
        slider_row("Opacity", self._ded_opacity, lambda v: f"{v}%")
        disp_btn = QPushButton("Show deconvolved channels")
        disp_btn.clicked.connect(self._emit_hed)
        lay.addWidget(disp_btn)
        self._show_h.toggled.connect(lambda _: self._emit_hed())
        self._show_e.toggled.connect(lambda _: self._emit_hed())
        self._ded_opacity.valueChanged.connect(lambda _: self._emit_hed())

        section("Tissue mask")
        self._edit_mask_btn = QPushButton("Edit tissue mask (paint / erase)")
        self._edit_mask_btn.setCheckable(True)
        self._edit_mask_btn.toggled.connect(lambda on: self.maskEditRequested.emit(bool(on)))
        commit_mask = QPushButton("Commit edited tissue mask")
        commit_mask.clicked.connect(lambda: self.maskCommitRequested.emit())
        lay.addWidget(self._edit_mask_btn)
        lay.addWidget(commit_mask)

        section("Convert threshold mask to ROIs")
        mrow = QHBoxLayout()
        mrow.setSpacing(8)
        mrow.addWidget(QLabel("Minimum area (px²)"))
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 10_000_000)
        self._min_area.setValue(50)
        mrow.addWidget(self._min_area, 1)
        lay.addLayout(mrow)
        brow = QHBoxLayout()
        brow.setSpacing(7)
        to_hu = QPushButton("Human ROIs")
        to_hu.clicked.connect(lambda: self.convertMaskRequested.emit(float(self._min_area.value()), "human"))
        to_ai = QPushButton("AI proposals")
        to_ai.clicked.connect(lambda: self.convertMaskRequested.emit(float(self._min_area.value()), "ai"))
        brow.addWidget(to_hu)
        brow.addWidget(to_ai)
        lay.addLayout(brow)

        section("Point counter")
        row = QHBoxLayout()
        row.setSpacing(7)
        new_pts = QPushButton("New points layer")
        new_pts.clicked.connect(lambda: self.pointsNewRequested.emit())
        clr_pts = QPushButton("Clear points")
        clr_pts.clicked.connect(lambda: self.pointsClearRequested.emit())
        row.addWidget(new_pts)
        row.addWidget(clr_pts)
        lay.addLayout(row)
        self._point_count = QLabel("0 points")
        lay.addWidget(self._point_count)
        commit_pts = QPushButton("Commit point count (audited)")
        commit_pts.clicked.connect(lambda: self.pointsCommitRequested.emit())
        lay.addWidget(commit_pts)
        lay.addStretch(1)

        self._contrast.valueChanged.connect(self._apply_display)
        self._brightness.valueChanged.connect(self._apply_display)
        self._gamma.valueChanged.connect(self._apply_gamma)
        self._threshold.valueChanged.connect(lambda v: self.thresholdMaskRequested.emit(float(v)))

    def _emit_hed(self):
        self.hedShowRequested.emit(
            bool(self._show_h.isChecked()),
            bool(self._show_e.isChecked()),
            self._ded_opacity.value() / 100.0,
        )

    def set_point_count(self, n: int):
        self._point_count.setText(f"{n} points")

    def _image_layer(self):
        return getattr(self._win, "_image_layer", None)

    def _apply_display(self):
        layer = self._image_layer()
        if layer is None:
            return
        data = layer.data
        if data is None or data.size == 0:
            return
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if vmax <= vmin:
            return
        mid = (vmin + vmax) / 2.0
        half = (vmax - vmin) / 2.0
        scale = self._contrast.value() / 100.0
        shift = self._brightness.value() / 100.0 * half
        layer.contrast_limits = (max(vmin, mid - half * scale) + shift,
                                 min(vmax, mid + half * scale) + shift)

    def _apply_gamma(self):
        layer = self._image_layer()
        if layer is not None:
            layer.gamma = self._gamma.value() / 100.0

    def refresh_histogram(self, data=None):
        if data is None:
            layer = self._image_layer()
            data = layer.data if layer is not None else None
        if data is None:
            return
        flat = np.asarray(data, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        counts, _ = np.histogram(flat, bins=HIST_BINS, range=(flat.min(), flat.max() + 1e-9))
        self._hist.set_counts(counts)
        self._auto_threshold(flat)

    def _auto_threshold(self, flat):
        import skimage.filters as filters
        try:
            thr = filters.threshold_otsu(flat)
        except Exception:  # noqa: BLE001
            thr = float(np.median(flat))
        self._threshold.setValue(int(thr))