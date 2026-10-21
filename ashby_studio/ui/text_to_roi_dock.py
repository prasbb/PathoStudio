"""Text -> ROI dock widget and background inference worker (M2.2).

Inference runs on a QThread; napari layer updates happen only on the GUI thread
via Qt signals. Results are transient "AI similarity candidates — research only"
until the user explicitly saves a candidate through AnnotationService.
"""
from __future__ import annotations

import threading
import uuid

from qtpy.QtCore import QThread, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..models.text_to_roi import (
    BATCH_DEFAULT,
    TILE_DEFAULT,
    PROPOSAL_MAX_DEFAULT,
    PROPOSAL_MIN_SEP_PX_DEFAULT,
    RetrievalCancelled,
)

LABEL = "AI similarity candidates — research only"


class TextToRoiWorker(QThread):
    """Compute patch embeddings and cosine scores off the GUI thread.

    Embeddings run in bounded batches (see ``batch_size``) with cooperative
    cancellation and progress reporting. ``finished_search``/``failed``/
    ``cancelled`` are emitted asynchronously and are delivered to the GUI on
    Qt's event loop; napari layer edits stay on the GUI thread.
    """

    finished_search = Signal(object, object, object, object)  # proposals, heatmap, tiles, evidence
    failed = Signal(str)
    cancelled = Signal(str)
    progress = Signal(int, int)  # (done_tiles, total_tiles)

    def __init__(self, encoder, cache, image_rgb, image_sha, query, tile_size, stride,
                 batch_size: int = BATCH_DEFAULT, tissue_mask=None,
                 max_proposals: int = PROPOSAL_MAX_DEFAULT,
                 min_sep_px: float = PROPOSAL_MIN_SEP_PX_DEFAULT, parent=None):
        super().__init__(parent)
        self._encoder = encoder
        self._cache = cache
        self._image_rgb = image_rgb
        self._image_sha = image_sha
        self._query = query
        self._tile_size = tile_size
        self._stride = stride
        self._batch_size = batch_size
        self._tissue_mask = tissue_mask
        self._max_proposals = max_proposals
        self._min_sep_px = min_sep_px
        self._cancel_evt = threading.Event()

    def cancel(self):
        """Request cancellation; the worker stops at the next batch boundary."""
        self._cancel_evt.set()

    def run(self):  # worker thread
        from ..models.text_to_roi import (
            run_retrieval, select_hotspot_proposals,
        )

        def progress(done: int, total: int):
            self.progress.emit(done, total)

        def is_cancelled() -> bool:
            return self._cancel_evt.is_set()

        try:
            tiles, scores, heatmap, evidence = run_retrieval(
                self._encoder, self._cache, self._image_rgb, self._image_sha,
                self._query, tile_size=self._tile_size, stride=self._stride,
                batch_size=self._batch_size, progress_callback=progress,
                is_cancelled=is_cancelled, tissue_mask=self._tissue_mask,
            )
            if self._cancel_evt.is_set():
                self.cancelled.emit("Search cancelled.")
                return
            proposals = select_hotspot_proposals(
                tiles, scores,
                [[t.x0, t.y0, t.x1, t.y1] for t in tiles],
                max_proposals=self._max_proposals, min_sep_px=self._min_sep_px,
            )
            self.finished_search.emit(proposals, heatmap, tiles, evidence)
        except RetrievalCancelled:
            self.cancelled.emit("Search cancelled.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class TextToRoiDock(QWidget):
    searchRequested = Signal(str)
    thresholdChanged = Signal(float)
    clearRequested = Signal()
    saveRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ashby_text_to_roi")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(9)
        lay.setSizeConstraint(QLayout.SetMinimumSize)

        def heading(text):
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600; margin-top: 6px; margin-bottom: 2px;")
            lay.addWidget(label)

        heading("Search pathology description")
        self.query = QLineEdit()
        self.query.setPlaceholderText("e.g. mineralized deposit near a gland")
        self.query.setToolTip("Find image regions resembling the entered morphology description")
        lay.addWidget(self.query)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(lambda: self.searchRequested.emit(self.query.text().strip()))
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clearRequested)
        row.addWidget(self._search_btn)
        row.addWidget(self._clear_btn)
        lay.addLayout(row)

        heading("Patch tiling")
        self._tile_combo = QComboBox()
        for label, val in (("224 px stride 112", (224, 112)), ("128 px stride 64", (128, 64)), ("256 px stride 256", (256, 256))):
            self._tile_combo.addItem(label, val)
        self._tile_combo.setCurrentIndex(0)
        self._tile_combo.setToolTip("Smaller stride increases overlap and compute cost")
        lay.addWidget(self._tile_combo)

        heading("Similarity threshold")
        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.valueChanged.connect(self._on_slider)
        self._thresh_label = QLabel("0.50")
        self._thresh_label.setMinimumWidth(42)
        self._thresh_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_row.addWidget(self._slider, 1)
        slider_row.addWidget(self._thresh_label)
        lay.addLayout(slider_row)

        note = QLabel(LABEL)
        note.setWordWrap(True)
        lay.addWidget(note)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        lay.addWidget(self._status)

        heading("Candidates")
        self.results = QListWidget()
        self.results.setMinimumHeight(150)
        self.results.setMaximumHeight(300)
        self.results.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.results.setAlternatingRowColors(True)
        lay.addWidget(self.results)
        save = QPushButton("Save shown candidates as AI proposals")
        save.clicked.connect(self.saveRequested)
        lay.addWidget(save)
        lay.addStretch(1)
        self._default_threshold = 0.5

    def _on_slider(self, value: int):
        self._thresh_label.setText(f"{value / 100:.2f}")
        self.thresholdChanged.emit(value / 100.0)

    def threshold(self) -> float:
        return self._slider.value() / 100.0

    def tile_params(self) -> tuple[int, int]:
        return self._tile_combo.currentData()

    def set_running(self, running: bool, detail: str = ""):
        self._search_btn.setEnabled(not running)
        prefix = "Searching local CONCH…"
        if running:
            self._status.setText((prefix + " " + detail).strip())
            self._status.setVisible(True)
        elif self._status.text().startswith(prefix):
            # Leave a failure/cancellation message visible after QThread.finished.
            self._status.clear()
            self._status.setVisible(False)

    def set_error(self, msg: str):
        self._status.setText(f"Failed: {msg}" if msg else "")
        self._status.setVisible(bool(msg))

    def new_run_id(self) -> str:
        return f"ttr-{uuid.uuid4().hex[:10]}"