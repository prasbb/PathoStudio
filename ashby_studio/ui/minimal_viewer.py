"""Minimal M0.1 desktop viewer: launch napari/PySide6 and open an H&E image.

Scope-limited by design (delivery-orchestrator): this module only hosts the
napari viewer and loads a local (or synthetic demo) H&E image read-only. ROI
tools, model selector, mock AI, persistence and audio panels are intentionally
NOT here — they belong to later tasks.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QLabel, QMainWindow, QToolBar

from ..domain.images import ImageAsset, load_image


class MinimalViewer(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.setWindowTitle("PathoSuite")
        self.viewer = napari_viewer
        self.setCentralWidget(self.viewer.window._qt_viewer)  # type: ignore[attr-defined]
        self.asset: ImageAsset | None = None
        self._image_layer = None

        self._build_toolbar()
        self.statusBar().showMessage("Research-only prototype — not for diagnostic use.  'Open H&E…' to load an image.")
        self.show_synthetic_demo()

    def _build_toolbar(self):
        tb = QToolBar("File")
        self.addToolBar(tb)
        act = tb.addAction("Open H&E…")
        act.triggered.connect(self.open_dialog)
        self._asset_label = QLabel("")
        tb.addWidget(self._asset_label)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open H&E image", "", "Images (*.png *.tif *.tiff)")
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> ImageAsset:
        asset = load_image(path, modality="he")
        self._attach(asset)
        return asset

    def show_synthetic_demo(self) -> ImageAsset:
        from ..imaging.generator import make_synthetic_he_image

        data, _ = make_synthetic_he_image()
        asset = ImageAsset(asset_id="demo-synthetic", path=Path("SYNTHETIC DEMO"), modality="he", data=data)
        self._attach(asset)
        return asset

    def _attach(self, asset: ImageAsset):
        self.asset = asset
        if self._image_layer is None:
            self._image_layer = self.viewer.add_image(asset.data, name="H&E (read-only)", rgb=True, metadata={"asset_id": asset.asset_id})
        else:
            self._image_layer.data = asset.data
            self._image_layer.name = "H&E (read-only)"
        self.statusBar().showMessage(
            f"{asset.path.name}  {asset.width}x{asset.height}  sha256 {asset.sha256[:8]}  (read-only)",
            8000,
        )