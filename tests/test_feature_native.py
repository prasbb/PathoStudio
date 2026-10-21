"""Opt-in native macOS viewer integration; avoids unsupported offscreen OpenGL."""
import os
from pathlib import Path
import socket

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ASHBY_TEST_NATIVE_GUI") != "1", reason="Native desktop integration is opt-in")


def test_native_feature_search_save_reopen(tmp_path,monkeypatch,qtbot):
    # Isolate napari preferences/caches from the user's application settings.
    monkeypatch.setenv("NAPARI_CONFIG", str(tmp_path / "settings.yaml"))
    import napari.utils._platformdirs as dirs
    for name in ("user_cache_dir","user_config_dir","user_data_dir","user_log_dir","user_state_dir"):
        monkeypatch.setattr(dirs,name,lambda: str(tmp_path / "napari"))
    import napari
    from qtpy.QtWidgets import QFileDialog, QMessageBox
    from ashby_studio.models.registry import ModelRegistry
    from ashby_studio.ui.main_window import MainWindow
    from test_feature_models import make_run
    adapter,asset,_ = make_run(tmp_path / "models")
    def deny(*args,**kwargs):
        raise AssertionError("Network access in native feature workflow")
    monkeypatch.setattr(socket.socket,"connect",deny)
    monkeypatch.setattr(socket,"getaddrinfo",deny)
    monkeypatch.setattr(QMessageBox,"critical",lambda *args: pytest.fail(str(args[2])))
    viewer = napari.Viewer(show=False)
    win = MainWindow(viewer,registry=ModelRegistry().add_adapter(adapter))
    qtbot.addWidget(win)
    try:
        win.show()
        win._set_asset(asset)
        win._show_panel("model")
        win.model_dock.combo.setCurrentIndex(win.model_dock.combo.findData(adapter.package.id))
        controls = win.model_dock.feature_controls
        controls.find.click()
        qtbot.waitUntil(lambda:win._feature_controller.worker is None,timeout=10000)
        assert len(win._feature_controller.runs) == 1
        controls.threshold.setValue(-2)
        controls.heatmap.setChecked(True)
        controls.save.click()
        assert len(win.service.annotations) == 1
        assert win.service.annotations[0].review_state == "suggested"
        saved_id = win.service.annotations[0].roi_id
        filename = str(tmp_path / "feature.ashbycase")
        monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *args: (filename,""))
        win.save_case()
        assert Path(filename).is_file()
        win._feature_controller.invalidate(clear_runs=True)
        monkeypatch.setattr(QFileDialog,"getOpenFileName",lambda *args: (filename,""))
        win.reopen_case()
        assert win.asset.sha256 == asset.sha256
        assert win.service.annotations[0].roi_id == saved_id
        assert len(win._feature_controller.runs) == 1
        assert win._feature_controller.box_layer is not None
        assert win._feature_controller.heat_layer is not None
        controls.save.click()
        assert len(win.service.annotations) == 1
        qtbot.wait(100)
        if os.environ.get("ASHBY_GUI_SCREENSHOT"):
            win.grab().save(os.environ["ASHBY_GUI_SCREENSHOT"])
    finally:
        win.close()
        viewer.close()
