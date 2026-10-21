"""Real Qt controls/controller with a synthetic scorer and lightweight canvas."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from qtpy.QtWidgets import QMainWindow

from ashby_studio.annotations.service import AnnotationService
from ashby_studio.models.registry import ModelRegistry, MockAdapter
from ashby_studio.models.features.scan import run_scan
from ashby_studio.ui.docks import ModelDock
from ashby_studio.ui.feature_controller import FeatureController
from test_feature_models import make_run, FakeFeatureAdapter


class Canvas:
    def __init__(self):
        self.layers = []

    def add_image(self,data,**kwargs):
        return self._add(data,kwargs)

    def add_shapes(self,data,**kwargs):
        return self._add(data,kwargs)

    def _add(self,data,kwargs):
        # identity equality avoids numpy elementwise comparisons in list.remove
        class Layer:
            pass
        layer = Layer()
        layer.data = data
        layer.kwargs = kwargs
        self.layers.append(layer)
        return layer


class Window(QMainWindow):
    def __init__(self,adapter,asset):
        super().__init__()
        self.registry = ModelRegistry().add_adapter(MockAdapter()).add_adapter(adapter)
        self.model_dock = ModelDock(self.registry,self)
        self.setCentralWidget(self.model_dock)
        self.asset = asset
        self.service = AnnotationService()
        self.viewer = Canvas()
        self.history_dock = SimpleNamespace(set_events=lambda e:None)
        self._feature_controller = FeatureController(self)
        self.model_dock.modelChanged.connect(self._feature_controller.model_changed)
        self.selected = []

    def _selected_annotations(self):
        return self.selected

    def _ensure_layer_order(self):
        pass

    def _rebuild_all(self):
        pass


@pytest.fixture
def feature_window(tmp_path,qtbot):
    adapter,asset,_ = make_run(tmp_path)
    win = Window(adapter,asset)
    qtbot.addWidget(win)
    win.show()
    yield win
    controller = win._feature_controller
    if controller.worker is not None:
        controller.cancel()
        qtbot.waitUntil(lambda: controller.worker is None,timeout=10000)


def select_feature(win):
    win.model_dock.combo.setCurrentIndex(win.model_dock.combo.findData("feature:necrosis:conch"))


def test_existing_selector_and_find_workflow(feature_window,qtbot):
    w = feature_window
    c = w.model_dock.feature_controls
    assert w.model_dock.current() == "none"
    assert not c.isVisible() and not w.model_dock.run_mock.isVisible()
    select_feature(w)
    assert c.isVisible() and c.find.text() == "Find Necrosis"
    c.size.setValue(64)
    c.stride.setValue(32)
    c.find.click()
    qtbot.waitUntil(lambda:w._feature_controller.worker is None,timeout=10000)
    assert len(w._feature_controller.runs) == 1
    assert w.service.annotations == []
    c.threshold.setValue(-2)
    assert w._feature_controller.box_layer is not None
    c.heatmap.setChecked(True)
    assert w._feature_controller.heat_layer is not None
    c.save.click()
    assert len(w.service.annotations) == 1
    annotation = w.service.annotations[0]
    assert annotation.review_state == "suggested" and annotation.origin == "ai"
    assert annotation.actor_id == "feature:necrosis:conch"
    c.save.click()
    assert len(w.service.annotations) == 1
    c.threshold.setValue(10)
    assert w._feature_controller.box_layer is None
    assert w._feature_controller.heat_layer is None
    assert len(w._feature_controller.active["tiles"]) == 1
    assert len(w.service.annotations) == 1
    actions = [e.action for e in w.service.events]
    assert "feature_run_complete" in actions and "feature_threshold_change" in actions
    w.model_dock.select_none()
    assert not c.isVisible() and w._feature_controller.active is None
    select_feature(w)
    assert w._feature_controller.active is not None


def test_roi_scope_requires_selection(feature_window):
    w = feature_window
    select_feature(w)
    c = w.model_dock.feature_controls
    c.scope.setCurrentIndex(1)
    c.find.click()
    assert "exactly one polygon" in c.status.text()
    assert w._feature_controller.worker is None


def test_late_result_is_discarded_for_new_image(feature_window):
    w = feature_window
    select_feature(w)
    controller = w._feature_controller
    run = run_scan(
        w.registry.adapter("feature:necrosis:conch"),w.asset.data,w.asset.sha256,w.asset.asset_id,size=32,stride=32)
    worker = SimpleNamespace(generation=controller.generation, image_sha=w.asset.sha256, asset_id=w.asset.asset_id,
                             case_id=w.service.case_id, model_id="feature:necrosis:conch", run_id="stale",
                             service=w.service, stop=SimpleNamespace(set=lambda:None))
    controller.worker = worker
    controller.invalidate(clear_runs=True)
    controller._result(run)
    assert controller.runs == [] and w.viewer.layers == []
    assert w.service.events[-1].action == "feature_run_discarded"
    controller.worker = None


def test_case_restore_keeps_raw_scores_and_deduplicates_proposals(feature_window,qtbot,tmp_path):
    from ashby_studio.storage.feature_runs import load_runs
    w = feature_window
    select_feature(w)
    c = w.model_dock.feature_controls
    c.find.click()
    qtbot.waitUntil(lambda:w._feature_controller.worker is None,timeout=10000)
    c.threshold.setValue(-2)
    c.save.click()
    controller = w._feature_controller
    controller.save(tmp_path)
    runs = load_runs(tmp_path,w.service.case_id)
    controller.restore(runs)
    c.save.click()
    assert len(w.service.annotations) == 1
    assert c.threshold.value() == -2
    assert controller.active["proposal_ids"]


def test_missing_musk_is_disabled(tmp_path,qtbot):
    from ashby_studio.models.features.conch import FeatureAdapter
    from ashby_studio.models.features.catalog import FeaturePackage
    adapter,asset,_ = make_run(tmp_path)
    d = dict(adapter.package.manifest)
    d["encoder"] = {**d["encoder"],"model_id":"musk"}
    package = FeaturePackage(adapter.package.root,d,adapter.package.manifest_sha256)
    w = Window(FeatureAdapter(package),asset)
    qtbot.addWidget(w)
    idx = w.model_dock.combo.findData("feature:necrosis:musk")
    assert idx >= 0 and not w.model_dock.combo.model().item(idx).isEnabled()


@pytest.mark.parametrize("action",["switch_model","cancel","close"])
def test_running_job_cancellation_lifecycle(feature_window,qtbot,action):
    w = feature_window
    select_feature(w)
    entered,release = threading.Event(),threading.Event()
    adapter = w.registry.adapter("feature:necrosis:conch")
    original = adapter.score
    def delayed(patches):
        entered.set()
        release.wait(timeout=5)
        return original(patches)
    adapter.score = delayed
    c = w.model_dock.feature_controls
    c.find.click()
    try:
        qtbot.waitUntil(entered.is_set,timeout=5000)
        if action == "switch_model":
            w.model_dock.select_none()
        elif action == "cancel":
            c.cancel.click()
        else:
            w.close()
            assert w._feature_controller._closing
    finally:
        release.set()
    qtbot.waitUntil(lambda:w._feature_controller.worker is None,timeout=10000)
    assert not w._feature_controller.runs and not w.viewer.layers
    assert "feature_run_cancelled" in [e.action for e in w.service.events]
