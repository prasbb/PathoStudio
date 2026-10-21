"""Coordinates feature jobs, overlays and case records without expanding MainWindow."""
from __future__ import annotations

import uuid
import threading

import numpy as np
from qtpy.QtCore import QObject, QThread, Signal, QEvent

from ..models.features.scan import run_scan, visible_tiles, score_heatmap, ScoreCache, ScanCancelled
from ..models.features.catalog import FEATURE_NAMES
from ..storage.feature_runs import save_runs


class FeatureWorker(QThread):
    result = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, int)

    def __init__(self, adapter, image, image_sha, asset_id, options, cache, parent=None):
        super().__init__(parent)
        self.adapter, self.image, self.image_sha, self.asset_id = adapter, image, image_sha, asset_id
        self.options, self.cache = options, cache
        self.stop = threading.Event()

    def run(self):
        try:
            result = run_scan(self.adapter, self.image, self.image_sha, self.asset_id,
                              **self.options, cache=self.cache, progress=self.progress.emit,
                              cancelled=self.stop.is_set)
            if self.stop.is_set():
                raise ScanCancelled()
            self.result.emit(result)
        except ScanCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.adapter.unload()


class FeatureController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.controls = window.model_dock.feature_controls
        self.runs = []
        self.active = None
        self.worker = None
        self.generation = 0
        self.cache = ScoreCache()
        self.box_layer = None
        self.heat_layer = None
        self._closing = False
        c = self.controls
        c.runRequested.connect(self.start)
        c.cancelRequested.connect(self.cancel)
        c.thresholdChanged.connect(self.filter_changed)
        c.overlayChanged.connect(self.render)
        c.saveRequested.connect(self.create_proposals)
        c.runSelected.connect(self.select_run)
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Close and self.worker is not None:
            self._closing = True
            self.cancel()
            self.controls.status.setText("Stopping feature search before closing…")
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def audit(self, action, run_id=None, *, actor="human", model_id=None, payload=None):
        self.window.service.record_model_event(action, run_id=run_id, actor_type=actor,
                                               model_id=model_id, payload=payload or {})
        self.window.history_dock.set_events(self.window.service.events)

    def model_changed(self, model_id):
        self.invalidate()
        meta = self.window.registry.get(model_id)
        feature = "feature_patch_scoring" in meta.tasks
        self.controls.setVisible(feature)
        if feature:
            self.controls.find.setText("Find " + meta.ui_name.split(" — ")[0])
            self.controls.find.setEnabled(meta.status == "available" and self.worker is None)
            self.audit("feature_model_select", model_id=model_id, payload={"model_id": model_id})
        self.refresh_runs()

    def refresh_runs(self, active_id=None):
        selected = self.window.model_dock.current()
        runs = [r for r in self.runs if r["provenance"]["model_id"] == selected]
        self.controls.set_runs(runs, active_id)
        self.select_run(self.controls.runs.currentData() or "")

    def select_run(self, run_id):
        asset = self.window.asset
        self.active = next((r for r in self.runs if r["provenance"]["run_id"] == run_id
                            and r["provenance"]["model_id"] == self.window.model_dock.current()
                            and asset is not None and r["image_sha256"] == asset.sha256
                            and r["provenance"]["image_asset_ids"] == [asset.asset_id]), None)
        if self.active:
            self.controls.threshold.blockSignals(True)
            self.controls.threshold.setValue(self.active["display_threshold"])
            self.controls.threshold.blockSignals(False)
        self.controls.save.setEnabled(self.active is not None and self.worker is None)
        self.render()

    def start(self):
        w,c = self.window,self.controls
        if self.worker is not None:
            c.status.setText("Wait for the current search to stop.")
            return
        if w.asset is None:
            c.status.setText("Open an image first.")
            return
        adapter = w.registry.adapter(w.model_dock.current())
        if adapter is None or "feature_patch_scoring" not in adapter.metadata().tasks:
            c.status.setText("Choose an available feature model.")
            return
        polygon = None
        if c.scope.currentData() == "roi":
            selected = w._selected_annotations()
            if len(selected) != 1 or selected[0].geometry_type != "polygon":
                c.status.setText("Select exactly one polygon ROI.")
                return
            polygon = [list(p) for p in selected[0].points]
        self.generation += 1
        token = self.generation
        run_id = f"feature-{uuid.uuid4().hex}"
        options = dict(size=c.size.value(), stride=c.stride.value(), polygon=polygon,
                       device=c.device.currentData(), threshold=c.threshold.value(), run_id=run_id)
        worker = FeatureWorker(adapter, w.asset.data, w.asset.sha256, w.asset.asset_id,
                               options, self.cache, self)
        worker.case_id = w.service.case_id
        worker.generation = token
        worker.model_id = adapter.metadata().id
        worker.run_id = run_id
        # Store the originating service so invalidated jobs finish in the right journal.
        worker.service = w.service
        self.worker = worker
        worker.result.connect(self._result)
        worker.failed.connect(self._error)
        worker.cancelled.connect(self._cancelled)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._done)
        c.set_running(True)
        c.progress.setRange(0, 0)
        c.status.setText("Validating local weights and preparing search…")
        self.audit("feature_run_start", run_id, model_id=worker.model_id,
                   payload={"model_id": worker.model_id, "image_asset_id": w.asset.asset_id, **options})
        worker.start()

    def _valid(self, worker):
        w = self.window
        return (worker is self.worker and worker.generation == self.generation
                and w.asset is not None and w.asset.sha256 == worker.image_sha
                and w.asset.asset_id == worker.asset_id and w.service.case_id == worker.case_id
                and w.model_dock.current() == worker.model_id)

    def _progress(self, done, total):
        if self.worker is not None and self._valid(self.worker):
            self.controls.progress.setRange(0, total)
            self.controls.progress.setValue(done)
            self.controls.status.setText(f"Searching {done}/{total} patches…")

    def _result(self, run):
        worker = self.worker
        if worker is None:
            return
        if not self._valid(worker):
            self._record_end(worker, "feature_run_discarded")
            return
        self.runs.append(run)
        self._record_end(worker, "feature_run_complete", {"provenance": run["provenance"], "tile_count": len(run["tiles"])})
        self.refresh_runs(run["provenance"]["run_id"])
        self.controls.status.setText("Search complete. Save Case to retain all scores and proposals.")

    def _record_end(self, worker, action, payload=None):
        worker.service.record_model_event(action, run_id=worker.run_id, actor_type="ai",
                                           model_id=worker.model_id, payload=payload or {})
        if worker.service is self.window.service:
            self.window.history_dock.set_events(worker.service.events)

    def _error(self, message):
        if self.worker is not None:
            self._record_end(self.worker, "feature_run_failed", {"reason": "local inference error"})
            if self._valid(self.worker):
                self.controls.status.setText("Search failed: " + message)

    def _cancelled(self):
        if self.worker is not None:
            self._record_end(self.worker, "feature_run_cancelled")
            if self._valid(self.worker):
                self.controls.status.setText("Search cancelled; previous completed results retained.")

    def _done(self):
        worker = self.worker
        self.worker = None
        self.controls.set_running(False)
        if self.controls.progress.maximum() == 0:
            self.controls.progress.setRange(0, 1)
            self.controls.progress.setValue(0)
        self.controls.find.setEnabled(self.window.registry.get(self.window.model_dock.current()).status == "available")
        self.controls.save.setEnabled(self.active is not None)
        if worker is not None:
            worker.deleteLater()
        if self._closing:
            self.window.close()

    def cancel(self):
        if self.worker is not None:
            self.worker.stop.set()

    def invalidate(self, clear_runs=False):
        self.cancel()
        self.generation += 1
        self.active = None
        if clear_runs:
            self.runs = []
        self.remove_layers()
        self.controls.summary.setText("No feature search results.")
        self.controls.save.setEnabled(False)
        self.controls.status.clear()

    def remove_layers(self):
        for attr in ("box_layer", "heat_layer"):
            layer = getattr(self, attr)
            if layer is not None and layer in self.window.viewer.layers:
                self.window.viewer.layers.remove(layer)
            setattr(self, attr, None)

    def filter_changed(self, threshold):
        if self.active is not None:
            old = self.active["display_threshold"]
            self.active["display_threshold"] = threshold
            self.audit("feature_threshold_change", self.active["provenance"]["run_id"],
                       payload={"previous_threshold": old, "threshold": threshold, "score_type": "logit"})
        self.render()

    def render(self):
        self.remove_layers()
        run,c = self.active,self.controls
        if run is None:
            c.summary.setText("No feature search results for this selection.")
            return
        tiles = visible_tiles(run, c.threshold.value())
        c.summary.setText(f"{len(tiles)} candidate patches shown · {len(run['tiles'])} raw scores retained.\n"
                          "Overlapping boxes are deduplicated for display; scores are uncalibrated.")
        viewer = self.window.viewer
        feature = FEATURE_NAMES.get(run["feature"], run["feature"])
        if c.heatmap.isChecked():
            heat, scale = score_heatmap(run)
            heat[heat < c.threshold.value()] = np.nan
            finite = heat[np.isfinite(heat)]
            if finite.size:
                lo,hi = float(finite.min()),float(finite.max())
                self.heat_layer = viewer.add_image(heat, name=f"{feature} patch scores (research only)",
                                                  colormap="magma", opacity=0.45, scale=(scale,scale),
                                                  contrast_limits=(lo, hi if hi>lo else lo+1))
        if tiles and c.boxes.isChecked():
            shapes = [np.array([[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]]], dtype=float)
                      for b in [t["box"] for t in tiles]]
            self.box_layer = viewer.add_shapes(shapes, shape_type="polygon", name=f"{feature} candidates (AI suggested)",
                                               edge_color="orange", face_color=[1,0.6,0,0.08], edge_width=2,
                                               properties={"score": [t["score"] for t in tiles]})
            self.box_layer.editable = False
        self.window._ensure_layer_order()

    def create_proposals(self):
        run,w = self.active,self.window
        if (run is None or self.worker is not None or w.asset is None or run["image_sha256"] != w.asset.sha256
                or run["provenance"]["image_asset_ids"] != [w.asset.asset_id]):
            return
        feature = FEATURE_NAMES.get(run["feature"], run["feature"])
        saved = 0
        for tile in visible_tiles(run, self.controls.threshold.value()):
            tid = tile["tile_id"]
            if tid in run["proposal_ids"]:
                continue
            x0,y0,x1,y1 = tile["box"]
            ann = w.service.create_polygon(
                [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], label="unclassified",
                name=f"{feature} candidate · {tile['score']:.3f}",
                image_asset_id=w.asset.asset_id, origin="ai", modality=w.asset.modality,
                notes=f"Research-only {feature} patch suggestion; uncalibrated logit={tile['score']}. "
                      "Not a segmentation or diagnosis. Chemical identity unconfirmed.",
                ai_meta={"model_run_id": run["provenance"]["run_id"], "actor_id": run["provenance"]["model_id"],
                         "evidence_ids": [f"{run['provenance']['run_id']}:{tid}"], "feature": run["feature"],
                         "tile_id": tid, "score": tile["score"], "score_type": "logit",
                         "threshold": self.controls.threshold.value(), "box": tile["box"]})
            run["proposal_ids"][tid] = ann.roi_id
            saved += 1
        self.audit("feature_proposals_create", run["provenance"]["run_id"], payload={"created": saved})
        w._rebuild_all()
        self.controls.status.setText(f"Created {saved} AI suggestions. Review in ROI Properties; Save Case to keep them.")

    def save(self, case_dir):
        save_runs(case_dir, self.window.service.case_id, self.runs)

    def restore(self, runs):
        self.invalidate(clear_runs=True)
        self.runs = runs
        self.refresh_runs()
