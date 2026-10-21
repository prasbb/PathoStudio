"""Feature packages, numerical semantics, bounded scans and portable evidence."""
import copy
import json
import socket
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ashby_studio.models.features.catalog import read_package, register_feature_models, digest
from ashby_studio.models.features.conch import FeatureAdapter, linear_scores, apply_adaptation
from ashby_studio.models.features.scan import run_scan, scan_tiles, visible_tiles, score_heatmap, ScoreCache, ScanCancelled
from ashby_studio.models.registry import ModelRegistry
from ashby_studio.domain.images import ImageAsset
from ashby_studio.annotations.service import AnnotationService
from ashby_studio.storage.repository import CaseRepository
from ashby_studio.storage.feature_runs import save_runs, load_runs


def make_package(root, feature="necrosis"):
    folder = root / feature / "conch"
    folder.mkdir(parents=True)
    (folder / "backbone.bin").write_bytes(b"synthetic backbone")
    (folder / "adaptation.safetensors").write_bytes(b"synthetic adaptation")
    d = {
        "kind": "ashby-feature-partial-v1", "feature": feature,
        "encoder": {"model_id": "conch", "model_version": "synthetic-test",
                    "weight_sha256": digest(b"synthetic backbone"), "preprocessing": "synthetic RGB",
                    "feature_options": "encode_image(proj_contrast=False,normalize=False)"},
        "feature_dimension": 512, "feature_normalization": "l2", "score_type": "uncalibrated_logit",
        "head": {"coef": [0.0]*512, "intercept": -1.0}, "backbone_filename": "backbone.bin",
        "adaptation": {"filename": "adaptation.safetensors", "sha256": digest(b"synthetic adaptation")},
        "training": {"trainable_parameter_names": ["weight"]},
    }
    (folder / "model.json").write_text(json.dumps(d))
    return read_package(folder / "model.json")


class FakeFeatureAdapter:
    def __init__(self, package):
        self.package = package
        self.device = "cpu"
        self.loads = self.calls = self.unloads = 0

    def metadata(self):
        from ashby_studio.models.registry import ModelMetadata
        return ModelMetadata(self.package.id, self.package.title, "available", "Synthetic test adapter",
                             ["feature_patch_scoring"], "logit", "none", "synthetic-test")

    def prepare(self, device):
        self.package.verify()
        self.device = device

    def evidence(self):
        return {"model_id": self.package.id, "model_version": "synthetic-test",
                "weight_sha256": self.package.manifest["encoder"]["weight_sha256"],
                "manifest_sha256": self.package.manifest_sha256, "adaptation_sha256": self.package.manifest["adaptation"]["sha256"],
                "preprocessing": "synthetic RGB", "device": self.device}

    def load(self, device):
        self.loads += 1

    def score(self, patches):
        self.calls += 1
        return np.array([float(p.mean())/127.5 - 1 for p in patches])

    def unload(self):
        self.unloads += 1


def make_run(root):
    adapter = FakeFeatureAdapter(make_package(root))
    image = np.zeros((32,64,3), dtype=np.uint8)
    image[:,32:] = 255
    asset = ImageAsset("synthetic", Path("SYNTHETIC"), "he", image)
    run = run_scan(adapter, image, asset.sha256, asset.asset_id, size=32, stride=32)
    return adapter, asset, run


def test_catalog_is_lazy_and_blocks_bad_packages(tmp_path, monkeypatch):
    make_package(tmp_path)
    monkeypatch.setattr(FeatureAdapter, "load", lambda *args: pytest.fail("model initialized during discovery"))
    reg = ModelRegistry()
    register_feature_models(reg, tmp_path)
    assert reg.all_entries()[0].id == "none"
    assert reg.get("feature:necrosis:conch").score_type == "logit"
    path = tmp_path / "necrosis/conch/model.json"
    d = json.loads(path.read_text())
    d["backbone_filename"] = "../../outside.bin"
    path.write_text(json.dumps(d))
    reg = ModelRegistry()
    register_feature_models(reg, tmp_path)
    assert reg.get("feature:necrosis:conch").status == "disabled"


def test_package_checks_both_weights_and_manifest(tmp_path):
    package = make_package(tmp_path)
    package.verify()
    package.backbone.write_bytes(b"modified backbone")
    with pytest.raises(ValueError, match="backbone checksum"):
        package.verify()
    package.backbone.write_bytes(b"synthetic backbone")
    package.adaptation.write_bytes(b"modified")
    with pytest.raises(ValueError, match="adaptation checksum"):
        package.verify()
    (package.root / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        package.verify()


def test_head_is_normalized_linear_logit_not_probability():
    np.testing.assert_allclose(linear_scores([[3,4],[-3,-4]], [2,0], -2), [-0.8,-3.2], atol=1e-6)
    with pytest.raises(ValueError):
        linear_scores([[0,0]], [2,0], 0)


def test_adaptation_replaces_parameters_and_rejects_mismatch():
    import torch
    model = torch.nn.Linear(2,1, bias=False)
    model.weight.data.fill_(4)
    state = {"weight": torch.tensor([[2.0,3.0]])}
    apply_adaptation(model, state, ["weight"])
    np.testing.assert_allclose(model.weight.detach().numpy(), [[2,3]])
    with pytest.raises(ValueError, match="keys"):
        apply_adaptation(model, state, ["missing"])
    with pytest.raises(ValueError, match="Incompatible"):
        apply_adaptation(model, {"weight": torch.zeros(1,3)}, ["weight"])


def test_scan_filter_negative_heatmap_and_source_unchanged(tmp_path):
    _,asset,run = make_run(tmp_path)
    original = asset.data.copy()
    assert [t["score"] for t in run["tiles"]] == [-1,1]
    assert len(visible_tiles(run,0)) == 1
    assert len(visible_tiles(run,-2)) == 2
    assert len(run["tiles"]) == 2
    heat,scale = score_heatmap(run)
    assert heat[0,0] == -1 and heat[0,40] == 1 and scale == 1
    np.testing.assert_array_equal(original,asset.data)
    assert run["provenance"]["score_type"] == "logit"
    assert run["provenance"]["calibration_id"] is None


def test_scan_cache_isolated_by_package_and_reuses_scores(tmp_path):
    a,asset,_ = make_run(tmp_path)
    cache = ScoreCache()
    options = dict(size=32,stride=32,cache=cache)
    run_scan(a,asset.data,asset.sha256,asset.asset_id,**options)
    calls = a.calls
    run_scan(a,asset.data,asset.sha256,asset.asset_id,**options)
    assert a.calls == calls
    b = FakeFeatureAdapter(make_package(tmp_path,"mucin"))
    run_scan(b,asset.data,asset.sha256,asset.asset_id,**options)
    assert b.calls == 1
    # A newly trained adaptation for the same feature must not reuse its scores.
    path = a.package.root / "model.json"
    manifest = json.loads(path.read_text())
    a.package.adaptation.write_bytes(b"new adaptation")
    manifest["adaptation"]["sha256"] = digest(b"new adaptation")
    path.write_text(json.dumps(manifest))
    newer = FakeFeatureAdapter(read_package(path))
    run_scan(newer,asset.data,asset.sha256,asset.asset_id,**options)
    assert newer.calls == 1


def test_roi_scope_bounds_and_budget():
    tiles = scan_tiles(200,200,20,20, [[50,50],[90,50],[90,90],[50,90]])
    assert len(tiles) == 4
    assert tiles[0] == [50,50,70,70]
    with pytest.raises(ValueError, match="exceeds"):
        scan_tiles(100000,100000,32,16)
    with pytest.raises(ValueError, match="stride"):
        scan_tiles(200,200,20,40)


def test_cancellation_unloads_model_and_has_no_completed_run(tmp_path):
    adapter,asset,_ = make_run(tmp_path)
    adapter.calls = 0
    stopped = []
    def progress(done,total):
        if done:
            stopped.append(True)
    with pytest.raises(ScanCancelled):
        run_scan(adapter,asset.data,asset.sha256,asset.asset_id,size=32,stride=32,batch_size=1,
                 progress=progress,cancelled=lambda: bool(stopped))
    assert adapter.calls == 1 and adapter.unloads >= 2


def test_offline_scan_and_discovery(tmp_path, monkeypatch):
    def denied(*args,**kwargs):
        pytest.fail("Network access")
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setattr(socket,"getaddrinfo",denied)
    make_run(tmp_path)
    register_feature_models(ModelRegistry(),tmp_path)


def test_portable_feature_records_and_reviews(tmp_path):
    _,asset,run = make_run(tmp_path / "models")
    service = AnnotationService()
    run_id = run["provenance"]["run_id"]
    ann = service.create_polygon([[0,0],[32,0],[32,32],[0,32]], image_asset_id=asset.asset_id,
                                 origin="ai", name="Necrosis candidate",
                                 ai_meta={"model_run_id":run_id,"actor_id":"feature:necrosis:conch",
                                          "evidence_ids":[run_id+":0"], "score":-1})
    run["proposal_ids"]["0"] = ann.roi_id
    assert service.events[-1].actor_type == "ai" and ann.review_state == "suggested"
    service.review(ann.roi_id,"accepted")
    assert service.events[-1].actor_type == "human" and ann.origin == "ai"
    assert service.events[-1].previous_event_sha256 == service.events[-2].event_sha256
    assert service.events[-2].event_sha256 is not None
    repo = CaseRepository(tmp_path / "case")
    repo.save(case_id=service.case_id,annotations=service.annotations,events=service.events,asset=asset,embed_image=True)
    save_runs(repo.case_dir,service.case_id,[run])
    archive = tmp_path / "saved.ashbycase"
    with zipfile.ZipFile(archive,"w") as z:
        for p in repo.case_dir.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(repo.case_dir))
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp_path / "fresh")
    reopened = CaseRepository(tmp_path / "fresh")
    cid,anns,events,image = reopened.load(resolve_image=reopened.case_dir / "images")
    back = load_runs(reopened.case_dir,cid)
    assert back == [run] and anns[0].evidence_ids == ann.evidence_ids
    assert image.sha256 == asset.sha256
    np.testing.assert_array_equal(image.data,asset.data)
    assert len(back[0]["tiles"]) == 2
    image.path.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="checksum"):
        reopened.load(resolve_image=reopened.case_dir / "images")


def test_sidecar_tampering_and_legacy_cases(tmp_path):
    _,asset,run = make_run(tmp_path / "models")
    assert load_runs(tmp_path,"case") == []
    save_runs(tmp_path,"case",[run])
    path = tmp_path / "feature_runs.json"
    d = json.loads(path.read_text())
    d["payload"]["runs"][0]["tiles"][0]["score"] = 999
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="checksum"):
        load_runs(tmp_path,"case")


def test_existing_provenance_contract_and_wrong_image_rejection(tmp_path):
    import jsonschema
    from ashby_studio.storage.feature_runs import validate_runs_for_asset
    _,asset,run = make_run(tmp_path)
    schema = json.loads((Path(__file__).resolve().parents[1] / "contracts/model-run.schema.json").read_text())
    jsonschema.validate(run["provenance"],schema)
    feature_schema = json.loads((Path(__file__).resolve().parents[1] / "contracts/feature-run.schema.json").read_text())
    # Resolve the local contract in memory; never fetch a schema over the network.
    feature_schema["properties"]["provenance"] = schema
    jsonschema.validate(run,feature_schema)
    validate_runs_for_asset([run],asset)
    asset.asset_id = "different-image"
    with pytest.raises(ValueError,match="do not match"):
        validate_runs_for_asset([run],asset)


def test_existing_case_can_resave_without_copying_onto_itself(tmp_path):
    _,asset,run = make_run(tmp_path / "models")
    repo = CaseRepository(tmp_path / "case")
    repo.save(case_id="case",annotations=[],events=[],asset=asset,embed_image=True)
    _,_,_,loaded = repo.load(resolve_image=repo.case_dir / "images")
    repo.save(case_id="case",annotations=[],events=[],asset=loaded,embed_image=True)
