"""M0.2 focused tests: manual polygon ROI CRUD + persistence.

Covers AC-003 (manual polygon/label edit) and AC-004 (save/reopen roundtrip).
These are pure service/repository-level tests and do not require a Qt display:

    QT_QPA_PLATFORM=offscreen PYOPENGL_PLATFORM=osmesa \
        .venv/bin/python -m pytest tests/test_annotation_workflow.py -q

All tests are UNTESTED in this session (not executed per implementation policy).
"""
from pathlib import Path

from ashby_studio.annotations.service import AnnotationService
from ashby_studio.domain.annotations import (
    DEFAULT_LABEL,
    STATE_ACCEPTED,
    STATE_MANUAL,
    STATE_SUGGESTED,
)
from ashby_studio.domain.images import ImageAsset
from ashby_studio.imaging.generator import make_synthetic_he_image
from ashby_studio.storage.repository import CaseRepository

TRIANGLE = [[10.0, 10.0], [20.0, 10.0], [15.0, 25.0]]


def _asset() -> ImageAsset:
    data, _ = make_synthetic_he_image(64, 64)
    return ImageAsset(asset_id="img-0001", path=Path("in-memory"), modality="he", data=data)


def test_create_polygon_manual_preserves_origin_state_and_label():
    svc = AnnotationService()
    ann = svc.create_polygon(
        TRIANGLE,
        label=DEFAULT_LABEL,
        image_asset_id="img-0001",
        origin="human",
    )
    assert ann.origin == "human"
    assert ann.review_state == STATE_MANUAL
    assert ann.actor_id == svc.actor_id
    assert ann.label == DEFAULT_LABEL
    assert ann.revision == 1
    assert ann.roi_id in svc.annotations
    assert svc.events[0].action == "roi_create"


def test_update_polygon_bumps_revision_and_journals():
    svc = AnnotationService()
    ann = svc.create_polygon(TRIANGLE, image_asset_id="img-0001", origin="human")
    moved = [[p[0] + 5.0, p[1] + 5.0] for p in ann.points]
    updated = svc.update_polygon(ann.roi_id, moved)
    assert updated.revision == ann.revision + 1
    assert updated.points == moved
    assert updated.review_state == STATE_MANUAL  # unchanged through geometry edit
    actions = [e.action for e in svc.events]
    assert actions.count("roi_edit") == 1
    # hash chain links consecutive events
    assert all(
        svc.events[i + 1].previous_event_sha256 == svc.events[i].event_sha256
        for i in range(len(svc.events) - 1)
    )


def test_label_change_updates_without_new_geometry_revision():
    svc = AnnotationService()
    ann = svc.create_polygon(TRIANGLE, image_asset_id="img-0001", origin="human")
    updated = svc.update_polygon(ann.roi_id, label="artifact")
    assert updated.label == "artifact"
    assert updated.points == TRIANGLE


def test_delete_removes_and_preserves_journal():
    svc = AnnotationService()
    ann = svc.create_polygon(TRIANGLE, image_asset_id="img-0001", origin="human")
    svc.delete(ann.roi_id)
    assert svc.find(ann.roi_id) is None
    assert svc.events[-1].action == "roi_delete"


def test_review_transitions_state_with_given_reason():
    svc = AnnotationService()
    ann = svc.create_polygon(TRIANGLE, image_asset_id="img-0001", origin="human")
    reviewed = svc.review(ann.roi_id, STATE_ACCEPTED, reason="confirmed")
    assert reviewed.review_state == STATE_ACCEPTED
    assert "confirmed" in reviewed.notes
    assert svc.events[-1].action == "roi_review"


def test_rename_annotation_is_audited_and_preserves_geometry():
    svc = AnnotationService()
    ann = svc.create_polygon(TRIANGLE, image_asset_id="img-0001", origin="human")
    renamed = svc.rename_annotation(ann.roi_id, "Tumor cluster")
    assert renamed.name == "Tumor cluster"
    assert renamed.points == TRIANGLE  # geometry untouched
    assert renamed.revision == ann.revision  # no geometry revision
    assert svc.events[-1].action == "roi_rename"
    assert svc.events[-1].payload["previous_name"] == ""
    # hash chain still links consecutive events
    assert all(
        svc.events[i + 1].previous_event_sha256 == svc.events[i].event_sha256
        for i in range(len(svc.events) - 1)
    )


def test_ai_edit_forks_human_revision_preserving_parent():
    svc = AnnotationService()
    ai = svc.create_polygon(
        TRIANGLE,
        image_asset_id="img-0001",
        origin="ai",
        ai_meta={"model_run_id": "mock-run-1", "revision": 1},
    )
    assert ai.review_state == STATE_SUGGESTED
    edited = svc.update_polygon(ai.roi_id, [[p[0] + 1.0, p[1] + 1.0] for p in ai.points])
    assert edited.origin == "human"
    assert edited.parent_roi_id == ai.roi_id
    assert edited.parent_revision == ai.revision
    assert edited.revision == ai.revision + 1
    # original AI record preserved intact
    assert svc.find(ai.roi_id).origin == "ai"
    assert [e.action for e in svc.events].count("roi_edit_fork_ai") == 1


def test_create_polygon_audits_name_and_provenance():
    svc = AnnotationService()
    ann = svc.create_polygon(
        TRIANGLE,
        label="candidate_calcification",
        name="T→R 0.912",
        notes="AI similarity candidate — research only. query='dense deposit'",
        image_asset_id="img-0001",
        origin="ai",
        ai_meta={"model_run_id": "ttr-abc", "revision": 1},
    )
    assert ann.name == "T→R 0.912"
    assert "research only" in ann.notes
    assert ann.model_run_id == "ttr-abc"
    ev = svc.events[-1]
    assert ev.action == "roi_create"
    assert ev.model_run_id == "ttr-abc"
    assert ev.payload["name"] == "T→R 0.912"
    assert "query='dense deposit'" in ev.payload["notes"]


def test_save_reopen_roundtrip_retains_geometry_label_and_events(tmp_path: Path):
    import json

    svc = AnnotationService()
    ann = svc.create_polygon(TRIANGLE, label="candidate_calcification", image_asset_id="img-0001", origin="human")
    svc.review(ann.roi_id, STATE_ACCEPTED)

    asset = _asset()
    repo = CaseRepository(tmp_path / "case")
    repo.save(
        case_id=svc.case_id,
        annotations=svc.annotations,
        events=svc.events,
        asset=asset,
        embed_image=True,
    )

    case_id, annotations, events, loaded_asset = repo.load(resolve_image=repo.case_dir / "images")
    assert case_id == svc.case_id
    assert len(annotations) == 1
    back = annotations[0]
    assert back.roi_id == ann.roi_id
    assert back.points == ann.points
    assert back.label == ann.label
    assert back.review_state == STATE_ACCEPTED
    assert len(events) == len(svc.events)
    assert [e.action for e in events] == [e.action for e in svc.events]
    assert loaded_asset is not None
    # JSON payload stores native-pixel decimals without lossy rounding
    payload = json.loads((repo.case_dir / "annotations.json").read_text())
    assert payload["annotations"][0]["geometry"]["points"] == TRIANGLE