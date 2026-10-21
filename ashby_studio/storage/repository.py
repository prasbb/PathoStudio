#!/usr/bin/env python3
"""M0 local JSON repository: exact save/reopen of annotations + events.

Upgraded to the versioned portable ``.ashbycase`` ZIP in M1 (case-bundle skill).
JSON working storage is acceptable for M0. Coordinates are stored as decimals in
native pixel space without precision loss on roundtrip.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from ..domain.annotations import Annotation, DEFAULT_LABEL
from ..domain.events import AuditEvent
from ..domain.images import ImageAsset, load_image

CASE_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, obj) -> None:
    tmp = tempfile.NamedTemporaryFile("w", prefix=f".{path.name}.", dir=str(path.parent), suffix=".tmp", delete=False)
    try:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        tmp.close()
        Path(tmp.name).rename(path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


class CaseRepository:
    """Read/write a M0 working case directory (kept inside a single folder)."""

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.annotations_path = self.case_dir / "annotations.json"
        self.events_path = self.case_dir / "events.json"
        self.images_manifest_path = self.case_dir / "images.json"

    # ---- write ---------------------------------------------------------------
    def save(
        self,
        *,
        case_id: str,
        annotations: list[Annotation],
        events: list[AuditEvent],
        asset: ImageAsset,
        embed_image: bool,
    ) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        img_ref = {"asset_id": asset.asset_id, "modality": asset.modality, "sha256": asset.sha256}
        if embed_image:
            from .image_assets import embed_asset
            img_ref.update(embed_asset(asset, self.case_dir))

        payload = {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": case_id,
            "image": img_ref,
            "annotations": [a.to_dict() for a in annotations],
            "actors": self._actors(annotations),
        }
        _atomic_write_json(self.annotations_path, payload)
        _atomic_write_json(self.events_path, {"schema_version": 1, "case_id": case_id, "events": [e.to_dict() for e in events]})
        _atomic_write_json(self.images_manifest_path, {"schema_version": 1, "case_id": case_id, "image": img_ref})

    def _copy_image(self, asset: ImageAsset, dest: Path) -> None:
        from PIL import Image

        Image.fromarray(asset.data).save(dest)  # original source file remains unmodified

    # ---- read ----------------------------------------------------------------
    def load(self, *, resolve_image: Path | None = None) -> tuple[str, list[Annotation], list[AuditEvent], ImageAsset | None]:
        payload = json.loads(self.annotations_path.read_text())
        if payload.get("schema_version", 0) > CASE_SCHEMA_VERSION:
            raise ValueError("annotation data was written by a newer app version; open read-only")
        case_id = payload["case_id"]
        annotations = [Annotation.from_dict(a) for a in payload.get("annotations", [])]
        events = self._load_events(case_id)

        asset: ImageAsset | None = None
        img = payload.get("image", {})
        if resolve_image is not None:
            from .image_assets import resolve_asset
            asset = resolve_asset(self.case_dir, img, resolve_image)
        return case_id, annotations, events, asset

    def _load_events(self, case_id: str) -> list[AuditEvent]:
        if not self.events_path.exists():
            return []
        data = json.loads(self.events_path.read_text())
        return [AuditEvent.from_dict(e) for e in data.get("events", [])]

    def _actors(self, annotations: list[Annotation]) -> dict:
        actors: dict[str, str] = {}
        for a in annotations:
            if a.actor_id:
                actors.setdefault(a.actor_id, "human" if a.origin == "human" else "ai")
        return actors


def new_case_id() -> str:
    return f"case-{uuid.uuid4().hex[:10]}"
