"""RoI annotation domain model.

Mirrors ``contracts/annotation.schema.json`` so saved records validate against
the portable schema. Own geometry in native pixel space; provenance fields keep
human-created and AI-proposed ROIs distinct (ADR-0001 / ADR-0002).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

ORIGIN_HUMAN = "human"
ORIGIN_AI = "ai"

# review states (see ARCHITECTURE state machine)
STATE_MANUAL = "manual"
STATE_SUGGESTED = "suggested"
STATE_EDITED = "edited"
STATE_ACCEPTED = "accepted"
STATE_REJECTED = "rejected"

REVIEW_STATES = {STATE_MANUAL, STATE_SUGGESTED, STATE_EDITED, STATE_ACCEPTED, STATE_REJECTED}

# Internal palette for compatibility with older case files. Legacy entries
# are deliberately NOT shown as class choices on a new case; the user defines
# all pathology classes. New ROIs start neutrally as "unclassified".
LABELS = {
    "unclassified": (0.55, 0.55, 0.55, 0.22),
    "candidate_calcification": (220 / 255, 220 / 255, 0 / 255, 1.0),   # yellow
    "candidate_crystal": (0, 191, 255, 1.0),                            # sky blue
    "artifact": (255, 0, 0, 1.0),                                       # red
    "other": (0, 0, 255, 1.0),                                          # blue
    "uncertain": (128, 0, 128, 1.0),                                    # purple
}
DEFAULT_LABEL = "unclassified"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _points_to_list(points: Any) -> list[list[float]]:
    """Normalize a polygon vertex array to [[x, y], ...] floats."""
    out = []
    for p in points:
        out.append([float(p[0]), float(p[1])])
    return out


@dataclass
class Annotation:
    """A single polygon (or mask-reference) ROI with provenance."""

    schema_version: int = 1
    roi_id: str = ""
    image_asset_id: str = ""
    coordinate_space: str = "native_pixel"
    geometry_type: str = "polygon"
    points: list[list[float]] = field(default_factory=list)
    mask_ref: str | None = None
    label: str = DEFAULT_LABEL
    origin: str = ORIGIN_HUMAN
    review_state: str = STATE_MANUAL
    revision: int = 1
    parent_roi_id: str | None = None
    parent_revision: int | None = None
    model_run_id: str | None = None
    actor_id: str | None = None
    modality: str = "he"
    notes: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow)
    updated_at: str | None = None

    # The display name is saved with the ROI; color is a UI-only attribute.
    name: str = field(default="", repr=False)
    color: tuple[float, float, float, float] = field(default=(1, 1, 1, 1), repr=False)

    def to_dict(self) -> dict:
        geometry: dict[str, Any]
        if self.geometry_type == "polygon":
            geometry = {"type": "polygon", "points": self.points}
        else:
            geometry = {"type": "raster_mask", "mask_ref": self.mask_ref or ""}
        return {
            "schema_version": self.schema_version,
            "roi_id": self.roi_id,
            "image_asset_id": self.image_asset_id,
            "coordinate_space": self.coordinate_space,
            "geometry": geometry,
            "label": self.label,
            "origin": self.origin,
            "review_state": self.review_state,
            "revision": self.revision,
            "parent_roi_id": self.parent_roi_id,
            "parent_revision": self.parent_revision,
            "model_run_id": self.model_run_id,
            "actor_id": self.actor_id,
            "modality": self.modality,
            "name": self.name,
            "notes": self.notes,
            "evidence_ids": list(self.evidence_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        geom = d.get("geometry", {})
        gtype = geom.get("type", "polygon")
        points = _points_to_list(geom.get("points", [])) if gtype == "polygon" else []
        mask_ref = geom.get("mask_ref") if gtype == "raster_mask" else None
        return cls(
            schema_version=d.get("schema_version", 1),
            roi_id=d["roi_id"],
            image_asset_id=d["image_asset_id"],
            coordinate_space=d.get("coordinate_space", "native_pixel"),
            geometry_type=gtype,
            points=points,
            mask_ref=mask_ref,
            label=d.get("label", DEFAULT_LABEL),
            origin=d.get("origin", ORIGIN_HUMAN),
            review_state=d.get("review_state", STATE_MANUAL),
            revision=d.get("revision", 1),
            parent_roi_id=d.get("parent_roi_id"),
            parent_revision=d.get("parent_revision"),
            model_run_id=d.get("model_run_id"),
            actor_id=d.get("actor_id"),
            modality=d.get("modality", "he"),
            name=d.get("name", ""),
            notes=d.get("notes", ""),
            evidence_ids=list(d.get("evidence_ids", [])),
            created_at=d.get("created_at", utcnow()),
            updated_at=d.get("updated_at"),
        )

    def as_napari_shape(self) -> list[list[float]]:
        return [[p[1], p[0]] for p in self.points]  # row,col for napari


def is_ai(annotation: Annotation) -> bool:
    return annotation.origin == ORIGIN_AI