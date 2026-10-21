"""Annotation service: single in-memory source of truth for ROIs and events.

The UI talks to this service (not direct widget state) so save/undo/audit can
later be layered in cleanly (app-bootstrap skill). M0 keeps a lightweight
list of :class:`Annotation` plus an append-only :class:`AuditEvent` journal.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..domain.annotations import (
    Annotation, DEFAULT_LABEL, STATE_ACCEPTED, STATE_REJECTED, utcnow,
)
from ..domain.events import AuditEvent


@dataclass
class AnnotationState:
    """Mutable snapshot captured before an edit to enable compensating events."""

    roi_id: str
    previous_revision: int
    previous_state: str
    previous_points: list[list[float]]
    previous_label: str


@dataclass
class AnnotationService:
    case_id: str = field(default_factory=lambda: f"case-{uuid.uuid4().hex[:10]}")
    actor_id: str = "h-0001"  # pseudonymous reviewer identity (M0 single-user)
    annotations: list[Annotation] = field(default_factory=list)
    events: list[AuditEvent] = field(default_factory=list)
    _last_event_hash: str | None = None

    # -- annotation CRUD ------------------------------------------------------
    def find(self, roi_id: str) -> Annotation | None:
        for a in self.annotations:
            if a.roi_id == roi_id:
                return a
        return None

    def create_polygon(
        self,
        points: list[list[float]],
        *,
        label: str = DEFAULT_LABEL,
        image_asset_id: str,
        modality: str = "he",
        origin: str = "human",
        name: str | None = None,
        notes: str = "",
        ai_meta: dict | None = None,
    ) -> Annotation:
        roi_id = str(uuid.uuid4())[:13]
        ann = Annotation(
            roi_id=roi_id,
            image_asset_id=image_asset_id,
            points=[[float(p[0]), float(p[1])] for p in points],
            label=label,
            origin=origin,
            review_state="manual" if origin == "human" else "suggested",
            modality=modality,
            name=name or "",
            notes=notes,
            actor_id=self.actor_id if origin == "human" else None,
        )
        if ai_meta:
            ann.model_run_id = ai_meta.get("model_run_id")
            if origin == "ai" and ai_meta.get("actor_id"):
                ann.actor_id = ai_meta["actor_id"]
                ann.evidence_ids = list(ai_meta.get("evidence_ids", []))
            if ai_meta.get("revision"):
                ann.revision = int(ai_meta["revision"])
        self.annotations.append(ann)
        self._emit(
            "roi_create",
            ann,
            payload={
                "origin": ann.origin,
                "name": ann.name,
                "notes": ann.notes,
                "model_run_id": ann.model_run_id,
                **({"ai_metadata": dict(ai_meta)} if ai_meta else {}),
            },
        )
        return ann

    def update_polygon(self, roi_id: str, points: list[list[float]] | None = None, *, label: str | None = None) -> Annotation:
        """Apply a human edit, creating a new revision of the ROI.

        For an AI-proposed ROI the original AI record is preserved; a derived
        human revision is written referencing the parent (ADR-0002)."""
        ann = self.find(roi_id)
        if ann is None:
            raise KeyError(roi_id)
        prev_rev = ann.revision
        prev_state = ann.review_state
        if ann.origin == "ai":
            # fork: new human-owned revision referencing the AI parent
            derived = Annotation(
                roi_id=roi_id,  # same id; revision chain distinguishes them
                image_asset_id=ann.image_asset_id,
                points=list(ann.points),
                label=ann.label,
                origin="human",
                review_state="edited",
                revision=prev_rev + 1,
                parent_roi_id=ann.roi_id,
                parent_revision=prev_rev,
                model_run_id=ann.model_run_id,
                actor_id=self.actor_id,
                modality=ann.modality,
                notes=ann.notes,
                name=ann.name,
                created_at=ann.created_at,
            )
            derived.updated_at = utcnow()
            self.annotations.append(derived)
            emissions = [("roi_edit_fork_ai", derived)]
            ann = derived
            roi_id = ann.roi_id
        else:
            ann.review_state = "edited" if ann.review_state == "manual" else ann.review_state
            emissions = [("roi_edit", ann)]

        if points is not None:
            ann.points = [[float(p[0]), float(p[1])] for p in points]
        if label is not None:
            ann.label = label
        ann.revision = prev_rev + 1
        ann.updated_at = utcnow()

        for action, target in emissions:
            self._emit(
                action,
                target,
                payload={"previous_revision": prev_rev, "new_revision": ann.revision},
            )
        return ann

    def delete(self, roi_id: str) -> None:
        ann = self.find(roi_id)
        if ann is None:
            raise KeyError(roi_id)
        self.annotations = [a for a in self.annotations if a.roi_id != roi_id]
        self._emit("roi_delete", ann, human_action=True)

    def identify(self, roi_id: str, actor_id: str) -> None:
        """Record who created/owns an ROI (provenance)."""
        ann = self.find(roi_id)
        if ann is not None:
            ann.actor_id = actor_id
            self._emit("roi_identified", ann, payload={"actor_id": actor_id})

    # -- review actions ---------------------------------------------------------
    def review(self, roi_id: str, state: str, *, reason: str = "") -> Annotation:
        """Record an explicit human review decision with an audit event.

        A second identical decision is a no-op, avoiding meaningless events.
        The caller decides which annotation origins are eligible for review.
        """
        if state not in {STATE_ACCEPTED, STATE_REJECTED}:
            raise ValueError("Review state must be 'accepted' or 'rejected'")
        ann = self.find(roi_id)
        if ann is None:
            raise KeyError(roi_id)
        if ann.review_state == state and not reason:
            return ann
        prev = ann.review_state
        ann.review_state = state
        ann.updated_at = utcnow()
        if reason:
            ann.notes = (ann.notes + "\n" + reason).strip()
        self._emit(
            "roi_review", ann,
            payload={"previous_state": prev, "new_state": state, "reason": reason},
            human_action=True,
        )
        return ann

    def update_notes(self, roi_id: str, notes: str) -> Annotation:
        """Save pathologist notes with an auditable, non-geometry mutation."""
        ann = self.find(roi_id)
        if ann is None:
            raise KeyError(roi_id)
        if not isinstance(notes, str):
            raise TypeError("ROI notes must be text")
        if len(notes) > 20_000:
            raise ValueError("ROI notes exceed 20,000 characters")
        if ann.notes == notes:
            return ann
        previous = ann.notes
        ann.notes = notes
        ann.updated_at = utcnow()
        self._emit(
            "roi_notes_update", ann,
            payload={"previous_notes": previous, "new_notes": notes},
            human_action=True,
        )
        return ann

    def record_point_count(
        self,
        *,
        count: int,
        points: list[list[float]] | None = None,
        notes: str = "",
        modality: str = "he",
    ) -> AuditEvent:
        """Audited manual point-counter entry (mineralization foci count).

        Appends a ``count_record`` event to the journal without fabricating an
        ROI. Preserves the hash chain and human actor attribution.
        """
        ev = AuditEvent(
            event_id=f"ev-{uuid.uuid4().hex[:12]}",
            case_id=self.case_id,
            actor_type="human",
            actor_id=self.actor_id,
            action="count_record",
            target_id="",
            model_run_id=None,
            new_revision=None,
            payload={
                "count": int(count),
                "points": points or [],
                "notes": notes,
                "modality": modality,
            },
            previous_event_sha256=self._last_event_hash,
        )
        self._last_event_hash = ev.event_sha256
        self.events.append(ev)
        return ev

    def all_ai(self) -> list[Annotation]:
        return [a for a in self.annotations if a.origin == "ai"]

    def all_human(self) -> list[Annotation]:
        return [a for a in self.annotations if a.origin == "human"]

    def rename_annotation(self, roi_id: str, name: str) -> Annotation:
        """Persist an audited ROI display name without changing geometry."""
        ann = self.find(roi_id)
        if ann is None:
            raise KeyError(roi_id)
        name = (name or "").strip()
        if not name or len(name) > 200:
            raise ValueError("ROI name must be 1-200 characters")
        previous = ann.name
        if previous == name:
            return ann
        ann.name = name
        ann.updated_at = utcnow()
        self._emit(
            "roi_rename",
            ann,
            payload={"previous_name": previous, "new_name": name},
            human_action=True,
        )
        return ann

    # -- journal --------------------------------------------------------------
    def record_model_event(self, action, *, run_id=None, actor_type="human", model_id=None, payload=None):
        """Record model lifecycle and display/review actions without fabricating an ROI."""
        if actor_type not in {"human", "ai", "system"}:
            raise ValueError("Unknown event actor")
        event = AuditEvent(
            event_id=f"ev-{uuid.uuid4().hex[:12]}", case_id=self.case_id,
            actor_type=actor_type, actor_id=self.actor_id if actor_type == "human" else model_id or "system",
            action=action, target_id=run_id or model_id or "", model_run_id=run_id,
            payload=payload or {}, previous_event_sha256=self._last_event_hash,
        )
        event.event_sha256 = event.to_dict()["event_sha256"]
        self._last_event_hash = event.event_sha256
        self.events.append(event)
        return event

    def _emit(
        self,
        action: str,
        target: Annotation,
        payload: dict | None = None,
        *,
        human_action: bool = False,
    ) -> AuditEvent:
        # The target's AI origin must never misattribute a pathologist's
        # review, rename, or notes to a model/system actor.
        actor_type = "human" if human_action or target.origin == "human" else "system"
        if not human_action and target.origin == "ai" and target.actor_id:
            actor_type = "ai"
        actor_id = self.actor_id if human_action else (
            target.actor_id or (self.actor_id if target.origin == "human" else "mock-ai")
        )
        ev = AuditEvent(
            event_id=f"ev-{uuid.uuid4().hex[:12]}",
            case_id=self.case_id,
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            target_id=target.roi_id,
            model_run_id=target.model_run_id,
            previous_revision=None,
            new_revision=target.revision,
            payload=payload or {},
            previous_event_sha256=self._last_event_hash,
        )
        ev.event_sha256 = ev.to_dict()["event_sha256"]
        self._last_event_hash = ev.event_sha256
        self.events.append(ev)
        return ev
