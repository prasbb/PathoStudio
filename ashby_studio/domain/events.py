"""Append-only audit event model (M0 lightweight).

Matches ``contracts/event.schema.json`` as closely as practical for M0. Human
edits of an AI proposal must create a *distinct* review event and preserve the
original AI record (ADR-0002); we never overwrite prior decisions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .annotations import utcnow


def _sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class AuditEvent:
    schema_version: int = 1
    event_id: str = ""
    case_id: str = ""
    timestamp: str = field(default_factory=utcnow)
    actor_type: str = "human"  # human | ai | system
    actor_id: str = ""
    action: str = ""
    target_id: str = ""
    model_run_id: str | None = None
    previous_revision: int | None = None
    new_revision: int | None = None
    payload: dict = field(default_factory=dict)
    previous_event_sha256: str | None = None
    event_sha256: str | None = None

    def to_dict(self) -> dict:
        payload = dict(self.payload)
        body = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "case_id": self.case_id,
            "timestamp": self.timestamp,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "target_id": self.target_id,
            "model_run_id": self.model_run_id,
            "previous_revision": self.previous_revision,
            "new_revision": self.new_revision,
            "payload": payload,
            "previous_event_sha256": self.previous_event_sha256,
            "event_sha256": _sha256(payload),
        }
        return body

    @classmethod
    def from_dict(cls, d: dict) -> "AuditEvent":
        return cls(
            schema_version=d.get("schema_version", 1),
            event_id=d["event_id"],
            case_id=d["case_id"],
            timestamp=d.get("timestamp", utcnow()),
            actor_type=d.get("actor_type", "system"),
            actor_id=d.get("actor_id", ""),
            action=d.get("action", ""),
            target_id=d.get("target_id", ""),
            model_run_id=d.get("model_run_id"),
            previous_revision=d.get("previous_revision"),
            new_revision=d.get("new_revision"),
            payload=d.get("payload", {}),
            previous_event_sha256=d.get("previous_event_sha256"),
            event_sha256=d.get("event_sha256"),
        )