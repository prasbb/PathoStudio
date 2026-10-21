"""Optional, checksummed feature-run case sidecar. Raw scores survive filtering."""
from __future__ import annotations

import hashlib
import json
import math
import re

from .repository import _atomic_write_json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_run(run):
    p = run["provenance"]
    if (run["schema_version"] != 1 or p["score_type"] != "logit"
            or p["schema_version"] != 1 or p["status"] != "completed" or p["calibration_id"] is not None):
        raise ValueError("Unsupported feature-run schema")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,59}", run["feature"]):
        raise ValueError("Invalid feature label")
    h,w = run["image_shape"]
    if type(h) is not int or type(w) is not int or min(h,w) < 1:
        raise ValueError("Invalid feature-run image dimensions")
    if not 1 <= len(run["tiles"]) <= 4096 or not math.isfinite(run["display_threshold"]):
        raise ValueError("Invalid feature-run limits")
    for key in ("run_id", "model_id", "run_at", "model_version", "preprocessing_version"):
        if not isinstance(p[key], str) or not p[key]:
            raise ValueError("Invalid model-run provenance")
    if not p["model_id"].startswith(f"feature:{run['feature']}:") or len(p["image_asset_ids"]) != 1:
        raise ValueError("Model-run feature or image identity mismatch")
    ids = set()
    for t in run["tiles"]:
        x0,y0,x1,y1 = t["box"]
        if (not all(type(v) is int for v in t["box"]) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)
                or not math.isfinite(t["score"]) or t["tile_id"] in ids):
            raise ValueError("Invalid feature-run tile")
        ids.add(t["tile_id"])
    if not isinstance(run["proposal_ids"], dict) or not set(run["proposal_ids"]).issubset(ids):
        raise ValueError("Invalid feature proposal references")
    if not re.fullmatch(r"[a-f0-9]{64}", run["image_sha256"]):
        raise ValueError("Invalid feature image hash")


def save_runs(case_dir, case_id, runs):
    for run in runs:
        validate_run(run)
    payload = {"schema_version": 1, "case_id": case_id, "runs": runs}
    _atomic_write_json(case_dir / "feature_runs.json", {"payload": payload, "sha256": _digest(payload)})


def load_runs(case_dir, case_id):
    path = case_dir / "feature_runs.json"
    if not path.exists():
        return []
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Feature run archive exceeds size limit")
    wrapper = json.loads(path.read_text())
    payload = wrapper["payload"]
    if wrapper["sha256"] != _digest(payload) or payload["case_id"] != case_id or payload["schema_version"] != 1:
        raise ValueError("Feature run archive checksum or identity mismatch")
    seen = set()
    for run in payload["runs"]:
        validate_run(run)
        rid = run["provenance"]["run_id"]
        if rid in seen:
            raise ValueError("Duplicate feature run identity")
        seen.add(rid)
    return payload["runs"]


def validate_runs_for_asset(runs, asset):
    if asset is None:
        return
    for run in runs:
        if (run["image_sha256"] != asset.sha256 or run["image_shape"] != list(asset.data.shape[:2])
                or run["provenance"]["image_asset_ids"] != [asset.asset_id]):
            raise ValueError("Feature results do not match the case image")
