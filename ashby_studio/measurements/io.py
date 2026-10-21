"""CSV export for measurement results (reproducible, unit-explicit)."""
from __future__ import annotations

import csv
from pathlib import Path

from .engine import MeasurementResult, Summary


def export_results_csv(
    path: str | Path,
    rows: list[MeasurementResult],
    summary: Summary | None,
    *,
    case_id: str = "",
    image_sha256: str = "",
) -> None:
    """Write a unit-aware CSV with header metadata and a results table."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ashby-studio", "measurement-export", "research-only, not for diagnosis"])
        w.writerow(["case_id", case_id])
        if summary is not None:
            w.writerow(["image_sha256", summary.image_sha256 or image_sha256])
            w.writerow(["denominator", summary.denominator_desc])
            w.writerow(["candidate_area_px2", summary.candidate_area_px2])
            w.writerow(["denominator_area_px2", summary.denominator_area_px2])
            w.writerow(["candidate_fraction_pct", summary.candidate_fraction_pct])
            w.writerow(["n_objects", summary.n_objects])
            w.writerow(["overlaps_deduped", summary.overlaps_deduped])
            w.writerow(["calibrated", summary.calibrated])
            w.writerow(["scale_um_per_px", summary.scale_um_per_px if summary.scale_um_per_px else ""])
            w.writerow(["filters", summary.filters])
        w.writerow([])
        header = ["roi_id", "name", "label", "origin", "review_state",
                  "area_px2", "area_um2", "area_mm2", "perimeter_px", "perimeter_um"]
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.roi_id, r.name, r.label, r.origin, r.review_state,
                f"{r.area_px2:.6g}", f"{r.area_um2:.6g}" if r.area_um2 is not None else "",
                f"{r.area_mm2:.6g}" if r.area_mm2 is not None else "",
                f"{r.perimeter_px:.6g}", f"{r.perimeter_um:.6g}" if r.perimeter_um is not None else "",
            ])