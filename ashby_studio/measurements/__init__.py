"""Measurement metrics (imported for convenience)."""
from .engine import (
    Scale,
    MeasurementResult,
    Summary,
    polygon_area_px2,
    polygon_perimeter_px,
    bbox,
    filter_annotations,
    generate_tissue_mask,
    rasterize_union,
    measure_rois,
    summarize_candidates,
    DEFAULT_CANDIDATE_CLASSES,
)

__all__ = [
    "Scale",
    "MeasurementResult",
    "Summary",
    "polygon_area_px2",
    "polygon_perimeter_px",
    "bbox",
    "filter_annotations",
    "generate_tissue_mask",
    "rasterize_union",
    "measure_rois",
    "summarize_candidates",
    "DEFAULT_CANDIDATE_CLASSES",
]