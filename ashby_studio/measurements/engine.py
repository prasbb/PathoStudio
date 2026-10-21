"""Pure measurement engine for Virchow's Suite (Phase 1).

All geometry stays in native pixel space; physical units (µm, mm) are produced
only when an explicit, user-entered calibration (µm/pixel) is supplied. Results
always carry an explicit denominator and never invent scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np

from ..domain.annotations import Annotation, is_ai

MM2_PER_UM2 = 1e-6

# Candidate labels treated as "mineralization candidates" by default. H&E can
# only flag candidates; chemical identity is not inferred here.
DEFAULT_CANDIDATE_CLASSES = ("candidate_calcification", "candidate_crystal")


@dataclass
class Scale:
    """Pixel calibration. ``um_per_px=None`` means uncalibrated (pixel units)."""
    um_per_px: float | None = None

    @property
    def calibrated(self) -> bool:
        return bool(self.um_per_px) and self.um_per_px > 0

    def px2_to_um2(self, area_px: float) -> float | None:
        if not self.calibrated:
            return None
        return area_px * (self.um_per_px ** 2)

    def px_to_um(self, length_px: float) -> float | None:
        if not self.calibrated:
            return None
        return length_px * self.um_per_px

    @classmethod
    def from_um_per_px(cls, um_per_px: float) -> "Scale":
        return cls(um_per_px=um_per_px)


# ---- geometry ---------------------------------------------------------------
def polygon_area_px2(points: Iterable[list[float]]) -> float:
    """Signed shoelace area in pixel², returned as absolute value."""
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    s1 = s2 = 0.0
    for i, (x, y) in enumerate(pts):
        xn, yn = pts[(i + 1) % len(pts)]
        s1 += x * yn
        s2 += xn * y
    return abs(s1 - s2) / 2.0


def polygon_perimeter_px(points: Iterable[list[float]]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i, (x, y) in enumerate(pts):
        xn, yn = pts[(i + 1) % len(pts)]
        total += ((x - xn) ** 2 + (y - yn) ** 2) ** 0.5
    return total


def bbox(points: Iterable[list[float]]) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) in integer pixel coordinates."""
    xs = [float(p[0]) for p in points] or [0.0]
    ys = [float(p[1]) for p in points] or [0.0]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def filter_annotations(
    annotations: list[Annotation],
    *,
    classes: Iterable[str] | None = None,
    origins: Iterable[str] | None = None,
    review_states: Iterable[str] | None = None,
) -> list[Annotation]:
    cset = set(classes) if classes is not None else None
    oset = set(origins) if origins is not None else None
    rset = set(review_states) if review_states is not None else None
    out = []
    for a in annotations:
        if cset is not None and a.label not in cset:
            continue
        if oset is not None and a.origin not in oset:
            continue
        if rset is not None and a.review_state not in rset:
            continue
        out.append(a)
    return out


# ---- masks ------------------------------------------------------------------
def generate_tissue_mask(rgb: np.ndarray, *, dark_is_tissue: bool = True) -> np.ndarray:
    """Otsu-based tissue/background proposal on the grayscale image.

    Returns a boolean H x W mask (True = tissue). Non-destructive: the source
    image is never modified.
    """
    import skimage.filters as filters

    gray = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    thr = filters.threshold_otsu(gray)
    if dark_is_tissue:
        return gray < thr
    return gray >= thr


def rasterize_union(
    annotations: list[Annotation],
    width: int,
    height: int,
) -> np.ndarray:
    """OR all ROI polygons into a single boolean mask; overlaps are deduped."""
    import skimage.draw as draw

    mask = np.zeros((height, width), dtype=bool)
    for a in annotations:
        pts = [[float(p[0]), float(p[1])] for p in a.points]
        if len(pts) < 3:
            continue
        rr, cc = draw.polygon([p[1] for p in pts], [p[0] for p in pts], shape=(height, width))
        mask[rr, cc] = True
    return mask


@dataclass
class MeasurementResult:
    roi_id: str
    name: str
    label: str
    origin: str
    review_state: str
    area_px2: float
    perimeter_px: float
    area_um2: float | None = None
    area_mm2: float | None = None
    perimeter_um: float | None = None

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class Summary:
    candidate_area_px2: float
    denominator_area_px2: float
    candidate_fraction_pct: float
    denominator_desc: str
    n_objects: int
    overlaps_deduped: bool
    calibrated: bool
    scale_um_per_px: float | None = None
    image_sha256: str = ""
    filters: dict = field(default_factory=dict)
    object_sizes_px2: list[float] = field(default_factory=list)


def measure_rois(
    annotations: list[Annotation],
    scale: Scale,
) -> list[MeasurementResult]:
    out = []
    for a in annotations:
        area_px = polygon_area_px2(a.points)
        per_px = polygon_perimeter_px(a.points)
        out.append(
            MeasurementResult(
                roi_id=a.roi_id,
                name=a.name or a.roi_id,
                label=a.label,
                origin="human" if a.origin == "human" else "ai",
                review_state=a.review_state,
                area_px2=area_px,
                perimeter_px=per_px,
                area_um2=scale.px2_to_um2(area_px),
                area_mm2=(scale.px2_to_um2(area_px) * MM2_PER_UM2) if scale.calibrated else None,
                perimeter_um=scale.px_to_um(per_px),
            )
        )
    return out


def summarize_candidates(
    annotations: list[Annotation],
    image_shape: tuple[int, int],
    *,
    scale: Scale,
    tissue_mask: np.ndarray | None = None,
    candidate_classes: Iterable[str] = DEFAULT_CANDIDATE_CLASSES,
    filters: dict | None = None,
) -> Summary:
    import skimage.measure as measure

    height, width = image_shape[:2]
    candidates = [a for a in annotations if a.label in set(candidate_classes)]
    union = rasterize_union(candidates, width, height)

    candidate_area = int(union.sum())
    if tissue_mask is not None and tissue_mask.shape == (height, width):
        denominator_area = int(tissue_mask.sum())
        analyzed = union & tissue_mask
        reported_candidate = int(analyzed.sum())
        denom_desc = "tissue mask (unioned, overlaps deduped)"
    else:
        denominator_area = height * width
        reported_candidate = candidate_area
        denom_desc = "whole image (no tissue mask)"

    # Object count + size distribution from connected components of the union.
    objs: list[float] = []
    n_objects = 0
    if reported_candidate > 0:
        labelled, n_objects = measure.label(analyzed if tissue_mask is not None else union, return_num=True)
        objs = [float((labelled == i).sum()) for i in range(1, n_objects + 1)]

    fraction = (reported_candidate / denominator_area * 100.0) if denominator_area else 0.0
    return Summary(
        candidate_area_px2=float(reported_candidate),
        denominator_area_px2=float(denominator_area),
        candidate_fraction_pct=fraction,
        denominator_desc=denom_desc,
        n_objects=int(n_objects),
        overlaps_deduped=True,
        calibrated=scale.calibrated,
        scale_um_per_px=scale.um_per_px,
        filters=filters or {},
        object_sizes_px2=objs,
    )