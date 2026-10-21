"""Binary-mask utilities: mask -> polygons and boolean ROI algebra.

All geometry is in native pixel space and returns lists of polygon vertex lists
in ``[x, y]`` order to match :class:`Annotation`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def mask_to_polygons(mask: np.ndarray, *, min_area: int = 0) -> list[list[list[float]]]:
    """Convert connected components of a boolean mask into outer-boundary polygons.

    Each returned polygon is ``[[x, y], ...]``. Components smaller than
    ``min_area`` (px) are dropped.
    """
    import skimage.measure

    labelled, n = skimage.measure.label(mask, return_num=True)
    out: list[list[list[float]]] = []
    for i in range(1, n + 1):
        comp = labelled == i
        if int(comp.sum()) < min_area:
            continue
        ys, xs = np.nonzero(comp)
        if ys.size == 0:
            continue
        contours = skimage.measure.find_contours(comp.astype(float), 0.5)
        if not contours:
            out.append(_bbox_polygon(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
            continue
        outer = max(contours, key=len)  # rows=y, cols=x
        out.append([[float(float(x1)), float(float(y1))] for (y1, x1) in outer])
    return out


def _bbox_polygon(x0: int, y0: int, x1: int, y1: int) -> list[list[float]]:
    return [[float(x0), float(y0)], [float(x1), float(y0)],
            [float(x1), float(y1)], [float(x0), float(y1)]]


def _rasterize(points: Iterable[list[float]], height: int, width: int) -> np.ndarray:
    import skimage.draw

    m = np.zeros((height, width), dtype=bool)
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 3:
        return m
    rr, cc = skimage.draw.polygon([p[1] for p in pts], [p[0] for p in pts], shape=(height, width))
    m[rr, cc] = True
    return m


def rois_boolean(
    points_list: list[list[list[float]]],
    op: str,
    width: int,
    height: int,
    *,
    min_area: int = 0,
) -> list[list[list[float]]]:
    """Union/intersection/subtract sets of ROI polygons, returning polygons.

    ``op`` is one of 'union', 'intersection', 'subtract'. 'subtract' removes the
    union of all ROIs after the first from the first.
    """
    if not points_list:
        return []
    masks = [_rasterize(pts, height, width) for pts in points_list]
    if op == "union":
        result = np.zeros((height, width), dtype=bool)
        for m in masks:
            result |= m
    elif op == "intersection":
        result = np.ones((height, width), dtype=bool)
        for m in masks:
            result &= m
    elif op == "subtract":
        result = masks[0].copy()
        for m in masks[1:]:
            result &= ~m
    else:
        raise ValueError(f"Unknown boolean op: {op!r}")
    return mask_to_polygons(result, min_area=min_area)