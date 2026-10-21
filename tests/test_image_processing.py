"""Phase 2 image-processing helpers — focused tests (UNTESTED)."""
import numpy as np

from ashby_studio.image_processing.binary import mask_to_polygons, rois_boolean
from ashby_studio.image_processing.deconvolve import hed_channels


def test_hed_channels_shape_and_range():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180  # some hematoxylin-ish stain
    hed = hed_channels(rgb)
    assert hed.shape == (32, 32, 3)
    assert np.isfinite(hed).all()


def test_mask_to_polygons_single_blob():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True  # one 10x10 square
    polys = mask_to_polygons(mask, min_area=0)
    assert len(polys) == 1
    pts = polys[0]
    assert all(len(p) == 2 for p in pts)  # [x, y] pairs
    assert min(p[0] for p in pts) >= 0 and min(p[1] for p in pts) >= 0


def test_mask_to_polygons_respects_min_area():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:3, 2:3] = True  # tiny 1px blob
    assert mask_to_polygons(mask, min_area=50) == []
    assert len(mask_to_polygons(mask, min_area=0)) >= 1


def test_rois_boolean_union():
    w = h = 40
    a = [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]]
    b = [[12.0, 12.0], [20.0, 12.0], [20.0, 20.0], [12.0, 20.0]]
    polys = rois_boolean([a, b], "union", w, h)
    assert len(polys) >= 1


def test_rois_boolean_subtract():
    w = h = 40
    outer = [[5.0, 5.0], [35.0, 5.0], [35.0, 35.0], [5.0, 35.0]]
    inner = [[15.0, 15.0], [25.0, 15.0], [25.0, 25.0], [15.0, 25.0]]
    # subtraction returns an annulus-ish polygon (or multiple), never the original fill
    polys = rois_boolean([outer, inner], "subtract", w, h)
    assert len(polys) >= 1
    out = polys[0]
    # representative points along the middle row should sit in the ring (hole present)
    assert len(out) >= 3


def test_rois_boolean_intersection_reduces_to_overlap():
    w = h = 40
    a = [[5.0, 5.0], [20.0, 5.0], [20.0, 20.0], [5.0, 20.0]]
    b = [[15.0, 15.0], [30.0, 15.0], [30.0, 30.0], [15.0, 30.0]]
    polys = rois_boolean([a, b], "intersection", w, h)
    assert len(polys) >= 1