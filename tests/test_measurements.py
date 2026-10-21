"""Phase 1 measurements — focused tests (UNTESTED in this session)."""
import numpy as np

from ashby_studio.domain.annotations import Annotation
from ashby_studio.measurements.engine import (
    Scale,
    bbox,
    filter_annotations,
    generate_tissue_mask,
    measure_rois,
    polygon_area_px2,
    polygon_perimeter_px,
    summarize_candidates,
)


def _ann(roi_id, pts, label="candidate_calcification", origin="human", state="accepted", name=""):
    return Annotation(roi_id=roi_id, image_asset_id="img", points=pts,
                      label=label, origin=origin, review_state=state, name=name)


def test_polygon_area_and_perimeter_known_square():
    pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert polygon_area_px2(pts) == 100.0
    assert polygon_perimeter_px(pts) == 40.0
    assert bbox(pts) == (0, 0, 10, 10)


def test_area_orientation_invariant():
    cw = [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]]
    ccw = list(reversed(cw))
    assert polygon_area_px2(cw) == polygon_area_px2(ccw) == 100.0


def test_calibration_units():
    scale = Scale(um_per_px=0.25)
    assert scale.calibrated
    # 100 px² * (0.25 µm/px)^2 = 6.25 µm²
    assert abs(scale.px2_to_um2(100.0) - 6.25) < 1e-9
    assert abs(scale.px_to_um(40.0) - 10.0) < 1e-12
    unc = Scale()
    assert not unc.calibrated
    assert unc.px2_to_um2(100.0) is None


def test_filter_by_class_origin_review():
    anns = [
        _ann("a", [[0, 0, 1, 0, 1, 1, 0, 1]], label="candidate_calcification", origin="human", state="accepted"),
        _ann("b", [[0, 0, 1, 0, 1, 1, 0, 1]], label="artifact", origin="ai", state="suggested"),
    ]
    assert len(filter_annotations(anns, classes=["candidate_calcification"])) == 1
    assert len(filter_annotations(anns, origins=["human"])) == 1
    assert len(filter_annotations(anns, review_states=["suggested"])) == 1


def test_tissue_mask(self=None):
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[:] = 230  # background white
    rgb[10:30, 10:30] = 90  # tissue darker
    mask = generate_tissue_mask(rgb)
    assert mask.shape == (40, 40)
    assert mask.dtype == bool
    # tissue region mostly True
    assert mask[20, 20]
    # far background corner mostly False
    assert not mask[0, 0]


def test_summary_overlap_dedup_and_fraction():
    # two overlapping candidate squares; union area must be < sum of the two.
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    overlaps = [
        _ann("c1", [[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]]),
        _ann("c2", [[40.0, 40.0], [90.0, 40.0], [90.0, 90.0], [40.0, 90.0]]),
    ]
    summary = summarize_candidates(overlaps, (100, 100), scale=Scale())
    assert summary.overlaps_deduped
    assert summary.candidate_area_px2 < 5000  # deduped below the 2500+2500 sum
    assert summary.candidate_area_px2 > 0
    assert summary.denominator_area_px2 == 10000.0
    assert 0.0 < summary.candidate_fraction_pct < 50.0

    # two disjoint candidate squares => counted as 2 objects (area not double-counted).
    disjoint = [
        _ann("d1", [[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]]),
        _ann("d2", [[10.0, 70.0], [60.0, 70.0], [60.0, 90.0], [10.0, 90.0]]),
    ]
    sd = summarize_candidates(disjoint, (100, 100), scale=Scale())
    assert sd.n_objects == 2
    assert sd.candidate_area_px2 > 0
    assert sd.candidate_area_px2 < 5000


def test_summary_tissue_denominator():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[0:50] = 80  # top half = tissue
    rgb[50:] = 220  # bottom half = background
    tissue = generate_tissue_mask(rgb)
    cand = _ann("x", [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    summary = summarize_candidates([cand], (100, 100), scale=Scale(), tissue_mask=tissue)
    assert "tissue mask" in summary.denominator_desc
    assert summary.denominator_area_px2 > 0


def test_measure_rois_calibrated_and_not():
    a = _ann("a", [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    unc = measure_rois([a], Scale())
    assert unc[0].area_um2 is None
    assert unc[0].area_px2 == 100.0
    cal = measure_rois([a], Scale(um_per_px=0.5))
    assert abs(cal[0].area_um2 - 25.0) < 1e-9