"""M2.2 Text -> ROI — focused tests (UNTESTED in this session)."""
import numpy as np

from ashby_studio.annotations.service import AnnotationService
from ashby_studio.models.embedding_cache import DiskEmbeddingCache
from ashby_studio.models.text_to_roi import (
    TILE_DEFAULT,
    build_heatmap,
    compute_patch_embeddings,
    dedupe_proposals,
    make_tiles,
    postprocess_heatmap,
    rank_query,
    run_retrieval,
    select_hotspot_proposals,
)


class FakeEncoder:
    weight_sha256 = "w" * 64
    model_version = "fake-conch"
    device = "cpu"

    def __init__(self):
        self.image_calls = 0
        self.dim = 8

    def embed_image_batch(self, patches):
        self.image_calls += 1
        out = []
        for i, _p in enumerate(patches):
            v = [0.0] * self.dim
            v[i % self.dim] = 1.0
            out.append(v)
        return np.asarray(out, dtype=np.float32)

    def embed_text(self, texts):
        v = [0.0] * self.dim
        v[0] = 1.0
        return np.asarray([v] * len(texts), dtype=np.float32)


def _img(h=64, w=64):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _tiles_roundtrip():
    img = _img()
    return make_tiles(img.shape[1], img.shape[0], TILE_DEFAULT, TILE_DEFAULT)


def test_make_tiles_spatial_coords_and_overlap():
    # small non-overlapping 2x2 grid
    tiles = make_tiles(64, 64, 32, 32)
    assert len(tiles) == 4
    assert tiles[0].x0 == 0 and tiles[0].y0 == 0
    assert tiles[0].x1 == 32 and tiles[0].y1 == 32
    # overlapping stride yields more tiles and clamps to bounds
    ov = make_tiles(64, 64, 32, 16)
    assert len(ov) > 4
    for t in ov:
        assert t.x0 >= 0 and t.y0 >= 0
        assert t.x1 <= 64 and t.y1 <= 64


def test_compute_patch_embeddings_caches(tmp_path):
    encoder = FakeEncoder()
    cache = DiskEmbeddingCache(tmp_path)
    img = _img()
    tiles = make_tiles(64, 64, 32, 32)

    v1 = compute_patch_embeddings(encoder, cache, img, tiles, "img-sha", 32, 32)
    calls = encoder.image_calls
    assert calls >= 1
    assert len(v1) == len(tiles)

    v2 = compute_patch_embeddings(encoder, cache, img, tiles, "img-sha", 32, 32)
    assert encoder.image_calls == calls  # served from cache, no re-encode
    for a, b in zip(v1, v2):
        assert np.allclose(a, b)

    # different geometry/identity must not reuse cache
    v3 = compute_patch_embeddings(encoder, cache, img, tiles, "other-sha", 32, 32)
    assert encoder.image_calls > calls


def test_rank_query_matches_expected_patch():
    q = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)[0]
    patches = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    ]
    scores = rank_query(q, patches)
    assert scores[0] > 0.99
    assert scores[1] < 0.01


def test_run_retrieval_end_to_end(tmp_path):
    encoder = FakeEncoder()
    cache = DiskEmbeddingCache(tmp_path)
    img = _img()
    tiles, scores, heatmap, evidence = run_retrieval(
        encoder, cache, img, "img-sha", "dense mineralized deposit",
        tile_size=32, stride=32,
    )
    assert len(tiles) == len(scores)
    assert heatmap.shape == (img.shape[0], img.shape[1])
    assert scores[0] > 0.99  # first tile aligned with query vector
    assert evidence["model_id"] == "conch"
    assert evidence["score_type"] == "cosine_similarity"
    assert evidence["calibration"] == "none"
    assert evidence["image_sha256"] == "img-sha"
    assert "research only" in evidence["label"].lower()
    assert "heatmap_postprocess" in evidence
    assert evidence["heatmap_postprocess"]["tissue_masked"] is False


def test_build_heatmap_aligned():
    img = _img()
    tiles = make_tiles(64, 64, 32, 32)
    scores = [1.0, 0.0, 0.0, 0.0]
    heat = build_heatmap(img, tiles, scores)
    assert heat.shape == (64, 64)
    assert heat[0, 0] == 1.0
    assert heat[0, 40] == 0.0  # other tile region stays 0


def test_dedupe_proposals_removes_overlap():
    boxes = [[0, 0, 10, 10], [2, 2, 12, 12], [20, 20, 30, 30]]
    scores = [1.0, 0.9, 0.8]
    kept = dedupe_proposals(boxes, scores, iou_threshold=0.3)
    by_score = {k["score"]: k["box"] for k in kept}
    # overlapping first two are deduped, keeping the higher-scored one
    assert len(kept) == 2
    assert [0, 0, 10, 10] in by_score.values()


def test_postprocess_heatmap_aligned_and_smoothed():
    img = _img()
    tiles = make_tiles(64, 64, 32, 32)
    scores = [1.0, 0.0, 0.0, 0.0]
    heat = build_heatmap(img, tiles, scores)
    out = postprocess_heatmap(heat, sigma=0.5)
    assert out.shape == (64, 64)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 + 1e-6
    # smoothing spreads the peak away from the single hot tile's corner
    assert out[32, 32] > 0.0  # adjacent region now nonzero after gaussian blur


def test_postprocess_heatmap_tissue_mask_zeroes_background():
    img = _img()
    tiles = make_tiles(64, 64, 32, 32)
    heat = build_heatmap(img, tiles, [1.0, 1.0, 1.0, 1.0])
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :32] = True  # only top-left quadrant is tissue
    # give non-tissue a nonzero heat so masking is observable
    heat[40:48, 40:48] = 1.0
    out = postprocess_heatmap(heat, mask, sigma=0.0)
    assert out[40, 40] == 0.0  # background zeroed despite nonzero raw heat
    assert out[4, 4] > 0.0     # tissue region preserved
    bad = np.ones((8, 8), dtype=bool)
    try:
        postprocess_heatmap(heat, bad)
        assert False, "should raise on mismatched mask"
    except ValueError:
        pass


def test_select_hotspot_proposals_limits_and_separates():
    tiles = make_tiles(64, 64, 32, 32)  # 4 tiles at top-left, top-right, bottom-left, bottom-right
    scores = [1.0, 0.9, 0.8, 0.7]
    boxes = [[t.x0, t.y0, t.x1, t.y1] for t in tiles]
    props = select_hotspot_proposals(tiles, scores, boxes, max_proposals=2, min_sep_px=8)
    assert len(props) <= 2
    assert props[0]["score"] == 1.0
    assert "tile_id" in props[0]
    # empty/edge guards
    try:
        select_hotspot_proposals(tiles, scores[:2], boxes, max_proposals=2)
        assert False, "must raise on length mismatch"
    except ValueError:
        pass


def test_select_hotspot_proposals_separation_prevents_crowding():
    # Two high-scoring adjacent tiles should merge into one hotspot at min_sep=large.
    tiles = [make_tiles(64, 64, 32, 32)[0], make_tiles(64, 76, 32, 32)[1]]
    scores = [1.0, 0.98]
    boxes = [[0, 0, 32, 32], [32, 0, 64, 32]]
    # min_sep smaller than both tile sizes -> both kept (distinct centers)
    props = select_hotspot_proposals(tiles, scores, boxes, max_proposals=10, min_sep_px=4)
    assert len(props) == 2
    # min_sep larger than the gap -> second suppressed by center-distance rule
    props2 = select_hotspot_proposals(tiles, scores, boxes, max_proposals=10, min_sep_px=100)
    assert len(props2) == 1


def test_provenance_of_saved_ai_candidate():
    svc = AnnotationService()
    ann = svc.create_polygon(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        image_asset_id="img-0001",
        origin="ai",
        ai_meta={"model_run_id": "ttr-abc123", "revision": 1},
    )
    assert ann.origin == "ai"
    assert ann.review_state == "suggested"  # unreviewed until human acts
    assert ann.model_run_id == "ttr-abc123"
    assert svc.events[-1].action == "roi_create"
    assert svc.events[-1].model_run_id == "ttr-abc123"