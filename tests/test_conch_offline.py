"""M2.1 offline CONCH ROI-to-text — targeted tests (UNTESTED in this session)."""
from pathlib import Path

import numpy as np

from ashby_studio.models.descriptions import (
    DEFAULT_CANDIDATES,
    CandidateDescriptions,
)
from ashby_studio.models.conch import ConchAdapter
from ashby_studio.models.embedding_cache import DiskEmbeddingCache
from ashby_studio.models.roi_extraction import extract_roi_crop, crop_content_sha
from ashby_studio.models.registry import build_default_registry, MODEL_CONCH


def test_descriptions_balanced():
    c = DEFAULT_CANDIDATES
    assert len(c.ids) == len(c.descriptions) == len(c.templates) > 0
    # clinical-safety: no explicit chemical confirmation or cancer diagnostic terms
    joined = " ".join(c.descriptions).lower()
    assert "calcium oxalate" not in joined
    assert "cancer" not in joined and "carcinoma" not in joined


def test_descriptions_templates_per_category():
    c = DEFAULT_CANDIDATES
    seen = set()
    for cid, _disp, tmpls in c.category_template_sets():
        assert cid not in seen and len(tmpls) >= 1
        seen.add(cid)
    assert seen == set(DEFAULT_CANDIDATES.ids)


def test_candidate_constructor_rejects_mismatch():
    try:
        CandidateDescriptions(descriptions=["a", "b"], ids=["x"])
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        CandidateDescriptions(
            descriptions=["a"], ids=["x"], templates=["a", "b"]
        )
        assert False, "should reject template id/length mismatch"
    except ValueError:
        pass


def test_embedding_cache_roundtrip(tmp_path: Path):
    cache = DiskEmbeddingCache(tmp_path)
    key = cache.image_key("conch", "w" * 64, "c" * 8)
    assert cache.get(key) is None
    cache.put(key, [0.1, 0.2, 0.3], meta={"model": "conch"})
    reloaded = DiskEmbeddingCache(tmp_path)
    assert reloaded.get(key) == [0.1, 0.2, 0.3]


def test_roi_extraction_masks_crop(tmp_path):
    rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
    from ashby_studio.domain.annotations import Annotation

    ann = Annotation(roi_id="r1", image_asset_id="i1", points=[[20.0, 20.0], [40.0, 20.0], [30.0, 40.0]])
    crop = extract_roi_crop(rgb, ann)
    assert crop.ndim == 3 and crop.shape[-1] == 3
    assert len(crop_content_sha(crop)) == 64
    assert crop.max() == 200 and crop.min() == 0  # background masked to 0


def test_registry_disables_conch_without_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("ASHBY_CONCH_WEIGHTS", raising=False)
    reg = build_default_registry()
    meta = reg.get(MODEL_CONCH)
    assert meta.status == "disabled"
    assert "weights" in meta.reason.lower()
    assert reg.adapter(MODEL_CONCH) is None


def test_conch_categories_mean_pool_ensembles(monkeypatch):
    """A 3-template category is embedded per-variant and mean-pooled."""
    adapter = ConchAdapter(weights_path=None, cache=None)  # no real load needed
    three = CandidateDescriptions(
        descriptions=["my category"], ids=["cat_0"],
        templates=[["p1", "p2", "p3"]],
    )

    def fake_embed_text(texts):
        rows = []
        for t in texts:
            i = int(t[1]) - 1
            v = [0.0, 0.0, 0.0]
            v[i] = 1.0
            rows.append(v)
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(adapter, "embed_text", fake_embed_text)
    cats = adapter._cached_category_vectors(three)
    assert len(cats) == 1
    pooled = np.asarray(cats[0])
    # mean([1,0,0],[0,1,0],[0,0,1]) = 1/3 each
    assert np.allclose(pooled, [1 / 3, 1 / 3, 1 / 3])


def test_rank_descriptions_evidence_reports_ensembling(monkeypatch):
    adapter = ConchAdapter(weights_path=None, cache=None)
    monkeypatch.setattr(
        adapter, "_cached_image_vector",
        lambda crop: [1.0, 0.0, 0.0, 0.0],
    )
    monkeypatch.setattr(
        adapter, "_cached_category_vectors",
        lambda candidates: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    two = CandidateDescriptions(
        descriptions=["a", "b"], ids=["x", "y"], templates=[["a1"], ["b1"]],
    )
    results, evidence = adapter.rank_descriptions(np.zeros((8, 8, 3), dtype=np.uint8), two)
    assert results[0][0] == "x"
    ens = evidence["text_ensembling"]
    assert ens["method"] == "mean_pool" and ens["templates_per_category"] == 1