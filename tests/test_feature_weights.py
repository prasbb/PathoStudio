"""Opt-in compatibility smoke checks; synthetic pixels, local weights, no network."""
import os
from pathlib import Path
import socket

import numpy as np
import pytest

from ashby_studio.domain.images import ImageAsset
from ashby_studio.imaging.generator import make_synthetic_he_image
from ashby_studio.models.features.catalog import FEATURE_NAMES, read_package
from ashby_studio.models.features.conch import FeatureAdapter
from ashby_studio.models.features.scan import run_scan


@pytest.mark.skipif(os.environ.get("ASHBY_TEST_FEATURE_WEIGHTS") != "1", reason="Opt-in local real-weight smoke test")
@pytest.mark.parametrize("feature", list(FEATURE_NAMES))
def test_real_feature_package_offline(feature, monkeypatch):
    def deny(*args,**kwargs):
        raise AssertionError("Network access during local feature inference")
    monkeypatch.setattr(socket.socket,"connect",deny)
    monkeypatch.setattr(socket,"create_connection",deny)
    monkeypatch.setattr(socket,"getaddrinfo",deny)
    root = Path(__file__).resolve().parents[1] / "feature_models"
    package = read_package(root / feature / "conch/model.json")
    adapter = FeatureAdapter(package)
    image,_ = make_synthetic_he_image(672,672)
    asset = ImageAsset("synthetic-feature-test",Path("SYNTHETIC"),"he",image)
    result = run_scan(adapter,image,asset.sha256,asset.asset_id,device="cpu")
    assert len(result["tiles"]) == 1 and np.isfinite(result["tiles"][0]["score"])
    assert result["provenance"]["parameters"]["adaptation_sha256"] == package.manifest["adaptation"]["sha256"]
    assert adapter._model is None
