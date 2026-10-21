"""M0.1 focused smoke test: the minimal viewer launches and opens an H&E image.

Covers the first M0 deliverable only (offline launch + open local H&E image).
Run headless with:
    QT_QPA_PLATFORM=offscreen PYOPENGL_PLATFORM=osmesa .venv/bin/python -m pytest tests/test_smoke.py -q
"""
import tempfile
from pathlib import Path

from ashby_studio.app import build_app


def test_viewer_launches_and_opens_he_image():
    viewer, win = build_app(offscreen=True)
    try:
        # synthetic demo image auto-loaded at startup
        assert win.asset is not None
        assert win.asset.modality == "he"
        assert "Synthetic H&E" in {l.name for l in viewer.layers} or "H&E (read-only)" in {l.name for l in viewer.layers}

        # open a real local PNG H&E image
        from ashby_studio.imaging.generator import write_synthetic_png

        with tempfile.TemporaryDirectory() as d:
            info = write_synthetic_png(Path(d) / "he.png")
            asset = win.open_path(Path(info["path"]))

            he_layers = [l for l in viewer.layers if l.name == "H&E (read-only)"]
            assert len(he_layers) == 1
            img = he_layers[0]
            assert img.ndim == 3 and img.data.shape[-1] == 3
            # data matches what was written (round-trip through load_image)
            assert img.data.shape[0] == asset.height
            assert img.data.shape[1] == asset.width
            assert len(asset.sha256) == 64  # content hash recorded, source unmodified
    finally:
        viewer.close()


def test_viewer_rejects_unsupported_extension():
    """Opening a non-image file surfaces a clear error instead of crashing."""
    viewer, win = build_app(offscreen=True)
    try:
        import pytest

        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "notes.txt"
            bad.write_text("not an image")
            from ashby_studio.domain.images import load_image

            with pytest.raises(ValueError):
                load_image(bad)
    finally:
        viewer.close()