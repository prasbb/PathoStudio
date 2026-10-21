"""Imaging package import regression tests.

Guards against regressions like a stale ``__all__`` in
``ashby_studio/imaging/__init__.py`` that referenced non-existent names
(``images``, ``load_image``), which broke ``from ashby_studio.imaging import *``
/ qualified imports and blocked startup of ``minimal_viewer``.
"""
import ashby_studio.imaging


def test_imaging_package_imports_star():
    """Star-import of the imaging package must expose only real names."""
    namespace = {}
    exec("from ashby_studio.imaging import *", namespace)
    assert "make_synthetic_he_image" in namespace
    assert "write_synthetic_png" in namespace
    assert "write_synthetic_tiff" in namespace
    assert "images" not in namespace
    assert "load_image" not in namespace


def test_imaging_package_all_matches_public_api():
    """__all__ must reference only attributes the package truly provides."""
    assert set(ashby_studio.imaging.__all__) == {
        "make_synthetic_he_image",
        "write_synthetic_png",
        "write_synthetic_tiff",
    }
    for name in ashby_studio.imaging.__all__:
        assert hasattr(ashby_studio.imaging, name), f"missing exported name: {name}"