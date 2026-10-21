"""Application factory: build a napari viewer + full annotation main window.

Kept separate from ``__main__`` so smoke tests and GUI tests can construct the
app in offscreen mode without touching argv/global state.

Offscreen mode (headless tests) sets ``QT_QPA_PLATFORM=offscreen`` and
``PYOPENGL_PLATFORM=osmesa`` when constructing the napari canvas with no
display. Note: on some macOS/vispy builds this combination still segfaults, so
offscreen tests are not guaranteed to pass on every machine; prefer a display
whenever possible.
"""
from __future__ import annotations

import os


def build_app(*, offscreen: bool | None = None, viewer=None) -> tuple:
    """Return ``(napari.Viewer, MainWindow)``.

    Constructs the full annotation workspace (:class:`MainWindow`) which hosts
    the napari viewer plus polygon ROI, mock-AI and persistence docks. If
    ``offscreen`` is true, configure headless Qt + software OpenGL. If ``viewer``
    is supplied it is reused (e.g. `napari._testing` or a pytest fixture);
    otherwise a new one is made.
    """
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import napari

    if viewer is None:
        viewer = napari.Viewer(show=False)

    from .ui.main_window import MainWindow

    win = MainWindow(viewer)
    return viewer, win