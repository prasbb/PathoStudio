"""Valid napari layer-mode helpers (pure, no Qt/napari import at module scope).

Napari ``Labels`` layers only accept the drawing/navigation modes below;
``"select"`` is NOT valid for a Labels layer and raising it previously broke
tissue-mask committing. ``Shapes`` layers keep their own ``"select"`` mode and
are intentionally untouched here.
"""
from __future__ import annotations

NAPARI_LABELS_MODES = frozenset({"pan_zoom", "paint", "erase", "fill", "pick"})


def labels_mode(mode: str) -> str:
    """Return ``mode`` if it is a valid napari Labels mode, else ``"pan_zoom"``.

    Guards against arbitrary strings (e.g. ``"select"``) that napari rejects
    with ``ValueError`` and asserts control back to navigation.
    """
    if mode not in NAPARI_LABELS_MODES:
        return "pan_zoom"
    return mode