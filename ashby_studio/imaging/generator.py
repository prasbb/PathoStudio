"""Synthetic demo fixtures.

Generate a small tissue-like RGB image with well-defined circular foci of known
coordinates/area, useful both as the app's demo image and for deterministic
tests. Explicitly labeled SYNTHETIC — never misrepresented as diagnostic
clinical breast tissue (see qa-clinical-offline skill).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TISSUE = np.array([232, 200, 195], dtype=np.uint8)      # pale eosin-ish stroma
NUCLEI = np.array([60, 62, 120], dtype=np.uint8)         # basophilic


def make_synthetic_he_image(
    width: int = 512,
    height: int = 384,
    *,
    seed: int = 0,
    blob_centers: list[tuple[int, int]] | None = None,
    blob_radius: int = 16,
) -> tuple[np.ndarray, list[dict]]:
    """Return (RGB image, list of known blob dicts with 'center' and 'radius').

    The blob list documents known candidate-ish foci so tests can assert exact
    coordinates; these are synthetic, not pathology findings.
    """
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = TISSUE
    # subtle background texture
    noise = rng.integers(-12, 13, size=(height, width, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # scattered nuclear dots
    n_dots = int((height * width) / (40 * 40))
    for _ in range(n_dots):
        cx, cy = rng.integers(0, width), rng.integers(0, height)
        r = rng.integers(1, 3)
        yi, xi = np.ogrid[:height, :width]
        mask = (xi - cx) ** 2 + (yi - cy) ** 2 <= r * r
        img[mask] = NUCLEI

    centers = blob_centers or [
        (width // 3, height // 3),
        (2 * width // 3, height // 2),
        (width // 2, 2 * height // 3),
    ]
    blobs: list[dict] = []
    for idx, (cx, cy) in enumerate(centers):
        r = blob_radius + idx * 2
        color = np.array([120, 120, 180], dtype=np.uint8)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[mask] = color
        blobs.append({"center": (int(cx), int(cy)), "radius": int(r)})
    return img, blobs


def write_synthetic_png(path: Path, width: int = 320, height: int = 256, **kw) -> dict:
    from PIL import Image

    img, blobs = make_synthetic_he_image(width, height, **kw)
    Image.fromarray(img).save(path)
    return {"path": str(path), "blobs": blobs}


def write_synthetic_tiff(path: Path, width: int = 320, height: int = 256, **kw) -> dict:
    import tifffile

    img, blobs = make_synthetic_he_image(width, height, **kw)
    tifffile.imwrite(path, img)
    return {"path": str(path), "blobs": blobs}