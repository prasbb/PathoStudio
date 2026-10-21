"""Read-only image asset loading and hashing.

M0 supports PNG and TIFF (single-page). Original pixels are never mutated.
Synthetic demo fixtures live in :mod:`ashby_studio.imaging.generator`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_EXTS = {".png", ".tif", ".tiff"}

# Modality identifiers supported by the UI. H&E is the only required one in M0;
# the selector is designed to extend to polarized light / IHC in M2 without
# implementing image registration yet (each modality is just another layer).
HE = "he"
POLARIZED_LIGHT = "polarized_light"
IHC = "ihc"
OTHER = "other"

MODALITIES = {
    HE: "H&E",
    POLARIZED_LIGHT: "Polarized Light",
    IHC: "IHC",
    OTHER: "Other",
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (original input is never modified)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()



@dataclass
class ImageAsset:
    """A file-backed or in-memory image with traceable content identity."""

    asset_id: str
    path: Path
    modality: str
    data: np.ndarray
    shape: tuple[int, ...] = field(init=False)
    sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"ImageAsset.data must be a numpy.ndarray, "
                f"got {type(self.data).__name__}"
            )

        if self.data.ndim not in (2, 3):
            raise ValueError(
                f"Unsupported image shape: {self.data.shape}"
            )

        self.shape = tuple(self.data.shape)

        if not self.sha256:
            if self.path.is_file():
                # Original image on disk
                self.sha256 = sha256_file(self.path)
            else:
                # In-memory image, such as synthetic demo
                arr = np.ascontiguousarray(self.data)

                hasher = hashlib.sha256()
                hasher.update(str(arr.shape).encode())
                hasher.update(str(arr.dtype).encode())
                hasher.update(arr.tobytes())

                self.sha256 = hasher.hexdigest()

    @property
    def width(self) -> int:
        return int(self.shape[1])

    @property
    def height(self) -> int:
        return int(self.shape[0])

    


def load_image(path: Path, *, modality: str = HE, asset_id: str | None = None) -> ImageAsset:
    """Load *path* as an RGB(A) numpy array for display in napari.

    Raises ValueError for unsupported extensions or unreadable content so the UI
    can show a clear error while preserving manual workflow.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported image extension {ext!r}; use one of {sorted(SUPPORTED_EXTS)}")

    if ext in {".png"}:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            data = np.asarray(im, dtype=np.uint8)
    else:  # tif/tiff
        import tifffile

        arr = tifffile.imread(path)
        if arr.ndim == 2:
            data = np.dstack([arr, arr, arr])  # replicate grayscale to RGB
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            data = arr[..., :3]
        else:
            raise ValueError(
                f"Unsupported TIFF shape {arr.shape}; expected HxW, HxWx3 or HxWx4"
            )
        data = data.astype(np.uint8)

    if data.dtype != np.uint8:
        data = data.astype(np.uint8)

    return ImageAsset(
        asset_id=asset_id or path.name,
        path=path,
        modality=modality,
        data=data,
    )