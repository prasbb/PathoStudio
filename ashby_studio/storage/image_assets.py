"""Portable image references with source identity and embedded-byte checksums."""
from pathlib import Path
import shutil

from ..domain.images import load_image, sha256_file


def embed_asset(asset, case_dir):
    suffix = asset.path.suffix.lower() if asset.path.is_file() else ".png"
    relative = f"images/{asset.sha256}{suffix}"
    dest = case_dir / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if asset.path.is_file():
        if asset.path.resolve() != dest.resolve():
            shutil.copyfile(asset.path, dest)
    else:
        from PIL import Image
        Image.fromarray(asset.data).save(dest)
    return {"embedded_path": relative, "embedded_sha256": sha256_file(dest)}


def resolve_asset(case_dir, reference, requested=None):
    embedded = reference.get("embedded_path")
    if requested is not None and requested.is_file():
        path = requested
    elif embedded:
        path = case_dir / embedded
        if not path.resolve().is_relative_to(case_dir.resolve()):
            raise ValueError("Embedded image escapes the case directory")
    else:
        return None
    if not path.is_file():
        return None
    expected = reference.get("embedded_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError("Embedded image checksum mismatch")
    asset = load_image(path, modality=reference.get("modality", "he"), asset_id=reference.get("asset_id"))
    # The source identity may be an original TIFF or synthetic-array hash. The
    # independently verified embedded-byte hash protects its portable rendition.
    asset.sha256 = reference.get("sha256") or asset.sha256
    return asset
