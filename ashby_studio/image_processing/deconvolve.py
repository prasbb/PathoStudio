"""H&E color deconvolution (Ruifrok & Johnston) via skimage.

Returns hematoxylin (H), eosin (E) and residual (DAB) channel estimates. The
source RGB image is never modified; channels are computed on demand.
"""
from __future__ import annotations

import numpy as np


def hed_channels(rgb: np.ndarray) -> np.ndarray:
    """Return (H, E, DAB) channels as a float HxWx3 optical-density estimate."""
    import skimage.color as color

    hed = color.rgb2hed(rgb)
    return np.asarray(hed, dtype=np.float64)