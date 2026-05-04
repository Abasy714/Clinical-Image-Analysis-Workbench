"""
Zoom dispatcher that routes zoom requests to either nearest-neighbor or bilinear interpolation.
Maintains the zoom factor and selected interpolation mode.
"""

# numpy — array passthrough
# processing.interpolation.nearest_neighbor — nearest_neighbor_resize
# processing.interpolation.bilinear — bilinear_resize

import numpy as np
from utils.error_handler import wrap_errors
from processing.interpolation.nearest_neighbor import nearest_neighbor_resize

INTERPOLATION_MODES = ['nearest', 'bilinear']


@wrap_errors
def apply_zoom(image: np.ndarray, zoom_factor: float, mode: str) -> np.ndarray:
    #error handling affirmations for input validation
    if image is None:
        raise ValueError("No image provided to apply_zoom.")
    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
    if mode not in INTERPOLATION_MODES:
        raise ValueError(f"Unknown interpolation mode: '{mode}'. Choose from {INTERPOLATION_MODES}")

    #avoids error at certain zoom levels (e.g. zoom 0.33 on a 100px image: int(33.0) = 33 but round(33.3) = 33 too)
    # 1 guarantees the output is never 0 pixels even at extreme zoom-out levels
    src_h, src_w = image.shape[:2]
    new_h = max(1, int(round(src_h * zoom_factor))) 
    new_w = max(1, int(round(src_w * zoom_factor)))

    #if the zoom factor is exactly 1, we can skip interpolation and return a copy of the OG image
    if zoom_factor == 1.0:
        return image.copy()

    if mode == 'nearest':
        return nearest_neighbor_resize(image, new_h, new_w)
    else:
        from processing.interpolation.bilinear import bilinear_resize
        return bilinear_resize(image, new_h, new_w)