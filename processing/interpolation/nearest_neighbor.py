"""
Nearest-neighbor interpolation for image resizing, implemented entirely from scratch.
Maps each output pixel to the closest input pixel using floor rounding.
"""

import numpy as np
from utils.error_handler import wrap_errors


@wrap_errors
def nearest_neighbor_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid output dimensions: {new_h}x{new_w}")
    #get original image dimensions,both greyscale and rgb images will have shape (H, W) or (H, W, C), so we take the first two dimensions as height and width
    src_h, src_w = image.shape[:2]

    #scaling factors
    scale_y = src_h / new_h     #input size / output size  (for rows)
    scale_x = src_w / new_w     #(for columns)
    
    out_row_idx = np.arange(new_h)    # output position (0, 1, 2, ... new_h-1)
    out_col_idx = np.arange(new_w)    # output position (0, 1, 2, ... new_w-1)

    #maps each output row to its corresponding source row. np.floor rounds down to the nearest integer,
    #output position × ratio = input position
    src_row_idx = np.floor(out_row_idx * scale_y).astype(np.int32)
    src_col_idx = np.floor(out_col_idx * scale_x).astype(np.int32)

    #clamps all indices to valid range to prevent out-of-bounds access.
    # This is important because due to rounding, some indices might end up being equal to src_h or src_w
    src_row_idx = np.clip(src_row_idx, 0, src_h - 1)
    src_col_idx = np.clip(src_col_idx, 0, src_w - 1)

    return image[np.ix_(src_row_idx, src_col_idx)]