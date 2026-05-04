"""
Non-linear median filter implemented entirely from scratch.
Replaces each pixel with the median of its local neighborhood — effective for salt-and-pepper noise.
"""

import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale, normalize_to_uint8


@wrap_errors
def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to a grayscale image from scratch.

    For each pixel, extracts the kernel_size x kernel_size neighborhood,
    sorts all values, and assigns the median as the output pixel value.
    Uses replicate padding to handle image borders without black edges.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square neighborhood (e.g. 3, 5, 7)

    Returns
    -------
    output : 2D uint8 array (H, W), same size as input
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')
    output = np.zeros((img_h, img_w), dtype=np.float64)

    for row in range(img_h):
        for col in range(img_w):
            neighborhood = padded[row: row + kernel_size,
                                  col: col + kernel_size]
            flat = neighborhood.flatten()
            flat.sort()
            output[row, col] = flat[len(flat) // 2]

    return normalize_to_uint8(output)
