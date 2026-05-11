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

    padded = np.pad(image, pad, mode='edge') # Replicate padding to avoid black borders

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    patches = as_strided(padded, shape=shape, strides=strides) 
    output = np.median(patches.reshape(img_h, img_w, -1), axis=2) #bey compute median across el flattened kernel dimension, badal makano 4 values hayb2a 3 bas wb3deen for a kernell size of 3 hayb2a 9 values hayb2a 3*3=9, then we take the median across that dimension to get the output pixel value
    return normalize_to_uint8(output)
