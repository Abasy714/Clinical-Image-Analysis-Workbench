"""
Order-statistic spatial filters implemented entirely from scratch.
No scipy, OpenCV, or skimage filtering functions are used.

Filters included:
    - min_filter      : minimum pixel in neighborhood (removes salt noise)
    - max_filter      : maximum pixel in neighborhood (removes pepper noise)
    - midpoint_filter : (min + max) / 2 of neighborhood
"""

import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale, normalize_to_uint8


@wrap_errors
def min_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a min filter to a grayscale image from scratch.

    Each output pixel is the minimum value in the neighborhood:
        g(x, y) = min{ f(s, t) }

    Effect: darkens the image, shrinks bright regions.
    Best for: removing salt noise (bright impulses) — the minimum
    in a window containing a salt pixel is always a normal pixel.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square window (must be odd, e.g. 3, 5, 7)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='edge')  # replicate border pixels to avoid black edges

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],  # move to next row of center pixels
        padded.strides[1],  # move to next col of center pixels
        padded.strides[0],  # move down inside the patch
        padded.strides[1],  # move right inside the patch
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    patches = patches.reshape(img_h, img_w, -1)  # flatten each window → (H, W, k*k)

    output = patches.min(axis=2)  # take minimum pixel in each neighborhood
    return normalize_to_uint8(output)


@wrap_errors
def max_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a max filter to a grayscale image from scratch.

    Each output pixel is the maximum value in the neighborhood:
        g(x, y) = max{ f(s, t) }

    Effect: brightens the image, shrinks dark regions.
    Best for: removing pepper noise (dark impulses) — the maximum
    in a window containing a pepper pixel is always a normal pixel.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square window (must be odd, e.g. 3, 5, 7)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='edge')  # replicate border pixels to avoid black edges

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],  # move to next row of center pixels
        padded.strides[1],  # move to next col of center pixels
        padded.strides[0],  # move down inside the patch
        padded.strides[1],  # move right inside the patch
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    patches = patches.reshape(img_h, img_w, -1)  # flatten each window → (H, W, k*k)

    output = patches.max(axis=2)  # take maximum pixel in each neighborhood
    return normalize_to_uint8(output)


@wrap_errors
def midpoint_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a midpoint filter to a grayscale image from scratch.

    Each output pixel is the average of the min and max values
    in the neighborhood:
        g(x, y) = (min{ f(s,t) } + max{ f(s,t) }) / 2

    Best for: randomly distributed noise like Gaussian or uniform noise.
    Combines properties of min and max filters.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square window (must be odd, e.g. 3, 5, 7)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='edge')  # replicate border pixels to avoid black edges

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],  # move to next row of center pixels
        padded.strides[1],  # move to next col of center pixels
        padded.strides[0],  # move down inside the patch
        padded.strides[1],  # move right inside the patch
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    patches = patches.reshape(img_h, img_w, -1)  # flatten each window → (H, W, k*k)

    # midpoint = (min + max) / 2 -- benakhod el minimum we el maximum men kol neighborhood we benhaseb el average
    output = (patches.min(axis=2) + patches.max(axis=2)) / 2.0
    return normalize_to_uint8(output)