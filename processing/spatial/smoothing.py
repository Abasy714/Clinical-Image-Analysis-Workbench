"""
Smoothing filters: average (box) filter and Gaussian filter, both implemented from scratch.
Uses the shared convolution engine — no scipy or OpenCV filtering functions permitted.
"""

import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale, normalize_to_uint8
from processing.spatial.convolution import convolve2d


def _build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Build a 2D Gaussian kernel from scratch.

    Uses the formula:
        G(x, y) = exp(-(x² + y²) / (2 * sigma²))
    then normalizes so all values sum to 1.

    Parameters
    ----------
    size  : kernel side length (must be odd)
    sigma : standard deviation of the Gaussian

    Returns
    -------
    kernel : 2D float64 array of shape (size, size), sums to 1.0
    """
    if size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {size}")
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")

    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)

    for row in range(size):
        for col in range(size):
            x = col - center
            y = row - center
            kernel[row, col] = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))

    # normalize so kernel sums to 1 (no net brightness change)
    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


@wrap_errors
def average_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a box (average) filter to a grayscale image.

    Every output pixel is the arithmetic mean of the kernel_size × kernel_size
    neighborhood centered on that pixel.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square kernel (e.g. 3, 5, 7)

    Returns
    -------
    result : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    # uniform kernel — all weights equal 1/n²
    n = kernel_size * kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / n

    raw = convolve2d(image, kernel)
    return normalize_to_uint8(raw)


@wrap_errors
def gaussian_filter(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian smoothing filter to a grayscale image.

    The Gaussian kernel is built from scratch using the 2D Gaussian formula
    and normalized to sum to 1 before convolution.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square kernel (must be odd, e.g. 3, 5, 7)
    sigma       : standard deviation controlling blur strength

    Returns
    -------
    result : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    kernel = _build_gaussian_kernel(kernel_size, sigma)
    raw = convolve2d(image, kernel)
    return normalize_to_uint8(raw)