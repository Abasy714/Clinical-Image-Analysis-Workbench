"""
Edge detection using Sobel and Prewitt operators, implemented from scratch.
Returns horizontal edges, vertical edges, and combined gradient magnitude.
"""

import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale, normalize_to_uint8
from processing.spatial.convolution import convolve2d

SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1],
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
], dtype=np.float64)

PREWITT_X = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
], dtype=np.float64)

PREWITT_Y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
], dtype=np.float64)


def combined_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute the gradient magnitude from horizontal and vertical components.
    Formula: magnitude = sqrt(Gx² + Gy²)
    """
    mag = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)
    return normalize_to_uint8(mag)


@wrap_errors
def sobel(image: np.ndarray) -> tuple:
    """
    Apply Sobel edge detection to a grayscale image.

    Parameters
    ----------
    image : 2D numpy array (H, W), grayscale uint8

    Returns
    -------
    gx        : 2D uint8 array — horizontal edges
    gy        : 2D uint8 array — vertical edges
    magnitude : 2D uint8 array — combined edge strength
    """
    validate_grayscale(image)
    raw_gx = convolve2d(image, SOBEL_X)
    raw_gy = convolve2d(image, SOBEL_Y)
    gx = normalize_to_uint8(raw_gx)
    gy = normalize_to_uint8(raw_gy)
    mag = combined_magnitude(raw_gx, raw_gy)
    return gx, gy, mag


@wrap_errors
def prewitt(image: np.ndarray) -> tuple:
    """
    Apply Prewitt edge detection to a grayscale image.

    Parameters
    ----------
    image : 2D numpy array (H, W), grayscale uint8

    Returns
    -------
    gx        : 2D uint8 array — horizontal edges
    gy        : 2D uint8 array — vertical edges
    magnitude : 2D uint8 array — combined edge strength
    """
    validate_grayscale(image)
    raw_gx = convolve2d(image, PREWITT_X)
    raw_gy = convolve2d(image, PREWITT_Y)
    gx = normalize_to_uint8(raw_gx)
    gy = normalize_to_uint8(raw_gy)
    mag = combined_magnitude(raw_gx, raw_gy)
    return gx, gy, mag