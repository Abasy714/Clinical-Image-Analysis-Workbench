"""
Morphological boundary extraction from scratch.
Formula: boundary = binary_image - erode(binary_image, SE)

This isolates the 1-pixel-thick outline of every foreground object,
which is useful for delineating anatomical structures (vessels, organs, cells).
No morphological libraries are used anywhere.
"""

import numpy as np
from .erosion_dilation import erode

def extract_boundary(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Extracts the morphological boundary of foreground objects.

    The boundary of a set A with structuring element B is defined as:
        β(A) = A − ε(A)   where ε denotes erosion.

    A pixel belongs to the boundary if it is foreground in the original image
    but would be removed (set to background) by erosion — i.e., it lies on
    the edge of a foreground region.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean (or 0/1 uint8) binary image.
    se : np.ndarray
        2-D boolean structuring element (typically a small square or cross).

    Returns
    -------
    np.ndarray
        2-D boolean image containing only the boundary pixels.
    """
    binary = binary.astype(bool)
    eroded = erode(binary, se)

    # Set-theoretic difference: pixels that are True in binary but False after erosion
    boundary = np.zeros(binary.shape, dtype=bool)
    rows, cols = binary.shape
    for r in range(rows):
        for c in range(cols):
            # A pixel is on the boundary if it is foreground and its eroded value is background
            if binary[r, c] and not eroded[r, c]:
                boundary[r, c] = True

    return boundary


def extract_boundary_vectorized(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Faster vectorized version of extract_boundary using NumPy boolean indexing.
    Mathematically identical to extract_boundary(); use this in the GUI pipeline
    for responsive performance on large images.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean or uint8 binary image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        2-D boolean boundary image.
    """
    binary = binary.astype(bool)
    eroded = erode(binary, se)
    # Boolean subtraction: True where binary is True AND eroded is False
    return binary & ~eroded