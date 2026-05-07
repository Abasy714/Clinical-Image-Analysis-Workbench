"""
Compound morphological operations: opening and closing.
Built exclusively on the custom erode() and dilate() functions — no libraries.

Opening  = erode  then dilate  → removes small foreground noise (salt noise)
Closing  = dilate then erode   → fills small background holes  (pepper noise)
"""

import numpy as np
from .erosion_dilation import erode, dilate

def opening(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Morphological opening: erosion followed by dilation with the same SE.

    Effect: removes small bright (foreground) blobs and thin protrusions
    without significantly changing the size of larger structures.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean input image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        Opened binary image, same shape as `binary`, dtype bool.
    """
    eroded = erode(binary, se)
    opened = dilate(eroded, se)
    return opened


def closing(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Morphological closing: dilation followed by erosion with the same SE.

    Effect: fills small dark (background) holes and gaps inside foreground
    structures — useful for sealing blood vessel cross-sections, etc.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean input image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        Closed binary image, same shape as `binary`, dtype bool.
    """
    dilated = dilate(binary, se)
    closed = erode(dilated, se)
    return closed