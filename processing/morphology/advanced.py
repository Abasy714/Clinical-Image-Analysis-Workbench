"""
Advanced morphological operations built on the custom erode() and dilate() primitives.
Implements:
  - Morphological Gradient
  - Top-Hat Transform    (white top-hat)
  - Bottom-Hat Transform (black top-hat / bottom-hat)

All operations are derived purely from erode() and dilate() — no libraries.
"""

import numpy as np
from .erosion_dilation import erode, dilate
from .opening_closing import opening, closing


# ---------------------------------------------------------------------------
# Morphological Gradient
# ---------------------------------------------------------------------------

def morphological_gradient(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Morphological gradient: dilation minus erosion.
    Formula: grad(A) = δ(A) − ε(A)

    Highlights the edges / transitions between foreground and background,
    producing a thicker boundary than simple boundary extraction.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean input image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        2-D boolean gradient image (True at edge regions).
    """
    binary = binary.astype(bool)
    dilated = dilate(binary, se)
    eroded  = erode(binary, se)
    # Gradient = pixels that are in dilation but not in erosion
    return dilated & ~eroded


# ---------------------------------------------------------------------------
# Top-Hat Transforms
# ---------------------------------------------------------------------------

def white_top_hat(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    White top-hat transform: original minus its opening.
    Formula: WTH(A) = A − (A ∘ B)   where ∘ denotes opening.

    Extracts small bright structures (foreground details smaller than the SE)
    that were removed by opening. Useful for isolating small bright lesions
    or calcifications in medical images.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean input image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        2-D boolean image of small bright features.
    """
    binary = binary.astype(bool)
    opened = opening(binary, se)
    # White top-hat = foreground pixels that survived original but were lost in opening
    return binary & ~opened


def black_top_hat(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Black top-hat (bottom-hat) transform: closing minus original.
    Formula: BTH(A) = (A • B) − A   where • denotes closing.

    Extracts small dark structures (background holes smaller than the SE)
    that were filled by closing. Useful for isolating small dark gaps,
    holes, or valleys inside anatomical structures.

    Parameters
    ----------
    binary : np.ndarray
        2-D boolean input image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        2-D boolean image of small dark features.
    """
    binary = binary.astype(bool)
    closed = closing(binary, se)
    # Black top-hat = background pixels that were filled by closing but absent originally
    return closed & ~binary


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_advanced_op(operation: str, binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    GUI-facing dispatcher: applies the named advanced morphological operation.

    Parameters
    ----------
    operation : str
        One of 'gradient', 'white_top_hat', 'black_top_hat' (case-insensitive).
    binary : np.ndarray
        2-D boolean input image.
    se : np.ndarray
        2-D boolean structuring element.

    Returns
    -------
    np.ndarray
        Result image as a 2-D boolean array.

    Raises
    ------
    ValueError
        If `operation` is not one of the supported names.
    """
    op = operation.strip().lower()
    if op == "gradient":
        return morphological_gradient(binary, se)
    elif op == "white_top_hat":
        return white_top_hat(binary, se)
    elif op == "black_top_hat":
        return black_top_hat(binary, se)
    else:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            "Choose 'gradient', 'white_top_hat', or 'black_top_hat'."
        )