"""
Generates binary structuring elements (SE) for morphological operations.
Supports square, cross, and disk shapes at user-defined sizes.
All logic implemented from scratch — no morphological libraries used.
"""

import numpy as np


def get_square_se(size: int) -> np.ndarray:
    """
    Creates a filled square structuring element of the given odd size.

    Parameters
    ----------
    size : int
        Side length of the square (must be a positive odd integer, e.g. 3, 5, 7).

    Returns
    -------
    np.ndarray
        2-D boolean array of shape (size, size) filled with True.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"SE size must be a positive odd integer; got {size}.")
    return np.ones((size, size), dtype=bool)


def get_cross_se(size: int) -> np.ndarray:
    """
    Creates a plus-shaped (cross) structuring element of the given odd size.
    Only the centre row and centre column are True; all other cells are False.

    Parameters
    ----------
    size : int
        Side length of the bounding box (must be a positive odd integer).

    Returns
    -------
    np.ndarray
        2-D boolean array of shape (size, size).
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"SE size must be a positive odd integer; got {size}.")
    se = np.zeros((size, size), dtype=bool)
    mid = size // 2
    se[mid, :] = True   # horizontal bar
    se[:, mid] = True   # vertical bar
    return se


def get_disk_se(size: int) -> np.ndarray:
    """
    Creates a disk (circular) structuring element whose diameter equals `size`.
    A pixel (r, c) is True when its Euclidean distance from the centre <= radius.

    Parameters
    ----------
    size : int
        Diameter of the disk (must be a positive odd integer).

    Returns
    -------
    np.ndarray
        2-D boolean array of shape (size, size).
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"SE size must be a positive odd integer; got {size}.")
    mid = size // 2
    radius = mid
    se = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if (r - mid) ** 2 + (c - mid) ** 2 <= radius ** 2:
                se[r, c] = True
    return se


def get_se(shape: str, size: int) -> np.ndarray:
    """
    Convenience dispatcher: returns the requested SE by name.

    Parameters
    ----------
    shape : str
        One of 'square', 'cross', or 'disk' (case-insensitive).
    size : int
        Side length / diameter (positive odd integer).

    Returns
    -------
    np.ndarray
        2-D boolean structuring element.
    """
    shape = shape.strip().lower()
    if shape == "square":
        return get_square_se(size)
    elif shape == "cross":
        return get_cross_se(size)
    elif shape == "disk":
        return get_disk_se(size)
    else:
        raise ValueError(
            f"Unknown SE shape '{shape}'. Choose 'square', 'cross', or 'disk'."
        )