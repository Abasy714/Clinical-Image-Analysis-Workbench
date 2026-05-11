
import numpy as np


def get_square_se(size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0: #size must be odd to allow center pixel
        raise ValueError(f"SE size must be a positive odd integer; got {size}.")
    return np.ones((size, size), dtype=bool) #2d array of ones (all true)


def get_cross_se(size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError(f"SE size must be a positive odd integer; got {size}.")
    se = np.zeros((size, size), dtype=bool) #2d array all false
    mid = size // 2
    se[mid, :] = True   # horizontal bar
    se[:, mid] = True   # vertical bar
    return se


def get_disk_se(size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError(f"SE size must be a positive odd integer; got {size}.")
    mid = size // 2
    radius = mid
    se = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if (r - mid) ** 2 + (c - mid) ** 2 <= radius ** 2: #circle equation
                se[r, c] = True
    return se


def get_se(shape: str, size: int) -> np.ndarray:
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