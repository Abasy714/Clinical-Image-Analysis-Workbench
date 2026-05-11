
import numpy as np
from .erosion_dilation import erode

def extract_boundary(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    eroded = erode(binary, se) #removes white pixels with any black neighbours(boundary) 
    boundary = np.zeros(binary.shape, dtype=bool)
    rows, cols = binary.shape
    for r in range(rows):
        for c in range(cols):
            if binary[r, c] and not eroded[r, c]: #pixel white in original and did not survive erosion
                boundary[r, c] = True

    return boundary


def extract_boundary_vectorized(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    eroded = erode(binary, se)
    return binary & ~eroded