
import numpy as np
from .erosion_dilation import erode, dilate

def opening(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    eroded = erode(binary, se)
    opened = dilate(eroded, se)
    return opened


def closing(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    dilated = dilate(binary, se)
    closed = erode(dilated, se)
    return closed