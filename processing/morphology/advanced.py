
import numpy as np
from .erosion_dilation import erode, dilate
from .opening_closing import opening, closing

def morphological_gradient(binary: np.ndarray, se: np.ndarray) -> np.ndarray: #like boundary extraction but thicker

    binary = binary.astype(bool)
    dilated = dilate(binary, se)
    eroded  = erode(binary, se)
    return dilated & ~eroded #dilated-eroded

def white_top_hat(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    opened = opening(binary, se)#removes small white structures that dont survive erosion
    return binary & ~opened #finds small bright structures that are smaller than se


def black_top_hat(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    closed = closing(binary, se) #closing fills black holes in white regions
    return closed & ~binary #finds small dark holes inside white structures that are smaller than se




def apply_advanced_op(operation: str, binary: np.ndarray, se: np.ndarray) -> np.ndarray:
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