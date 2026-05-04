"""
Histogram computation from scratch without using numpy.histogram or cv2.calcHist.
Supports 8-bit and 16-bit grayscale images.
"""
import numpy as np
from utils.image_utils import validate_grayscale
from utils.error_handler import wrap_errors


@wrap_errors
def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    validate_grayscale(image)
    flat = image.flatten().astype(np.int32)
    flat = np.clip(flat, 0, bins - 1)
    return np.bincount(flat, minlength=bins).astype(np.int64)


@wrap_errors
def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(histogram).astype(np.float64)
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    return cdf