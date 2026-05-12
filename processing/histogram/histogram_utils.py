"""
Histogram computation from scratch without using numpy.histogram or cv2.calcHist.
Supports 8-bit and 16-bit grayscale images.
"""
import numpy as np
from utils.image_utils import normalize_to_uint8, validate_grayscale
from utils.error_handler import wrap_errors


@wrap_errors
def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    validate_grayscale(image)
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    data = normalize_to_uint8(image)
    flat = data.ravel().astype(np.uint16)
    bin_idx = (flat * int(bins)) // 256
    bin_idx = np.clip(bin_idx, 0, int(bins) - 1)
    return np.bincount(bin_idx, minlength=int(bins)).astype(np.int64)


@wrap_errors
def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(histogram).astype(np.float64)
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    return cdf
