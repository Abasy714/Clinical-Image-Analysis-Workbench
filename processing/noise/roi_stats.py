# STATUS: STUB — Phase 2
"""
Statistical analysis of a rectangular ROI drawn on the image.
mean, variance, histogram, SNR, entropy, CNR, PSNR, MSE.
"""
import numpy as np
from utils.image_utils import validate_grayscale
from utils.error_handler import wrap_errors
from processing.histogram.histogram_utils import compute_histogram


@wrap_errors
def extract_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    validate_grayscale(image)
    img_h, img_w = image.shape[:2]

    x  = max(0, min(x, img_w - 1))
    y  = max(0, min(y, img_h - 1))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))

    return image[y:y2, x:x2]


@wrap_errors
def compute_roi_stats(image: np.ndarray, x: int, y: int, w: int, h: int) -> dict:
    validate_grayscale(image)
    roi = extract_roi(image, x, y, w, h)

    if roi.size == 0:
        return {
            'mean':        0.0,
            'variance':    0.0,
            'histogram':   np.zeros(256, dtype=np.int64),
            'pixel_count': 0,
        }

    flat = roi.flatten().astype(np.float64)

    mean     = float(np.sum(flat) / flat.size)
    variance = float(np.sum((flat - mean) ** 2) / flat.size)
    histogram = compute_histogram(roi)

    return {
        'mean':        mean,
        'variance':    variance,
        'histogram':   histogram,
        'pixel_count': int(roi.size),
    }