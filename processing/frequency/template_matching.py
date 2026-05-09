"""
Fourier-domain template matching via cross-correlation.
Locates a user-cropped template within the full image and returns the best-match bounding box.
"""

import numpy as np
from utils.image_utils import validate_grayscale
from utils.error_handler import wrap_errors


@wrap_errors
def fourier_cross_correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Compute Fourier-domain cross-correlation between image and template.

    The template is zero-padded to the image shape, then:
      fft2(image) * conj(fft2(template_padded)) → ifft2 → fftshift → abs

    Returns
    -------
    np.ndarray float64 correlation map, same shape as image.
    Peak location gives the best-match position.
    """
    validate_grayscale(image)
    H, W = image.shape[:2]

    tmpl_padded = np.zeros((H, W), dtype=np.float64)
    th = min(template.shape[0], H)
    tw = min(template.shape[1], W)
    tmpl_padded[:th, :tw] = template[:th, :tw].astype(np.float64)

    F_image = np.fft.fft2(image.astype(np.float64))
    F_tmpl  = np.fft.fft2(tmpl_padded)

    cross = F_image * np.conj(F_tmpl)
    corr  = np.abs(np.fft.fftshift(np.fft.ifft2(cross)))

    return corr


@wrap_errors
def find_best_match(correlation_map: np.ndarray) -> tuple:
    """Return (row, col) of the peak in the correlation map."""
    return np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
