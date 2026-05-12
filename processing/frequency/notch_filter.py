"""
Notch reject filter generation for periodic noise removal in the frequency domain.
Supports ideal, Butterworth, and Gaussian notch shapes.
"""

import numpy as np
from utils.error_handler import wrap_errors


@wrap_errors
def create_notch_filter(shape_hw: tuple, notch_centers: list,
                         radius: int = 10,
                         filter_shape: str = 'ideal',
                         order: int = 2) -> np.ndarray:
    """
    Create a combined notch reject filter mask for all notch centers.

    Parameters
    ----------
    shape_hw      : (H, W) — spectrum shape
    notch_centers : list of (row, col) tuples in the shifted spectrum
    radius        : notch cutoff radius D0
    filter_shape  : 'ideal', 'butterworth', or 'gaussian'
    order         : Butterworth filter order

    Returns
    -------
    np.ndarray float64 — product of per-notch masks (0 inside notch, 1 outside)
    """
    H, W = shape_hw
    cols = np.arange(W)
    rows = np.arange(H)
    C, R = np.meshgrid(cols, rows)
    kind = filter_shape.strip().lower()
    combined = np.ones((H, W), dtype=np.float64)

    for (v, u) in notch_centers:
        D = np.sqrt((R - v) ** 2 + (C - u) ** 2)
        if kind == 'butterworth':
            mask = 1.0 - 1.0 / (1.0 + (D / max(radius, 1e-10)) ** (2 * order))
        elif kind == 'gaussian':
            mask = 1.0 - np.exp(-D ** 2 / (2.0 * max(radius, 1) ** 2))
        else:
            mask = np.where(D <= radius, 0.0, 1.0)
        combined *= mask

    return combined


@wrap_errors
def apply_notch_filter(shifted_fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply a shifted FFT by a notch mask."""
    return shifted_fft * mask
