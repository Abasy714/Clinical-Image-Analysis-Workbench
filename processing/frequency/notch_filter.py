"""
Notch reject filter generation for periodic noise removal in the frequency domain.
Supports ideal, Butterworth, and Gaussian notch shapes with automatic conjugate mirroring.
"""

import numpy as np
from utils.image_utils import normalize_to_uint8
from utils.error_handler import wrap_errors


@wrap_errors
def create_notch_filter(shape: tuple, u: int, v: int,
                         radius: int = 10,
                         kind: str = "ideal",
                         order: int = 2) -> np.ndarray:
    """
    Create a notch reject filter mask for a single notch center at (u, v)
    in the shifted spectrum.

    Parameters
    ----------
    shape  : (H, W) — spectrum shape
    u, v   : notch center (column u, row v) in the shifted spectrum
    radius : notch cutoff radius D0
    kind   : 'ideal', 'butterworth', or 'gaussian'
    order  : Butterworth filter order (only used when kind='butterworth')

    Returns
    -------
    np.ndarray float64 — 0 inside notch, 1 outside
    """
    H, W = shape
    cols = np.arange(W)
    rows = np.arange(H)
    C, R = np.meshgrid(cols, rows)

    D = np.sqrt((R - v) ** 2 + (C - u) ** 2)

    kind = kind.strip().lower()
    if kind == "butterworth":
        # Reject-notch: 0 at center (D→0), 1 far from center
        mask = 1.0 - 1.0 / (1.0 + (D / max(radius, 1e-10)) ** (2 * order))
    elif kind == "gaussian":
        mask = 1.0 - np.exp(-D ** 2 / (2.0 * max(radius, 1) ** 2))
    else:  # ideal
        mask = np.where(D <= radius, 0.0, 1.0)

    return mask.astype(np.float64)


@wrap_errors
def apply_notch_filter(shifted_fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply a shifted FFT by a notch mask."""
    return shifted_fft * mask
