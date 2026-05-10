# STATUS: IMPLEMENTED
"""
Frequency-domain high-pass filters (Ideal, Butterworth, Gaussian).
All masks are generated from scratch and returned as float64 in [0, 1].
"""

import numpy as np
from utils.error_handler import wrap_errors

SUPPORTED_HIGHPASS_KINDS = ("ideal", "butterworth", "gaussian")
SUPPORTED_LOWPASS_KINDS = ("ideal", "butterworth", "gaussian")
SUPPORTED_BANDREJECT_KINDS = ("ideal", "butterworth", "gaussian")
SUPPORTED_BANDPASS_KINDS = ("ideal", "butterworth", "gaussian")


def _distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Return distance-to-center grid for a shifted frequency plane."""
    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape, got {shape}")
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid shape dimensions: {shape}")

    cy = h // 2
    cx = w // 2
    y = np.arange(h, dtype=np.float64) - cy
    x = np.arange(w, dtype=np.float64) - cx
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx * xx + yy * yy)

"---------------------------------------------------HP FILTERS---------------------------------------------------"

@wrap_errors
def ideal_high_pass_filter(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Ideal high-pass filter: pass frequencies where D > D0."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    d = _distance_grid(shape)
    return (d > float(cutoff)).astype(np.float64)


@wrap_errors
def butterworth_high_pass_filter(shape: tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth high-pass filter."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    d = _distance_grid(shape)
    eps = 1e-12
    ratio = float(cutoff) / np.maximum(d, eps)
    return 1.0 / (1.0 + np.power(ratio, 2 * int(order)))


@wrap_errors
def gaussian_high_pass_filter(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Gaussian high-pass filter."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")

    d = _distance_grid(shape)
    return 1.0 - np.exp(-(d * d) / (2.0 * float(cutoff) * float(cutoff)))


@wrap_errors
def create_high_pass_filter(
    shape: tuple[int, int],
    cutoff: float,
    kind: str = "ideal",
    order: int = 2,
) -> np.ndarray:
    """Dispatch high-pass filter creation by kind."""
    kind_norm = kind.strip().lower()
    if kind_norm not in SUPPORTED_HIGHPASS_KINDS:
        raise ValueError(f"Unsupported high-pass kind '{kind}'. Choose from {SUPPORTED_HIGHPASS_KINDS}.")

    if kind_norm == "ideal":
        return ideal_high_pass_filter(shape, cutoff)
    if kind_norm == "butterworth":
        return butterworth_high_pass_filter(shape, cutoff, order)
    return gaussian_high_pass_filter(shape, cutoff)


@wrap_errors
def apply_frequency_filter(shifted_fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a frequency mask to a shifted FFT."""
    if shifted_fft is None or mask is None:
        raise ValueError("shifted_fft and mask must not be None.")
    if shifted_fft.shape != mask.shape:
        raise ValueError(f"Shape mismatch: shifted_fft {shifted_fft.shape} vs mask {mask.shape}")
    return shifted_fft * mask

"---------------------------------------------------LP FILTERS---------------------------------------------------"

@wrap_errors
def ideal_low_pass_filter(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Ideal low-pass filter: pass frequencies where D <= D0."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    d = _distance_grid(shape)
    return (d <= float(cutoff)).astype(np.float64)


@wrap_errors
def butterworth_low_pass_filter(shape: tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth low-pass filter."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    d = _distance_grid(shape)
    eps = 1e-12
    ratio = np.maximum(d, eps) / float(cutoff)
    return 1.0 / (1.0 + np.power(ratio, 2 * int(order)))


@wrap_errors
def gaussian_low_pass_filter(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Gaussian low-pass filter."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")

    d = _distance_grid(shape)
    return np.exp(-(d * d) / (2.0 * float(cutoff) * float(cutoff)))


@wrap_errors
def create_low_pass_filter(
    shape: tuple[int, int],
    cutoff: float,
    kind: str = "ideal",
    order: int = 2,
) -> np.ndarray:
    """Dispatch low-pass filter creation by kind."""
    kind_norm = kind.strip().lower()
    if kind_norm not in SUPPORTED_LOWPASS_KINDS:
        raise ValueError(f"Unsupported low-pass kind '{kind}'. Choose from {SUPPORTED_LOWPASS_KINDS}.")

    if kind_norm == "ideal":
        return ideal_low_pass_filter(shape, cutoff)
    if kind_norm == "butterworth":
        return butterworth_low_pass_filter(shape, cutoff, order)
    return gaussian_low_pass_filter(shape, cutoff)

"---------------------------------------------------BR FILTERS---------------------------------------------------"

@wrap_errors
def ideal_band_reject_filter(shape: tuple[int, int], center: float, bandwidth: float) -> np.ndarray:
    """Ideal band-reject filter."""
    if center <= 0:
        raise ValueError(f"center must be > 0, got {center}")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}")

    d = _distance_grid(shape)
    half_bw = float(bandwidth) / 2.0
    lower = float(center) - half_bw
    upper = float(center) + half_bw
    # Reject the ring region [lower, upper] and pass everything else.
    return np.where((d >= lower) & (d <= upper), 0.0, 1.0).astype(np.float64)


@wrap_errors
def butterworth_band_reject_filter(
    shape: tuple[int, int],
    center: float,
    bandwidth: float,
    order: int = 2,
) -> np.ndarray:
    """Butterworth band-reject filter."""
    if center <= 0:
        raise ValueError(f"center must be > 0, got {center}")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    d = _distance_grid(shape)
    eps = 1e-12
    denom = np.maximum(np.abs((d * d) - (float(center) * float(center))), eps)
    ratio = (d * float(bandwidth)) / denom
    return 1.0 / (1.0 + np.power(ratio, 2 * int(order)))


@wrap_errors
def gaussian_band_reject_filter(shape: tuple[int, int], center: float, bandwidth: float) -> np.ndarray:
    """Gaussian band-reject filter."""
    if center <= 0:
        raise ValueError(f"center must be > 0, got {center}")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}")

    d = _distance_grid(shape)
    eps = 1e-12
    numer = (d * d) - (float(center) * float(center))
    denom = np.maximum(d * float(bandwidth), eps)
    return 1.0 - np.exp(-(numer * numer) / (2.0 * denom * denom))


@wrap_errors
def create_band_reject_filter(
    shape: tuple[int, int],
    center: float,
    bandwidth: float,
    kind: str = "ideal",
    order: int = 2,
) -> np.ndarray:
    """Dispatch band-reject filter creation by kind."""
    kind_norm = kind.strip().lower()
    if kind_norm not in SUPPORTED_BANDREJECT_KINDS:
        raise ValueError(f"Unsupported band-reject kind '{kind}'. Choose from {SUPPORTED_BANDREJECT_KINDS}.")

    if kind_norm == "ideal":
        return ideal_band_reject_filter(shape, center, bandwidth)
    if kind_norm == "butterworth":
        return butterworth_band_reject_filter(shape, center, bandwidth, order)
    return gaussian_band_reject_filter(shape, center, bandwidth)


@wrap_errors
def ideal_band_pass_filter(shape: tuple[int, int], center: float, bandwidth: float) -> np.ndarray:
    """Ideal band-pass filter."""
    brf = ideal_band_reject_filter(shape, center, bandwidth)
    return 1.0 - brf


@wrap_errors
def butterworth_band_pass_filter(
    shape: tuple[int, int],
    center: float,
    bandwidth: float,
    order: int = 2,
) -> np.ndarray:
    """Butterworth band-pass filter."""
    brf = butterworth_band_reject_filter(shape, center, bandwidth, order)
    return 1.0 - brf


@wrap_errors
def gaussian_band_pass_filter(shape: tuple[int, int], center: float, bandwidth: float) -> np.ndarray:
    """Gaussian band-pass filter."""
    brf = gaussian_band_reject_filter(shape, center, bandwidth)
    return 1.0 - brf


@wrap_errors
def create_band_pass_filter(
    shape: tuple[int, int],
    center: float,
    bandwidth: float,
    kind: str = "ideal",
    order: int = 2,
) -> np.ndarray:
    """Dispatch band-pass filter creation by kind."""
    kind_norm = kind.strip().lower()
    if kind_norm not in SUPPORTED_BANDPASS_KINDS:
        raise ValueError(f"Unsupported band-pass kind '{kind}'. Choose from {SUPPORTED_BANDPASS_KINDS}.")

    if kind_norm == "ideal":
        return ideal_band_pass_filter(shape, center, bandwidth)
    if kind_norm == "butterworth":
        return butterworth_band_pass_filter(shape, center, bandwidth, order)
    return gaussian_band_pass_filter(shape, center, bandwidth)
