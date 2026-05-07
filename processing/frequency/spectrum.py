"""
Computes and formats the FFT magnitude and phase spectrum of an image for display.
Uses numpy's fft2 and fftshift — permitted built-ins for frequency domain work.
"""

import numpy as np
from utils.image_utils import validate_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors


@wrap_errors
def compute_spectrum(image: np.ndarray) -> tuple:
    validate_grayscale(image)
    base = image.astype(np.float64)

    fft         = np.fft.fft2(base)
    shifted_fft = np.fft.fftshift(fft)

    log_magnitude = np.log1p(np.abs(shifted_fft))
    phase         = np.angle(shifted_fft)

    return shifted_fft, log_magnitude, phase


@wrap_errors
def spectrum_to_display(shifted_fft: np.ndarray) -> np.ndarray:
    log_magnitude = np.log1p(np.abs(shifted_fft))
    return normalize_to_uint8(log_magnitude)


@wrap_errors
def phase_to_display(shifted_fft: np.ndarray) -> np.ndarray:
    phase = np.angle(shifted_fft)
    phase_shifted = phase + np.pi
    return normalize_to_uint8(phase_shifted)


@wrap_errors
def inverse_spectrum(shifted_fft: np.ndarray) -> np.ndarray:
    unshifted = np.fft.ifftshift(shifted_fft)
    spatial   = np.fft.ifft2(unshifted)
    real      = np.real(spatial)
    return normalize_to_uint8(np.clip(real, 0, 255))