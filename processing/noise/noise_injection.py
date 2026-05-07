# STATUS: STUB — Phase 2
"""
Synthetic noise injection from scratch: Gaussian, Uniform, Rayleigh, 
Exponential, and Salt & Pepper noise models.
"""

import numpy as np
from utils.image_utils import validate_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors


def _box_muller(size: tuple) -> np.ndarray:
    u1 = np.random.uniform(0.0, 1.0, size)
    u2 = np.random.uniform(0.0, 1.0, size)
    u1 = np.clip(u1, 1e-10, 1.0)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


@wrap_errors
def add_gaussian_noise(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    validate_grayscale(image)
    base = image.astype(np.float64)
    noise = mean + sigma * _box_muller(image.shape)
    noisy = base + noise
    return normalize_to_uint8(np.clip(noisy, 0, 255))


@wrap_errors
def add_uniform_noise(image: np.ndarray, low: float, high: float) -> np.ndarray:
    validate_grayscale(image)
    if low >= high:
        raise ValueError(f"low must be less than high. Got low={low}, high={high}")
    base = image.astype(np.float64)
    noise = np.random.uniform(low, high, image.shape)
    noisy = base + noise
    return normalize_to_uint8(np.clip(noisy, 0, 255))


@wrap_errors
def add_rayleigh_noise(image: np.ndarray, scale: float) -> np.ndarray:
    validate_grayscale(image)
    base = image.astype(np.float64)
    u = np.random.uniform(0.0, 1.0, image.shape)
    u = np.clip(u, 1e-10, 1.0)
    noise = scale * np.sqrt(-2.0 * np.log(u))
    noisy = base + noise
    return normalize_to_uint8(np.clip(noisy, 0, 255))


@wrap_errors
def add_exponential_noise(image: np.ndarray, scale: float) -> np.ndarray:
    validate_grayscale(image)
    base = image.astype(np.float64)
    u = np.random.uniform(0.0, 1.0, image.shape)
    u = np.clip(u, 1e-10, 1.0)
    noise = -scale * np.log(u)
    noisy = base + noise
    return normalize_to_uint8(np.clip(noisy, 0, 255))


@wrap_errors
def add_salt_and_pepper_noise(image: np.ndarray,
                               salt_prob: float,
                               pepper_prob: float) -> np.ndarray:
    validate_grayscale(image)
    if not (0.0 <= salt_prob <= 1.0 and 0.0 <= pepper_prob <= 1.0):
        raise ValueError("Probabilities must be between 0 and 1.")
    noisy = image.copy().astype(np.float64)
    random_map = np.random.uniform(0.0, 1.0, image.shape)
    noisy[random_map < pepper_prob] = 0
    noisy[random_map > 1.0 - salt_prob] = 255
    return normalize_to_uint8(noisy)