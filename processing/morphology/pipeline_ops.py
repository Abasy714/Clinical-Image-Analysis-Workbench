"""GUI-facing morphology operations for grayscale image inputs."""

from __future__ import annotations

import numpy as np

from utils.image_utils import normalize_to_uint8, to_grayscale, validate_grayscale

from .advanced import black_top_hat, morphological_gradient, white_top_hat
from .boundary_extraction import extract_boundary_vectorized
from .erosion_dilation import dilate, erode
from .opening_closing import closing, opening
from .structuring_element import get_se


def _prepare_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("No image loaded.")
    gray = to_grayscale(image)
    validate_grayscale(gray)
    return normalize_to_uint8(gray)


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return 128
    prob = hist / total
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    w0 = cum_prob
    w1 = 1.0 - cum_prob
    mu0 = np.where(w0 > 0, cum_mean / w0, 0.0)
    mu1 = np.where(w1 > 0, (cum_mean[-1] - cum_mean) / w1, 0.0)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    return int(np.argmax(sigma_b))


def binarize_for_morphology(
    image: np.ndarray,
    threshold: int | None = 128,
) -> np.ndarray:
    gray = _prepare_gray(image)
    t = _otsu_threshold(gray) if threshold is None else int(np.clip(threshold, 0, 255))
    return gray > t


def apply_morphology(
    image: np.ndarray,
    operation: str,
    se_shape: str = "square",
    se_size: int = 3,
    threshold: int | None = 128,
) -> np.ndarray:
    binary = binarize_for_morphology(image, threshold)
    se = get_se(se_shape, int(se_size))
    op = operation.strip().lower()

    if op == "erode":
        result = erode(binary, se)
    elif op == "dilate":
        result = dilate(binary, se)
    elif op == "open":
        result = opening(binary, se)
    elif op == "close":
        result = closing(binary, se)
    elif op == "boundary":
        result = extract_boundary_vectorized(binary, se)
    elif op == "gradient":
        result = morphological_gradient(binary, se)
    elif op in {"white_top_hat", "white top hat"}:
        result = white_top_hat(binary, se)
    elif op in {"black_top_hat", "black top hat"}:
        result = black_top_hat(binary, se)
    else:
        raise ValueError(f"Unknown morphology operation: {operation}")

    return normalize_to_uint8(result.astype(np.uint8) * 255)
