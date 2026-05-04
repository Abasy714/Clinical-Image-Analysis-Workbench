"""
Local (block-based) histogram equalization implemented from scratch.
Divides the image into blocks of user-defined size and equalizes each independently.
"""

import numpy as np
from utils.image_utils import validate_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors
from processing.histogram.histogram_utils import compute_histogram, compute_cdf


def _equalize_block(block: np.ndarray, clip_limit: float = 3.0, num_levels: int = 256) -> np.ndarray:
    hist = compute_histogram(block, bins=num_levels)

    avg_count = block.size / num_levels
    max_count = max(1, int(clip_limit * avg_count))

    excess = 0
    for i in range(num_levels):
        if hist[i] > max_count:
            excess += hist[i] - max_count
            hist[i] = max_count

    redistribute = excess // num_levels
    remainder = excess % num_levels
    hist = hist + redistribute
    hist[:remainder] += 1

    cdf = compute_cdf(hist)
    lut = np.round(cdf * (num_levels - 1)).astype(np.uint8)
    equalized = lut[block.astype(np.int32)]
    return equalized

@wrap_errors
def local_histogram_equalization(image: np.ndarray, block_size: int, clip_limit: float = 3.0) -> np.ndarray:
    validate_grayscale(image)

    if block_size < 2:
        raise ValueError(f"block_size must be at least 2, got {block_size}.")
    if block_size > image.shape[0] or block_size > image.shape[1]:
        raise ValueError(f"block_size ({block_size}) exceeds image dimensions {image.shape}.")

    image = normalize_to_uint8(image)
    height, width = image.shape
    output = np.zeros_like(image)

    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            block = image[row : row + block_size, col : col + block_size]
            equalized_block = _equalize_block(block, clip_limit=clip_limit)
            output[row : row + block_size, col : col + block_size] = equalized_block

    return normalize_to_uint8(output)