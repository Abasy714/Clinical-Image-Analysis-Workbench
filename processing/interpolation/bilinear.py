"""
Bilinear interpolation for image resizing and geometric transformations, implemented from scratch.
Computes weighted average of four neighboring pixels for sub-pixel accuracy.
"""

from __future__ import annotations
import numpy as np

# numpy — coordinate grids, vectorized weighted average computation

# FUNCTIONS
# def bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
#   — compute scale factors -> generate fractional coordinate grid
#   — extract four neighbors (floor/ceil) -> compute weights -> return blended result
# def bilinear_sample(image: np.ndarray, y: float, x: float) -> float:
#   — sample a single sub-pixel location using bilinear weighting
#   — used by geometric transforms that need per-pixel sampling

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate bilinear_resize and bilinear_sample
# Note: bilinear_sample is called per-pixel by rotation.py and shearing.py — keep it fast
# Note: caller (zoom.py or geometric modules) is responsible for normalize_to_uint8 on output


def bilinear_sample(image: np.ndarray, y: float, x: float) -> float:
	"""
	Sample a grayscale image at non-integer coordinates using bilinear interpolation.

	Args:
		image: 2D grayscale image (uint8 or float) as a numpy array.
		y: Source y coordinate (row) in float.
		x: Source x coordinate (column) in float.

	Returns:
		Interpolated value as float (no casting).
	"""
	height, width = image.shape

	x1 = int(np.floor(x))
	y1 = int(np.floor(y))
	x2 = x1 + 1
	y2 = y1 + 1

	# Compute fractional offsets using the original floor values.
	dx = x - x1
	dy = y - y1

	# Clamp indices to valid boundaries.
	x1c = max(0, min(x1, width - 1))
	x2c = max(0, min(x2, width - 1))
	y1c = max(0, min(y1, height - 1))
	y2c = max(0, min(y2, height - 1))

	q11 = image[y1c, x1c]
	q21 = image[y1c, x2c]
	q12 = image[y2c, x1c]
	q22 = image[y2c, x2c]

	# Bilinear blend of the four neighbors.
	return (
		q11 * (1.0 - dx) * (1.0 - dy)
		+ q21 * dx * (1.0 - dy)
		+ q12 * (1.0 - dx) * dy
		+ q22 * dx * dy
	)


def bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
	"""
	Resize a grayscale image using bilinear interpolation (no external libraries).

	Args:
		image: 2D grayscale image (uint8 or float) as a numpy array.
		new_h: Target height.
		new_w: Target width.

	Returns:
		Resized image as uint8.
	"""
	old_h, old_w = image.shape
	if new_h <= 0 or new_w <= 0:
		raise ValueError("new_h and new_w must be positive")

	scale_y = old_h / new_h
	scale_x = old_w / new_w

	output = np.empty((new_h, new_w), dtype=np.float64)

	for i in range(new_h):
		src_y = i * scale_y
		for j in range(new_w):
			src_x = j * scale_x
			output[i, j] = bilinear_sample(image, src_y, src_x)

	return np.clip(output, 0, 255).astype(np.uint8)