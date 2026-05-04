"""
Bilinear interpolation for image resizing and geometric transformations, implemented from scratch.
Computes weighted average of four neighboring pixels for sub-pixel accuracy.
"""

from __future__ import annotations
import numpy as np

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
	h, w = image.shape
	if new_h <= 0 or new_w <= 0:
		raise ValueError("new_h and new_w must be positive")
	image = image.astype(np.float64)

	# Vectorized coordinate grid — replaces O(H*W) Python loop
	row_coords = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
	col_coords = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5

	r0 = np.floor(row_coords).astype(np.int32)
	c0 = np.floor(col_coords).astype(np.int32)
	r1 = r0 + 1
	c1 = c0 + 1

	r0 = np.clip(r0, 0, h - 1)
	r1 = np.clip(r1, 0, h - 1)
	c0 = np.clip(c0, 0, w - 1)
	c1 = np.clip(c1, 0, w - 1)

	dr = (row_coords - np.floor(row_coords))[:, np.newaxis]  # (new_h, 1)
	dc = (col_coords - np.floor(col_coords))[np.newaxis, :]  # (1, new_w)

	top_left     = image[np.ix_(r0, c0)]
	top_right    = image[np.ix_(r0, c1)]
	bottom_left  = image[np.ix_(r1, c0)]
	bottom_right = image[np.ix_(r1, c1)]

	output = (
		top_left     * (1 - dr) * (1 - dc) +
		top_right    * (1 - dr) * dc +
		bottom_left  * dr       * (1 - dc) +
		bottom_right * dr       * dc
	)

	return np.clip(output, 0, 255).astype(np.uint8)