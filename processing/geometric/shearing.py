"""
Image shearing from scratch using inverse mapping and bilinear interpolation.
Applies horizontal and/or vertical shearing transforms without built-in functions.
"""

import numpy as np
from processing.interpolation.bilinear import bilinear_sample
from utils.image_utils import validate_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors


@wrap_errors
def shear_image(image: np.ndarray, shear_x: float, shear_y: float) -> np.ndarray:
	validate_grayscale(image)

	h, w = image.shape
	image = image.astype(np.float64)

	# Vectorized inverse mapping — replaces O(H*W) Python loop
	out_y, out_x = np.mgrid[0:h, 0:w].astype(np.float64)
	src_x = out_x - shear_x * out_y
	src_y = out_y - shear_y * out_x

	x0 = np.floor(src_x).astype(np.int32)
	y0 = np.floor(src_y).astype(np.int32)
	x1 = x0 + 1
	y1 = y0 + 1

	x0c = np.clip(x0, 0, w - 1)
	x1c = np.clip(x1, 0, w - 1)
	y0c = np.clip(y0, 0, h - 1)
	y1c = np.clip(y1, 0, h - 1)

	dx = src_x - np.floor(src_x)
	dy = src_y - np.floor(src_y)

	valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

	output = (
		image[y0c, x0c] * (1 - dy) * (1 - dx) +
		image[y0c, x1c] * (1 - dy) * dx +
		image[y1c, x0c] * dy * (1 - dx) +
		image[y1c, x1c] * dy * dx
	)
	output[~valid] = 0

	return normalize_to_uint8(output)