"""
Image rotation from scratch using inverse mapping and bilinear interpolation.
Rotates the image by a user-defined angle in degrees without using built-in rotation functions.
"""

# numpy — coordinate transformation, output array construction
# math — cos, sin for rotation matrix computation
# processing.interpolation.bilinear — bilinear_sample

# FUNCTIONS
# def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
#   — compute rotation matrix -> for each output pixel compute inverse-mapped source coordinate
#   — sample source image using bilinear_sample -> fill output array

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before inverse mapping loop
# normalize_to_uint8(output)          # call on the rotated array before returning
# @wrap_errors                        # decorate rotate_image
# Note: bilinear_sample will return float values — output array must be float64 during construction

import numpy as np
import math
from processing.interpolation.bilinear import bilinear_sample
from utils.image_utils import validate_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors


@wrap_errors
def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
	validate_grayscale(image)

	h, w = image.shape
	image = image.astype(np.float64)

	angle_rad = math.radians(angle_deg)
	cos_a = math.cos(angle_rad)
	sin_a = math.sin(angle_rad)
	cy, cx = h / 2.0, w / 2.0

	# Vectorized inverse mapping — replaces O(H*W) Python loop
	out_y, out_x = np.mgrid[0:h, 0:w].astype(np.float64)
	out_y_c = out_y - cy
	out_x_c = out_x - cx

	src_x_c =  cos_a * out_x_c + sin_a * out_y_c
	src_y_c = -sin_a * out_x_c + cos_a * out_y_c
	src_x = src_x_c + cx
	src_y = src_y_c + cy

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