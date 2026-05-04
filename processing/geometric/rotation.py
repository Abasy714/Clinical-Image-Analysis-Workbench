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

	theta = math.radians(angle_deg)
	h, w = image.shape
	cx = w / 2.0
	cy = h / 2.0

	output = np.zeros((h, w), dtype=np.float64)

	cos_t = math.cos(theta)
	sin_t = math.sin(theta)

	for i in range(h):
		for j in range(w):
			# shift to center before inverse mapping
			x = j - cx
			y = i - cy

			# inverse mapping from destination to source
			src_x = x * cos_t + y * sin_t
			src_y = -x * sin_t + y * cos_t

			# shift back to image coordinates
			src_x += cx
			src_y += cy

			if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
				# bilinear interpolation at non-integer coordinates
				output[i, j] = bilinear_sample(image, src_y, src_x)

	return normalize_to_uint8(output)