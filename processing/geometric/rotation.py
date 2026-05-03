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