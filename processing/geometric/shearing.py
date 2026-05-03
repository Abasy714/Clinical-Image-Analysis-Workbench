"""
Image shearing from scratch using inverse mapping and bilinear interpolation.
Applies horizontal and/or vertical shearing transforms without built-in functions.
"""

# numpy — coordinate transformation, output array construction
# processing.interpolation.bilinear — bilinear_sample

# FUNCTIONS
# def shear_image(image: np.ndarray, shear_x: float, shear_y: float) -> np.ndarray:
#   — build shear transformation matrix -> inverse-map output coords to source
#   — sample using bilinear_sample -> return sheared image

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before inverse mapping loop
# normalize_to_uint8(output)          # call on sheared array before returning
# @wrap_errors                        # decorate shear_image