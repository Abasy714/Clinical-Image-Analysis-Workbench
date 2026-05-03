"""
Edge detection using Sobel and Prewitt operators, implemented from scratch.
Returns horizontal edges, vertical edges, and combined gradient magnitude.
"""

# numpy — kernel definitions, magnitude computation (sqrt of sum of squares)
# processing.spatial.convolution — convolve2d

# CONSTANTS
# SOBEL_X, SOBEL_Y — 3x3 Sobel kernels
# PREWITT_X, PREWITT_Y — 3x3 Prewitt kernels

# FUNCTIONS
# def sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#   — apply Sobel X and Y kernels -> compute magnitude -> return (Gx, Gy, magnitude)
# def prewitt(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#   — apply Prewitt X and Y kernels -> return (Gx, Gy, magnitude)
# def combined_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
#   — compute sqrt(Gx^2 + Gy^2) -> clip to valid range

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at the top of sobel and prewitt
# normalize_to_uint8(Gx)             # call on horizontal result before returning
# normalize_to_uint8(Gy)             # call on vertical result before returning
# normalize_to_uint8(magnitude)       # call on combined magnitude before returning
# @wrap_errors                        # decorate sobel, prewitt, combined_magnitude