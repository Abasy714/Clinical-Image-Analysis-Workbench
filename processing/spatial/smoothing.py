"""
Smoothing filters: average (box) filter and Gaussian filter, both implemented from scratch.
Uses the shared convolution engine — no scipy or OpenCV filtering functions permitted.
"""

# numpy — kernel construction, array operations
# processing.spatial.convolution — convolve2d

# FUNCTIONS
# def average_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
#   — build uniform kernel of ones / kernel_size^2 -> call convolve2d
# def gaussian_filter(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
#   — build 2D Gaussian kernel from scratch using sigma -> normalize -> call convolve2d
# def _build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
#   — compute Gaussian weights using exp(-(x^2+y^2)/(2*sigma^2)) -> normalize to sum=1

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at the top of average_filter and gaussian_filter
# normalize_to_uint8(result)          # call on convolve2d output before returning
# @wrap_errors                        # decorate both filter functions