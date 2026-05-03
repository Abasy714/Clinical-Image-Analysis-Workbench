"""
2D convolution implemented entirely from scratch using nested loops and numpy.
This is the core engine used by all spatial filters and the Harris corner detector.
"""

# numpy — array operations, padding, kernel application

# CONSTANTS
# PADDING_MODES = ['zero', 'reflect', 'replicate']

# FUNCTIONS
# def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = 'zero') -> np.ndarray:
#   — pad image -> slide kernel over every pixel -> accumulate weighted sum -> return result
#   — must handle both odd and even kernel sizes
# def _pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str) -> np.ndarray:
#   — apply zero / reflect / replicate padding based on mode

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate convolve2d — mismatched kernel/image shapes will crash
# Note: convolve2d returns float64 — callers must call normalize_to_uint8 on the result
# Note: no other utils needed — this is a pure math function, keep dependencies minimal
# Note: this function is called by smoothing.py, edge_detection.py, and harris_detector.py