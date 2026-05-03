"""
Non-linear median filter implemented entirely from scratch.
Replaces each pixel with the median of its local neighborhood — effective for salt-and-pepper noise.
"""

# numpy — neighborhood extraction, sorting, median computation

# FUNCTIONS
# def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
#   — pad image -> for each pixel, extract neighborhood -> sort -> assign median value
#   — must handle both grayscale and edge pixels correctly

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before padding and neighborhood loop
# normalize_to_uint8(result)          # call on output array before returning
# @wrap_errors                        # decorate median_filter