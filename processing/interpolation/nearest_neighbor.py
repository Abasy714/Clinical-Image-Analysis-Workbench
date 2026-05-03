"""
Nearest-neighbor interpolation for image resizing, implemented entirely from scratch.
Maps each output pixel to the closest input pixel using floor rounding.
"""

# numpy — array operations for coordinate mapping and pixel assignment

# FUNCTIONS
# def nearest_neighbor_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
#   — compute scale factors -> generate output coordinate grid
#   — map to nearest input coords using np.floor -> index and return result

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate nearest_neighbor_resize — invalid dimensions will crash
# Note: no image_utils needed here — input/output are raw numpy arrays, caller handles normalization