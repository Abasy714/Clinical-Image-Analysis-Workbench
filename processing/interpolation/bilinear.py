"""
Bilinear interpolation for image resizing and geometric transformations, implemented from scratch.
Computes weighted average of four neighboring pixels for sub-pixel accuracy.
"""

# numpy — coordinate grids, vectorized weighted average computation

# FUNCTIONS
# def bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
#   — compute scale factors -> generate fractional coordinate grid
#   — extract four neighbors (floor/ceil) -> compute weights -> return blended result
# def bilinear_sample(image: np.ndarray, y: float, x: float) -> float:
#   — sample a single sub-pixel location using bilinear weighting
#   — used by geometric transforms that need per-pixel sampling

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate bilinear_resize and bilinear_sample
# Note: bilinear_sample is called per-pixel by rotation.py and shearing.py — keep it fast
# Note: caller (zoom.py or geometric modules) is responsible for normalize_to_uint8 on output