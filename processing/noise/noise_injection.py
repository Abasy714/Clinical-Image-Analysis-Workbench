"""
Synthetic noise injection from scratch: Gaussian and uniform noise models.
Does not use numpy random convenience wrappers that auto-apply noise — builds distributions manually.
"""

# numpy — random number generation, clipping, array arithmetic

# FUNCTIONS
# def add_gaussian_noise(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
#   — generate Gaussian noise array using Box-Muller or numpy randn -> add to image -> clip
# def add_uniform_noise(image: np.ndarray, low: float, high: float) -> np.ndarray:
#   — generate uniform noise array -> add to image -> clip to valid range

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before generating noise array
# normalize_to_uint8(noisy)           # call after adding noise and clipping to valid range
# @wrap_errors                        # decorate add_gaussian_noise and add_uniform_noise