"""
Fourier-domain template matching via normalized cross-correlation.
Locates a user-cropped template within the full image and returns the best-match bounding box.
"""

# numpy — fft2, ifft2, fftshift, conjugate, argmax, unravel_index

# FUNCTIONS
# def fourier_cross_correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
#   — zero-pad template to image size -> fft2 both -> multiply image FFT by conjugate of template FFT
#   — ifft2 -> take real part -> return correlation map
# def find_best_match(correlation_map: np.ndarray, template_shape: tuple) -> tuple[int, int, int, int]:
#   — find argmax of correlation map -> compute bounding box (x, y, w, h) centered on peak

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before fourier_cross_correlate
# validate_grayscale(template)        # call on template too — must also be single channel
# normalize_to_uint8(corr_display)    # call if you want to visualize the correlation map
# @wrap_errors                        # decorate fourier_cross_correlate and find_best_match