"""
Computes and formats the FFT magnitude spectrum of an image for display.
Uses numpy's fft2 and fftshift — permitted built-ins for frequency domain work.
"""

# numpy — fft2, ifft2, fftshift, ifftshift, log scaling, array ops

# FUNCTIONS
# def compute_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#   — apply fft2 -> fftshift -> return (shifted_fft, log_magnitude_display)
# def spectrum_to_display(shifted_fft: np.ndarray) -> np.ndarray:
#   — compute log(1 + |F|) -> normalize to 0-255 for display
# def inverse_spectrum(shifted_fft: np.ndarray) -> np.ndarray:
#   — ifftshift -> ifft2 -> take real part -> clip to valid range

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at the top of compute_spectrum
# normalize_to_uint8(display)         # call on the log-magnitude array inside spectrum_to_display
# normalize_to_uint8(spatial)         # call on the real part of ifft2 result in inverse_spectrum
# @wrap_errors                        # decorate compute_spectrum and inverse_spectrum